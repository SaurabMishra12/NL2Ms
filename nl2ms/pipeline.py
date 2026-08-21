"""Experiment orchestration: phase sequencing, checkpointing, resume.

The controlling idea is that the notebook is *restartable at any point*. Every
phase writes its outputs before the next begins, every per-sample unit of work
is recorded in the manifest, and every phase begins by asking the manifest
what has already been done. Losing a Kaggle session costs the work in flight,
never the work completed.

Phase completion is itself checkpointed (``checkpoints/phase_<name>.json``),
so a phase that produces a single aggregate artefact is skipped wholesale on
resume rather than recomputed.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import critical as crit
from . import extraction as ext
from . import geometry as geo
from . import interventions as interv
from . import jspace as jsp
from . import signals as sig
from . import stats as st
from .config import ExperimentConfig
from .datasets_build import (ANSWER_UNDEFINED, Sample, build_dataset,
                             dataset_summary, load_samples, save_samples,
                             stratified_subset)
from .env import (capture_environment, build_max_memory, detect_hardware,
                  print_gpu_report, save_environment, summarise_environment)
from .modeling import ModelWrapper, build_answer_spec, load_model
from .registry import save_registry, write_lineage
from .runtime import (BudgetExceeded, Heartbeat, RuntimeController,
                      SAFE_STOP_MESSAGE, free_gpu_memory, save_runtime_report)
from .storage import (CHECK_RECOMPUTE, CHECK_RESUME, CHECK_SKIP,
                      STATUS_COMPLETE, STATUS_CORRUPTED, STATUS_FAILED,
                      ErrorLog, ExperimentPaths, Manifest, ManifestRecord,
                      ShardReader, ShardWriter, estimate_storage,
                      file_checksum, load_json, make_backup_archive, read_jsonl,
                      save_json, save_parquet)

PHASE_GENERATION = "generation"
PHASE_ANALYSIS = "analysis"
PHASE_JSPACE = "jspace"
PHASE_INTERVENTION = "intervention"


class Experiment:
    """Holds all run state and exposes one method per protocol phase."""

    def __init__(self, config: ExperimentConfig, *, repo_dir: str = ".") -> None:
        self.config = config
        self.paths = ExperimentPaths(config.output_root).ensure()
        self.repo_dir = repo_dir
        self.manifest = Manifest(self.paths.manifest_jsonl)
        self.errors = ErrorLog(self.paths.errors_jsonl)
        self.controller = RuntimeController(
            config.runtime.max_runtime_hours,
            reserve_minutes=config.runtime.reserve_minutes_for_finalisation)
        self.heartbeat = Heartbeat(self.paths.heartbeat,
                                   interval=config.runtime.heartbeat_seconds,
                                   controller=self.controller,
                                   experiment_root=self.paths.root)
        self.environment: Dict[str, Any] = {}
        self.model: Optional[ModelWrapper] = None
        self.samples: List[Sample] = []
        self.generations: Dict[str, Any] = {}
        self.results: Dict[str, Any] = {}
        self.phases_completed: Dict[str, Any] = {}
        self.statistical_tests: List[Dict[str, Any]] = []
        self.figures: List[Dict[str, Any]] = []
        self.missing_analyses: List[str] = []

    # ==================================================================
    # Phase bookkeeping
    # ==================================================================
    def _phase_checkpoint(self, name: str) -> Path:
        return self.paths.checkpoints / f"phase_{name}.json"

    def phase_done(self, name: str) -> bool:
        """Has this phase already completed under the *same* configuration?

        The config hash guard matters: resuming with changed measurement
        settings must recompute, not silently reuse results produced under
        different definitions.
        """
        p = self._phase_checkpoint(name)
        if not p.exists():
            return False
        try:
            rec = load_json(p)
        except Exception:
            return False
        if rec.get("config_hash") != self.config.config_hash():
            print(f"  phase '{name}' was completed under a different "
                  f"configuration -- recomputing")
            return False
        return rec.get("status") == STATUS_COMPLETE

    def mark_phase(self, name: str, **extra: Any) -> None:
        payload = {
            "phase": name, "status": STATUS_COMPLETE, "timestamp": time.time(),
            "config_hash": self.config.config_hash(),
            "elapsed_seconds": self.controller.elapsed,
        }
        payload.update(extra)
        save_json(self._phase_checkpoint(name), payload)
        self.phases_completed[name] = payload

    def _log(self, message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        print(f"[{stamp}] {message}", flush=True)

    # ==================================================================
    # PHASE 0 -- environment and reproducibility
    # ==================================================================
    def phase0_environment(self, *, strict_determinism: bool = False) -> Dict[str, Any]:
        self._log("PHASE 0: environment and reproducibility")
        self.environment = capture_environment(self.config.seed,
                                               repo_dir=self.repo_dir,
                                               strict_determinism=strict_determinism)
        save_environment(self.environment, self.paths.config / "environment.json")
        self.config.save(self.paths.config / "experiment_config.json")
        save_registry(self.paths.config / "signal_registry.json")
        print(summarise_environment(self.environment))
        missing = self.environment["missing_required_packages"]
        if missing:
            raise RuntimeError(f"required packages missing: {missing}")
        self.heartbeat.beat(force=True, current_phase="phase0_environment")
        self.mark_phase("phase0_environment")
        return self.environment

    # ==================================================================
    # PHASE 1 -- dataset preparation
    # ==================================================================
    def phase1_dataset(self, *, pilot: bool = False) -> List[Sample]:
        self._log("PHASE 1: dataset preparation")
        path = self.paths.datasets / ("pilot_samples.jsonl" if pilot
                                      else "samples.jsonl")
        if path.exists() and self.phase_done("phase1_dataset" + ("_pilot" if pilot else "")):
            self.samples = load_samples(path)
            self._log(f"  reusing {len(self.samples)} samples from {path.name}")
            return self.samples

        samples, provenance = build_dataset(
            self.config.datasets, self.config.model.name,
            self.config.model.revision, self.config.prompt_template_id)
        if pilot:
            samples = stratified_subset(samples, self.config.pilot_n_samples,
                                        seed=self.config.seed)
        self.samples = samples
        save_samples(samples, path)
        summary = dataset_summary(samples)
        save_json(self.paths.datasets / ("pilot_provenance.json" if pilot
                                         else "provenance.json"),
                  {"provenance": provenance, "summary": summary})
        for w in provenance.get("warnings", []):
            self._log(f"  WARNING {w}")
        self._log(f"  {summary['n_samples']} samples: {summary['by_dataset']}")
        self._log(f"  answer specs: {summary['by_answer_spec']}")
        self.mark_phase("phase1_dataset" + ("_pilot" if pilot else ""),
                        n_samples=len(samples), summary=summary)
        return samples

    # ==================================================================
    # Model loading
    # ==================================================================
    def load(self) -> ModelWrapper:
        if self.model is not None:
            return self.model
        self._log(f"loading model {self.config.model.name}")
        hardware = detect_hardware()
        max_memory = (build_max_memory(hardware)
                      if self.config.model.device_map == "auto" else
                      self.config.model.max_memory)
        self.model = load_model(self.config.model, max_memory=max_memory)
        print_gpu_report()
        verification = self.model.verify_capture()
        save_json(self.paths.config / "model_verification.json",
                  {"architecture": self.model.arch.to_dict(),
                   "load_info": self.model.load_info,
                   "capture_verification": verification})
        if not verification.get("pathway_verified"):
            # Not fatal -- some architectures legitimately lack a final norm --
            # but every logit-lens number downstream inherits this caveat, so
            # it is surfaced loudly rather than buried in a JSON file.
            self._log("  WARNING: logit-lens pathway did NOT reproduce the "
                      "model's own logits. Lens results are unreliable for "
                      "this architecture; see config/model_verification.json")
        else:
            self._log(f"  logit-lens pathway verified "
                      f"(relative diff "
                      f"{verification.get('relative_logit_diff', float('nan')):.2e})")
        return self.model

    # ==================================================================
    # Storage / runtime planning (protocol sections 37, 58)
    # ==================================================================
    def plan(self, n_samples: int) -> Dict[str, Any]:
        self._log("PLANNING: storage and runtime estimate")
        model = self.load()
        arch = model.arch
        flags = self.config.effective_flags()
        ext_cfg = self.config.extraction
        n_positions = min(ext_cfg.max_generated_positions + 3,
                          self.config.generation.max_new_tokens + 3)
        seq_len = 256
        estimate = estimate_storage(
            n_samples=n_samples, n_layers=arch.n_layers,
            hidden_size=arch.hidden_size,
            n_positions=3 if not flags["save_all_token_positions"] else n_positions,
            n_heads=arch.n_heads, seq_len=seq_len,
            top_k=ext_cfg.logit_lens_top_k, flags=flags,
            shard_size=ext_cfg.shard_size,
            hidden_bytes=2, output_root=self.paths.root,
            min_free_gb=self.config.runtime.min_free_disk_gb,
            n_full_attention_samples=ext_cfg.save_full_attention_for_n_samples,
            vocab_size=arch.vocab_size,
        )
        print(f"  estimated GB        : {estimate.estimated_gb:.2f}")
        print(f"  available GB        : {estimate.available_gb:.2f}")
        print(f"  estimated shards    : {estimate.n_shards}")
        print(f"  per sample MB       : {estimate.per_sample_mb:.2f}")
        for k, v in sorted(estimate.breakdown.items()):
            if v > 0.001:
                print(f"    - {k:<24} {v:7.3f} GB")
        print(f"  {estimate.detail}")

        plan_rows = [
            {"phase": PHASE_GENERATION, "samples": n_samples,
             "estimated_storage_gb": 0.01,
             "estimated_gpu_gb": _weight_gb(arch, self.config)},
            {"phase": PHASE_ANALYSIS, "samples": n_samples,
             "estimated_storage_gb": estimate.estimated_gb,
             "estimated_gpu_gb": _weight_gb(arch, self.config) + 2.0},
            {"phase": PHASE_JSPACE,
             "samples": min(n_samples, self.config.jspace.max_samples),
             "estimated_storage_gb": 0.01,
             "estimated_gpu_gb": _weight_gb(arch, self.config) + 3.0},
            {"phase": PHASE_INTERVENTION,
             "samples": min(n_samples, self.config.interventions.max_samples),
             "estimated_storage_gb": 0.02,
             "estimated_gpu_gb": _weight_gb(arch, self.config) + 3.0},
        ]
        from .runtime import benchmark_plan
        rows = benchmark_plan(self.controller, plan_rows)
        print()
        print(self.controller.planning_table(rows))
        payload = {"storage": estimate.__dict__, "plan": rows,
                   "n_samples": n_samples}
        save_json(self.paths.logs / "plan.json", payload)
        if not estimate.sufficient:
            raise RuntimeError(
                f"insufficient disk: {estimate.detail}. Lower storage_level, "
                f"reduce max_generated_positions, or reduce sample counts.")
        return payload

    # ==================================================================
    # PHASE 2 -- baseline generation
    # ==================================================================
    def phase2_generation(self, samples: Optional[Sequence[Sample]] = None
                          ) -> Dict[str, Any]:
        self._log("PHASE 2: baseline generation")
        samples = list(samples if samples is not None else self.samples)
        model = self.load()
        out_path = self.paths.generations / "generations.jsonl"

        existing = {r["sample_id"]: r for r in read_jsonl(out_path)}
        todo = [s for s in samples
                if self.manifest.check(s.sample_id, PHASE_GENERATION) != CHECK_SKIP
                or s.sample_id not in existing]
        self._log(f"  {len(existing)} already generated, {len(todo)} to do")

        batch_size = max(1, self.config.generation.batch_size)
        stopped_early = False
        for start in range(0, len(todo), batch_size):
            batch = todo[start:start + batch_size]
            if not self.controller.can_afford(PHASE_GENERATION, len(batch)):
                self._log(f"  {SAFE_STOP_MESSAGE}")
                stopped_early = True
                break
            t0 = time.time()
            try:
                results = ext.generate_batch(model, batch, self.config)
            except Exception as exc:
                # A whole batch failing is usually OOM; retry one at a time so
                # a single pathological sample cannot kill its neighbours.
                self._log(f"  batch failed ({type(exc).__name__}); retrying "
                          f"individually")
                free_gpu_memory()
                results = []
                for s in batch:
                    try:
                        results.extend(ext.generate_batch(model, [s], self.config))
                    except Exception as inner:
                        self.errors.log(s.sample_id, PHASE_GENERATION, inner)
                        self.manifest.record(ManifestRecord(
                            s.sample_id, PHASE_GENERATION, STATUS_FAILED,
                            time.time(), error=f"{type(inner).__name__}: {inner}",
                            model=self.config.model.name, seed=self.config.seed))
            elapsed = (time.time() - t0) / max(1, len(batch))
            for r in results:
                from .storage import append_jsonl
                append_jsonl(out_path, r.to_dict())
                existing[r.sample_id] = r.to_dict()
                self.controller.record(PHASE_GENERATION, elapsed)
                self.manifest.record(ManifestRecord(
                    r.sample_id, PHASE_GENERATION, STATUS_COMPLETE, time.time(),
                    model=self.config.model.name, seed=self.config.seed,
                    runtime_seconds=elapsed,
                    extra={"correct": r.correct, "parse_status": r.parse_status,
                           "generation_length": r.generation_length}))
            self.heartbeat.beat(current_phase=PHASE_GENERATION,
                                completed_samples=len(existing),
                                remaining_samples=len(todo) - start - len(batch),
                                current_sample=batch[-1].sample_id)

        self.generations = existing
        graded = [r for r in existing.values() if r.get("correct") is not None]
        n_correct = sum(1 for r in graded if r["correct"])
        self._log(f"  {len(existing)} generations; {len(graded)} graded; "
                  f"accuracy {n_correct}/{len(graded)}"
                  + (f" ({n_correct / len(graded):.1%})" if graded else ""))
        if not stopped_early:
            self.mark_phase("phase2_generation", n_generations=len(existing),
                            n_graded=len(graded), n_correct=n_correct)
        return existing

    # ==================================================================
    # PHASES 3-10 -- per-sample analysis
    # ==================================================================
    def phase3_analysis(self, samples: Optional[Sequence[Sample]] = None
                        ) -> Dict[str, Any]:
        self._log("PHASES 3-10: hidden states, logit lens, entropy, JSD, "
                  "trajectory, attention")
        samples = list(samples if samples is not None else self.samples)
        model = self.load()
        by_id = {s.sample_id: s for s in samples}

        shard_writer = ShardWriter(self.paths.hidden_states, "hidden",
                                   self.config.extraction.shard_size,
                                   manifest_path=self.paths.shard_manifest)
        already_sharded = shard_writer.existing_sample_ids()

        n_full_attn = self.config.extraction.save_full_attention_for_n_samples
        full_attn_done = len(list(self.paths.attention.glob("full_*.npz")))

        todo: List[Sample] = []
        for s in samples:
            if s.sample_id not in self.generations:
                continue
            decision = self.manifest.check(
                s.sample_id, PHASE_ANALYSIS,
                validator=ext.validate_analysis_file)
            if decision == CHECK_SKIP:
                continue
            if decision == CHECK_RECOMPUTE:
                self._log(f"  checkpoint corrupted for {s.sample_id}; recomputing")
                self.manifest.record(ManifestRecord(
                    s.sample_id, PHASE_ANALYSIS, STATUS_CORRUPTED, time.time()))
            if self.manifest.failure_count(s.sample_id, PHASE_ANALYSIS) >= \
                    self.config.runtime.max_sample_failures:
                continue
            todo.append(s)

        n_done = len(self.manifest.completed_ids(PHASE_ANALYSIS))
        self._log(f"  {n_done} already analysed, {len(todo)} to do")

        stopped_early = False
        for i, sample in enumerate(todo):
            if not self.controller.can_afford(PHASE_ANALYSIS, 1):
                self._log(f"  {SAFE_STOP_MESSAGE}")
                stopped_early = True
                break
            gen = _generation_from_dict(self.generations[sample.sample_id])
            out_path = self.paths.entropy / f"{sample.sample_id}.npz"
            t0 = time.time()
            try:
                full_path = None
                if full_attn_done < n_full_attn:
                    full_path = self.paths.attention / f"full_{sample.sample_id}.npz"
                analysis = ext.analyse_sample(model, sample, gen, self.config,
                                              save_full_attention_path=full_path)
                if full_path is not None and full_path.exists():
                    full_attn_done += 1
                ext.save_analysis(analysis, out_path)
                if analysis.hidden is not None and \
                        sample.sample_id not in already_sharded:
                    shard_writer.add(sample.sample_id, {
                        "hidden": analysis.hidden,
                        "positions": np.asarray(analysis.hidden_positions,
                                                dtype=np.int32),
                    })
                elapsed = time.time() - t0
                self.controller.record(PHASE_ANALYSIS, elapsed)
                self.manifest.record(ManifestRecord(
                    sample.sample_id, PHASE_ANALYSIS, STATUS_COMPLETE,
                    time.time(), output_path=str(out_path),
                    checksum=file_checksum(out_path),
                    model=self.config.model.name, seed=self.config.seed,
                    runtime_seconds=elapsed,
                    extra={"diagnostics": analysis.diagnostics}))
                del analysis
            except Exception as exc:
                free_gpu_memory()
                self.errors.log(sample.sample_id, PHASE_ANALYSIS, exc,
                                dataset=sample.dataset)
                self.manifest.record(ManifestRecord(
                    sample.sample_id, PHASE_ANALYSIS, STATUS_FAILED, time.time(),
                    error=f"{type(exc).__name__}: {exc}",
                    model=self.config.model.name, seed=self.config.seed,
                    runtime_seconds=time.time() - t0))
                self._log(f"  FAILED {sample.sample_id}: "
                          f"{type(exc).__name__}: {str(exc)[:120]}")
            if (i + 1) % 8 == 0:
                shard_writer.flush()
                gc.collect()
            self.heartbeat.beat(current_phase=PHASE_ANALYSIS,
                                current_sample=sample.sample_id,
                                completed_samples=n_done + i + 1,
                                remaining_samples=len(todo) - i - 1,
                                last_successful_checkpoint=str(out_path))
        shard_writer.flush()
        free_gpu_memory()

        completed = self.manifest.completed_ids(PHASE_ANALYSIS)
        self._log(f"  analysis complete for {len(completed)} samples")
        if not stopped_early:
            self.mark_phase("phase3_analysis", n_completed=len(completed))
        return {"n_completed": len(completed), "stopped_early": stopped_early}

    # ==================================================================
    # Loading analysed results back
    # ==================================================================
    def load_analyses(self, sample_ids: Optional[Sequence[str]] = None
                      ) -> Dict[str, Dict[str, Any]]:
        ids = list(sample_ids) if sample_ids is not None \
            else self.manifest.completed_ids(PHASE_ANALYSIS)
        out: Dict[str, Dict[str, Any]] = {}
        for sid in ids:
            path = self.paths.entropy / f"{sid}.npz"
            if not path.exists():
                continue
            try:
                out[sid] = ext.load_analysis(path)
            except Exception as exc:
                self.errors.log(sid, "load_analysis", exc)
        return out

    def hidden_bank(self, sample_ids: Sequence[str], position: str = "answer"
                    ) -> Tuple[Optional[np.ndarray], List[str]]:
        """Assemble ``(n_samples, n_depth, d)`` at one role position.

        Samples whose stored hidden tensors have a different depth or width
        are dropped rather than padded -- that mismatch means a different
        model, and stacking them would fabricate a comparison.
        """
        reader = ShardReader(self.paths.hidden_states, "hidden")
        rows: List[np.ndarray] = []
        kept: List[str] = []
        expected_shape: Optional[Tuple[int, int]] = None
        for sid, arrays in reader.iter_samples(sample_ids):
            h = arrays.get("hidden")
            if h is None or h.ndim != 3:
                continue
            # Stored positions are [last_input, answer, final] (deduplicated
            # and sorted); pick by role using the saved position list.
            idx = h.shape[1] - 1 if position == "final" else \
                (0 if position == "last_input" else min(1, h.shape[1] - 1))
            vec = h[:, idx, :].astype(np.float32)
            if expected_shape is None:
                expected_shape = vec.shape
            elif vec.shape != expected_shape:
                continue
            rows.append(vec)
            kept.append(sid)
        if not rows:
            return None, []
        return np.stack(rows, axis=0), kept

    # ==================================================================
    # PHASE 7 / 18-20 -- population geometry
    # ==================================================================
    def phase7_geometry(self, sample_ids: Sequence[str],
                        correct_map: Dict[str, Optional[bool]]) -> Dict[str, Any]:
        self._log("PHASES 7/18-20: latent-space geometry")
        bank, kept = self.hidden_bank(sample_ids)
        if bank is None or bank.shape[0] < 5:
            self.missing_analyses.append(
                "population geometry (needs >= 5 samples with stored hidden states)")
            return {"status": "insufficient_samples",
                    "n": 0 if bank is None else int(bank.shape[0])}
        labels = np.array([correct_map.get(s) for s in kept], dtype=object)
        result = geo.geometry_across_layers(bank, labels, cfg=self.config.geometry)
        pca = geo.pca_trajectory(bank, n_components=3)
        payload = {
            "status": "ok",
            "n_samples_used": result["n_samples_used"],
            "sample_ids": kept,
            "profiles": {k: np.asarray(v).tolist()
                         for k, v in result["profiles"].items()},
            "layers": result["layers"].tolist(),
        }
        save_json(self.paths.geometry / "layer_geometry.json", payload)
        self._log(f"  geometry computed over {result['n_samples_used']} samples, "
                  f"{len(result['profiles'])} profiles")
        self.mark_phase("phase7_geometry", n_samples=result["n_samples_used"])
        return {"status": "ok", "result": result, "pca": pca,
                "bank": bank, "kept": kept}

    # ==================================================================
    # PHASES 22/23 -- J-space
    # ==================================================================
    def phase22_jspace(self, samples: Sequence[Sample]) -> Dict[str, Any]:
        if not self.config.jspace.enabled:
            self.missing_analyses.append("J-space (disabled in configuration)")
            return {"status": "disabled"}
        self._log("PHASES 22/23: local sensitivity (J-space)")
        model = self.load()
        jcfg = self.config.jspace
        eligible = [s for s in samples if s.sample_id in self.generations]
        subset = eligible[:jcfg.max_samples]
        out: Dict[str, Dict[str, Any]] = {}
        stopped_early = False

        for i, sample in enumerate(subset):
            decision = self.manifest.check(sample.sample_id, PHASE_JSPACE)
            path = self.paths.j_space / f"{sample.sample_id}.npz"
            if decision == CHECK_SKIP and path.exists():
                try:
                    out[sample.sample_id] = _load_jspace(path)
                    continue
                except Exception:
                    pass
            if not self.controller.can_afford(PHASE_JSPACE, 1):
                self._log(f"  {SAFE_STOP_MESSAGE}")
                stopped_early = True
                break
            t0 = time.time()
            try:
                import torch
                gen = _generation_from_dict(self.generations[sample.sample_id])
                full = list(gen.input_ids) + list(gen.generated_ids)
                ids = torch.tensor([full], dtype=torch.long, device=model.device)
                mask = torch.ones_like(ids)
                spec = build_answer_spec(sample, model.tokenizer)
                plan = ext.plan_positions(len(gen.input_ids), gen.generated_ids,
                                          spec, modes=self.config.extraction.modes,
                                          max_generated=self.config.extraction
                                          .max_generated_positions)
                pos = plan.positions[plan.answer_index
                                     if plan.answer_index is not None
                                     else plan.last_input_index]
                result = jsp.jacobian_descriptors(
                    model, ids, mask, token_position=pos,
                    n_probes=jcfg.n_probe_directions, epsilons=jcfg.epsilons,
                    seed=jcfg.probe_seed, relative=jcfg.relative_perturbation,
                    layer_chunk=8)
                result["epsilon_consistency"] = jsp.epsilon_consistency(result)
                _save_jspace(path, result)
                out[sample.sample_id] = result
                elapsed = time.time() - t0
                self.controller.record(PHASE_JSPACE, elapsed)
                self.manifest.record(ManifestRecord(
                    sample.sample_id, PHASE_JSPACE, STATUS_COMPLETE, time.time(),
                    output_path=str(path), checksum=file_checksum(path),
                    model=self.config.model.name, runtime_seconds=elapsed))
                del ids, mask
            except Exception as exc:
                free_gpu_memory()
                self.errors.log(sample.sample_id, PHASE_JSPACE, exc)
                self.manifest.record(ManifestRecord(
                    sample.sample_id, PHASE_JSPACE, STATUS_FAILED, time.time(),
                    error=f"{type(exc).__name__}: {exc}"))
            self.heartbeat.beat(current_phase=PHASE_JSPACE,
                                current_sample=sample.sample_id,
                                completed_samples=len(out),
                                remaining_samples=len(subset) - i - 1)
        free_gpu_memory()
        self._log(f"  J-space computed for {len(out)} samples")
        if not stopped_early:
            self.mark_phase("phase22_jspace", n_samples=len(out))
        return {"status": "ok", "results": out}

    # ==================================================================
    # PHASE 11 -- critical layer detection
    # ==================================================================
    def phase11_critical(self, analyses: Dict[str, Dict[str, Any]],
                         jspace_results: Dict[str, Dict[str, Any]],
                         n_layers: int) -> Dict[str, Any]:
        self._log("PHASES 11/26/27: critical-layer detection")
        detections: Dict[str, Dict[str, crit.DetectorResult]] = {}
        consensus: Dict[str, crit.ConsensusResult] = {}
        summaries: Dict[str, Dict[str, Any]] = {}

        for sid, data in analyses.items():
            profiles = dict(data["profiles"])
            extra: Dict[str, np.ndarray] = {}
            jr = jspace_results.get(sid)
            if jr and jr.get("status") == "ok":
                amp = np.asarray(jr["amplification_mean"], dtype=np.float64)
                # J-space is indexed by block (n_layers); profiles include the
                # embedding row, so pad to align the two index spaces.
                extra["jspace_amplification"] = np.concatenate([[np.nan], amp])
            det = crit.detect_all(profiles, extra_profiles=extra)
            cons = crit.consensus(det, n_layers + 1)
            detections[sid] = det
            consensus[sid] = cons
            summary = crit.summarise_detections(det, cons, n_layers + 1)
            summary["transition_strength"] = crit.transition_strength(profiles)
            summaries[sid] = summary

        agreement = crit.agreement_matrix(list(detections.values()))
        save_json(self.paths.critical_layers / "agreement.json", {
            "methods": agreement["methods"],
            "agreement_rate": np.asarray(agreement["agreement_rate"]).tolist(),
            "n_comparisons": np.asarray(agreement["n_comparisons"]).tolist(),
            "tolerance": agreement["tolerance"],
            "mean_abs_difference": agreement["mean_abs_difference"],
            "note": agreement["note"],
        })
        n_cons = sum(1 for c in consensus.values()
                     if c.critical_layer_consensus is not None)
        self._log(f"  consensus reached for {n_cons}/{len(consensus)} samples")
        self.mark_phase("phase11_critical", n_samples=len(summaries),
                        n_with_consensus=n_cons)
        return {"detections": detections, "consensus": consensus,
                "summaries": summaries, "agreement": agreement}

    # ==================================================================
    # PHASE 13/31/32 -- causal intervention
    # ==================================================================
    def phase13_interventions(self, samples: Sequence[Sample],
                              critical: Dict[str, Any],
                              bank: Optional[np.ndarray]) -> Dict[str, Any]:
        if not self.config.interventions.enabled:
            self.missing_analyses.append("causal intervention (disabled)")
            return {"status": "disabled"}
        self._log("PHASES 13/31/32: causal intervention")
        model = self.load()
        icfg = self.config.interventions
        eligible = [s for s in samples
                    if s.sample_id in self.generations
                    and s.answer_spec_type != ANSWER_UNDEFINED]
        subset = eligible[:icfg.max_samples]
        results: List[Dict[str, Any]] = []
        verifications: List[Dict[str, Any]] = []
        stopped_early = False

        for i, sample in enumerate(subset):
            path = self.paths.interventions / f"{sample.sample_id}.json"
            if self.manifest.check(sample.sample_id, PHASE_INTERVENTION) == CHECK_SKIP \
                    and path.exists():
                try:
                    results.append(load_json(path))
                    continue
                except Exception:
                    pass
            if not self.controller.can_afford(PHASE_INTERVENTION, 1):
                self._log(f"  {SAFE_STOP_MESSAGE}")
                stopped_early = True
                break
            t0 = time.time()
            try:
                gen = _generation_from_dict(self.generations[sample.sample_id])
                spec = build_answer_spec(sample, model.tokenizer)
                cons = critical["consensus"].get(sample.sample_id)
                cl = cons.critical_layer_consensus if cons else None
                if cl is not None:
                    # Detector layers index the profile array (embedding at 0);
                    # intervention layers index transformer blocks.
                    cl = max(0, min(model.arch.n_layers - 1, cl - 1))
                plan = ext.plan_positions(len(gen.input_ids), gen.generated_ids,
                                          spec, modes=self.config.extraction.modes,
                                          max_generated=self.config.extraction
                                          .max_generated_positions)
                pos = plan.positions[plan.answer_index
                                     if plan.answer_index is not None
                                     else plan.last_input_index]
                cross = _cross_sample_direction(model, samples, sample, i)
                result = interv.run_interventions(
                    model, sample, gen, spec, self.config,
                    critical_layer=cl, token_position=pos,
                    hidden_bank=bank, cross_sample_dir=cross,
                    seed=self.config.seed + i)
                save_json(path, result)
                results.append(result)
                # Validate the cheap next-token proxy against real decoding for
                # a few samples.
                if len(verifications) < 5 and cl is not None:
                    verifications.append(interv.verify_with_regeneration(
                        model, sample, gen, spec, self.config, layer=cl,
                        epsilon=icfg.epsilons[-1], seed=self.config.seed + i))
                elapsed = time.time() - t0
                self.controller.record(PHASE_INTERVENTION, elapsed)
                self.manifest.record(ManifestRecord(
                    sample.sample_id, PHASE_INTERVENTION, STATUS_COMPLETE,
                    time.time(), output_path=str(path),
                    checksum=file_checksum(path), runtime_seconds=elapsed))
            except Exception as exc:
                free_gpu_memory()
                self.errors.log(sample.sample_id, PHASE_INTERVENTION, exc)
                self.manifest.record(ManifestRecord(
                    sample.sample_id, PHASE_INTERVENTION, STATUS_FAILED,
                    time.time(), error=f"{type(exc).__name__}: {exc}"))
            self.heartbeat.beat(current_phase=PHASE_INTERVENTION,
                                current_sample=sample.sample_id,
                                completed_samples=len(results),
                                remaining_samples=len(subset) - i - 1)
        free_gpu_memory()
        df = interv.aggregate_interventions(results)
        if len(df):
            save_parquet(self.paths.interventions / "intervention_outcomes.parquet", df)
        if verifications:
            save_json(self.paths.interventions / "regeneration_validation.json",
                      {"validations": verifications})
        self._log(f"  {len(results)} samples intervened; {len(df)} outcomes")
        if not stopped_early:
            self.mark_phase("phase13_interventions", n_samples=len(results),
                            n_outcomes=int(len(df)))
        return {"status": "ok", "results": results, "df": df,
                "verifications": verifications}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _weight_gb(arch: Any, config: Any) -> float:
    """Rough weight footprint given the quantization setting."""
    bytes_per = {"4bit": 0.55, "8bit": 1.05}.get(config.model.quantization, 2.0)
    # Parameter count is dominated by the blocks plus the embedding/unembedding.
    n_params = (arch.n_layers * 12 * arch.hidden_size ** 2 +
                2 * arch.vocab_size * arch.hidden_size)
    return n_params * bytes_per / (1024 ** 3)


def _generation_from_dict(d: Dict[str, Any]):
    from .modeling import GenerationResult
    fields = {k: d.get(k) for k in GenerationResult.__dataclass_fields__}
    fields.setdefault("parse_status", "unknown")
    return GenerationResult(**fields)


def _cross_sample_direction(model: Any, samples: Sequence[Sample],
                            current: Sample, index: int) -> Optional[Any]:
    """Answer direction from a *different* sample, for the control arm."""
    others = [s for s in samples
              if s.sample_id != current.sample_id
              and s.answer_spec_type != ANSWER_UNDEFINED]
    if not others:
        return None
    other = others[index % len(others)]
    spec = build_answer_spec(other, model.tokenizer)
    return interv.answer_direction(model, spec)


def _save_jspace(path: Path, result: Dict[str, Any]) -> Path:
    from .storage import save_npz
    arrays: Dict[str, np.ndarray] = {}
    for eps, mat in result.get("descriptors", {}).items():
        arrays[f"desc::{eps}"] = np.asarray(mat, dtype=np.float32)
    for eps, mat in result.get("descriptors_to_final", {}).items():
        arrays[f"final::{eps}"] = np.asarray(mat, dtype=np.float32)
    for key in ["amplification_mean", "amplification_max", "amplification_spread",
                "amplification_delta", "clean_norms"]:
        if key in result:
            arrays[f"prof::{key}"] = np.asarray(result[key], dtype=np.float32)
    arrays["meta::primary_epsilon"] = np.asarray([result.get("primary_epsilon", 0.0)])
    arrays["meta::n_probes"] = np.asarray([result.get("n_probes", 0)])
    arrays["meta::probe_seed"] = np.asarray([result.get("probe_seed", 0)])
    arrays["meta::token_position"] = np.asarray([result.get("token_position", 0)])
    return save_npz(path, arrays)


def _load_jspace(path: Path) -> Dict[str, Any]:
    from .storage import load_npz
    raw = load_npz(path)
    out: Dict[str, Any] = {"status": "ok", "descriptors": {},
                           "descriptors_to_final": {}}
    for key, value in raw.items():
        section, _, name = key.partition("::")
        if section == "desc":
            out["descriptors"][name] = value.astype(np.float64)
        elif section == "final":
            out["descriptors_to_final"][name] = value.astype(np.float64)
        elif section == "prof":
            out[name] = value.astype(np.float64)
        elif section == "meta":
            out[name] = value[0] if len(value) else None
    out["primary_epsilon"] = float(out.get("primary_epsilon", 0.0))
    out["n_probes"] = int(out.get("n_probes", 0))
    return out
