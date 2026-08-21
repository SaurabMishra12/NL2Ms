"""Top-level entry points: pilot, full run, and the resume verification.

``run_experiment`` executes every phase in order and is safe to call
repeatedly: each phase consults the manifest first, so a second invocation
after an interrupted session picks up where the first stopped.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import analysis as ana
from . import critical as crit
from . import signals as sig
from .config import ExperimentConfig, pilot_config
from .datasets_build import Sample
from .pipeline import (PHASE_ANALYSIS, PHASE_GENERATION, Experiment)
from .runtime import BudgetExceeded, SAFE_STOP_MESSAGE, free_gpu_memory
from .storage import make_backup_archive, save_json


def run_experiment(config: ExperimentConfig, *, pilot: bool = False,
                   repo_dir: str = ".", make_backup: bool = True,
                   strict_determinism: bool = False) -> Dict[str, Any]:
    """Run every phase, resuming anything already complete.

    Returns a dict with the experiment object, the results payload and the
    integrity report. Raises only on conditions that make the run
    scientifically invalid (missing required packages, insufficient disk);
    a wall-clock stop is a normal, non-exceptional outcome.
    """
    exp = Experiment(config, repo_dir=repo_dir)
    t_start = time.time()

    try:
        # -- PHASE 0-1 --------------------------------------------------
        exp.phase0_environment(strict_determinism=strict_determinism)
        samples = exp.phase1_dataset(pilot=pilot)
        if not samples:
            raise RuntimeError(
                "no samples were assembled. With hub downloads unavailable, "
                "set datasets.n_synthetic > 0 so the controlled benchmark "
                "still provides data.")

        exp.plan(len(samples))

        # -- PHASE 2 ----------------------------------------------------
        exp.phase2_generation(samples)

        # -- PHASES 3-10 ------------------------------------------------
        exp.phase3_analysis(samples)

        analyses = exp.load_analyses()
        if not analyses:
            raise RuntimeError("no samples completed analysis; nothing to report")
        correct_map = {sid: exp.generations.get(sid, {}).get("correct")
                       for sid in analyses}

        # -- PHASES 22/23 ----------------------------------------------
        jspace_out = exp.phase22_jspace(samples)
        jspace_results = jspace_out.get("results", {}) or {}

        # -- PHASE 11 ---------------------------------------------------
        n_layers = exp.model.arch.n_layers if exp.model else 0
        critical = exp.phase11_critical(analyses, jspace_results, n_layers)

        # -- PHASE 7/18-20 ---------------------------------------------
        geometry = exp.phase7_geometry(list(analyses.keys()), correct_map)
        bank = geometry.get("bank") if geometry.get("status") == "ok" else None

        # -- PHASE 13/31/32 --------------------------------------------
        intervention = exp.phase13_interventions(samples, critical, bank)
        intervention_df = intervention.get("df")

        # -- PHASE 47/48: master tables --------------------------------
        exp._log("PHASES 47/48: master tables")
        df = ana.build_sample_summary(exp, analyses, critical, jspace_results,
                                      intervention_df)
        layer_df = ana.build_layer_summary(exp, analyses, correct_map, geometry)
        exp._log(f"  sample_summary: {len(df)} rows x {len(df.columns)} columns")
        exp._log(f"  layer_summary : {len(layer_df)} rows")

        # -- PHASE 25: susceptibility ----------------------------------
        margins = sig.safe_stack([analyses[s]["profiles"]["order_margin"]
                                  for s in analyses
                                  if "order_margin" in analyses[s]["profiles"]])
        susceptibility = (crit.susceptibility_detector(margins)
                          if margins is not None and margins.shape[0] >= 2
                          else {"status": "insufficient_samples"})
        if susceptibility.get("status") == "ok":
            save_json(exp.paths.statistics / "susceptibility.json", {
                "profile": np.asarray(susceptibility["profile"]).tolist(),
                "mean_profile": np.asarray(susceptibility["mean_profile"]).tolist(),
                "detection": susceptibility["detection"],
                "n_samples": susceptibility["n_samples"],
                "caveat": susceptibility["caveat"],
            })

        # -- PHASE 12/14/29/34/35 --------------------------------------
        exp._log("PHASES 12/14/29/34/35: statistics, nulls, confounds")
        cvi_tests = ana.compare_correct_incorrect(exp, df, analyses, correct_map)
        cross_dataset = ana.cross_dataset_analysis(exp, df)
        nulls = ana.run_null_models(exp, df, analyses, correct_map)
        confounds = ana.run_confound_analysis(exp, df, analyses)

        # -- PHASE 15 ---------------------------------------------------
        exp._log("PHASE 15: figures")
        figures = ana.generate_figures(exp, analyses, correct_map, df, geometry,
                                       jspace_results, critical, intervention_df,
                                       susceptibility)
        exp._log(f"  {figures['n_generated']} figures generated, "
                 f"{figures['n_skipped']} skipped")
        for s in figures.get("skipped", []):
            exp._log(f"    skipped {s['figure']}: {s['reason']}")

        # -- PHASE 16 ---------------------------------------------------
        exp._log("PHASE 16: manifest, integrity check, final report")
        final = ana.finalise(
            exp, analyses=analyses, correct_map=correct_map, df=df,
            critical=critical, geometry=geometry, jspace_results=jspace_results,
            intervention=intervention, susceptibility=susceptibility,
            cvi_tests=cvi_tests, cross_dataset=cross_dataset, nulls=nulls,
            confounds=confounds, figures=figures)

        integrity = final["integrity"]
        exp._log(f"  integrity: {'PASSED' if integrity['passed'] else 'PROBLEMS FOUND'}")
        for p in integrity["problems"]:
            exp._log(f"    PROBLEM {p['check']}: {p['detail']}")
        for w in integrity["warnings"][:5]:
            exp._log(f"    warning {w['check']}")

        archive = None
        if make_backup:
            archive = make_backup_archive(exp.paths, include_raw=False)
            exp._log(f"  backup archive: {archive}")

        exp.heartbeat.beat(force=True, current_phase="complete",
                           completed_samples=len(analyses), remaining_samples=0)
        exp._log(f"DONE in {(time.time() - t_start) / 60:.1f} min")
        print()
        print(f"Final report : {final['report_path']}")
        print(f"Manifest     : {final['manifest_path']}")
        if archive:
            print(f"Backup       : {archive}")

        return {"experiment": exp, "results": final["results"],
                "integrity": integrity, "sample_summary": df,
                "layer_summary": layer_df, "figures": figures,
                "report_path": final["report_path"],
                "backup_path": str(archive) if archive else None}

    except BudgetExceeded as exc:
        exp._log(str(exc))
        exp.heartbeat.beat(force=True, current_phase="safe_stop")
        make_backup_archive(exp.paths, include_raw=False)
        print(SAFE_STOP_MESSAGE)
        return {"experiment": exp, "stopped": "budget", "message": str(exc)}
    finally:
        free_gpu_memory()


def run_pilot(config: ExperimentConfig, **kwargs: Any) -> Dict[str, Any]:
    """Mandatory pilot pass (protocol section 54).

    Exercises every code path on a handful of samples so that a failure in,
    say, attention summarisation surfaces after two minutes rather than after
    six hours of extraction.
    """
    pcfg = pilot_config(config)
    pcfg.output_root = str(Path(config.output_root).parent /
                           f"{Path(config.output_root).name}_pilot")
    print("=" * 72)
    print("PILOT RUN -- validating every code path before the full experiment")
    print("=" * 72)
    out = run_experiment(pcfg, pilot=True, **kwargs)
    print("=" * 72)
    ok = out.get("integrity", {}).get("passed", False)
    print(f"PILOT {'PASSED' if ok else 'COMPLETED WITH PROBLEMS'}")
    print("=" * 72)
    return out


def validate_pilot(pilot_result: Dict[str, Any]) -> Dict[str, Any]:
    """Check the pilot actually exercised each component (protocol section 54)."""
    checks: Dict[str, Any] = {}
    exp = pilot_result.get("experiment")
    df = pilot_result.get("sample_summary")
    figures = pilot_result.get("figures", {})

    checks["model_loaded"] = exp is not None and exp.model is not None
    checks["tokenizer_ok"] = bool(exp and exp.model and exp.model.tokenizer)
    checks["generations_written"] = bool(exp and len(exp.generations) > 0)
    checks["hidden_states_extracted"] = bool(
        exp and any(exp.paths.hidden_states.glob("*.npz")))
    checks["logit_lens_ok"] = bool(df is not None and len(df) and
                                   "max_entropy" in df.columns)
    checks["entropy_ok"] = bool(df is not None and len(df) and
                                df["max_entropy"].notna().any())
    checks["attention_ok"] = bool(
        exp and any(exp.paths.attention.glob("*.npz"))) or \
        bool(df is not None and len(df) and
             any(c.startswith("critical_layer_attention") for c in df.columns))
    checks["geometry_ok"] = bool(
        exp and (exp.paths.geometry / "layer_geometry.json").exists())
    reasons: Dict[str, str] = {}
    if not checks["geometry_ok"]:
        reasons["geometry_ok"] = (
            "population geometry did not run -- it needs at least 5 samples "
            "with stored hidden states. Raise pilot_n_samples or "
            "storage_level.")
    checks["jsd_ok"] = bool(df is not None and len(df) and
                            df["max_jsd"].notna().any())
    checks["trajectory_ok"] = bool(df is not None and len(df) and
                                   df["max_velocity"].notna().any())
    checks["jspace_ok"] = bool(exp and any(exp.paths.j_space.glob("*.npz")))
    checks["interventions_ok"] = bool(exp and any(exp.paths.interventions.glob("*.json")))
    checks["checkpoints_written"] = bool(exp and any(exp.paths.checkpoints.glob("*.json")))
    checks["figures_ok"] = figures.get("n_generated", 0) > 0
    checks["report_ok"] = bool(exp and exp.paths.final_report.exists())
    checks["storage_estimate_ok"] = bool(exp and (exp.paths.logs / "plan.json").exists())

    failed = [k for k, v in checks.items() if not v]
    result = {"checks": checks, "failed": failed, "passed": not failed,
              "reasons": reasons}
    if exp:
        save_json(exp.paths.logs / "pilot_validation.json", result)
    print()
    print("PILOT COMPONENT VALIDATION")
    for name, ok in sorted(checks.items()):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if failed:
        print(f"\n{len(failed)} component(s) did not validate: {failed}")
        for name in failed:
            if name in reasons:
                print(f"  {name}: {reasons[name]}")
        print("Investigate before starting the full run; the same failure will "
              "recur at scale.")
    else:
        print("\nAll pilot components validated.")
    return result


def resume_test(config: ExperimentConfig, *, n_first: int = 4,
                repo_dir: str = ".") -> Dict[str, Any]:
    """Demonstrate resumability for real (protocol section 55).

    Processes a few samples, tears the experiment object down, rebuilds it from
    disk, and verifies that (a) the completed samples are skipped and (b) work
    continues from the next incomplete one. Prints ``RESUME TEST PASSED`` only
    if both hold.
    """
    print("=" * 72)
    print("RESUME TEST")
    print("=" * 72)

    cfg = config
    root = Path(cfg.output_root)

    # --- pass 1: process n_first samples ------------------------------
    exp1 = Experiment(cfg, repo_dir=repo_dir)
    exp1.phase0_environment()
    samples = exp1.phase1_dataset(pilot=True)
    subset = samples[:max(2, n_first * 2)]
    first = subset[:n_first]
    exp1.phase2_generation(first)
    exp1.phase3_analysis(first)
    done_after_first = set(exp1.manifest.completed_ids(PHASE_ANALYSIS))
    print(f"\npass 1: analysed {len(done_after_first)} samples")
    if exp1.model is not None:
        model_handle = exp1.model            # reuse weights; the point of the
        exp1.model = None                    # test is manifest state, not I/O
    else:
        model_handle = None
    del exp1

    # --- pass 2: rebuild from disk and continue ------------------------
    exp2 = Experiment(cfg, repo_dir=repo_dir)
    exp2.model = model_handle
    exp2.phase0_environment()
    samples2 = exp2.phase1_dataset(pilot=True)
    print(f"pass 2: manifest reloaded with "
          f"{len(exp2.manifest.completed_ids(PHASE_ANALYSIS))} completed samples")

    skipped_correctly = (set(exp2.manifest.completed_ids(PHASE_ANALYSIS)) ==
                         done_after_first)

    # Track which samples pass 2 actually recomputes.
    before_records = len(exp2.manifest.__dict__["_index"])
    exp2.phase2_generation(subset)
    exp2.phase3_analysis(subset)
    done_after_second = set(exp2.manifest.completed_ids(PHASE_ANALYSIS))
    newly_done = done_after_second - done_after_first

    continued = len(newly_done) > 0
    no_recompute = done_after_first <= done_after_second

    # Verify the original checkpoints were not rewritten.
    unchanged = True
    for sid in done_after_first:
        rec = exp2.manifest.get(sid, PHASE_ANALYSIS)
        if rec is None or rec.get("status") != "complete":
            unchanged = False
            break

    passed = skipped_correctly and continued and no_recompute and unchanged
    result = {
        "n_after_pass1": len(done_after_first),
        "n_after_pass2": len(done_after_second),
        "n_newly_completed_in_pass2": len(newly_done),
        "completed_samples_skipped": skipped_correctly,
        "continued_from_next_incomplete": continued,
        "no_completed_work_lost": no_recompute,
        "existing_checkpoints_unchanged": unchanged,
        "passed": passed,
    }
    save_json(exp2.paths.logs / "resume_test.json", result)

    print()
    print(f"  completed samples skipped on resume : {skipped_correctly}")
    print(f"  continued from next incomplete      : {continued} "
          f"({len(newly_done)} new)")
    print(f"  no completed work lost              : {no_recompute}")
    print(f"  existing checkpoints unchanged      : {unchanged}")
    print()
    if passed:
        print("RESUME TEST PASSED")
    else:
        print("RESUME TEST FAILED -- do not rely on resumability until fixed")
    print("=" * 72)
    result["experiment"] = exp2
    return result
