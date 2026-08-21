"""Phases 16, 49, 50, 64 -- manifest, integrity check, and the final report.

The report generator is written to be incapable of endorsing the hypothesis.
It takes computed statistics as input and renders them with fixed hedged
language; there is no branch anywhere that emits "a phase transition was
observed". The strongest phrasing available to it is "consistent with
phase-transition-like behaviour", and every such statement is paired with the
null-model comparison that would undercut it.

The mandatory closing section, *WHAT WOULD INVALIDATE THE PHASE-TRANSITION
HYPOTHESIS?*, is generated from the run's own numbers so that it names the
specific results that already point against the hypothesis, rather than
listing generic caveats.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .storage import (ExperimentPaths, dir_size_gb, file_checksum, load_json,
                      read_jsonl, save_json)


# ---------------------------------------------------------------------------
# Integrity check (protocol section 64)
# ---------------------------------------------------------------------------
def check_integrity(paths: ExperimentPaths, *, expected_samples: int,
                    manifest: Any, errors: Sequence[Dict[str, Any]],
                    figures: Sequence[Dict[str, Any]],
                    sample_ids: Sequence[str]) -> Dict[str, Any]:
    """Verify the experiment directory is complete and self-consistent."""
    report: Dict[str, Any] = {
        "checked_at": time.time(),
        "checked_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checks": {},
        "problems": [],
        "warnings": [],
    }

    def check(name: str, ok: bool, detail: Any = None, *,
              severity: str = "problem") -> None:
        report["checks"][name] = {"ok": bool(ok), "detail": detail}
        if not ok:
            target = report["problems"] if severity == "problem" else report["warnings"]
            target.append({"check": name, "detail": detail})

    # -- sample accounting ---------------------------------------------
    counts = manifest.phase_counts()
    report["phase_counts"] = counts
    completed = set(manifest.completed_ids("analysis"))
    failed = set(manifest.failed_ids("analysis"))
    requested = set(sample_ids)
    report["n_expected"] = expected_samples
    report["n_requested"] = len(requested)
    report["n_completed"] = len(completed)
    report["n_failed"] = len(failed)
    report["n_skipped"] = len(requested - completed - failed)
    report["failed_sample_ids"] = sorted(failed)

    check("no_duplicate_sample_ids", len(requested) == len(sample_ids),
          {"n_unique": len(requested), "n_total": len(sample_ids)})
    check("all_samples_accounted_for",
          len(completed | failed) >= len(requested),
          {"unaccounted": sorted(requested - completed - failed)[:20]},
          severity="warning")

    # -- readability of manifests and checkpoints ----------------------
    unreadable: List[str] = []
    for p in [paths.manifest_jsonl, paths.shard_manifest,
              paths.config / "experiment_config.json",
              paths.config / "environment.json"]:
        if not p.exists():
            unreadable.append(f"{p} (missing)")
            continue
        try:
            if p.suffix == ".jsonl":
                read_jsonl(p)
            else:
                load_json(p)
        except Exception as exc:
            unreadable.append(f"{p} ({type(exc).__name__})")
    check("manifests_and_config_readable", not unreadable, unreadable)

    # -- checkpoint files parse and checksums match --------------------
    bad_checkpoints: List[Dict[str, Any]] = []
    checked = 0
    for record in read_jsonl(paths.manifest_jsonl):
        out_path = record.get("output_path")
        if not out_path or record.get("status") != "complete":
            continue
        p = Path(out_path)
        checked += 1
        if not p.exists():
            bad_checkpoints.append({"path": out_path, "issue": "missing"})
            continue
        expected = record.get("checksum")
        if expected:
            try:
                if file_checksum(p) != expected:
                    bad_checkpoints.append({"path": out_path,
                                            "issue": "checksum_mismatch"})
            except OSError as exc:
                bad_checkpoints.append({"path": out_path,
                                        "issue": f"unreadable: {exc}"})
    check("checkpoints_intact", not bad_checkpoints,
          {"n_checked": checked, "bad": bad_checkpoints[:20]})

    # -- numerical sanity ----------------------------------------------
    nan_report = scan_for_nonfinite(paths)
    report["nonfinite_scan"] = nan_report
    check("nonfinite_values_documented",
          nan_report["n_undocumented_files"] == 0,
          {"undocumented": nan_report["undocumented"][:10],
           "documented_cases": nan_report["documented_cases"]},
          severity="warning")

    # -- shard consistency ---------------------------------------------
    shard_issues = check_shards(paths)
    report["shards"] = shard_issues
    check("shard_shapes_consistent", not shard_issues["inconsistent"],
          shard_issues["inconsistent"][:10], severity="warning")

    # -- outputs exist --------------------------------------------------
    check("figures_generated", len(figures) > 0, {"n_figures": len(figures)},
          severity="warning")
    check("config_saved", (paths.config / "experiment_config.json").exists())
    check("environment_saved", (paths.config / "environment.json").exists())
    check("final_report_generated", paths.final_report.exists(),
          severity="warning")

    report["n_errors_logged"] = len(errors)
    report["error_types"] = _count_by(errors, "exception_type")
    report["storage_gb"] = dir_size_gb(paths.root)
    report["passed"] = len(report["problems"]) == 0
    return report


def scan_for_nonfinite(paths: ExperimentPaths, limit: int = 200) -> Dict[str, Any]:
    """Find NaN/Inf outside the documented cases.

    Documented cases (expected by construction, not data problems):

    * index 0 of any layer-pair quantity -- the embedding row has no previous
      layer, so ``jsd_prev_layer[0]`` and friends are NaN;
    * index 0 of attention profiles -- the embedding has no attention;
    * the final layer of ``jspace_amplification`` -- no ``l+1`` to measure into;
    * ``inf`` in ``kl_prev_layer`` where the supports genuinely differ.
    """
    documented = {
        "jsd_prev_layer": [0], "kl_prev_layer": [0], "jsd_first_layer": [0],
        "traj_velocity": [], "attn_": [0], "jspace_amplification": [-1],
    }
    undocumented: List[Dict[str, Any]] = []
    n_files = 0
    for npz in list(paths.derived.rglob("*.npz"))[:limit]:
        n_files += 1
        try:
            with np.load(npz, allow_pickle=False) as z:
                for key in z.files:
                    if not key.startswith("profiles::"):
                        continue
                    name = key.split("::", 1)[1]
                    arr = np.asarray(z[key], dtype=np.float64)
                    bad = np.where(~np.isfinite(arr))[0]
                    if bad.size == 0:
                        continue
                    allowed = None
                    for prefix, idxs in documented.items():
                        if name.startswith(prefix):
                            allowed = set(i % arr.size for i in idxs)
                            break
                    if allowed is not None and set(bad.tolist()) <= allowed:
                        continue
                    undocumented.append({
                        "file": str(npz), "signal": name,
                        "n_nonfinite": int(bad.size), "size": int(arr.size),
                        "indices": bad[:10].tolist(),
                    })
        except Exception as exc:
            undocumented.append({"file": str(npz), "error": str(exc)})
    return {
        "n_files_scanned": n_files,
        "n_undocumented_files": len(undocumented),
        "undocumented": undocumented,
        "documented_cases": {
            "layer_pair_index_0": "no previous layer exists for the embedding row",
            "attention_index_0": "the embedding produces no attention",
            "jspace_final_layer": "no layer l+1 to measure amplification into",
            "kl_infinite": "supports genuinely differ; JSD is the primary measure",
        },
    }


def check_shards(paths: ExperimentPaths) -> Dict[str, Any]:
    """Verify shard metadata against the files on disk."""
    inconsistent: List[Dict[str, Any]] = []
    total_samples = 0
    shapes: Dict[str, set] = {}
    for meta_file in paths.hidden_states.rglob("*.meta.json"):
        try:
            meta = load_json(meta_file)
        except Exception as exc:
            inconsistent.append({"meta": str(meta_file), "issue": str(exc)})
            continue
        path = Path(meta.get("path", ""))
        if not path.exists():
            inconsistent.append({"meta": str(meta_file), "issue": "shard_missing"})
            continue
        total_samples += meta.get("n_samples", 0)
        for key, shape in (meta.get("shapes") or {}).items():
            suffix = key.split("::")[-1]
            shapes.setdefault(suffix, set()).add(tuple(shape))
    for suffix, seen in shapes.items():
        # Only activation tensors carry a meaningful shape contract. 1-D
        # sidecars such as the saved position list legitimately vary in length
        # (a sample whose answer position coincides with its last input
        # position stores one fewer position), so checking them would report a
        # permanent false problem.
        tensors = {s for s in seen if len(s) >= 3}
        if len(tensors) < 2:
            continue
        # Middle axis (position count) may vary for the reason above; depth
        # and hidden width may not -- those differing means a different model.
        depth_dims = {s[0] for s in tensors}
        hidden_dims = {s[-1] for s in tensors}
        if len(depth_dims) > 1 or len(hidden_dims) > 1:
            inconsistent.append({"key": suffix,
                                 "depth_dims": sorted(depth_dims),
                                 "hidden_dims": sorted(hidden_dims)})
    return {"n_shard_samples": total_samples, "inconsistent": inconsistent,
            "n_distinct_keys": len(shapes)}


def _count_by(records: Sequence[Dict[str, Any]], key: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in records:
        v = str(r.get(key, "unknown"))
        out[v] = out.get(v, 0) + 1
    return out


# ---------------------------------------------------------------------------
# Experiment manifest (protocol section 49)
# ---------------------------------------------------------------------------
def build_experiment_manifest(paths: ExperimentPaths, *, config: Any,
                              environment: Dict[str, Any],
                              arch: Optional[Dict[str, Any]],
                              dataset_summary: Dict[str, Any],
                              phases_completed: Dict[str, Any],
                              manifest: Any,
                              errors: Sequence[Dict[str, Any]],
                              figures: Sequence[Dict[str, Any]],
                              statistical_tests: Sequence[Dict[str, Any]],
                              runtime: Dict[str, Any]) -> Dict[str, Any]:
    """The single document that answers "what was this run?"."""
    return {
        "experiment_name": config.experiment_name,
        "created_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_hash": config.config_hash(),
        "config": config.to_dict(),
        "model": {
            "name": config.model.name,
            "revision": config.model.revision,
            "tokenizer": config.model.resolved_tokenizer_name(),
            "tokenizer_revision": config.model.resolved_tokenizer_revision(),
            "quantization": config.model.quantization,
            "dtype": config.model.dtype,
            "device_map": config.model.device_map,
            "architecture": arch,
        },
        "environment": environment,
        "git_commit": environment.get("git_commit"),
        "data": dataset_summary,
        "phases_completed": phases_completed,
        "sample_accounting": {
            "phase_counts": manifest.phase_counts(),
            "n_completed_analysis": len(manifest.completed_ids("analysis")),
            "n_failed_analysis": len(manifest.failed_ids("analysis")),
            "failed_sample_ids": sorted(manifest.failed_ids("analysis")),
        },
        "errors": {
            "n_errors": len(errors),
            "by_type": _count_by(errors, "exception_type"),
            "by_phase": _count_by(errors, "phase"),
            "log_path": str(paths.errors_jsonl),
        },
        "raw_tensor_locations": {
            "hidden_states": str(paths.hidden_states),
            "attention": str(paths.attention),
            "generations": str(paths.generations),
            "shard_manifest": str(paths.shard_manifest),
        },
        "figures": list(figures),
        "statistical_tests": list(statistical_tests),
        "runtime": runtime,
        "storage_gb": dir_size_gb(paths.root),
        "seeds": {
            "master": config.seed,
            "generation": config.generation.seed,
            "synthetic_data": config.datasets.synthetic_seed,
            "jspace_probes": config.jspace.probe_seed,
            "bootstrap": config.stats.bootstrap_seed,
        },
    }


# ---------------------------------------------------------------------------
# Final report (protocol section 50)
# ---------------------------------------------------------------------------
def _fmt(v: Any, digits: int = 3) -> str:
    if v is None:
        return "not computed"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return "not computed" if not np.isfinite(v) else f"{float(v):.{digits}f}"
    return str(v)


def _effect_label(g: Optional[float]) -> str:
    """Conventional magnitude bands, stated as conventions."""
    if g is None or not np.isfinite(g):
        return "not estimable"
    a = abs(g)
    if a < 0.2:
        return "negligible by Cohen's convention"
    if a < 0.5:
        return "small by Cohen's convention"
    if a < 0.8:
        return "medium by Cohen's convention"
    return "large by Cohen's convention"


def generate_final_report(paths: ExperimentPaths, *, results: Dict[str, Any]
                          ) -> Path:
    """Render ``FINAL_REPORT.md`` from computed results only."""
    R = results
    L: List[str] = []
    add = L.append

    add("# Critical transitions in transformer reasoning dynamics")
    add("")
    add("## Automatically generated report")
    add("")
    add(f"*Generated {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}*")
    add("")
    add("> This report is produced mechanically from the run's own numbers. "
        "It does not assert that the phase-transition hypothesis is true, and "
        "the language is deliberately constrained to *consistent with*, "
        "*associated with* and *candidate critical transition*. The closing "
        "section lists the evidence in this run that points **against** the "
        "hypothesis.")
    add("")

    # -- 0. Research question ------------------------------------------
    add("## 0. Research question")
    add("")
    add("**Hypothesis under test.** Successful reasoning in transformer "
        "language models may contain localized critical transitions in "
        "representation dynamics, analogous to phase-transition-like "
        "behaviour.")
    add("")
    add("**Operational question.** When the model \"figures something out\", "
        "does a measurable critical transition occur in its hidden-state, "
        "logit, attention, latent-space or dynamical trajectory before the "
        "final answer is produced?")
    add("")
    add("The experiment is designed so that *no transition*, *a gradual "
        "transition*, *multiple transitions*, and *an artefactual transition "
        "caused by the unembedding geometry* all remain reachable "
        "conclusions.")
    add("")

    # -- 1-2. Samples ---------------------------------------------------
    acc = R.get("accounting", {})
    add("## 1. Samples and accounting")
    add("")
    add("| quantity | value |")
    add("| --- | --- |")
    add(f"| model | `{R.get('model_name')}` (revision `{R.get('model_revision')}`) |")
    add(f"| architecture | {_fmt(R.get('n_layers'))} layers, "
        f"hidden {_fmt(R.get('hidden_size'))}, {_fmt(R.get('n_heads'))} heads |")
    add(f"| samples requested | {_fmt(acc.get('n_requested'))} |")
    add(f"| samples analysed | {_fmt(acc.get('n_completed'))} |")
    add(f"| samples failed | {_fmt(acc.get('n_failed'))} |")
    add(f"| samples not reached (budget/interrupt) | {_fmt(acc.get('n_skipped'))} |")
    add(f"| correct | {_fmt(acc.get('n_correct'))} |")
    add(f"| incorrect | {_fmt(acc.get('n_incorrect'))} |")
    add(f"| ungraded (unparsed or no ground truth) | {_fmt(acc.get('n_ungraded'))} |")
    add("")
    if acc.get("n_ungraded"):
        add(f"> {acc['n_ungraded']} generations could not be graded. These are "
            "excluded from every correct-vs-incorrect comparison rather than "
            "counted as incorrect.")
        add("")
    by_ds = acc.get("by_dataset") or {}
    if by_ds:
        add("Per dataset:")
        add("")
        add("| dataset | n | accuracy |")
        add("| --- | --- | --- |")
        for ds, info in sorted(by_ds.items()):
            add(f"| {ds} | {info.get('n')} | {_fmt(info.get('accuracy'))} |")
        add("")

    # -- 3. Transitions -------------------------------------------------
    tr = R.get("transitions", {})
    add("## 2. Were abrupt transitions observed?")
    add("")
    shapes = tr.get("shape_counts") or {}
    total_shapes = sum(shapes.values()) or 1
    add("Classification of each sample's dominant profile shape:")
    add("")
    add("| shape | n | fraction |")
    add("| --- | --- | --- |")
    for shape, n in sorted(shapes.items(), key=lambda kv: -kv[1]):
        add(f"| {shape} | {n} | {n / total_shapes:.1%} |")
    add("")
    add(f"- Samples classified `sharp` (peak within {tr.get('sharp_max_width', 2)} "
        f"layers): **{shapes.get('sharp', 0)}** "
        f"({shapes.get('sharp', 0) / total_shapes:.1%})")
    add(f"- Samples classified `flat` (no peak distinguishable from the "
        f"profile's own variation): **{shapes.get('flat', 0)}**")
    add("")
    fp = R.get("null_models", {}).get("detector_false_positive", {})
    if fp.get("status") == "ok":
        add(f"**Calibration.** On structureless AR(1) curves of the same depth, "
            f"the same classifier labels {fp['false_sharp_rate']:.1%} of curves "
            f"`sharp`. Any observed `sharp` fraction must be read against this "
            f"floor.")
        add("")

    # -- 4-5. Critical layer location -----------------------------------
    add("## 3. Critical-layer location")
    add("")
    cons = tr.get("consensus", {})
    add(f"- Samples with a detector consensus: **{_fmt(cons.get('n_with_consensus'))}** "
        f"of {_fmt(cons.get('n_total'))} "
        f"({_fmt(cons.get('fraction_with_consensus'), 3)})")
    add(f"- Samples with **no** consensus (detectors disagreed): "
        f"**{_fmt(cons.get('n_without_consensus'))}**")
    add("")
    if cons.get("mean_normalised") is not None:
        ci = cons.get("ci", {})
        add(f"Among samples with a consensus, the critical layer sits at "
            f"normalised depth **{_fmt(cons.get('mean_normalised'))}** "
            f"(95% CI [{_fmt(ci.get('ci_low'))}, {_fmt(ci.get('ci_high'))}], "
            f"median {_fmt(cons.get('median_normalised'))}, "
            f"SD {_fmt(cons.get('std_normalised'))}).")
        add("")
        add("Normalised depth `l/L` is used throughout so the number remains "
            "meaningful if a second model of different depth is added.")
        add("")
    else:
        add("No consensus critical layer could be computed. This is reported "
            "as-is; no fallback statistic is substituted.")
        add("")

    # -- 6. Method agreement --------------------------------------------
    add("## 4. Agreement between detection methods")
    add("")
    agree = tr.get("agreement", {})
    if agree.get("mean_abs_difference"):
        add("Mean absolute difference in detected layer, per detector pair "
            "(lower = more agreement):")
        add("")
        add("| detector pair | mean absolute layer difference |")
        add("| --- | --- |")
        items = sorted(agree["mean_abs_difference"].items(), key=lambda kv: kv[1])
        for pair, v in items[:12]:
            add(f"| {pair.replace('|', ' vs ')} | {_fmt(v, 2)} |")
        add("")
    add(f"- Mean pairwise agreement rate (within tolerance): "
        f"{_fmt(tr.get('mean_pairwise_agreement'))}")
    add("")
    add("> `order_parameter_growth` and `margin_growth` are computed from the "
        "same underlying profile. Their agreement is arithmetic, not "
        "corroborative, and should be discounted when reading the matrix.")
    add("")

    # -- 7. Correct vs incorrect ----------------------------------------
    add("## 5. Correct vs incorrect")
    add("")
    cvi = R.get("correct_vs_incorrect", [])
    if cvi:
        add("| measure | n correct | n incorrect | effect size | 95% CI | "
            "p (corrected) | significant |")
        add("| --- | --- | --- | --- | --- | --- | --- |")
        for row in cvi:
            ci = f"[{_fmt(row.get('diff_ci_low'))}, {_fmt(row.get('diff_ci_high'))}]"
            add(f"| {row.get('label')} | {_fmt(row.get('n_a'))} | "
                f"{_fmt(row.get('n_b'))} | {_fmt(row.get('effect_size'))} | "
                f"{ci} | {_fmt(row.get('p_value_corrected'), 4)} | "
                f"{_fmt(row.get('significant'))} |")
        add("")
        n_sig = sum(1 for r in cvi if r.get("significant") is True)
        add(f"{n_sig} of {len(cvi)} comparisons survive "
            f"{cvi[0].get('correction_method', 'FDR')} correction across this "
            f"family of {len(cvi)} tests.")
        add("")
    else:
        add("No correct-vs-incorrect comparison could be run (insufficient "
            "graded samples in one or both groups).")
        add("")

    # -- 8. Cross-dataset -----------------------------------------------
    add("## 6. Cross-dataset comparison")
    add("")
    xd = R.get("cross_dataset", {})
    if xd.get("per_dataset"):
        add("| dataset | n | mean normalised critical layer | sharp fraction |")
        add("| --- | --- | --- | --- |")
        for ds, info in sorted(xd["per_dataset"].items()):
            add(f"| {ds} | {_fmt(info.get('n'))} | "
                f"{_fmt(info.get('mean_normalised_critical_layer'))} | "
                f"{_fmt(info.get('sharp_fraction'))} |")
        add("")
        add(f"Between-dataset test of critical-layer location: "
            f"{_fmt(xd.get('test_description'))}, "
            f"p = {_fmt(xd.get('p_value'), 4)}.")
        add("")
        add("This distinguishes a **universal** transition from a "
            "**task-specific** one, and the answer changes what the phenomenon "
            "could be.")
        add("")
    else:
        add("Only one dataset contributed enough samples; the "
            "universal-vs-task-specific question cannot be addressed by this "
            "run.")
        add("")

    # -- 9. Cross-model -------------------------------------------------
    add("## 7. Cross-model comparison")
    add("")
    xm = R.get("cross_model", {})
    if xm.get("models"):
        add("| model | layers | mean l/L | n |")
        add("| --- | --- | --- | --- |")
        for m, info in sorted(xm["models"].items()):
            add(f"| {m} | {_fmt(info.get('n_layers'))} | "
                f"{_fmt(info.get('mean_normalised'))} | {_fmt(info.get('n'))} |")
        add("")
    else:
        add("**Only one model was run.** Nothing in this report can speak to "
            "universality across architectures. Any depth-normalised location "
            "reported above is a property of this one model.")
        add("")

    # -- 10. Causal -----------------------------------------------------
    add("## 8. Causal intervention")
    add("")
    ci_res = R.get("causal", {})
    if ci_res.get("status") == "ok":
        add(f"- Perturbation outcomes recorded: {_fmt(ci_res.get('n_outcomes'))} "
            f"across {_fmt(ci_res.get('n_samples'))} samples")
        add(f"- Mean output JSD at the **critical** layer: "
            f"{_fmt(ci_res.get('critical_mean'), 4)}")
        add(f"- Mean output JSD at a **random control** layer: "
            f"{_fmt(ci_res.get('random_control_mean'), 4)}")
        d = ci_res.get("cohens_d_critical_vs_random")
        add(f"- Critical vs random-layer effect size: {_fmt(d)} "
            f"({_effect_label(d)})")
        p = ci_res.get("p_value_corrected", ci_res.get("p_value"))
        add(f"- p (corrected): {_fmt(p, 4)}")
        add("")
        if d is not None and np.isfinite(d) and abs(d) < 0.2:
            add("> The critical layer is **not** measurably more sensitive to "
                "perturbation than a randomly chosen layer at matched "
                "magnitude. This is evidence against a causally privileged "
                "critical layer, whatever the correlational profiles show.")
            add("")
        if ci_res.get("regeneration_agreement") is not None:
            add(f"- Next-token proxy agreed with full re-generation on "
                f"{_fmt(ci_res['regeneration_agreement'], 3)} of validated cases. "
                "Where this is low, the sweep's conclusions do not transfer to "
                "the model's actual behaviour.")
            add("")
    else:
        add(f"Causal intervention did not run to completion "
            f"(`{ci_res.get('status', 'not run')}`). Without it, every result "
            "above is correlational and cannot support a claim that any layer "
            "*causes* the answer.")
        add("")

    # -- 11. Null models ------------------------------------------------
    add("## 9. Null-model comparisons")
    add("")
    nulls = R.get("null_models", {})
    if nulls:
        add("| null model | observed | null 95th pct | p | exceeds null |")
        add("| --- | --- | --- | --- | --- |")
        for name, res in sorted(nulls.items()):
            if not isinstance(res, dict) or res.get("status") != "ok":
                continue
            if name == "detector_false_positive":
                continue  # different schema; reported in section 2 instead
            obs = res.get("observed", res.get("observed_sharpness"))
            add(f"| {name} | {_fmt(obs)} | {_fmt(res.get('null_p95'))} | "
                f"{_fmt(res.get('p_value'), 4)} | "
                f"{_fmt(res.get('exceeds_null'))} |")
        add("")
        failed_nulls = [n for n, r in nulls.items()
                        if isinstance(r, dict) and r.get("exceeds_null") is False
                        and n != "detector_false_positive"]
        if failed_nulls:
            add(f"> The observed signal does **not** exceed the null for: "
                f"{', '.join(failed_nulls)}. For those measures the apparent "
                "structure is within what a structureless process produces.")
            add("")

    # -- 12-13. Statistics ----------------------------------------------
    add("## 10. Statistical summary")
    add("")
    tests = R.get("statistical_tests", [])
    add(f"- Tests run: {len(tests)}")
    add(f"- Correction: {R.get('correction_method', 'fdr_bh')} across each "
        "test family")
    n_sig = sum(1 for t in tests if t.get("significant") is True)
    add(f"- Surviving correction: {n_sig}")
    add(f"- Bootstrap resamples: {_fmt(R.get('n_bootstrap'))}; permutations: "
        f"{_fmt(R.get('n_permutation'))}")
    add("")
    add("Effect sizes and confidence intervals are the primary evidence; "
        "p-values are reported for completeness. At these sample sizes a "
        "small p-value with a negligible effect size is not a finding.")
    add("")

    # -- 14. Confounds ---------------------------------------------------
    add("## 11. Confounds tested")
    add("")
    conf = R.get("confounds", {})
    if conf.get("correlations"):
        add("Spearman correlation between the critical-layer location and "
            "candidate confounds:")
        add("")
        add("| confound | rho | n |")
        add("| --- | --- | --- |")
        for name, res in sorted(conf["correlations"].items()):
            if isinstance(res, dict) and res.get("status") == "ok":
                add(f"| {name} | {_fmt(res.get('spearman_rho'))} | "
                    f"{_fmt(res.get('n'))} |")
        add("")
    for key, title in [("adjusted", "Effect after controlling for prompt "
                                    "length, generation length and baseline "
                                    "confidence"),
                       ("matched", "Effect on a confound-matched subsample")]:
        info = conf.get(key)
        if isinstance(info, dict) and info.get("status") == "ok":
            add(f"**{title}.** Raw effect size "
                f"{_fmt(info.get('raw_effect_size'))} -> adjusted "
                f"{_fmt(info.get('adjusted_effect_size'))} "
                f"(attenuation {_fmt(info.get('effect_attenuation'))}).")
            add("")
    nn = R.get("no_norm_control", {})
    if nn.get("mean_correlation") is not None:
        add(f"**Final-normalisation control.** Correlation between the "
            f"with-norm and without-norm entropy profiles: "
            f"{_fmt(nn['mean_correlation'])} (mean over "
            f"{_fmt(nn.get('n_samples'))} samples). A low value would mean the "
            "layer-wise entropy shape is largely produced by the final "
            "normalisation rather than by the representation.")
        add("")

    # -- 15-16. Failures and gaps ---------------------------------------
    add("## 12. Failed samples and missing analyses")
    add("")
    add(f"- Samples that failed: {_fmt(acc.get('n_failed'))}")
    err_types = R.get("error_types") or {}
    if err_types:
        add("- Failure types: " + ", ".join(f"`{k}` ({v})"
                                            for k, v in sorted(err_types.items())))
    missing = R.get("missing_analyses") or []
    if missing:
        add("- Analyses **not** completed in this run:")
        for m in missing:
            add(f"  - {m}")
    else:
        add("- All configured analyses completed.")
    add("")

    # -- 17. Storage/runtime --------------------------------------------
    add("## 13. Storage and runtime")
    add("")
    rt = R.get("runtime", {})
    add(f"- Wall-clock elapsed: {_fmt(rt.get('elapsed_hours'), 2)} h")
    add(f"- Storage used: {_fmt(R.get('storage_gb'), 2)} GB")
    thr = rt.get("throughput", {})
    for phase, stats in sorted(thr.items()):
        if stats.get("seconds_per_unit"):
            add(f"- `{phase}`: {_fmt(stats['seconds_per_unit'], 2)} s/sample "
                f"({_fmt(stats.get('units_per_hour'), 1)} samples/h)")
    add("")

    # -- 18-19. Interpretation ------------------------------------------
    add("## 14. Interpretation")
    add("")
    add("What the measurements *do* support, at most:")
    add("")
    add("- Layer-wise profiles of entropy, order parameter, distributional "
        "movement and representation geometry were computed for every sample "
        "and are internally consistent.")
    if shapes.get("sharp", 0) and fp.get("false_sharp_rate") is not None:
        frac = shapes.get("sharp", 0) / total_shapes
        if frac > fp["false_sharp_rate"] * 2:
            add(f"- The fraction of samples with a localised peak ({frac:.1%}) "
                f"exceeds the structureless-null floor "
                f"({fp['false_sharp_rate']:.1%}). This is **consistent with** "
                "localised change in the measured quantities.")
        else:
            add(f"- The fraction with a localised peak ({frac:.1%}) is not "
                f"clearly above the structureless-null floor "
                f"({fp['false_sharp_rate']:.1%}).")
    add("")
    add("## 15. Alternative explanations")
    add("")
    add("Each of these would produce results resembling a critical transition "
        "without any transition existing in the computation:")
    add("")
    add("1. **Unembedding geometry.** The logit lens sharpens toward the final "
        "layer because that is where the readout is calibrated, not because "
        "the representation reorganises. Addressed by the no-norm and "
        "shuffled-unembedding controls; read those before the main result.")
    add("2. **Residual-norm growth.** Raw velocity and distance measures grow "
        "with depth architecturally. Scale-free counterparts "
        "(`velocity_normalised`, `turning_angle`, kNN-Jaccard) are the ones "
        "that carry information.")
    add("3. **Averaging artefact.** Gradual transitions occurring at different "
        "layers average into a sharp population curve. Individual "
        "trajectories are plotted alongside every mean for this reason.")
    add("4. **Peak-finding on smooth noise.** Any smooth curve has a maximum. "
        "The AR(1) null quantifies how often that maximum looks 'sharp'.")
    add("5. **Answer-token identity and frequency.** Common answer tokens are "
        "resolved earlier in depth regardless of reasoning.")
    add("6. **Generation-length and prompt-length effects.** Longer prompts "
        "shift where in the sequence the answer forms; the confound analysis "
        "tests this directly.")
    add("7. **Quantization and precision.** 4-bit weights add noise that is "
        "not uniform across layers.")
    add("")

    # -- 20. Evidence against -------------------------------------------
    add("## 16. Evidence AGAINST the hypothesis in this run")
    add("")
    against = R.get("evidence_against") or []
    if against:
        for item in against:
            add(f"- {item}")
    else:
        add("- No pre-specified disconfirming criterion was triggered. This is "
            "*not* the same as confirmation: it means the run did not produce "
            "results that meet the invalidation criteria below.")
    add("")

    # -- Mandatory closing section --------------------------------------
    add("## WHAT WOULD INVALIDATE THE PHASE-TRANSITION HYPOTHESIS?")
    add("")
    add("Stated in advance of interpretation, and evaluated against this run "
        "where the data permit:")
    add("")
    add("1. **The transition is not localised.** If most samples classify as "
        "`distributed` or `diffuse` rather than `sharp`, the phenomenon is a "
        "gradual accumulation and the phase-transition framing does not apply. "
        f"*This run: {shapes.get('sharp', 0)} sharp vs "
        f"{shapes.get('distributed', 0) + shapes.get('diffuse', 0)} "
        "distributed/diffuse.*")
    add("")
    add("2. **Detectors disagree.** If independent measures locate the "
        "transition at unrelated depths, there is no single underlying event "
        "and the 'critical layer' is an artefact of whichever measure was "
        f"chosen. *This run: consensus reached for "
        f"{_fmt(cons.get('n_with_consensus'))} of {_fmt(cons.get('n_total'))} "
        "samples.*")
    add("")
    add("3. **The signal does not exceed matched nulls.** If layer-permuted or "
        "AR(1)-null curves produce equally prominent peaks, the peak carries "
        "no information about the computation.")
    add("")
    add("4. **Perturbation at the critical layer is no more effective than at "
        "a random layer.** This is the decisive test. Correlational sharpness "
        "without causal privilege means the transition is a readout of "
        "computation happening elsewhere, not the computation itself. "
        f"*This run: critical-vs-random effect size "
        f"{_fmt(ci_res.get('cohens_d_critical_vs_random'))}.*")
    add("")
    add("5. **The effect disappears when the final norm is removed.** If the "
        "no-norm control shows no transition, the phenomenon lives in the "
        "readout geometry rather than in the residual stream.")
    add("")
    add("6. **Correct and incorrect answers show the same dynamics.** If "
        "trajectories are indistinguishable, whatever the transition marks is "
        "not the difference between reasoning that succeeds and reasoning that "
        "fails.")
    add("")
    add("7. **The location is fully explained by confounds.** If controlling "
        "for prompt length, generation length and baseline confidence removes "
        "the effect, the transition is a property of the prompt, not of "
        "reasoning.")
    add("")
    add("8. **It does not replicate across models.** A transition at a "
        "particular normalised depth in one model, absent in another of "
        "similar capability, is a fact about that model's training, not about "
        "transformer reasoning.")
    add("")
    add("9. **It is token-local rather than computation-local.** If the "
        "'transition' appears only at the final answer token and nowhere in "
        "the preceding generated positions, it describes answer emission "
        "rather than a change in internal state.")
    add("")
    add("---")
    add("")
    add("### Terminology used in this report")
    add("")
    add("- *candidate critical layer* -- the peak of a measured profile. A "
        "location, not a mechanism.")
    add("- *empirical susceptibility-like measure* -- across-sample variance "
        "of an order-parameter analogue. No fluctuation-dissipation relation "
        "is established and no critical exponent is claimed.")
    add("- *symmetry-breaking index* -- an operational concentration measure. "
        "No symmetry group is being broken in any formal sense.")
    add("- *J-space* -- the space of finite-difference directional sensitivity "
        "descriptors defined in `nl2ms/jspace.py`. Not a Jacobian in full.")
    add("- *order parameter* -- used by analogy for the answer margin. Its "
        "closed-set and open-vocabulary forms are on different scales and are "
        "never pooled.")
    add("")
    add(f"Complete provenance: `{paths.experiment_manifest.name}`, "
        f"`config/experiment_config.json`, `config/signal_registry.json`. "
        f"Every figure has a matching `.json` naming its source samples.")
    add("")

    text = "\n".join(L)
    from .storage import atomic_path
    with atomic_path(paths.final_report) as tmp:
        tmp.write_text(text)
    if not paths.final_report.exists() or paths.final_report.stat().st_size == 0:
        raise IOError("final report write verification failed")
    save_json(paths.reports / "experiment_summary.json", R)
    return paths.final_report


def collect_evidence_against(R: Dict[str, Any]) -> List[str]:
    """Enumerate disconfirming findings from the computed results.

    This is the counterweight to every optimistic reading: it inspects the
    same numbers and reports what argues against the hypothesis.
    """
    out: List[str] = []
    tr = R.get("transitions", {})
    shapes = tr.get("shape_counts") or {}
    total = sum(shapes.values())
    if total:
        sharp_frac = shapes.get("sharp", 0) / total
        diffuse = (shapes.get("distributed", 0) + shapes.get("diffuse", 0)) / total
        if sharp_frac < 0.25:
            out.append(f"Only {sharp_frac:.1%} of samples show a localised "
                       f"(`sharp`) transition; {diffuse:.1%} are distributed or "
                       "diffuse, which is more consistent with gradual "
                       "accumulation than with a localised critical transition.")
        fp = R.get("null_models", {}).get("detector_false_positive", {})
        if fp.get("status") == "ok" and sharp_frac <= fp["false_sharp_rate"]:
            out.append(f"The `sharp` fraction ({sharp_frac:.1%}) does not "
                       f"exceed the structureless-null floor "
                       f"({fp['false_sharp_rate']:.1%}).")

    cons = tr.get("consensus", {})
    frac = cons.get("fraction_with_consensus")
    if frac is not None and frac < 0.5:
        out.append(f"Detectors reached consensus on only {frac:.1%} of samples; "
                   "independent measures largely disagree about where any "
                   "transition occurs.")
    if cons.get("std_normalised") is not None and cons["std_normalised"] > 0.2:
        out.append(f"Critical-layer location varies widely across samples "
                   f"(SD {cons['std_normalised']:.3f} in normalised depth), "
                   "which is difficult to reconcile with a single "
                   "architecturally-fixed transition.")

    causal = R.get("causal", {})
    d = causal.get("cohens_d_critical_vs_random")
    if d is not None and np.isfinite(d) and abs(d) < 0.2:
        out.append(f"Perturbing the candidate critical layer is no more "
                   f"effective than perturbing a random layer at matched "
                   f"magnitude (effect size {d:.3f}). The correlational "
                   "transition has no demonstrated causal privilege.")
    if causal.get("status") not in ("ok", None):
        out.append("Causal intervention did not complete, so no causal claim "
                   "is supported by this run.")

    nulls = R.get("null_models", {})
    for name, res in nulls.items():
        if name == "detector_false_positive":
            continue
        if isinstance(res, dict) and res.get("exceeds_null") is False:
            pv = res.get("p_value")
            pv_str = f"{pv:.3f}" if isinstance(pv, float) and np.isfinite(pv) \
                else str(pv)
            out.append(f"The observed signal does not exceed the `{name}` null "
                       f"(p = {pv_str}).")

    cvi = R.get("correct_vs_incorrect", [])
    if cvi:
        n_sig = sum(1 for r in cvi if r.get("significant") is True)
        if n_sig == 0:
            out.append(f"None of the {len(cvi)} correct-vs-incorrect "
                       "comparisons survives multiple-comparison correction; "
                       "the dynamics of successful and unsuccessful answers "
                       "are not distinguishable in this run.")

    nn = R.get("no_norm_control", {})
    if nn.get("mean_correlation") is not None and nn["mean_correlation"] < 0.5:
        out.append(f"With-norm and without-norm entropy profiles correlate at "
                   f"only {nn['mean_correlation']:.3f}, indicating the final "
                   "normalisation contributes substantially to the apparent "
                   "layer-wise structure.")

    conf = R.get("confounds", {})
    adj = conf.get("adjusted")
    if isinstance(adj, dict) and adj.get("effect_attenuation") is not None:
        if np.isfinite(adj["effect_attenuation"]) and adj["effect_attenuation"] > 0.5:
            out.append(f"Controlling for prompt length, generation length and "
                       f"baseline confidence attenuates the effect by "
                       f"{adj['effect_attenuation']:.1%}, so much of it is "
                       "attributable to those confounds.")

    if R.get("n_models", 1) < 2:
        out.append("Only one model was measured, so no claim about "
                   "architecture-general behaviour is supported.")

    acc = R.get("accounting", {})
    if acc.get("n_completed", 0) < 50:
        out.append(f"Only {acc.get('n_completed')} samples completed analysis, "
                   "which is too few for stable population-level estimates "
                   "such as susceptibility and effective rank.")
    return out
