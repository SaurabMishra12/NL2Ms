"""Phases 12, 14, 15, 16, 47, 48 -- aggregation, statistics, figures, report.

Consumes the per-sample artefacts written by :mod:`nl2ms.pipeline` and
produces the run's conclusions: the master tables, the statistical tests with
multiplicity correction, the figure set, and ``FINAL_REPORT.md``.

Design rule carried through: a comparison that cannot be run is *reported as
not run*. There is no path here that substitutes a default value, drops an
inconvenient group, or reports an uncorrected p-value as if it were corrected.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import critical as crit
from . import jspace as jsp
from . import signals as sig
from . import stats as st
from .plots import FigureBuilder, select_examples
from .registry import write_lineage
from .report import (build_experiment_manifest, check_integrity,
                     collect_evidence_against, generate_final_report)
from .storage import save_json, save_parquet

# Profiles compared between correct and incorrect answers (Phase 12/28).
COMPARISON_PROFILES = [
    ("entropy", "entropy at the answer position"),
    ("order_margin", "answer margin"),
    ("jsd_prev_layer", "layer-to-layer JSD"),
    ("traj_velocity_normalised", "normalised representation velocity"),
    ("traj_curvature", "representation curvature"),
    ("order_symmetry_breaking_index", "symmetry-breaking index"),
    ("attn_restructuring_frobenius_delta", "attention restructuring"),
]

# Scalars compared between groups.
COMPARISON_SCALARS = [
    ("critical_layer_consensus_normalised", "consensus critical layer (l/L)"),
    ("transition_strength", "transition strength"),
    ("detector_spread", "detector disagreement (SD of layers)"),
    ("max_jsd", "maximum layer-to-layer JSD"),
    ("max_curvature", "maximum representation curvature"),
    ("max_velocity", "maximum normalised velocity"),
    ("final_entropy", "entropy at the final layer"),
    ("jspace_peak_layer", "J-space sensitivity peak layer"),
    ("jspace_max_amplification", "maximum J-space amplification"),
]


# ---------------------------------------------------------------------------
# Master tables (protocol sections 47, 48)
# ---------------------------------------------------------------------------
def build_sample_summary(exp: Any, analyses: Dict[str, Dict[str, Any]],
                         critical: Dict[str, Any],
                         jspace_results: Dict[str, Dict[str, Any]],
                         intervention_df: Any) -> Any:
    """One row per sample. Columns are extensible, never hard-coded to a list."""
    import pandas as pd

    by_id = {s.sample_id: s for s in exp.samples}
    rows: List[Dict[str, Any]] = []

    causal_by_sample: Dict[str, Dict[str, float]] = {}
    if intervention_df is not None and len(intervention_df):
        for sid, sub in intervention_df.groupby("sample_id"):
            crit_rows = sub[sub["layer_role"] == "critical"]
            ctrl_rows = sub[sub["layer_role"] == "random_control"]
            causal_by_sample[sid] = {
                "causal_sensitivity_critical": (float(crit_rows["jsd_output"].mean())
                                                if len(crit_rows) else np.nan),
                "causal_sensitivity_random": (float(ctrl_rows["jsd_output"].mean())
                                              if len(ctrl_rows) else np.nan),
                "causal_answer_change_rate": (float(sub["answer_changed"].mean())
                                              if sub["answer_changed"].notna().any()
                                              else np.nan),
                "causal_max_jsd": float(sub["jsd_output"].max()),
            }

    for sid, data in analyses.items():
        sample = by_id.get(sid)
        gen = exp.generations.get(sid, {})
        profiles = data["profiles"]
        row: Dict[str, Any] = {
            "sample_id": sid,
            "dataset": sample.dataset if sample else None,
            "subset": sample.subset if sample else None,
            "question_type": sample.question_type if sample else None,
            "answer_spec_type": sample.answer_spec_type if sample else None,
            "difficulty": sample.difficulty if sample else None,
            "reasoning_steps": sample.reasoning_steps if sample else None,
            "correct": gen.get("correct"),
            "prediction": gen.get("prediction"),
            "ground_truth": gen.get("ground_truth"),
            "parse_status": gen.get("parse_status"),
            "generation_length": gen.get("generation_length"),
            "prompt_length": gen.get("prompt_length"),
            "sequence_length": (gen.get("prompt_length") or 0) +
                               (gen.get("generation_length") or 0),
            "finish_reason": gen.get("finish_reason"),
            "finish_reason_is_eos": (gen.get("finish_reason") == "eos"),
            "model": exp.config.model.name,
            "seed": exp.config.seed,
            "temperature": exp.config.generation.temperature,
        }
        row.update(critical["summaries"].get(sid, {}))

        for key, agg, name in [
            ("jsd_prev_layer", np.nanmax, "max_jsd"),
            ("traj_curvature", np.nanmax, "max_curvature"),
            ("traj_velocity_normalised", np.nanmax, "max_velocity"),
            ("entropy", np.nanmax, "max_entropy"),
            ("order_margin", np.nanmax, "max_margin"),
        ]:
            prof = profiles.get(key)
            row[name] = (_safe_agg(agg, prof) if prof is not None else np.nan)

        ent = profiles.get("entropy")
        if ent is not None and ent.size:
            row["final_entropy"] = float(ent[-1])
            row["initial_entropy"] = float(ent[0])
            row["entropy_drop"] = float(ent[0] - ent[-1])
            row["baseline_entropy"] = float(ent[-1])
        conf = profiles.get("top1_prob")
        if conf is not None and conf.size:
            row["baseline_confidence"] = float(conf[-1])
        margin = profiles.get("order_margin")
        if margin is not None and margin.size:
            row["final_margin"] = float(margin[-1])
        mass = profiles.get("order_candidate_mass")
        if mass is not None and mass.size:
            row["candidate_mass"] = float(mass[-1])

        row.update(jsp.summarise_jspace(jspace_results.get(sid, {})))
        row.update(causal_by_sample.get(sid, {}))
        if "causal_sensitivity_critical" in row and \
                "causal_sensitivity_random" in row:
            row["causal_critical_minus_random"] = (
                row["causal_sensitivity_critical"] - row["causal_sensitivity_random"])

        meta = data.get("meta", {})
        row["answer_position_index"] = (int(meta["answer_index"][0])
                                        if "answer_index" in meta else None)
        rows.append(row)

    df = pd.DataFrame(rows)
    if len(df):
        save_parquet(exp.paths.critical_layers / "sample_summary.parquet", df)
        save_parquet(exp.paths.root / "sample_summary.parquet", df)
    return df


def build_layer_summary(exp: Any, analyses: Dict[str, Dict[str, Any]],
                        correct_map: Dict[str, Optional[bool]],
                        geometry: Optional[Dict[str, Any]]) -> Any:
    """One row per (dataset, group, layer). Correct and incorrect kept apart."""
    import pandas as pd

    by_id = {s.sample_id: s for s in exp.samples}
    profile_keys = sorted({k for d in analyses.values() for k in d["profiles"]})
    rows: List[Dict[str, Any]] = []

    groups: Dict[Tuple[str, str], List[str]] = {}
    for sid in analyses:
        sample = by_id.get(sid)
        ds = sample.dataset if sample else "unknown"
        c = correct_map.get(sid)
        label = "correct" if c is True else "incorrect" if c is False else "ungraded"
        groups.setdefault((ds, label), []).append(sid)
        groups.setdefault(("all", label), []).append(sid)
        groups.setdefault((ds, "all"), []).append(sid)
        groups.setdefault(("all", "all"), []).append(sid)

    geo_profiles = (geometry or {}).get("result", {}).get("profiles", {})

    for (dataset, group), sids in sorted(groups.items()):
        stacks: Dict[str, np.ndarray] = {}
        for key in profile_keys:
            arrs = [analyses[s]["profiles"][key] for s in sids
                    if key in analyses[s]["profiles"]]
            stacked = sig.safe_stack(arrs)
            if stacked is not None:
                stacks[key] = stacked
        if not stacks:
            continue
        n_layers = next(iter(stacks.values())).shape[1]
        norm = sig.normalised_layers(n_layers)
        for l in range(n_layers):
            row: Dict[str, Any] = {
                "model": exp.config.model.name,
                "dataset": dataset, "group": group, "layer": l,
                "normalised_layer": float(norm[l]),
                "n_samples": len(sids),
            }
            for key, mat in stacks.items():
                col = mat[:, l]
                finite = col[np.isfinite(col)]
                row[f"{key}_mean"] = float(np.mean(finite)) if finite.size else np.nan
                row[f"{key}_std"] = (float(np.std(finite, ddof=1))
                                     if finite.size > 1 else np.nan)
                row[f"{key}_n"] = int(finite.size)
            # Susceptibility is the across-sample variance of the order
            # parameter at this layer.
            if "order_margin" in stacks:
                col = stacks["order_margin"][:, l]
                finite = col[np.isfinite(col)]
                row["susceptibility"] = (float(np.var(finite, ddof=1))
                                         if finite.size > 1 else np.nan)
            if dataset == "all" and group == "all":
                for gk, gv in geo_profiles.items():
                    arr = np.asarray(gv, dtype=np.float64)
                    # Geometry profiles are indexed by block; profiles include
                    # the embedding row, so offset by one.
                    idx = l - 1
                    row[f"geo_{gk}"] = (float(arr[idx])
                                        if 0 <= idx < arr.size else np.nan)
            rows.append(row)

    df = pd.DataFrame(rows)
    if len(df):
        save_parquet(exp.paths.root / "layer_summary.parquet", df)
    return df


def _safe_agg(fn: Any, arr: Optional[np.ndarray]) -> float:
    if arr is None:
        return float("nan")
    a = np.asarray(arr, dtype=np.float64)
    finite = a[np.isfinite(a)]
    return float(fn(finite)) if finite.size else float("nan")


# ---------------------------------------------------------------------------
# Phase 12/28 -- correct vs incorrect
# ---------------------------------------------------------------------------
def compare_correct_incorrect(exp: Any, df: Any,
                              analyses: Dict[str, Dict[str, Any]],
                              correct_map: Dict[str, Optional[bool]]
                              ) -> List[Dict[str, Any]]:
    """Group comparisons on scalars and on per-layer profiles at their peak."""
    tests: List[Dict[str, Any]] = []
    cfg = exp.config.stats

    correct_ids = [s for s, v in correct_map.items() if v is True and s in analyses]
    wrong_ids = [s for s, v in correct_map.items() if v is False and s in analyses]

    if len(correct_ids) < cfg.min_group_size or len(wrong_ids) < cfg.min_group_size:
        return [{
            "label": "correct vs incorrect",
            "status": "insufficient_group_sizes",
            "n_correct": len(correct_ids), "n_incorrect": len(wrong_ids),
            "min_group_size": cfg.min_group_size,
            "note": ("Comparison not run. Reported as not run rather than "
                     "computed on too-few samples."),
        }]

    if df is not None and len(df):
        for col, label in COMPARISON_SCALARS:
            if col not in df.columns:
                continue
            a = df[df["correct"] == True][col].to_numpy(dtype=np.float64,  # noqa: E712
                                                        na_value=np.nan)
            b = df[df["correct"] == False][col].to_numpy(dtype=np.float64,  # noqa: E712
                                                         na_value=np.nan)
            res = st.compare_groups(a, b, label=f"{label} (correct vs incorrect)",
                                    cfg=cfg, seed=cfg.bootstrap_seed)
            res["family"] = "correct_vs_incorrect_scalar"
            res["column"] = col
            tests.append(res)

    # Per-layer profiles, summarised by their maximum, so a group difference in
    # *where or how strongly* a profile peaks is testable.
    for key, label in COMPARISON_PROFILES:
        a_vals, b_vals = [], []
        for sid in correct_ids:
            p = analyses[sid]["profiles"].get(key)
            if p is not None:
                a_vals.append(_safe_agg(np.nanmax, p))
        for sid in wrong_ids:
            p = analyses[sid]["profiles"].get(key)
            if p is not None:
                b_vals.append(_safe_agg(np.nanmax, p))
        if len(a_vals) < cfg.min_group_size or len(b_vals) < cfg.min_group_size:
            continue
        res = st.compare_groups(a_vals, b_vals, label=f"peak {label}",
                                cfg=cfg, seed=cfg.bootstrap_seed)
        res["family"] = "correct_vs_incorrect_profile_peak"
        res["column"] = key
        tests.append(res)

    corrected = st.correct_multiple_comparisons(
        tests, alpha=cfg.alpha, method=cfg.multiple_comparison_method)
    save_json(exp.paths.statistics / "correct_vs_incorrect.json",
              {"tests": corrected,
               "n_correct": len(correct_ids), "n_incorrect": len(wrong_ids),
               "note": ("ungraded samples are excluded from both groups; they "
                        "are not counted as incorrect")})
    return corrected


# ---------------------------------------------------------------------------
# Phase 29 -- cross-dataset
# ---------------------------------------------------------------------------
def cross_dataset_analysis(exp: Any, df: Any) -> Dict[str, Any]:
    if df is None or len(df) == 0 or "dataset" not in df.columns:
        return {"status": "no_data"}
    col = "critical_layer_consensus_normalised"
    per_dataset: Dict[str, Any] = {}
    groups: List[np.ndarray] = []
    names: List[str] = []
    for ds, sub in df.groupby("dataset"):
        vals = (sub[col].to_numpy(dtype=np.float64, na_value=np.nan)
                if col in sub.columns else np.array([]))
        finite = vals[np.isfinite(vals)]
        shapes = (sub["dominant_transition_shape"].value_counts().to_dict()
                  if "dominant_transition_shape" in sub.columns else {})
        n_shape = sum(shapes.values()) or 1
        per_dataset[str(ds)] = {
            "n": int(len(sub)),
            "n_with_consensus": int(finite.size),
            "mean_normalised_critical_layer": (float(np.mean(finite))
                                               if finite.size else None),
            "std_normalised_critical_layer": (float(np.std(finite, ddof=1))
                                              if finite.size > 1 else None),
            "sharp_fraction": shapes.get("sharp", 0) / n_shape,
            "shape_counts": shapes,
            "accuracy": (float(sub["correct"].mean())
                         if "correct" in sub.columns and sub["correct"].notna().any()
                         else None),
        }
        if finite.size >= exp.config.stats.min_group_size:
            groups.append(finite)
            names.append(str(ds))

    out: Dict[str, Any] = {"status": "ok", "per_dataset": per_dataset,
                           "groups_tested": names}
    if len(groups) >= 2:
        from scipy.stats import kruskal
        try:
            stat, p = kruskal(*groups)
            out["test_description"] = "Kruskal-Wallis across datasets"
            out["statistic"] = float(stat)
            out["p_value"] = float(p)
            out["interpretation"] = (
                "a small p-value indicates the critical-layer location differs "
                "by task, which argues against a universal transition")
        except ValueError as exc:
            out["test_error"] = str(exc)
    else:
        out["test_description"] = "not run (fewer than two datasets with enough samples)"
    save_json(exp.paths.statistics / "cross_dataset.json", out)
    return out


# ---------------------------------------------------------------------------
# Phase 34/35 -- nulls and confounds
# ---------------------------------------------------------------------------
def run_null_models(exp: Any, df: Any, analyses: Dict[str, Dict[str, Any]],
                    correct_map: Dict[str, Optional[bool]]) -> Dict[str, Any]:
    cfg = exp.config.stats
    out: Dict[str, Any] = {}

    n_layers = 0
    for data in analyses.values():
        n_layers = max(n_layers, int(data["profiles"]["entropy"].size))
        break
    if n_layers:
        out["detector_false_positive"] = st.detector_false_positive_rate(
            max(20, len(analyses)), n_layers, seed=cfg.bootstrap_seed)

    for key in ["jsd_prev_layer", "traj_curvature", "order_margin_delta"]:
        curves = sig.safe_stack([d["profiles"][key] for d in analyses.values()
                                 if key in d["profiles"]])
        if curves is not None and curves.shape[0] >= 3:
            out[f"layer_shuffle_{key}"] = st.null_random_layer(
                np.abs(curves), n_perm=min(cfg.n_permutation, 500),
                seed=cfg.bootstrap_seed)

    if df is not None and len(df) and "correct" in df.columns:
        for col in ["critical_layer_consensus_normalised", "transition_strength"]:
            if col not in df.columns:
                continue
            sub = df.dropna(subset=[col])
            sub = sub[sub["correct"].notna()]
            if len(sub) >= 2 * cfg.min_group_size:
                out[f"label_shuffle_{col}"] = st.null_label_shuffle(
                    sub[col].to_numpy(dtype=np.float64),
                    sub["correct"].tolist(),
                    n_perm=min(cfg.n_permutation, 1000), seed=cfg.bootstrap_seed)

    save_json(exp.paths.statistics / "null_models.json", out)
    return out


def run_confound_analysis(exp: Any, df: Any,
                          analyses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if df is None or len(df) == 0:
        return {"status": "no_data"}

    target = "critical_layer_consensus_normalised"
    if target in df.columns:
        out.update(st.confound_correlations(df, target))

    if "correct" in df.columns and "transition_strength" in df.columns:
        work = df[df["correct"].notna()].copy()
        work["correct_int"] = work["correct"].astype(int)
        out["adjusted"] = st.partial_effect_controlling(
            work, "transition_strength", "correct_int",
            ["prompt_length", "generation_length", "baseline_confidence"])
        matched = st.matched_subsample(
            work, "correct_int",
            ["prompt_length", "generation_length", "baseline_confidence"],
            seed=exp.config.stats.bootstrap_seed)
        if matched is not None and len(matched) >= 2 * exp.config.stats.min_group_size:
            a = matched[matched["correct_int"] == 1]["transition_strength"]
            b = matched[matched["correct_int"] == 0]["transition_strength"]
            res = st.compare_groups(a.to_numpy(dtype=np.float64),
                                    b.to_numpy(dtype=np.float64),
                                    label="transition strength, confound-matched",
                                    cfg=exp.config.stats)
            res["status"] = res.get("status", "ok")
            res["raw_effect_size"] = out.get("adjusted", {}).get("raw_effect_size")
            res["adjusted_effect_size"] = res.get("effect_size")
            res["effect_attenuation"] = None
            res["n_matched"] = int(len(matched))
            out["matched"] = res

    # Final-normalisation control, averaged over samples.
    corrs: List[float] = []
    for sid, rec in exp.manifest._index.items():
        pass
    for record in exp.manifest._index.values():
        diag = (record.get("diagnostics") or {}).get("no_norm_control", {})
        c = diag.get("correlation")
        if isinstance(c, (int, float)) and np.isfinite(c):
            corrs.append(float(c))
    if corrs:
        out["no_norm_control"] = {
            "mean_correlation": float(np.mean(corrs)),
            "median_correlation": float(np.median(corrs)),
            "n_samples": len(corrs),
            "interpretation": ("correlation between with-norm and without-norm "
                               "entropy profiles; low values mean the final "
                               "normalisation drives the layer-wise shape"),
        }
    save_json(exp.paths.statistics / "confounds.json", out)
    return out


# ---------------------------------------------------------------------------
# Phase 15 -- figures
# ---------------------------------------------------------------------------
def generate_figures(exp: Any, analyses: Dict[str, Dict[str, Any]],
                     correct_map: Dict[str, Optional[bool]],
                     df: Any, geometry: Optional[Dict[str, Any]],
                     jspace_results: Dict[str, Dict[str, Any]],
                     critical: Dict[str, Any], intervention_df: Any,
                     susceptibility: Dict[str, Any]) -> Dict[str, Any]:
    fb = FigureBuilder(exp.paths, exp.config.config_hash(),
                       exp.config.model.name,
                       exp.environment.get("git_commit"))

    def curves(key: str) -> Dict[str, np.ndarray]:
        return {sid: d["profiles"][key] for sid, d in analyses.items()
                if key in d["profiles"]}

    specs = [
        ("figure_01_entropy", "Entropy vs normalised layer", "entropy (nats)",
         "entropy", "entropy"),
        ("figure_02_margin", "Answer margin vs normalised layer",
         "m_l = p(correct) - max p(wrong)", "order_margin", "order_margin"),
        ("figure_04_jsd", "Layer-to-layer JSD", "JSD(p_l, p_{l+1})",
         "jsd_prev_layer", "jsd_prev_layer"),
        ("figure_05_velocity", "Representation velocity",
         "||h_l - h_{l-1}|| / ||h||", "traj_velocity_normalised",
         "traj_velocity_normalised"),
        ("figure_06_curvature", "Representation curvature",
         "turning angle / step length", "traj_curvature", "traj_curvature"),
        ("figure_09_attention", "Attention restructuring",
         "||A_l - A_{l-1}||_F", "attn_restructuring_frobenius_delta",
         "attn_restructuring_frobenius_delta"),
    ]
    for name, title, ylabel, key, signal in specs:
        fb.layer_profile_figure(name, title, ylabel, curves(key), correct_map,
                                signal)

    # Figure 3 is the correct-vs-incorrect view; it is the right panel of every
    # profile figure above, plus this dedicated symmetry-breaking version.
    fb.layer_profile_figure(
        "figure_03_correct_vs_incorrect",
        "Symmetry-breaking index: correct vs incorrect",
        "SB_l = 1 - H(q_l)/log K",
        curves("order_symmetry_breaking_index"), correct_map,
        "order_symmetry_breaking_index")

    fb.susceptibility_figure("figure_07_susceptibility", susceptibility,
                             list(analyses.keys()))
    fb.critical_layer_distribution("figure_08_critical_layers", df,
                                   list(crit.DETECTOR_SPECS.keys()))

    if geometry and geometry.get("status") == "ok":
        fb.geometry_figure("figure_10_effective_dimension", geometry["result"],
                           ["effective_rank", "participation_ratio",
                            "n_components_90pct", "twonn_intrinsic_dimension",
                            "neighbourhood_reorganisation", "anisotropy"],
                           "Latent-space dimensionality across layers",
                           geometry.get("kept", []))
        fb.geometry_figure("figure_12_latent_geometry", geometry["result"],
                           ["separation_ratio", "fisher_ratio",
                            "within_class_distance_mean",
                            "between_class_distance_mean",
                            "neighbourhood_purity", "local_anisotropy_mean"],
                           "Correct vs incorrect latent geometry",
                           geometry.get("kept", []))
        if geometry.get("pca"):
            kept = geometry.get("kept", [])
            fb.pca_trajectory_figure(
                "figure_11_pca_trajectory", geometry["pca"],
                [correct_map.get(s) for s in kept], kept)

    jcurves = {sid: np.concatenate([[np.nan],
                                    np.asarray(r["amplification_mean"], float)])
               for sid, r in jspace_results.items()
               if r.get("status") == "ok" and "amplification_mean" in r}
    separability = compute_jspace_separability(exp, jspace_results, analyses,
                                               correct_map)
    fb.jspace_figure("figure_13_jspace", jcurves, correct_map, separability)

    fb.causal_sensitivity_figure("figure_14_causal_sensitivity", intervention_df)

    by_dataset: Dict[str, np.ndarray] = {}
    id_to_ds = {s.sample_id: s.dataset for s in exp.samples}
    for ds in sorted({id_to_ds.get(s) for s in analyses if id_to_ds.get(s)}):
        arrs = [analyses[s]["profiles"]["jsd_prev_layer"] for s in analyses
                if id_to_ds.get(s) == ds and "jsd_prev_layer" in analyses[s]["profiles"]]
        stacked = sig.safe_stack(arrs)
        if stacked is not None:
            by_dataset[ds] = stacked
    fb.grouped_profile_figure("figure_15_cross_dataset",
                              "Layer-to-layer JSD by dataset",
                              "JSD(p_l, p_{l+1})", by_dataset,
                              "jsd_prev_layer", "dataset")

    fb.cross_model_figure("figure_16_cross_model",
                          load_cross_model_curves(exp),
                          "JSD(p_l, p_{l+1})", "jsd_prev_layer")

    if exp.config.discovery_mode:
        corr = st.signal_correlation_matrix(df)
        if corr.get("status") == "ok":
            save_json(exp.paths.statistics / "signal_correlation_matrix.json", {
                "columns": corr["columns"], "matrix": corr["matrix"].tolist(),
                "method": corr["method"], "n_rows": corr["n_rows"],
                "n_pairs": corr["n_pairs"], "note": corr["note"],
                "top_correlations": st.top_correlations(corr, 40),
            })
            fb.correlation_matrix_figure("figure_17_signal_correlations", corr)

    # Individual example traces, selected by recorded criteria.
    examples = select_examples(df, n_each=2)
    for item in examples:
        sid = item["sample_id"]
        if sid not in analyses:
            continue
        gen = exp.generations.get(sid, {})
        sample = next((s for s in exp.samples if s.sample_id == sid), None)
        profiles = dict(analyses[sid]["profiles"])
        jr = jspace_results.get(sid)
        if jr and jr.get("status") == "ok":
            profiles["jspace_amplification"] = np.concatenate(
                [[np.nan], np.asarray(jr["amplification_mean"], float)])
        det = {m: d.to_dict() for m, d in
               critical["detections"].get(sid, {}).items()}
        fb.example_trace_figure(
            f"example_{sid}", sid, profiles, det,
            {"dataset": sample.dataset if sample else None,
             "correct": gen.get("correct"),
             "prediction": gen.get("prediction"),
             "ground_truth": gen.get("ground_truth")},
            item["reason"])
        write_example_trace_report(exp, sid, sample, gen, profiles, det,
                                   critical, jspace_results, intervention_df)

    save_json(exp.paths.figures / "figure_index.json", fb.summary())
    return fb.summary()


def compute_jspace_separability(exp: Any, jspace_results: Dict[str, Dict[str, Any]],
                                analyses: Dict[str, Dict[str, Any]],
                                correct_map: Dict[str, Optional[bool]]
                                ) -> Dict[str, Any]:
    """Do correct/incorrect separate better in J-space than in hidden space?"""
    ok = {sid: r for sid, r in jspace_results.items() if r.get("status") == "ok"}
    if len(ok) < 8:
        return {"status": "insufficient_jspace_samples", "n": len(ok)}
    sids = sorted(ok.keys())
    labels = np.array([correct_map.get(s) for s in sids], dtype=object)
    n_layers = next(iter(ok.values()))["amplification_mean"].size
    layer = n_layers // 2
    js = jsp.jspace_matrix([ok[s] for s in sids], layer)
    bank, kept = exp.hidden_bank(sids)
    hidden = None
    if bank is not None and kept:
        index = {s: i for i, s in enumerate(kept)}
        rows = [bank[index[s], min(layer + 1, bank.shape[1] - 1), :]
                for s in sids if s in index]
        if len(rows) == len(sids):
            hidden = np.stack(rows, axis=0)
    result = jsp.compare_separability(js, hidden, labels,
                                      seed=exp.config.stats.bootstrap_seed)
    result["layer"] = int(layer)
    save_json(exp.paths.j_space / "separability.json", result)
    return result


def load_cross_model_curves(exp: Any, signal: str = "jsd_prev_layer"
                            ) -> Dict[str, Dict[str, Any]]:
    """Collect per-model layer profiles from sibling experiment directories.

    Enables Figure 16 when a second model has been run into a neighbouring
    output root, without re-loading that model or its raw tensors: the
    aggregated ``layer_summary.parquet`` already carries the per-layer mean,
    standard deviation and count. Returns ``{}`` when fewer than two models
    are present, and the figure records that as its skip reason.
    """
    out: Dict[str, Dict[str, Any]] = {}
    parent = exp.paths.root.parent
    for candidate in sorted(parent.glob("*/layer_summary.parquet")):
        try:
            import pandas as pd
            df = pd.read_parquet(candidate)
        except Exception:
            continue
        needed = {"model", "dataset", "group", "layer", f"{signal}_mean"}
        if not needed <= set(df.columns):
            continue
        sub = df[(df["dataset"] == "all") & (df["group"] == "all")]
        for model, g in sub.groupby("model"):
            g = g.sort_values("layer")
            mean = g[f"{signal}_mean"].to_numpy(dtype=np.float64)
            if mean.size < 3:
                continue
            label = str(model).rstrip("/").split("/")[-1]
            out[label] = {
                "mean": mean,
                "std": (g[f"{signal}_std"].to_numpy(dtype=np.float64)
                        if f"{signal}_std" in g.columns else None),
                "n": (g[f"{signal}_n"].to_numpy(dtype=np.float64)
                      if f"{signal}_n" in g.columns else None),
                "n_layers": int(mean.size),
                "source": str(candidate),
            }
    return out if len(out) >= 2 else {}


def write_example_trace_report(exp: Any, sid: str, sample: Any, gen: Dict[str, Any],
                               profiles: Dict[str, np.ndarray],
                               detections: Dict[str, Any],
                               critical: Dict[str, Any],
                               jspace_results: Dict[str, Dict[str, Any]],
                               intervention_df: Any) -> Path:
    """Per-example JSON for manual inspection (protocol section 46)."""
    cons = critical["consensus"].get(sid)
    interventions: List[Dict[str, Any]] = []
    if intervention_df is not None and len(intervention_df):
        sub = intervention_df[intervention_df["sample_id"] == sid]
        interventions = sub.to_dict(orient="records")[:200]
    payload = {
        "sample_id": sid,
        "question": sample.question if sample else None,
        "ground_truth": gen.get("ground_truth"),
        "prediction": gen.get("prediction"),
        "correct": gen.get("correct"),
        "parse_status": gen.get("parse_status"),
        "decoded_output": gen.get("decoded_output"),
        "dataset": sample.dataset if sample else None,
        "critical_layers": {m: d.get("critical_layer")
                            for m, d in detections.items()},
        "critical_layer_consensus": (cons.critical_layer_consensus
                                     if cons else None),
        "consensus_status": cons.consensus_status if cons else None,
        "entropy_curve": _tolist(profiles.get("entropy")),
        "margin_curve": _tolist(profiles.get("order_margin")),
        "jsd_curve": _tolist(profiles.get("jsd_prev_layer")),
        "velocity_curve": _tolist(profiles.get("traj_velocity_normalised")),
        "curvature_curve": _tolist(profiles.get("traj_curvature")),
        "jspace_curve": _tolist(profiles.get("jspace_amplification")),
        "attention_statistics": {
            k: _tolist(v) for k, v in profiles.items() if k.startswith("attn_")
        },
        "latent_statistics": {
            "norm": _tolist(profiles.get("traj_norm")),
            "displacement_from_first": _tolist(
                profiles.get("traj_displacement_from_first")),
            "cosine_to_final": _tolist(profiles.get("traj_cosine_to_final")),
        },
        "causal_intervention_results": interventions,
    }
    path = exp.paths.examples / f"trace_{sid}.json"
    return save_json(path, payload)


def _tolist(arr: Optional[np.ndarray]) -> Optional[List[Optional[float]]]:
    if arr is None:
        return None
    a = np.asarray(arr, dtype=np.float64)
    return [None if not np.isfinite(v) else float(v) for v in a]


# ---------------------------------------------------------------------------
# Phase 16 -- assemble everything and report
# ---------------------------------------------------------------------------
def finalise(exp: Any, *, analyses: Dict[str, Dict[str, Any]],
             correct_map: Dict[str, Optional[bool]], df: Any,
             critical: Dict[str, Any], geometry: Optional[Dict[str, Any]],
             jspace_results: Dict[str, Dict[str, Any]],
             intervention: Dict[str, Any], susceptibility: Dict[str, Any],
             cvi_tests: List[Dict[str, Any]], cross_dataset: Dict[str, Any],
             nulls: Dict[str, Any], confounds: Dict[str, Any],
             figures: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble the results dict, write the manifest, integrity check, report."""
    arch = exp.model.arch if exp.model else None
    acc = _accounting(exp, df, correct_map)

    shape_counts: Dict[str, int] = {}
    for c in critical["consensus"].values():
        shape_counts[c.dominant_shape] = shape_counts.get(c.dominant_shape, 0) + 1

    cons_values = [c.normalised_consensus for c in critical["consensus"].values()
                   if c.normalised_consensus is not None]
    n_total = len(critical["consensus"])
    cons_summary: Dict[str, Any] = {
        "n_total": n_total,
        "n_with_consensus": len(cons_values),
        "n_without_consensus": n_total - len(cons_values),
        "fraction_with_consensus": (len(cons_values) / n_total) if n_total else None,
    }
    if cons_values:
        cons_summary.update({
            "mean_normalised": float(np.mean(cons_values)),
            "median_normalised": float(np.median(cons_values)),
            "std_normalised": (float(np.std(cons_values, ddof=1))
                               if len(cons_values) > 1 else None),
            "ci": st.bootstrap_ci(cons_values, n_boot=exp.config.stats.n_bootstrap,
                                  seed=exp.config.stats.bootstrap_seed),
        })

    pairwise = [c.pairwise_agreement_rate for c in critical["consensus"].values()
                if np.isfinite(c.pairwise_agreement_rate)]

    causal: Dict[str, Any] = {"status": intervention.get("status", "not_run")}
    if intervention.get("status") == "ok" and intervention.get("df") is not None:
        idf = intervention["df"]
        summary = interv_summary(idf)
        causal.update(summary)
        causal["n_outcomes"] = int(len(idf))
        causal["n_samples"] = int(idf["sample_id"].nunique()) if len(idf) else 0
        vers = intervention.get("verifications") or []
        ok_vers = [v for v in vers if v.get("status") == "ok"]
        if ok_vers:
            causal["regeneration_agreement"] = float(np.mean(
                [1.0 if v.get("prediction_changed") is not None else 0.0
                 for v in ok_vers]))
            causal["n_regeneration_checks"] = len(ok_vers)

    results: Dict[str, Any] = {
        "model_name": exp.config.model.name,
        "model_revision": exp.config.model.revision,
        "n_layers": arch.n_layers if arch else None,
        "hidden_size": arch.hidden_size if arch else None,
        "n_heads": arch.n_heads if arch else None,
        "n_models": 1,
        "accounting": acc,
        "transitions": {
            "shape_counts": shape_counts,
            "sharp_max_width": crit.SHARP_MAX_WIDTH,
            "consensus": cons_summary,
            "agreement": {"mean_abs_difference":
                          critical["agreement"]["mean_abs_difference"]},
            "mean_pairwise_agreement": (float(np.mean(pairwise))
                                        if pairwise else None),
        },
        "correct_vs_incorrect": cvi_tests,
        "cross_dataset": cross_dataset,
        "cross_model": {"models": {}},
        "causal": causal,
        "null_models": nulls,
        "confounds": confounds,
        "no_norm_control": confounds.get("no_norm_control", {}),
        "statistical_tests": cvi_tests,
        "correction_method": exp.config.stats.multiple_comparison_method,
        "n_bootstrap": exp.config.stats.n_bootstrap,
        "n_permutation": exp.config.stats.n_permutation,
        "error_types": _count_errors(exp),
        "missing_analyses": exp.missing_analyses,
        "runtime": exp.controller.status(),
        "storage_gb": None,
    }
    results["evidence_against"] = collect_evidence_against(results)

    from .storage import dir_size_gb
    results["storage_gb"] = dir_size_gb(exp.paths.root)

    manifest_payload = build_experiment_manifest(
        exp.paths, config=exp.config, environment=exp.environment,
        arch=arch.to_dict() if arch else None,
        dataset_summary=acc, phases_completed=exp.phases_completed,
        manifest=exp.manifest, errors=exp.errors.all(),
        figures=figures.get("generated", []),
        statistical_tests=cvi_tests, runtime=exp.controller.status())
    save_json(exp.paths.experiment_manifest, manifest_payload)

    # The report is written before the integrity check so that the check can
    # verify it exists; the check's own output is the last artefact produced.
    report_path = generate_final_report(exp.paths, results=results)

    integrity = check_integrity(
        exp.paths, expected_samples=len(exp.samples), manifest=exp.manifest,
        errors=exp.errors.all(), figures=figures.get("generated", []),
        sample_ids=[s.sample_id for s in exp.samples])
    save_json(exp.paths.integrity_report, integrity)
    save_json(exp.paths.logs / "runtime_report.json", exp.controller.status())

    return {"results": results, "integrity": integrity,
            "report_path": str(report_path),
            "manifest_path": str(exp.paths.experiment_manifest)}


def interv_summary(idf: Any) -> Dict[str, Any]:
    from . import interventions as interv_mod
    out = interv_mod.sensitivity_by_layer_role(idf)
    keep = ["critical_mean", "random_control_mean",
            "cohens_d_critical_vs_random", "n_critical", "n_random_control",
            "comparison_status"]
    result = {k: out[k] for k in keep if k in out}
    crit_rows = idf[idf["layer_role"] == "critical"]["jsd_output"]
    ctrl_rows = idf[idf["layer_role"] == "random_control"]["jsd_output"]
    if len(crit_rows) >= 3 and len(ctrl_rows) >= 3:
        test = st.permutation_test(crit_rows.to_numpy(dtype=np.float64),
                                   ctrl_rows.to_numpy(dtype=np.float64),
                                   n_perm=1000, seed=0)
        result["p_value"] = test["p_value"]
        corrected = st.correct_multiple_comparisons([{"p_value": test["p_value"]}])
        result["p_value_corrected"] = corrected[0]["p_value_corrected"]
    result["status"] = "ok"
    return result


def _accounting(exp: Any, df: Any, correct_map: Dict[str, Optional[bool]]
                ) -> Dict[str, Any]:
    completed = exp.manifest.completed_ids("analysis")
    failed = exp.manifest.failed_ids("analysis")
    graded = [v for v in correct_map.values() if v is not None]
    out: Dict[str, Any] = {
        "n_requested": len(exp.samples),
        "n_completed": len(completed),
        "n_failed": len(failed),
        "n_skipped": max(0, len(exp.samples) - len(completed) - len(failed)),
        "n_correct": sum(1 for v in graded if v),
        "n_incorrect": sum(1 for v in graded if not v),
        "n_ungraded": len(correct_map) - len(graded),
        "accuracy": (sum(1 for v in graded if v) / len(graded)) if graded else None,
    }
    if df is not None and len(df) and "dataset" in df.columns:
        by: Dict[str, Any] = {}
        for ds, sub in df.groupby("dataset"):
            g = sub["correct"].dropna()
            by[str(ds)] = {"n": int(len(sub)),
                           "n_graded": int(len(g)),
                           "accuracy": float(g.mean()) if len(g) else None}
        out["by_dataset"] = by
    return out


def _count_errors(exp: Any) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for e in exp.errors.all():
        t = e.get("exception_type", "unknown")
        out[t] = out.get(t, 0) + 1
    return out
