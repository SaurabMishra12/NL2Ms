"""Phases 15, 44, 45 -- figure generation with lineage.

Every figure writes a matching ``<name>.json`` recording the samples, signal
definitions and parameters behind it, so a plot can always be traced back to
the data that produced it.

Plotting conventions used throughout:

* The x-axis is **normalised depth** ``l/L``, because raw layer index is not
  comparable across models and the cross-model figures would otherwise be
  meaningless.
* Group means are drawn with bootstrap bands, never bare. A mean curve with
  no uncertainty invites reading structure into sampling noise.
* Individual trajectories are overlaid where the sample count permits, since
  an averaged "sharp transition" can be produced entirely by averaging
  gradual transitions that occur at different layers -- a failure mode this
  experiment must be able to detect.
* Nothing is annotated as a "phase transition". Peaks are labelled
  "candidate critical layer".
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .registry import write_lineage
from .signals import normalised_layers, safe_stack
from .stats import bootstrap_curve

FIGURE_DPI = 150
CORRECT_COLOR = "#1b7837"
INCORRECT_COLOR = "#b2182b"
NEUTRAL_COLOR = "#2166ac"
NULL_COLOR = "#777777"


def _setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": FIGURE_DPI,
        "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.autolayout": True,
    })
    return plt


def _save(fig, path: Path, *, also_pdf: bool = True) -> List[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    fig.savefig(path, bbox_inches="tight")
    written.append(str(path))
    if also_pdf:
        pdf = path.with_suffix(".pdf")
        fig.savefig(pdf, bbox_inches="tight")
        written.append(str(pdf))
    import matplotlib.pyplot as plt
    plt.close(fig)
    for p in written:
        if not Path(p).exists() or Path(p).stat().st_size == 0:
            raise IOError(f"figure write verification failed: {p}")
    return written


def _legend(ax, **kwargs):
    """Add a legend only when something is labelled, to avoid a noisy warning."""
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(**kwargs)


def _boxplot(ax, data, labels, **kwargs):
    """Version-tolerant boxplot: matplotlib 3.9 renamed ``labels``.

    Kaggle images and current matplotlib disagree on this keyword, and the
    figure phase must not die on a cosmetic API rename.
    """
    try:
        return ax.boxplot(data, tick_labels=labels, **kwargs)
    except TypeError:
        return ax.boxplot(data, labels=labels, **kwargs)


def _band(ax, x, curves, color, label, *, alpha=0.18, seed=0):
    """Mean curve with a pointwise bootstrap band."""
    if curves is None or curves.shape[0] == 0:
        return None
    stats = bootstrap_curve(curves, n_boot=400, seed=seed)
    ax.plot(x, stats["mean"], color=color, lw=2,
            label=f"{label} (n={stats['n']})")
    ax.fill_between(x, stats["ci_low"], stats["ci_high"], color=color,
                    alpha=alpha, linewidth=0)
    return stats


def _split_by_correct(curves_by_sample: Dict[str, np.ndarray],
                      correct_map: Dict[str, Optional[bool]]
                      ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],
                                 List[str], List[str]]:
    """Split curves into correct / incorrect, dropping ungraded samples.

    Ungraded samples (unparsed generations, ambiguous items) are excluded
    rather than assigned to "incorrect" -- treating an unreadable answer as a
    wrong answer would bias every correct-vs-incorrect comparison.
    """
    c_ids = [sid for sid, v in correct_map.items()
             if v is True and sid in curves_by_sample]
    w_ids = [sid for sid, v in correct_map.items()
             if v is False and sid in curves_by_sample]
    c = safe_stack([curves_by_sample[s] for s in c_ids]) if c_ids else None
    w = safe_stack([curves_by_sample[s] for s in w_ids]) if w_ids else None
    return c, w, c_ids, w_ids


# ---------------------------------------------------------------------------
class FigureBuilder:
    """Generates the figure set and records what each one came from."""

    def __init__(self, paths: Any, config_hash: str, model_name: str,
                 git_commit: Optional[str] = None) -> None:
        self.paths = paths
        self.config_hash = config_hash
        self.model_name = model_name
        self.git_commit = git_commit
        self.generated: List[Dict[str, Any]] = []
        self.skipped: List[Dict[str, Any]] = []
        self.plt = _setup()

    def _record(self, name: str, files: List[str], *, sample_ids: List[str],
                signals: List[str], parameters: Dict[str, Any],
                produced_by: str) -> None:
        meta_path = self.paths.figures / f"{name}.json"
        write_lineage(meta_path, artefact=name, produced_by=produced_by,
                      config_hash=self.config_hash, sample_ids=sample_ids,
                      signals=signals, parameters=parameters,
                      git_commit=self.git_commit, model_name=self.model_name,
                      extra={"files": files})
        self.generated.append({"figure": name, "files": files,
                               "metadata": str(meta_path),
                               "n_samples": len(sample_ids)})

    def _skip(self, name: str, reason: str) -> None:
        self.skipped.append({"figure": name, "reason": reason})

    # -- generic layer-profile figure ----------------------------------
    def layer_profile_figure(self, name: str, title: str, ylabel: str,
                             curves_by_sample: Dict[str, np.ndarray],
                             correct_map: Dict[str, Optional[bool]],
                             signal: str, *, show_individual: bool = True,
                             max_individual: int = 40) -> None:
        """Correct vs incorrect mean curves, with individual trajectories."""
        if not curves_by_sample:
            return self._skip(name, "no curves available")
        stacked = safe_stack(list(curves_by_sample.values()))
        if stacked is None:
            return self._skip(name, "curves have inconsistent lengths")
        n_layers = stacked.shape[1]
        x = normalised_layers(n_layers)

        c, w, c_ids, w_ids = _split_by_correct(curves_by_sample, correct_map)
        fig, axes = self.plt.subplots(1, 2, figsize=(11, 4), sharey=True)

        ax = axes[0]
        _band(ax, x, stacked, NEUTRAL_COLOR, "all samples")
        if show_individual:
            ids = list(curves_by_sample.keys())[:max_individual]
            for sid in ids:
                col = (CORRECT_COLOR if correct_map.get(sid) is True else
                       INCORRECT_COLOR if correct_map.get(sid) is False
                       else NULL_COLOR)
                ax.plot(x, curves_by_sample[sid], color=col, alpha=0.13, lw=0.7)
        ax.set_title(f"{title}\nall samples + individual trajectories")
        ax.set_xlabel("normalised depth  l/L")
        ax.set_ylabel(ylabel)
        _legend(ax, fontsize=7)

        ax = axes[1]
        drawn = False
        if c is not None and c.shape[0] >= 2:
            _band(ax, x, c, CORRECT_COLOR, "correct"); drawn = True
        if w is not None and w.shape[0] >= 2:
            _band(ax, x, w, INCORRECT_COLOR, "incorrect"); drawn = True
        if not drawn:
            ax.text(0.5, 0.5, "too few graded samples\nfor a group comparison",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_title("correct vs incorrect (bootstrap 95% band, pointwise)")
        ax.set_xlabel("normalised depth  l/L")
        _legend(ax, fontsize=7)

        files = _save(fig, self.paths.figures / f"{name}.png")
        self._record(name, files, sample_ids=list(curves_by_sample.keys()),
                     signals=[signal],
                     parameters={"n_layers": n_layers, "bootstrap_n": 400,
                                 "n_correct": len(c_ids), "n_incorrect": len(w_ids),
                                 "band": "pointwise 95% bootstrap CI of the mean"},
                     produced_by="plots.FigureBuilder.layer_profile_figure")

    # -- FIGURE 7: susceptibility --------------------------------------
    def susceptibility_figure(self, name: str, susceptibility: Dict[str, Any],
                              sample_ids: List[str]) -> None:
        if susceptibility.get("status") != "ok":
            return self._skip(name, susceptibility.get("status", "unavailable"))
        chi = np.asarray(susceptibility["profile"], dtype=np.float64)
        mean = np.asarray(susceptibility.get("mean_profile"), dtype=np.float64)
        x = normalised_layers(chi.size)
        fig, ax = self.plt.subplots(figsize=(7, 4))
        ax.plot(x, chi, color=NEUTRAL_COLOR, lw=2, label="Var_x[m_l]")
        det = susceptibility.get("detection", {})
        peak = det.get("critical_layer")
        if peak is not None:
            ax.axvline(x[peak], color="black", ls="--", lw=1,
                       label=f"candidate critical layer (l={peak})")
        ax2 = ax.twinx()
        if mean is not None and mean.size == chi.size:
            ax2.plot(x, mean, color=NULL_COLOR, lw=1, ls=":", label="mean m_l")
            ax2.set_ylabel("mean order parameter", color=NULL_COLOR)
        ax.set_xlabel("normalised depth  l/L")
        ax.set_ylabel("empirical susceptibility-like measure")
        ax.set_title("Across-sample variance of the order parameter\n"
                     "(descriptive; not a physical susceptibility)")
        _legend(ax, fontsize=7, loc="upper left")
        files = _save(fig, self.paths.figures / f"{name}.png")
        self._record(name, files, sample_ids=sample_ids,
                     signals=["susceptibility_margin"],
                     parameters={"shape": det.get("transition_shape"),
                                 "caveat": susceptibility.get("caveat")},
                     produced_by="plots.FigureBuilder.susceptibility_figure")

    # -- FIGURE 8: critical-layer distribution -------------------------
    def critical_layer_distribution(self, name: str, df: Any,
                                    methods: Sequence[str]) -> None:
        cols = [f"critical_layer_{m}_normalised" for m in methods
                if f"critical_layer_{m}_normalised" in df.columns]
        if not cols:
            return self._skip(name, "no critical-layer columns present")
        fig, axes = self.plt.subplots(1, 2, figsize=(11, 4))
        ax = axes[0]
        data, labels = [], []
        for col, m in zip(cols, methods):
            v = df[col].to_numpy(dtype=np.float64, na_value=np.nan)
            v = v[np.isfinite(v)]
            if v.size:
                data.append(v); labels.append(m.replace("_", "\n"))
        if data:
            _boxplot(ax, data, labels, showmeans=True)
        ax.set_ylabel("normalised critical layer  l/L")
        ax.set_title("Critical-layer location by detection method")
        ax.tick_params(axis="x", labelsize=6, rotation=0)
        ax.set_ylim(0, 1)

        ax = axes[1]
        if "critical_layer_consensus_normalised" in df.columns:
            v = df["critical_layer_consensus_normalised"].to_numpy(
                dtype=np.float64, na_value=np.nan)
            finite = v[np.isfinite(v)]
            n_null = int(np.sum(~np.isfinite(v)))
            if finite.size:
                ax.hist(finite, bins=min(20, max(5, finite.size // 3)),
                        color=NEUTRAL_COLOR, alpha=0.8)
            ax.set_title(f"Consensus critical layer\n"
                         f"({finite.size} with consensus, {n_null} without)")
        ax.set_xlabel("normalised critical layer  l/L")
        ax.set_xlim(0, 1)
        files = _save(fig, self.paths.figures / f"{name}.png")
        self._record(name, files,
                     sample_ids=df["sample_id"].tolist() if "sample_id" in df else [],
                     signals=["critical_layer_consensus"],
                     parameters={"methods": list(methods)},
                     produced_by="plots.FigureBuilder.critical_layer_distribution")

    # -- FIGURE 11: PCA trajectory -------------------------------------
    def pca_trajectory_figure(self, name: str, pca: Dict[str, Any],
                              correct: Sequence[Optional[bool]],
                              sample_ids: List[str]) -> None:
        if pca.get("status") != "ok":
            return self._skip(name, pca.get("status", "unavailable"))
        proj = pca["projection"]            # (n_samples, n_layers, k)
        if proj.shape[2] < 2:
            return self._skip(name, "fewer than 2 principal components")
        fig, ax = self.plt.subplots(figsize=(6.5, 5.5))
        for i in range(min(proj.shape[0], 60)):
            col = (CORRECT_COLOR if correct[i] is True else
                   INCORRECT_COLOR if correct[i] is False else NULL_COLOR)
            ax.plot(proj[i, :, 0], proj[i, :, 1], color=col, alpha=0.35, lw=0.8)
            ax.scatter(proj[i, -1, 0], proj[i, -1, 1], color=col, s=8, zorder=3)
        evr = pca["explained_variance_ratio"]
        ax.set_xlabel(f"PC1 ({evr[0]:.1%} of variance)")
        ax.set_ylabel(f"PC2 ({evr[1]:.1%} of variance)")
        ax.set_title("Layer-wise trajectories in a shared PCA basis\n"
                     "(green = correct, red = incorrect, dot = final layer)")
        files = _save(fig, self.paths.figures / f"{name}.png")
        self._record(name, files, sample_ids=sample_ids, signals=[],
                     parameters={"n_components": int(proj.shape[2]),
                                 "basis": "shared across all layers",
                                 "explained_variance_ratio": evr.tolist()},
                     produced_by="plots.FigureBuilder.pca_trajectory_figure")

    # -- FIGURE 14: causal sensitivity ---------------------------------
    def causal_sensitivity_figure(self, name: str, df: Any,
                                  metric: str = "jsd_output") -> None:
        if df is None or len(df) == 0 or metric not in df.columns:
            return self._skip(name, "no intervention data")
        fig, axes = self.plt.subplots(1, 2, figsize=(11.5, 4.2))

        ax = axes[0]
        for kind, sub in df.groupby("perturbation_kind"):
            grouped = sub.groupby("normalised_layer")[metric].agg(["mean", "count"])
            grouped = grouped[grouped["count"] >= 2]
            if len(grouped):
                ax.plot(grouped.index, grouped["mean"], marker="o", ms=3,
                        lw=1.4, label=str(kind))
        ax.set_xlabel("normalised depth  l/L")
        ax.set_ylabel(f"{metric} (clean vs perturbed)")
        ax.set_title("Causal effect of perturbation by layer")
        ax.legend(fontsize=6.5)

        ax = axes[1]
        if "layer_role" in df.columns:
            roles, data = [], []
            for role, sub in df.groupby("layer_role"):
                v = sub[metric].to_numpy(dtype=np.float64, na_value=np.nan)
                v = v[np.isfinite(v)]
                if v.size >= 2:
                    roles.append(str(role)); data.append(v)
            if data:
                _boxplot(ax, data, roles, showmeans=True)
                ax.tick_params(axis="x", labelsize=6, rotation=30)
        ax.set_ylabel(metric)
        ax.set_title("By layer role\n(critical vs random control is the "
                     "comparison that matters)")
        files = _save(fig, self.paths.figures / f"{name}.png")
        sids = sorted(set(df["sample_id"].tolist())) if "sample_id" in df else []
        self._record(name, files, sample_ids=sids, signals=["jsd_output"],
                     parameters={"metric": metric,
                                 "n_outcomes": int(len(df))},
                     produced_by="plots.FigureBuilder.causal_sensitivity_figure")

    # -- FIGURE 15/16: cross-dataset and cross-model -------------------
    def grouped_profile_figure(self, name: str, title: str, ylabel: str,
                               groups: Dict[str, np.ndarray],
                               signal: str, group_label: str) -> None:
        """One mean curve per group (dataset or model), on normalised depth."""
        groups = {k: v for k, v in groups.items()
                  if v is not None and v.ndim == 2 and v.shape[0] >= 2}
        if not groups:
            return self._skip(name, f"no {group_label} group had >= 2 samples")
        fig, ax = self.plt.subplots(figsize=(7.5, 4.5))
        colors = self.plt.cm.viridis(np.linspace(0, 0.85, len(groups)))
        for (label, curves), color in zip(sorted(groups.items()), colors):
            x = normalised_layers(curves.shape[1])
            _band(ax, x, curves, color, label)
        ax.set_xlabel("normalised depth  l/L")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\n(normalised depth makes {group_label}s comparable)")
        _legend(ax, fontsize=7)
        files = _save(fig, self.paths.figures / f"{name}.png")
        self._record(name, files, sample_ids=[], signals=[signal],
                     parameters={"groups": {k: int(v.shape[0])
                                            for k, v in groups.items()},
                                 "group_label": group_label},
                     produced_by="plots.FigureBuilder.grouped_profile_figure")

    # -- FIGURE 16: cross-model ----------------------------------------
    def cross_model_figure(self, name: str, models: Dict[str, Dict[str, Any]],
                           ylabel: str, signal: str) -> None:
        """One curve per model on normalised depth, with a standard-error band.

        Unlike the other grouped figures this one receives pre-aggregated
        (mean, std, n) per layer rather than per-sample curves, because the
        source is another run's ``layer_summary.parquet`` -- raw per-sample
        data for a different model is not loaded into this process.

        Normalised depth is what makes the comparison legible at all: a
        transition at layer 14 of 28 and at layer 16 of 32 are the same
        depth, and raw layer index would hide that.
        """
        usable = {k: v for k, v in models.items()
                  if v.get("mean") is not None and len(v["mean"]) >= 3}
        if len(usable) < 2:
            return self._skip(name, "fewer than two models have layer summaries "
                                    "in this experiment root")
        fig, ax = self.plt.subplots(figsize=(7.5, 4.5))
        colors = self.plt.cm.viridis(np.linspace(0, 0.85, len(usable)))
        for (label, info), color in zip(sorted(usable.items()), colors):
            mean = np.asarray(info["mean"], dtype=np.float64)
            x = normalised_layers(mean.size)
            n_layers = info.get("n_layers", mean.size)
            ax.plot(x, mean, color=color, lw=2,
                    label=f"{label} ({n_layers} layers)")
            std = info.get("std")
            n = info.get("n")
            if std is not None and n is not None and len(np.atleast_1d(n)):
                std = np.asarray(std, dtype=np.float64)
                counts = np.asarray(n, dtype=np.float64)
                counts = counts[np.isfinite(counts) & (counts > 0)]
                se = std / np.sqrt(max(1.0, float(np.median(counts))
                                       if counts.size else 1.0))
                ax.fill_between(x, mean - 1.96 * se, mean + 1.96 * se,
                                color=color, alpha=0.16, linewidth=0)
        ax.set_xlabel("normalised depth  l/L")
        ax.set_ylabel(ylabel)
        ax.set_title("Cross-model comparison on normalised depth\n"
                     "(band = 1.96 x SE of the layer mean)")
        _legend(ax, fontsize=7)
        files = _save(fig, self.paths.figures / f"{name}.png")
        self._record(name, files, sample_ids=[], signals=[signal],
                     parameters={"models": {k: v.get("n_layers")
                                            for k, v in usable.items()}},
                     produced_by="plots.FigureBuilder.cross_model_figure")

    # -- FIGURE 13: J-space --------------------------------------------
    def jspace_figure(self, name: str, curves_by_sample: Dict[str, np.ndarray],
                      correct_map: Dict[str, Optional[bool]],
                      separability: Optional[Dict[str, Any]] = None) -> None:
        if not curves_by_sample:
            return self._skip(name, "no J-space data")
        stacked = safe_stack(list(curves_by_sample.values()))
        if stacked is None:
            return self._skip(name, "inconsistent J-space profile lengths")
        x = normalised_layers(stacked.shape[1])
        c, w, c_ids, w_ids = _split_by_correct(curves_by_sample, correct_map)
        fig, axes = self.plt.subplots(1, 2, figsize=(11, 4))
        ax = axes[0]
        for sid, curve in list(curves_by_sample.items())[:40]:
            col = (CORRECT_COLOR if correct_map.get(sid) is True else
                   INCORRECT_COLOR if correct_map.get(sid) is False else NULL_COLOR)
            ax.plot(x, curve, color=col, alpha=0.2, lw=0.7)
        if c is not None and c.shape[0] >= 2:
            _band(ax, x, c, CORRECT_COLOR, "correct")
        if w is not None and w.shape[0] >= 2:
            _band(ax, x, w, INCORRECT_COLOR, "incorrect")
        ax.axhline(1.0, color="black", lw=0.8, ls=":",
                   label="amplification = 1 (no growth)")
        ax.set_xlabel("normalised depth  l/L")
        ax.set_ylabel("mean directional amplification  ||J_l v|| / ||v||")
        ax.set_title("J-space trajectories")
        _legend(ax, fontsize=7)

        ax = axes[1]
        if separability and separability.get("jspace_status") == "ok":
            names, vals, errs = [], [], []
            for tag in ["jspace", "hidden_matched_dim", "hidden_full"]:
                if separability.get(f"{tag}_status") == "ok":
                    names.append(tag.replace("_", "\n"))
                    vals.append(separability[f"{tag}_balanced_accuracy"])
                    errs.append(separability.get(f"{tag}_balanced_accuracy_std", 0))
            if names:
                ax.bar(names, vals, yerr=errs, color=[NEUTRAL_COLOR] * len(names),
                       capsize=4, alpha=0.85)
                ax.axhline(0.5, color="black", ls="--", lw=1, label="chance")
                ax.set_ylim(0, 1)
                ax.set_ylabel("cross-validated balanced accuracy")
                ax.set_title("Correct/incorrect separability\nJ-space vs hidden space")
                _legend(ax, fontsize=7)
        else:
            ax.text(0.5, 0.5, "separability comparison unavailable",
                    ha="center", va="center", transform=ax.transAxes)
        files = _save(fig, self.paths.figures / f"{name}.png")
        self._record(name, files, sample_ids=list(curves_by_sample.keys()),
                     signals=["jspace_amplification", "jspace_descriptor"],
                     parameters={"separability": separability or {}},
                     produced_by="plots.FigureBuilder.jspace_figure")

    # -- FIGURE 10/12: geometry ----------------------------------------
    def geometry_figure(self, name: str, geometry: Dict[str, Any],
                        keys: Sequence[str], title: str,
                        sample_ids: List[str]) -> None:
        profiles = geometry.get("profiles", {})
        usable = [k for k in keys if k in profiles
                  and np.isfinite(np.asarray(profiles[k], float)).any()]
        if not usable:
            return self._skip(name, "no finite geometry profiles")
        n = len(usable)
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = self.plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows),
                                      squeeze=False)
        for i, key in enumerate(usable):
            ax = axes[i // ncols][i % ncols]
            y = np.asarray(profiles[key], dtype=np.float64)
            x = normalised_layers(y.size)
            ax.plot(x, y, color=NEUTRAL_COLOR, marker="o", ms=2.5, lw=1.3)
            ax.set_title(key.replace("_", " "), fontsize=8)
            ax.set_xlabel("l/L", fontsize=7)
        for j in range(n, nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")
        fig.suptitle(title, fontsize=10)
        files = _save(fig, self.paths.figures / f"{name}.png")
        self._record(name, files, sample_ids=sample_ids,
                     signals=["effective_rank", "neighbourhood_reorganisation",
                              "twonn_intrinsic_dimension"],
                     parameters={"keys": usable,
                                 "n_samples_used": geometry.get("n_samples_used")},
                     produced_by="plots.FigureBuilder.geometry_figure")

    # -- correlation matrix (discovery mode) ---------------------------
    def correlation_matrix_figure(self, name: str, result: Dict[str, Any]) -> None:
        if result.get("status") != "ok":
            return self._skip(name, result.get("status", "unavailable"))
        M = np.asarray(result["matrix"], dtype=np.float64)
        cols = result["columns"]
        if M.shape[0] > 40:
            M, cols = M[:40, :40], cols[:40]
        fig, ax = self.plt.subplots(figsize=(max(6, 0.28 * len(cols)),
                                             max(5, 0.28 * len(cols))))
        im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=90, fontsize=5)
        ax.set_yticks(range(len(cols)))
        ax.set_yticklabels(cols, fontsize=5)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, label=f"{result['method']} rho")
        ax.set_title("Signal correlation matrix (exploratory)\n"
                     f"{result['n_pairs']} pairs; no individual cell is tested",
                     fontsize=9)
        files = _save(fig, self.paths.figures / f"{name}.png", also_pdf=False)
        self._record(name, files, sample_ids=[], signals=[],
                     parameters={"method": result["method"],
                                 "n_rows": result["n_rows"],
                                 "n_pairs": result["n_pairs"]},
                     produced_by="plots.FigureBuilder.correlation_matrix_figure")

    # -- individual example traces (protocol section 45) ---------------
    def example_trace_figure(self, name: str, sample_id: str,
                             profiles: Dict[str, np.ndarray],
                             detections: Dict[str, Any],
                             header: Dict[str, Any],
                             selection_reason: str) -> None:
        keys = [k for k in ["entropy", "order_margin", "jsd_prev_layer",
                            "traj_velocity_normalised", "traj_curvature",
                            "jspace_amplification"]
                if k in profiles and np.isfinite(np.asarray(profiles[k], float)).any()]
        if not keys:
            return self._skip(name, "no finite profiles for this sample")
        fig, axes = self.plt.subplots(len(keys), 1, figsize=(7, 1.7 * len(keys)),
                                      sharex=True, squeeze=False)
        cand_layers = {m: d.get("critical_layer") for m, d in detections.items()
                       if isinstance(d, dict)}
        for i, key in enumerate(keys):
            ax = axes[i][0]
            y = np.asarray(profiles[key], dtype=np.float64)
            x = normalised_layers(y.size)
            ax.plot(x, y, color=NEUTRAL_COLOR, lw=1.5)
            for m, l in cand_layers.items():
                if l is not None and 0 <= l < y.size:
                    ax.axvline(x[l], color="black", alpha=0.18, lw=1)
            ax.set_ylabel(key.replace("_", "\n"), fontsize=7)
        axes[-1][0].set_xlabel("normalised depth  l/L")
        correct = header.get("correct")
        status = ("correct" if correct is True else
                  "incorrect" if correct is False else "ungraded")
        fig.suptitle(
            f"{sample_id}  [{header.get('dataset')}]  {status}\n"
            f"selected because: {selection_reason}\n"
            "vertical lines = candidate critical layers from each detector",
            fontsize=8)
        files = _save(fig, self.paths.examples / f"{name}.png", also_pdf=False)
        self._record(name, files, sample_ids=[sample_id],
                     signals=keys,
                     parameters={"selection_reason": selection_reason,
                                 "candidate_layers": cand_layers,
                                 **{k: v for k, v in header.items()
                                    if k in ("dataset", "correct", "prediction",
                                             "ground_truth")}},
                     produced_by="plots.FigureBuilder.example_trace_figure")

    def summary(self) -> Dict[str, Any]:
        return {"n_generated": len(self.generated),
                "generated": self.generated,
                "n_skipped": len(self.skipped),
                "skipped": self.skipped}


def select_examples(df: Any, n_each: int = 2) -> List[Dict[str, str]]:
    """Choose examples to plot by *recorded criteria*, never by eye.

    Selecting "illustrative" traces manually is how cherry-picking happens.
    Each example here is chosen by an explicit, reproducible rule, and the
    rule travels with the figure.
    """
    out: List[Dict[str, str]] = []
    if df is None or len(df) == 0 or "sample_id" not in df.columns:
        return out

    def take(sub: Any, reason: str, ascending: bool, col: str) -> None:
        if col not in sub.columns:
            return
        s = sub.dropna(subset=[col]).sort_values(col, ascending=ascending)
        for sid in s["sample_id"].head(n_each).tolist():
            out.append({"sample_id": sid, "reason": reason})

    take(df, "strongest transition (highest transition_strength)", False,
         "transition_strength")
    take(df, "weakest transition (lowest transition_strength)", True,
         "transition_strength")
    if "correct" in df.columns:
        take(df[df["correct"] == True], "correct with weakest transition", True,   # noqa: E712
             "transition_strength")
        take(df[df["correct"] == False], "incorrect with strongest transition", False,  # noqa: E712
             "transition_strength")
    take(df, "largest causal sensitivity at the critical layer", False,
         "causal_sensitivity_critical")
    take(df, "smallest causal sensitivity at the critical layer", True,
         "causal_sensitivity_critical")
    take(df, "highest detector disagreement (largest spread)", False,
         "detector_spread")

    seen: set = set()
    unique: List[Dict[str, str]] = []
    for item in out:
        if item["sample_id"] not in seen:
            seen.add(item["sample_id"])
            unique.append(item)
    return unique
