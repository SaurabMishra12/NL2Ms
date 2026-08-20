"""Phases 14, 33, 34, 35 -- statistics, null models, confound control.

Reporting discipline enforced here:

* Every comparison returns an **effect size with a confidence interval**, not
  just a p-value. The p-value is reported too, but it is the least
  informative of the three at these sample sizes.
* Every family of tests goes through :func:`correct_multiple_comparisons`.
  Eight critical-layer detectors x four datasets x two groups is over sixty
  tests; without correction, several "significant" results are guaranteed by
  construction.
* Null models are first-class. A signal is only interesting to the extent it
  exceeds a matched null, and the nulls here are matched on the things most
  likely to produce a spurious peak: layer-index structure, class balance,
  and profile smoothness.
"""

from __future__ import annotations

import contextlib

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .signals import EPS, argmax_safe, derivative


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _quiet_nan():
    """Suppress NumPy's all-NaN-slice warnings where all-NaN is expected."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(invalid="ignore"):
            yield


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Standardised mean difference with pooled SD (independent groups)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    na, nb = a.size, b.size
    pooled = np.sqrt(((na - 1) * np.var(a, ddof=1) +
                      (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    if pooled < EPS:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / pooled)


def hedges_g(a: Sequence[float], b: Sequence[float]) -> float:
    """Cohen's d with the small-sample bias correction.

    Preferred here because per-dataset correct/incorrect groups are often
    under 30, where d is biased upward by several percent.
    """
    d = cohens_d(a, b)
    if not np.isfinite(d):
        return d
    na = int(np.sum(np.isfinite(np.asarray(a, float))))
    nb = int(np.sum(np.isfinite(np.asarray(b, float))))
    df = na + nb - 2
    if df <= 1:
        return float("nan")
    return float(d * (1 - 3 / (4 * df - 1)))


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Non-parametric effect size in [-1, 1]; robust to non-normality.

    Reported alongside g because layer-index distributions are bounded and
    often bimodal, where a standardised mean difference is hard to interpret.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    # O(n log n) via ranks rather than the O(n*m) double loop.
    combined = np.concatenate([a, b])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, combined.size + 1)
    # Average ranks for ties.
    _, inv, counts = np.unique(combined, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size)
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    ra = ranks[:a.size].sum()
    u = ra - a.size * (a.size + 1) / 2.0
    return float(2.0 * u / (a.size * b.size) - 1.0)


def rank_biserial(differences: Sequence[float]) -> float:
    """Paired effect size matching the Wilcoxon signed-rank test."""
    d = np.asarray(differences, dtype=np.float64)
    d = d[np.isfinite(d) & (d != 0)]
    if d.size == 0:
        return float("nan")
    from scipy.stats import rankdata
    r = rankdata(np.abs(d))
    total = r.sum()
    if total < EPS:
        return float("nan")
    return float((r[d > 0].sum() - r[d < 0].sum()) / total)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def bootstrap_ci(values: Sequence[float], statistic: Callable = np.mean, *,
                 n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
                 ) -> Dict[str, Any]:
    """Percentile bootstrap confidence interval."""
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return {"point": float(statistic(x)) if x.size else float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"),
                "n": int(x.size), "status": "insufficient_data"}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    stats = np.array([statistic(x[i]) for i in idx])
    return {
        "point": float(statistic(x)),
        "ci_low": float(np.percentile(stats, 100 * alpha / 2)),
        "ci_high": float(np.percentile(stats, 100 * (1 - alpha / 2))),
        "n": int(x.size), "n_boot": n_boot, "alpha": alpha, "status": "ok",
    }


def bootstrap_difference(a: Sequence[float], b: Sequence[float], *,
                         n_boot: int = 2000, alpha: float = 0.05,
                         seed: int = 0) -> Dict[str, Any]:
    """CI for the difference in means between two independent groups."""
    x = np.asarray(a, dtype=np.float64); x = x[np.isfinite(x)]
    y = np.asarray(b, dtype=np.float64); y = y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return {"difference": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n_a": int(x.size), "n_b": int(y.size),
                "status": "insufficient_data"}
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = (np.mean(rng.choice(x, x.size, replace=True)) -
                    np.mean(rng.choice(y, y.size, replace=True)))
    return {
        "difference": float(np.mean(x) - np.mean(y)),
        "ci_low": float(np.percentile(diffs, 100 * alpha / 2)),
        "ci_high": float(np.percentile(diffs, 100 * (1 - alpha / 2))),
        "n_a": int(x.size), "n_b": int(y.size), "n_boot": n_boot,
        "status": "ok",
    }


def bootstrap_curve(curves: np.ndarray, *, n_boot: int = 1000,
                    alpha: float = 0.05, seed: int = 0) -> Dict[str, np.ndarray]:
    """Pointwise bootstrap band for a set of per-layer curves.

    Pointwise, not simultaneous: the band answers "where is the mean at this
    layer", not "does the whole curve lie inside". Labelled as such in the
    figures so it is not over-read.
    """
    C = np.asarray(curves, dtype=np.float64)
    if C.ndim != 2 or C.shape[0] < 2:
        n = C.shape[-1] if C.ndim else 0
        nan = np.full(n, np.nan)
        return {"mean": nan, "ci_low": nan, "ci_high": nan, "n": 0}
    rng = np.random.default_rng(seed)
    n_samples, n_layers = C.shape
    boot = np.empty((n_boot, n_layers))
    # Columns that are all-NaN (a layer where no sample has a defined value,
    # e.g. the embedding row of a layer-pair quantity) stay NaN by design.
    with _quiet_nan():
        for i in range(n_boot):
            idx = rng.integers(0, n_samples, size=n_samples)
            boot[i] = np.nanmean(C[idx], axis=0)
        mean = np.nanmean(C, axis=0)
        lo = np.nanpercentile(boot, 100 * alpha / 2, axis=0)
        hi = np.nanpercentile(boot, 100 * (1 - alpha / 2), axis=0)
    return {"mean": mean, "ci_low": lo, "ci_high": hi,
            "n": n_samples, "n_boot": n_boot}


# ---------------------------------------------------------------------------
# Hypothesis tests
# ---------------------------------------------------------------------------
def permutation_test(a: Sequence[float], b: Sequence[float], *,
                     statistic: Callable[[np.ndarray, np.ndarray], float] = None,
                     n_perm: int = 2000, seed: int = 0,
                     alternative: str = "two-sided") -> Dict[str, Any]:
    """Label-shuffling test for a difference between two groups.

    The p-value uses the ``(r + 1) / (n + 1)`` convention, which is the
    unbiased estimator and cannot return exactly zero -- an honest floor of
    ``1/(n_perm+1)`` rather than a misleading ``p = 0``.
    """
    x = np.asarray(a, dtype=np.float64); x = x[np.isfinite(x)]
    y = np.asarray(b, dtype=np.float64); y = y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return {"statistic": float("nan"), "p_value": float("nan"),
                "n_a": int(x.size), "n_b": int(y.size),
                "status": "insufficient_data"}
    if statistic is None:
        def statistic(u, v):
            return float(np.mean(u) - np.mean(v))
    observed = statistic(x, y)
    pooled = np.concatenate([x, y])
    rng = np.random.default_rng(seed)
    n_a = x.size
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        s = statistic(pooled[:n_a], pooled[n_a:])
        if alternative == "two-sided":
            count += abs(s) >= abs(observed) - 1e-15
        elif alternative == "greater":
            count += s >= observed - 1e-15
        else:
            count += s <= observed + 1e-15
    return {
        "statistic": float(observed),
        "p_value": float((count + 1) / (n_perm + 1)),
        "n_a": int(x.size), "n_b": int(y.size), "n_perm": n_perm,
        "alternative": alternative, "status": "ok",
        "p_value_floor": float(1.0 / (n_perm + 1)),
    }


def compare_groups(a: Sequence[float], b: Sequence[float], *, label: str = "",
                   cfg: Any = None, paired: bool = False, seed: int = 0
                   ) -> Dict[str, Any]:
    """The standard comparison bundle: n, effect size, CI, and p-value."""
    from scipy import stats as sps

    n_boot = getattr(cfg, "n_bootstrap", 2000) if cfg else 2000
    n_perm = getattr(cfg, "n_permutation", 2000) if cfg else 2000
    alpha = getattr(cfg, "alpha", 0.05) if cfg else 0.05
    min_n = getattr(cfg, "min_group_size", 5) if cfg else 5

    x = np.asarray(a, dtype=np.float64); x = x[np.isfinite(x)]
    y = np.asarray(b, dtype=np.float64); y = y[np.isfinite(y)]
    out: Dict[str, Any] = {
        "label": label, "n_a": int(x.size), "n_b": int(y.size),
        "mean_a": float(np.mean(x)) if x.size else float("nan"),
        "mean_b": float(np.mean(y)) if y.size else float("nan"),
        "median_a": float(np.median(x)) if x.size else float("nan"),
        "median_b": float(np.median(y)) if y.size else float("nan"),
        "paired": bool(paired),
    }
    if x.size < min_n or y.size < min_n:
        out["status"] = "below_min_group_size"
        out["min_group_size"] = min_n
        return out

    if paired:
        n = min(x.size, y.size)
        diff = x[:n] - y[:n]
        try:
            stat, p = sps.wilcoxon(x[:n], y[:n])
            out["test"] = "wilcoxon_signed_rank"
            out["statistic"] = float(stat)
            out["p_value"] = float(p)
        except ValueError as exc:
            out["test"] = "wilcoxon_signed_rank"
            out["p_value"] = float("nan")
            out["test_error"] = str(exc)
        out["effect_size_name"] = "rank_biserial"
        out["effect_size"] = rank_biserial(diff)
        out.update({f"diff_{k}": v for k, v in
                    bootstrap_ci(diff, n_boot=n_boot, alpha=alpha,
                                 seed=seed).items()})
    else:
        stat, p = sps.mannwhitneyu(x, y, alternative="two-sided")
        out["test"] = "mann_whitney_u"
        out["statistic"] = float(stat)
        out["p_value"] = float(p)
        out["effect_size_name"] = "hedges_g"
        out["effect_size"] = hedges_g(x, y)
        out["cliffs_delta"] = cliffs_delta(x, y)
        out.update({f"diff_{k}": v for k, v in
                    bootstrap_difference(x, y, n_boot=n_boot, alpha=alpha,
                                         seed=seed).items()})
        perm = permutation_test(x, y, n_perm=n_perm, seed=seed)
        out["permutation_p_value"] = perm["p_value"]
        out["permutation_p_floor"] = perm.get("p_value_floor")
    out["status"] = "ok"
    return out


def correct_multiple_comparisons(results: Sequence[Dict[str, Any]], *,
                                 alpha: float = 0.05, method: str = "fdr_bh",
                                 key: str = "p_value") -> List[Dict[str, Any]]:
    """Apply FDR (or Bonferroni) across a family of tests, in place-ish.

    Every returned record gains ``p_value_corrected``, ``significant`` and
    ``correction_method``. Tests that could not run keep ``significant=None``
    rather than being silently dropped from the denominator.
    """
    out = [dict(r) for r in results]
    idx = [i for i, r in enumerate(out)
           if isinstance(r.get(key), float) and np.isfinite(r.get(key))]
    if not idx:
        for r in out:
            r["p_value_corrected"] = None
            r["significant"] = None
            r["correction_method"] = method
        return out

    pvals = np.array([out[i][key] for i in idx], dtype=np.float64)
    n = pvals.size
    if method == "bonferroni":
        corrected = np.minimum(pvals * n, 1.0)
    else:  # Benjamini-Hochberg
        order = np.argsort(pvals)
        ranked = pvals[order]
        adj = ranked * n / np.arange(1, n + 1)
        adj = np.minimum.accumulate(adj[::-1])[::-1]
        corrected = np.empty_like(adj)
        corrected[order] = np.minimum(adj, 1.0)

    for slot, i in enumerate(idx):
        out[i]["p_value_corrected"] = float(corrected[slot])
        out[i]["significant"] = bool(corrected[slot] < alpha)
        out[i]["correction_method"] = method
        out[i]["n_tests_in_family"] = int(n)
    for i, r in enumerate(out):
        if i not in idx:
            r["p_value_corrected"] = None
            r["significant"] = None
            r["correction_method"] = method
    return out


# ---------------------------------------------------------------------------
# Null models (protocol section 34)
# ---------------------------------------------------------------------------
def null_label_shuffle(values: Sequence[float], labels: Sequence[Any], *,
                       n_perm: int = 1000, seed: int = 0,
                       statistic: Optional[Callable] = None) -> Dict[str, Any]:
    """Null 3: shuffle correct/incorrect labels, keep values fixed.

    Directly tests whether the correct-vs-incorrect difference exceeds what
    an arbitrary partition of the same values would produce.
    """
    v = np.asarray(values, dtype=np.float64)
    lab = np.asarray(labels)
    valid = np.isfinite(v) & np.array([l is not None for l in lab])
    v, lab = v[valid], lab[valid]
    classes = np.unique(lab)
    if len(classes) != 2 or v.size < 4:
        return {"status": "insufficient_data", "n": int(v.size)}
    if statistic is None:
        def statistic(a, b):
            return abs(float(np.mean(a) - np.mean(b)))
    mask = lab == classes[0]
    observed = statistic(v[mask], v[~mask])
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(mask)
        null[i] = statistic(v[perm], v[~perm])
    return {
        "status": "ok", "observed": float(observed),
        "null_mean": float(np.mean(null)), "null_std": float(np.std(null)),
        "null_p95": float(np.percentile(null, 95)),
        "p_value": float((np.sum(null >= observed) + 1) / (n_perm + 1)),
        "z_score": float((observed - np.mean(null)) / max(np.std(null), EPS)),
        "exceeds_null": bool(observed > np.percentile(null, 95)),
        "n": int(v.size), "n_perm": n_perm,
    }


def null_random_layer(curves: np.ndarray, *, n_perm: int = 1000, seed: int = 0
                      ) -> Dict[str, Any]:
    """Null 1/5: is the peak more concentrated than for shuffled layer order?

    Independently permuting the layer axis of each curve destroys any
    depth-ordered structure while preserving each curve's marginal
    distribution of values exactly. If the real peak's prominence is not
    above this null, the "transition" is just the largest of L noisy values.
    """
    C = np.asarray(curves, dtype=np.float64)
    if C.ndim != 2 or C.shape[0] < 2:
        return {"status": "insufficient_data"}
    from .signals import transition_sharpness

    def prominence(mat: np.ndarray) -> float:
        with _quiet_nan():
            mean_curve = np.nanmean(mat, axis=0)
        s = transition_sharpness(mean_curve)
        return float(s["sharpness_ratio"]) if np.isfinite(s["sharpness_ratio"]) \
            else float("nan")

    observed = prominence(C)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = np.empty_like(C)
        for r in range(C.shape[0]):
            shuffled[r] = rng.permutation(C[r])
        null[i] = prominence(shuffled)
    finite = null[np.isfinite(null)]
    if not np.isfinite(observed) or finite.size == 0:
        return {"status": "degenerate", "observed": float(observed)}
    return {
        "status": "ok", "observed_sharpness": float(observed),
        "null_mean": float(np.mean(finite)), "null_p95": float(np.percentile(finite, 95)),
        "p_value": float((np.sum(finite >= observed) + 1) / (finite.size + 1)),
        "exceeds_null": bool(observed > np.percentile(finite, 95)),
        "n_curves": int(C.shape[0]), "n_perm": n_perm,
        "description": "layer order permuted within each curve",
    }


def null_smooth_random_walk(n_curves: int, n_layers: int, *, seed: int = 0,
                            smoothness: float = 0.8) -> np.ndarray:
    """Null 6: smooth random walks with no transition.

    Peak-finding on *any* smooth curve returns a layer. This generates curves
    with realistic autocorrelation but no built-in transition, so the
    detectors' false-positive rate can be measured directly rather than
    assumed to be zero.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros((n_curves, n_layers))
    for i in range(n_curves):
        x = 0.0
        for l in range(n_layers):
            x = smoothness * x + rng.normal()
            out[i, l] = x
    return out


def detector_false_positive_rate(n_curves: int, n_layers: int, *,
                                 seed: int = 0, n_repeats: int = 20
                                 ) -> Dict[str, Any]:
    """Fraction of *structureless* curves a detector calls 'sharp'.

    This is the single most important null in the suite: it puts a number on
    how often the detection machinery invents a transition where none exists.
    """
    from .critical import classify_shape
    from .signals import transition_sharpness

    shapes: Dict[str, int] = {}
    total = 0
    for r in range(n_repeats):
        curves = null_smooth_random_walk(n_curves, n_layers, seed=seed + r)
        for c in curves:
            s = transition_sharpness(c)
            shape = classify_shape(s["width_half_max"], s["sharpness_ratio"],
                                   n_layers)
            shapes[shape] = shapes.get(shape, 0) + 1
            total += 1
    return {
        "status": "ok", "n_curves": total,
        "shape_counts": shapes,
        "false_sharp_rate": shapes.get("sharp", 0) / max(1, total),
        "flat_rate": shapes.get("flat", 0) / max(1, total),
        "n_layers": n_layers,
        "description": ("smooth AR(1) random walks with no transition; any "
                        "'sharp' classification here is a false positive"),
    }


# ---------------------------------------------------------------------------
# Confound analysis (protocol section 35)
# ---------------------------------------------------------------------------
CONFOUND_COLUMNS = [
    "prompt_length", "generation_length", "sequence_length",
    "baseline_confidence", "baseline_entropy", "answer_token_frequency_rank",
    "finish_reason_is_eos", "candidate_mass",
]


def confound_correlations(df: Any, target: str,
                          confounds: Optional[Sequence[str]] = None
                          ) -> Dict[str, Any]:
    """Spearman correlation between a target measure and each confound.

    Spearman rather than Pearson because prompt length and rank variables are
    heavily skewed and monotone association is the relevant question.
    """
    from scipy.stats import spearmanr

    confounds = list(confounds or CONFOUND_COLUMNS)
    out: Dict[str, Any] = {"target": target, "correlations": {}}
    if df is None or target not in df.columns:
        return {**out, "status": "target_unavailable"}
    y = df[target].to_numpy(dtype=np.float64, na_value=np.nan)
    for c in confounds:
        if c not in df.columns:
            out["correlations"][c] = {"status": "column_absent"}
            continue
        x = df[c].to_numpy(dtype=np.float64, na_value=np.nan)
        both = np.isfinite(x) & np.isfinite(y)
        if both.sum() < 5 or np.std(x[both]) < EPS or np.std(y[both]) < EPS:
            out["correlations"][c] = {"status": "insufficient_variation",
                                      "n": int(both.sum())}
            continue
        rho, p = spearmanr(x[both], y[both])
        out["correlations"][c] = {"status": "ok", "spearman_rho": float(rho),
                                  "p_value": float(p), "n": int(both.sum())}
    out["status"] = "ok"
    return out


def partial_effect_controlling(df: Any, target: str, group_col: str,
                               controls: Sequence[str]) -> Dict[str, Any]:
    """Group difference in ``target`` after linearly removing ``controls``.

    Answers: does the correct-vs-incorrect difference survive controlling for
    prompt length, generation length and baseline confidence? If it does not,
    the difference is attributable to those, not to the representation
    dynamics.
    """
    out: Dict[str, Any] = {"target": target, "group_col": group_col,
                           "controls": list(controls)}
    if df is None or target not in df.columns or group_col not in df.columns:
        return {**out, "status": "columns_unavailable"}

    usable = [c for c in controls if c in df.columns]
    sub = df[[target, group_col] + usable].copy()
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 10 or sub[group_col].nunique() < 2:
        return {**out, "status": "insufficient_data", "n": int(len(sub))}

    y = sub[target].to_numpy(dtype=np.float64)
    g = sub[group_col].to_numpy()
    classes = np.unique(g)
    if len(classes) != 2:
        return {**out, "status": "group_not_binary", "n_classes": int(len(classes))}

    raw_g = hedges_g(y[g == classes[0]], y[g == classes[1]])
    out["raw_effect_size"] = raw_g
    out["n"] = int(len(sub))
    out["controls_used"] = usable

    if not usable:
        return {**out, "status": "no_controls_available",
                "adjusted_effect_size": raw_g}

    X = sub[usable].to_numpy(dtype=np.float64)
    X = np.column_stack([np.ones(len(X)), X])
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return {**out, "status": "regression_failed"}
    residual = y - X @ beta
    adj_g = hedges_g(residual[g == classes[0]], residual[g == classes[1]])
    out["adjusted_effect_size"] = adj_g
    out["effect_attenuation"] = (float(1 - abs(adj_g) / abs(raw_g))
                                 if np.isfinite(raw_g) and abs(raw_g) > EPS
                                 else float("nan"))
    out["status"] = "ok"
    out["interpretation"] = (
        "large attenuation means the group difference is largely explained by "
        "the control variables rather than by representation dynamics")
    return out


def matched_subsample(df: Any, group_col: str, match_cols: Sequence[str], *,
                      n_bins: int = 4, seed: int = 0) -> Any:
    """Coarsened exact matching on the confounds.

    Bins each control into quantiles and keeps an equal number from each group
    within every occupied cell. Cruder than propensity matching but
    transparent, and it does not require a correctly specified model.
    """
    import pandas as pd

    if df is None or group_col not in df.columns:
        return df
    usable = [c for c in match_cols if c in df.columns]
    if not usable:
        return df
    work = df.copy()
    keys = []
    for c in usable:
        try:
            work[f"__bin_{c}"] = pd.qcut(work[c], q=n_bins, labels=False,
                                         duplicates="drop")
        except (ValueError, TypeError):
            work[f"__bin_{c}"] = 0
        keys.append(f"__bin_{c}")
    rng = np.random.default_rng(seed)
    kept: List[Any] = []
    for _, cell in work.groupby(keys, dropna=False):
        counts = cell[group_col].value_counts()
        if len(counts) < 2:
            continue
        k = int(counts.min())
        for value in counts.index:
            rows = cell[cell[group_col] == value]
            take = rows.sample(n=k, random_state=int(rng.integers(0, 2**31)))
            kept.append(take)
    if not kept:
        return work.iloc[0:0].drop(columns=keys)
    out = pd.concat(kept, axis=0)
    return out.drop(columns=keys)


# ---------------------------------------------------------------------------
# Discovery-mode correlation matrix
# ---------------------------------------------------------------------------
def signal_correlation_matrix(df: Any, columns: Optional[Sequence[str]] = None,
                              method: str = "spearman") -> Dict[str, Any]:
    """All-pairs correlations among the numeric signals (protocol section 59).

    Explicitly exploratory. The number of implied tests is large, so a
    Bonferroni threshold for the whole matrix is reported and no individual
    cell is called significant on its own.
    """
    import pandas as pd

    if df is None or len(df) == 0:
        return {"status": "no_data"}
    numeric = df.select_dtypes(include=[np.number])
    if columns:
        numeric = numeric[[c for c in columns if c in numeric.columns]]
    numeric = numeric.loc[:, numeric.std(numeric_only=True) > EPS]
    if numeric.shape[1] < 2:
        return {"status": "insufficient_columns", "n_columns": int(numeric.shape[1])}
    corr = numeric.corr(method=method, min_periods=5)
    n_pairs = corr.shape[0] * (corr.shape[0] - 1) // 2
    return {
        "status": "ok",
        "columns": list(corr.columns),
        "matrix": corr.to_numpy(),
        "method": method,
        "n_rows": int(len(numeric)),
        "n_pairs": int(n_pairs),
        "bonferroni_alpha": float(0.05 / max(1, n_pairs)),
        "note": ("exploratory; with %d pairs, individual cells are not tested "
                 "and should be treated as hypothesis-generating only" % n_pairs),
    }


def top_correlations(result: Dict[str, Any], k: int = 25) -> List[Dict[str, Any]]:
    if result.get("status") != "ok":
        return []
    M = result["matrix"]
    cols = result["columns"]
    out: List[Dict[str, Any]] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = M[i, j]
            if np.isfinite(v):
                out.append({"a": cols[i], "b": cols[j], "correlation": float(v)})
    out.sort(key=lambda d: -abs(d["correlation"]))
    return out[:k]
