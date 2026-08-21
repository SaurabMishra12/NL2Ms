"""Pure NumPy signal definitions -- entropy, divergence, dynamics, symmetry.

Everything here is a deterministic function of arrays with no model, no GPU
and no disk access, which is what makes these definitions testable and
auditable. Each function corresponds to an entry in
:mod:`nl2ms.registry.SIGNAL_REGISTRY`.

Numerical conventions used throughout:

* Probabilities are computed in float64 from float32 logits. Softmax over a
  150k vocabulary in float16 loses enough precision to distort entropy in the
  third decimal, which is the scale at which layer-to-layer differences live.
* ``0 log 0 = 0`` is applied explicitly rather than relying on ``nan_to_num``,
  so a genuine NaN from bad input still surfaces as a NaN.
* Derivatives across layers use non-uniform-safe central differences at
  interior points and one-sided differences at the boundaries.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-12


# ---------------------------------------------------------------------------
# Probability utilities
# ---------------------------------------------------------------------------
def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax in float64."""
    x = np.asarray(logits, dtype=np.float64)
    x = x - np.max(x, axis=axis, keepdims=True)
    np.exp(x, out=x)
    total = np.sum(x, axis=axis, keepdims=True)
    return x / np.maximum(total, EPS)


def log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    x = x - np.max(x, axis=axis, keepdims=True)
    return x - np.log(np.maximum(np.sum(np.exp(x), axis=axis, keepdims=True), EPS))


def renormalise(p: np.ndarray, axis: int = -1) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    total = np.sum(p, axis=axis, keepdims=True)
    return p / np.maximum(total, EPS)


# ---------------------------------------------------------------------------
# Entropy (protocol section 14)
# ---------------------------------------------------------------------------
def shannon_entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    """H = -sum p log p, in nats, with 0 log 0 := 0."""
    p = np.asarray(p, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log(p), 0.0)
    return -np.sum(terms, axis=axis)


def normalised_entropy(p: np.ndarray, n_categories: Optional[int] = None,
                       axis: int = -1) -> np.ndarray:
    """H / log(V): 0 = fully concentrated, 1 = uniform over V categories."""
    p = np.asarray(p, dtype=np.float64)
    V = n_categories if n_categories is not None else p.shape[axis]
    if V <= 1:
        return np.zeros(shannon_entropy(p, axis=axis).shape)
    return shannon_entropy(p, axis=axis) / np.log(V)


def top_k_entropy(p: np.ndarray, k: int, axis: int = -1) -> np.ndarray:
    """Entropy of the renormalised top-k mass.

    Reported separately from full-vocabulary entropy because when only top-k
    probabilities are stored, this is the quantity that is actually
    identifiable; conflating the two would overstate precision.
    """
    p = np.asarray(p, dtype=np.float64)
    k = min(k, p.shape[axis])
    idx = np.argsort(-p, axis=axis)
    top = np.take_along_axis(p, np.take(idx, np.arange(k), axis=axis), axis=axis)
    return shannon_entropy(renormalise(top, axis=axis), axis=axis)


def derivative(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """First difference with central differences at interior points.

    ``np.gradient`` semantics: same shape as the input, so a per-layer array
    stays aligned with its layer index instead of shrinking by one.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.shape[axis] < 2:
        return np.zeros_like(x)
    return np.gradient(x, axis=axis)


def curvature_1d(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Second difference (discrete curvature of a scalar layer profile)."""
    return derivative(derivative(x, axis=axis), axis=axis)


def entropy_profile(probs_by_layer: np.ndarray, vocab_size: Optional[int] = None
                    ) -> Dict[str, np.ndarray]:
    """Entropy, its normalised form, and its first two layer derivatives.

    ``probs_by_layer`` has shape ``(n_layers, n_categories)``.
    """
    H = shannon_entropy(probs_by_layer, axis=-1)
    out = {
        "entropy": H,
        "entropy_normalised": normalised_entropy(probs_by_layer, vocab_size, axis=-1),
        "entropy_delta": derivative(H),
        "entropy_curvature": curvature_1d(H),
    }
    return out


# ---------------------------------------------------------------------------
# Distributional distance (protocol section 15)
# ---------------------------------------------------------------------------
def kl_divergence(p: np.ndarray, q: np.ndarray, axis: int = -1) -> np.ndarray:
    """KL(p||q) in nats.

    Returns ``inf`` where ``q`` is zero but ``p`` is not: that is the honest
    answer, and it is left to the caller (rather than clipped away) so an
    invalid comparison is visible instead of silently finite.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    support_violation = (p > 0) & (q <= 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        safe_p = np.where(p > 0, p, 1.0)
        safe_q = np.where(q > 0, q, 1.0)
        terms = np.where(p > 0, p * np.log(safe_p / safe_q), 0.0)
        terms = np.where(support_violation, np.inf, terms)
    return np.sum(terms, axis=axis)


def js_divergence(p: np.ndarray, q: np.ndarray, axis: int = -1,
                  base: Optional[float] = None) -> np.ndarray:
    """Jensen-Shannon divergence: symmetric, bounded, always finite.

    Preferred over KL for layer-to-layer comparison precisely because the
    supports differ: an early layer can assign essentially zero mass to the
    token a late layer prefers, which makes KL infinite and useless.
    """
    p = renormalise(np.asarray(p, dtype=np.float64), axis=axis)
    q = renormalise(np.asarray(q, dtype=np.float64), axis=axis)
    m = 0.5 * (p + q)
    jsd = 0.5 * kl_divergence(p, m, axis=axis) + 0.5 * kl_divergence(q, m, axis=axis)
    jsd = np.maximum(jsd, 0.0)  # clip float noise below zero
    if base is not None:
        jsd = jsd / np.log(base)
    return jsd


def js_distance(p: np.ndarray, q: np.ndarray, axis: int = -1) -> np.ndarray:
    """Square root of JSD -- a true metric, in [0, sqrt(ln 2)]."""
    return np.sqrt(np.maximum(js_divergence(p, q, axis=axis), 0.0))


def layerwise_jsd(probs_by_layer: np.ndarray) -> Dict[str, np.ndarray]:
    """JSD between consecutive layers, plus cumulative and from-first movement.

    ``probs_by_layer``: ``(n_layers, n_categories)``.

    ``jsd_consecutive[l]`` is JSD(layer l, layer l+1) and has length
    ``n_layers - 1``; it is padded to ``n_layers`` with a leading zero so that
    it can be stacked with per-layer quantities without index confusion.
    """
    P = np.asarray(probs_by_layer, dtype=np.float64)
    n = P.shape[0]
    if n < 2:
        zeros = np.zeros(n)
        return {"jsd_consecutive": zeros, "jsd_cumulative": zeros,
                "jsd_from_first": zeros, "kl_consecutive": zeros}
    consecutive = js_divergence(P[:-1], P[1:], axis=-1)
    kl = kl_divergence(P[:-1], P[1:], axis=-1)
    padded = np.concatenate([[0.0], consecutive])
    kl_padded = np.concatenate([[0.0], kl])
    return {
        "jsd_consecutive": padded,
        "jsd_cumulative": np.cumsum(padded),
        "jsd_from_first": np.concatenate([[0.0],
                                          js_divergence(P[0][None, :], P[1:], axis=-1)]),
        "jsd_from_last": np.concatenate([js_divergence(P[:-1], P[-1][None, :], axis=-1),
                                         [0.0]]),
        "kl_consecutive": kl_padded,
    }


# ---------------------------------------------------------------------------
# Order parameter and answer competition (protocol sections 13, 24)
# ---------------------------------------------------------------------------
def order_parameter_closed_set(cand_probs: np.ndarray, correct_index: int
                               ) -> Dict[str, np.ndarray]:
    """Margin-family statistics over an enumerated candidate set.

    ``cand_probs``: ``(n_layers, n_candidates)`` -- raw (not renormalised)
    probabilities of each candidate's first token.

    ``m_l = p(correct) - max_wrong p(wrong)`` is computed on the *renormalised*
    within-candidate distribution, so it measures competition among the
    answers the task offers rather than competition against the whole
    vocabulary. The unnormalised mass is reported separately as
    ``candidate_mass``, because a large margin over a candidate set that
    collectively holds 0.1% of the probability mass means something very
    different from the same margin at 90%.
    """
    P = np.asarray(cand_probs, dtype=np.float64)
    n_layers, K = P.shape
    mass = np.sum(P, axis=-1)
    Q = renormalise(P, axis=-1)

    correct_p = Q[:, correct_index]
    wrong = np.delete(Q, correct_index, axis=1)
    best_wrong = np.max(wrong, axis=1) if wrong.size else np.zeros(n_layers)

    margin = correct_p - best_wrong
    with np.errstate(divide="ignore", invalid="ignore"):
        log_odds = np.log(np.maximum(correct_p, EPS)) - \
            np.log(np.maximum(1.0 - correct_p, EPS))

    order = np.argsort(-Q, axis=1)
    rank = np.array([int(np.where(order[l] == correct_index)[0][0]) + 1
                     for l in range(n_layers)])
    sorted_q = np.sort(Q, axis=1)[:, ::-1]
    top1 = sorted_q[:, 0]
    top2 = sorted_q[:, 1] if K > 1 else np.zeros(n_layers)

    return {
        "margin": margin,
        "correct_prob": correct_p,
        "best_wrong_prob": best_wrong,
        "log_odds": log_odds,
        "rank": rank.astype(np.float64),
        "confidence": top1,
        "top1_minus_top2": top1 - top2,
        "candidate_mass": mass,
        "candidate_entropy": shannon_entropy(Q, axis=-1),
        "candidate_entropy_normalised": normalised_entropy(Q, K, axis=-1),
        "prob_variance": np.var(Q, axis=1),
        "gini": gini_concentration(Q),
        "symmetry_breaking_index": 1.0 - normalised_entropy(Q, K, axis=-1),
        "margin_delta": derivative(margin),
        "margin_curvature": curvature_1d(margin),
    }


def order_parameter_open_vocab(full_probs: np.ndarray, correct_token_id: int
                               ) -> Dict[str, np.ndarray]:
    """Margin-family statistics against the entire vocabulary.

    Used where no distractor set exists (GSM8K). ``m_l = p(correct) -
    max_{v != correct} p(v)``: negative until the correct token becomes the
    argmax. This is *not* comparable in scale to the closed-set margin and is
    never pooled with it.
    """
    P = np.asarray(full_probs, dtype=np.float64)
    n_layers, V = P.shape
    correct_p = P[:, correct_token_id]
    masked = P.copy()
    masked[:, correct_token_id] = -np.inf
    best_other = np.max(masked, axis=1)
    best_other = np.where(np.isfinite(best_other), best_other, 0.0)

    order = np.argsort(-P, axis=1)
    rank = np.array([int(np.where(order[l] == correct_token_id)[0][0]) + 1
                     for l in range(n_layers)])
    top1 = np.max(P, axis=1)
    margin = correct_p - best_other
    with np.errstate(divide="ignore", invalid="ignore"):
        log_odds = np.log(np.maximum(correct_p, EPS)) - \
            np.log(np.maximum(1.0 - correct_p, EPS))
    return {
        "margin": margin,
        "correct_prob": correct_p,
        "best_wrong_prob": best_other,
        "log_odds": log_odds,
        "rank": rank.astype(np.float64),
        "confidence": top1,
        "margin_delta": derivative(margin),
        "margin_curvature": curvature_1d(margin),
    }


def gini_concentration(p: np.ndarray, axis: int = -1) -> np.ndarray:
    """Gini coefficient of a probability vector: 0 = uniform, ->1 = peaked."""
    p = np.sort(np.asarray(p, dtype=np.float64), axis=axis)
    n = p.shape[axis]
    if n < 2:
        return np.zeros(p.shape[:-1] if axis == -1 else p.shape)
    index = np.arange(1, n + 1, dtype=np.float64)
    shape = [1] * p.ndim
    shape[axis] = n
    index = index.reshape(shape)
    total = np.sum(p, axis=axis, keepdims=True)
    numer = np.sum((2 * index - n - 1) * p, axis=axis, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = numer / np.maximum(n * total, EPS)
    return np.squeeze(g, axis=axis)


def pairwise_candidate_distances(cand_probs: np.ndarray) -> np.ndarray:
    """Mean pairwise |p_i - p_j| across candidates, per layer.

    A scalar summary of how far the candidate distribution is from a tie.
    """
    P = renormalise(np.asarray(cand_probs, dtype=np.float64), axis=-1)
    n_layers, K = P.shape
    if K < 2:
        return np.zeros(n_layers)
    out = np.zeros(n_layers)
    for l in range(n_layers):
        diffs = np.abs(P[l][:, None] - P[l][None, :])
        out[l] = diffs[np.triu_indices(K, k=1)].mean()
    return out


# ---------------------------------------------------------------------------
# Representation trajectory dynamics (protocol section 21)
# ---------------------------------------------------------------------------
def trajectory_metrics(H: np.ndarray) -> Dict[str, np.ndarray]:
    """Geometry of the layer-indexed path h_1 -> h_2 -> ... -> h_L.

    ``H``: ``(n_layers, hidden_dim)`` in float64.

    Both raw and norm-normalised velocities are returned. Residual-stream norm
    grows roughly monotonically with depth in most decoder-only models, so raw
    velocity trivially increases with layer index; the normalised version
    divides that growth out and is the one to trust for locating a transition.
    """
    H = np.asarray(H, dtype=np.float64)
    n_layers = H.shape[0]
    out: Dict[str, np.ndarray] = {}

    norms = np.linalg.norm(H, axis=1)
    out["norm"] = norms

    if n_layers < 2:
        zeros = np.zeros(n_layers)
        for key in ["velocity", "velocity_normalised", "acceleration",
                    "cosine_displacement", "turning_angle", "curvature",
                    "path_length", "displacement_from_first",
                    "displacement_from_last"]:
            out[key] = zeros.copy()
        return out

    diffs = np.diff(H, axis=0)                       # (L-1, d)
    velocity = np.linalg.norm(diffs, axis=1)         # (L-1,)
    # Pad to length L with a leading zero: velocity[l] is the step INTO layer l.
    out["velocity"] = np.concatenate([[0.0], velocity])

    denom = np.maximum(0.5 * (norms[:-1] + norms[1:]), EPS)
    out["velocity_normalised"] = np.concatenate([[0.0], velocity / denom])

    out["acceleration"] = derivative(out["velocity"])

    cos_disp = np.zeros(n_layers)
    for l in range(n_layers - 1):
        cos_disp[l + 1] = _cosine(H[l], H[l + 1])
    out["cosine_displacement"] = cos_disp

    # Turning angle between consecutive displacement vectors: the local
    # direction change of the trajectory, independent of step size.
    turning = np.zeros(n_layers)
    curvature = np.zeros(n_layers)
    for l in range(1, n_layers - 1):
        v1, v2 = diffs[l - 1], diffs[l]
        c = _cosine(v1, v2)
        angle = float(np.arccos(np.clip(c, -1.0, 1.0)))
        turning[l] = angle
        step = 0.5 * (np.linalg.norm(v1) + np.linalg.norm(v2))
        curvature[l] = angle / max(step, EPS)
    out["turning_angle"] = turning
    out["curvature"] = curvature

    out["path_length"] = np.concatenate([[0.0], np.cumsum(velocity)])
    out["displacement_from_first"] = np.linalg.norm(H - H[0][None, :], axis=1)
    out["displacement_from_last"] = np.linalg.norm(H - H[-1][None, :], axis=1)
    out["cosine_to_final"] = np.array([_cosine(H[l], H[-1]) for l in range(n_layers)])
    out["cosine_to_first"] = np.array([_cosine(H[l], H[0]) for l in range(n_layers)])
    return out


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < EPS or nb < EPS:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.maximum(norms, EPS)
    return Xn @ Xn.T


# ---------------------------------------------------------------------------
# Susceptibility-like measures (protocol section 25)
# ---------------------------------------------------------------------------
def empirical_susceptibility(values_by_sample: np.ndarray) -> Dict[str, np.ndarray]:
    """Across-sample variance of a per-layer quantity.

    Named "susceptibility-like" deliberately: in statistical physics
    susceptibility is the response of an order parameter to a conjugate field,
    with a fluctuation-dissipation identity behind the variance formula.
    Nothing here establishes such an identity. This is the *variance of an
    empirical order-parameter analogue across a finite sample of prompts*, and
    a peak in it is a descriptive observation, not a critical exponent.

    ``values_by_sample``: ``(n_samples, n_layers)``.
    """
    V = np.asarray(values_by_sample, dtype=np.float64)
    if V.ndim != 2 or V.shape[0] < 2:
        n_layers = V.shape[-1] if V.ndim else 0
        nan = np.full(n_layers, np.nan)
        return {"susceptibility": nan, "mean": nan, "n_samples": 0}
    with np.errstate(invalid="ignore"):
        var = np.nanvar(V, axis=0, ddof=1)
        mean = np.nanmean(V, axis=0)
        std = np.nanstd(V, axis=0, ddof=1)
    return {
        "susceptibility": var,
        "mean": mean,
        "std": std,
        "n_samples": V.shape[0],
        "susceptibility_delta": derivative(var),
        # Scale-free version: a variance peak can otherwise just track a mean
        # peak, since many of these quantities are non-negative.
        "coefficient_of_variation": std / np.maximum(np.abs(mean), EPS),
    }


# ---------------------------------------------------------------------------
# Peak / transition detection helpers
# ---------------------------------------------------------------------------
def argmax_safe(x: np.ndarray) -> Optional[int]:
    """Index of the maximum, or ``None`` if the profile is entirely invalid."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or np.all(~np.isfinite(x)):
        return None
    masked = np.where(np.isfinite(x), x, -np.inf)
    return int(np.argmax(masked))


def transition_sharpness(profile: np.ndarray, peak_index: Optional[int] = None
                         ) -> Dict[str, float]:
    """How concentrated a per-layer profile is around its peak.

    ``peak_fraction`` is the peak's share of total mass; ``width_half_max`` is
    the number of layers at or above half the peak. A single-layer spike and a
    broad plateau can share a maximum but differ completely here, which is the
    distinction protocol section 27 asks for.
    """
    x = np.asarray(profile, dtype=np.float64)
    finite = np.isfinite(x)
    if not finite.any():
        return {"peak_index": None, "peak_value": float("nan"),
                "peak_fraction": float("nan"), "width_half_max": float("nan"),
                "sharpness_ratio": float("nan")}
    xf = np.where(finite, x, 0.0)
    # Shift to non-negative so "fraction of mass" is meaningful for signed
    # profiles (e.g. entropy derivative).
    shifted = xf - np.min(xf)
    idx = peak_index if peak_index is not None else int(np.argmax(shifted))
    peak = float(shifted[idx])
    total = float(np.sum(shifted))
    half = peak / 2.0
    width = int(np.sum(shifted >= half)) if peak > 0 else 0
    others = np.delete(shifted, idx)
    mean_other = float(np.mean(others)) if others.size else 0.0
    return {
        "peak_index": int(idx),
        "peak_value": float(xf[idx]),
        "peak_fraction": peak / total if total > EPS else float("nan"),
        "width_half_max": float(width),
        "sharpness_ratio": peak / mean_other if mean_other > EPS else float("inf"),
    }


def detect_interval(profile: np.ndarray, threshold_fraction: float = 0.5
                    ) -> Tuple[Optional[int], Optional[int]]:
    """Contiguous layer interval where a profile stays above a relative threshold.

    Implements the critical *region* notion: the interval is grown outward
    from the peak while the profile remains above
    ``min + threshold_fraction * (max - min)``.
    """
    x = np.asarray(profile, dtype=np.float64)
    finite = np.isfinite(x)
    if not finite.any():
        return None, None
    xf = np.where(finite, x, -np.inf)
    peak_idx = int(np.argmax(xf))
    lo_val, hi_val = float(np.min(xf[finite])), float(xf[peak_idx])
    if not np.isfinite(lo_val) or hi_val <= lo_val:
        return peak_idx, peak_idx
    cutoff = lo_val + threshold_fraction * (hi_val - lo_val)
    start = peak_idx
    while start - 1 >= 0 and xf[start - 1] >= cutoff:
        start -= 1
    end = peak_idx
    while end + 1 < len(xf) and xf[end + 1] >= cutoff:
        end += 1
    return start, end


def normalised_layers(n_layers: int) -> np.ndarray:
    """l/L in [0, 1], the only depth coordinate valid across models."""
    if n_layers <= 1:
        return np.zeros(n_layers)
    return np.arange(n_layers, dtype=np.float64) / (n_layers - 1)


def safe_stack(profiles: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    """Stack equal-length profiles, returning ``None`` on shape mismatch.

    Shape mismatch means different models or different layer counts got mixed;
    padding them together would silently fabricate comparisons.
    """
    arrays = [np.asarray(p, dtype=np.float64) for p in profiles if p is not None]
    if not arrays:
        return None
    lengths = {a.shape[0] for a in arrays}
    if len(lengths) != 1:
        return None
    return np.stack(arrays, axis=0)
