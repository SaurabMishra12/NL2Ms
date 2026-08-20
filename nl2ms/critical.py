"""Phases 11 and 26-27 -- critical-layer and critical-region detection.

Eight independent detectors run on every sample. They are deliberately
redundant and deliberately *not* combined into a single number unless they
agree, because the central failure mode of this kind of analysis is picking
whichever heuristic produces the cleanest picture.

Detectors
---------
============================  =========================================
method                        peak of
============================  =========================================
``order_parameter_growth``    |d m_l / dl|
``entropy_derivative``        |d H_l / dl|
``jsd``                       JSD(p_l || p_{l+1})
``representation_curvature``  trajectory curvature
``perturbation_sensitivity``  J-space mean amplification
``susceptibility``            across-sample Var[m_l]  (population-level)
``attention_restructuring``   ||A_l - A_{l-1}||_F
``margin_growth``             d m_l / dl (signed, growth only)
============================  =========================================

Consensus rule
--------------
A ``critical_layer_consensus`` is emitted **only** when at least
``min_methods_agreeing`` detectors fall within ``agreement_tolerance`` layers
of each other. Otherwise the field is ``None`` and ``consensus_status``
explains why. A median over disagreeing detectors would always produce a
number, and that number would mean nothing.

Region vs layer
---------------
Every detector also returns an interval. ``transition_shape`` classifies the
profile as ``sharp`` (peak concentrated in <= 2 layers), ``distributed``
(a broad elevated region) or ``flat`` (no peak distinguishable from the
profile's own variation). A "critical layer" reported without this
classification would hide the difference between a genuine localised
transition and a gradual ramp.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .signals import (EPS, argmax_safe, detect_interval, normalised_layers,
                      transition_sharpness)

# method -> (profile key, whether to take |.| before peak-finding)
DETECTOR_SPECS: Dict[str, Tuple[str, bool]] = {
    "order_parameter_growth": ("order_margin_delta", True),
    "entropy_derivative": ("entropy_delta", True),
    "jsd": ("jsd_prev_layer", False),
    "representation_curvature": ("traj_curvature", False),
    "perturbation_sensitivity": ("jspace_amplification", False),
    "attention_restructuring": ("attn_restructuring_frobenius_delta", False),
    "margin_growth": ("order_margin_delta", False),
    "velocity": ("traj_velocity_normalised", False),
}

# Population-level detector; computed once over all samples, not per sample.
POPULATION_DETECTORS = {"susceptibility": "susceptibility_margin"}

SHARP_MAX_WIDTH = 2
FLAT_MIN_SHARPNESS = 1.5


@dataclass
class DetectorResult:
    method: str
    profile_key: str
    critical_layer: Optional[int]
    normalised_layer: Optional[float]
    interval_start: Optional[int]
    interval_end: Optional[int]
    peak_value: float
    peak_fraction: float
    width_half_max: float
    sharpness_ratio: float
    transition_shape: str
    status: str

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


def classify_shape(width_half_max: float, sharpness_ratio: float,
                   n_layers: int) -> str:
    """Sharp / distributed / flat, from peak width and prominence.

    ``flat`` is the honest label when a profile has a maximum but that maximum
    is not distinguishable from the profile's ordinary variation. Reporting a
    "critical layer" for such a profile would be reading structure into noise.
    """
    if np.isnan(sharpness_ratio):
        return "flat"
    # An infinite ratio is the *sharpest* possible profile (every non-peak
    # layer sits at the baseline), not an undefined one. Treating it as
    # non-finite and falling through to "flat" would misclassify the cleanest
    # transitions the instrument can see.
    if np.isinf(sharpness_ratio):
        return "sharp" if width_half_max <= SHARP_MAX_WIDTH else "distributed"
    if sharpness_ratio < FLAT_MIN_SHARPNESS:
        return "flat"
    if not np.isfinite(width_half_max):
        return "undetermined"
    if width_half_max <= SHARP_MAX_WIDTH:
        return "sharp"
    if width_half_max >= 0.5 * n_layers:
        return "diffuse"
    return "distributed"


def detect_from_profile(profile: np.ndarray, method: str, profile_key: str, *,
                        use_abs: bool = False,
                        threshold_fraction: float = 0.5,
                        exclude_boundary: int = 1) -> DetectorResult:
    """Locate the peak of one profile and describe its shape.

    ``exclude_boundary`` drops the first and last entries before peak-finding.
    Boundary layers are systematically special -- the embedding row has no
    previous layer (NaN derivatives) and the final layer is where the
    unembedding is actually applied -- so a peak there is usually an artefact
    of the measurement rather than a transition in the computation.
    """
    x = np.asarray(profile, dtype=np.float64)
    n = x.size
    if n == 0:
        return DetectorResult(method, profile_key, None, None, None, None,
                              float("nan"), float("nan"), float("nan"),
                              float("nan"), "undetermined", "empty_profile")
    work = np.abs(x) if use_abs else x.copy()
    if exclude_boundary > 0 and n > 2 * exclude_boundary + 1:
        work[:exclude_boundary] = np.nan
        work[-exclude_boundary:] = np.nan
    if not np.isfinite(work).any():
        return DetectorResult(method, profile_key, None, None, None, None,
                              float("nan"), float("nan"), float("nan"),
                              float("nan"), "undetermined", "all_nan")

    peak = argmax_safe(work)
    sharp = transition_sharpness(work, peak)
    start, end = detect_interval(work, threshold_fraction)
    norm = normalised_layers(n)
    shape = classify_shape(sharp["width_half_max"], sharp["sharpness_ratio"], n)
    return DetectorResult(
        method=method, profile_key=profile_key, critical_layer=peak,
        normalised_layer=float(norm[peak]) if peak is not None else None,
        interval_start=start, interval_end=end,
        peak_value=float(sharp["peak_value"]),
        peak_fraction=float(sharp["peak_fraction"]),
        width_half_max=float(sharp["width_half_max"]),
        sharpness_ratio=float(sharp["sharpness_ratio"]),
        transition_shape=shape, status="ok",
    )


def detect_all(profiles: Dict[str, np.ndarray], *,
               extra_profiles: Optional[Dict[str, np.ndarray]] = None,
               threshold_fraction: float = 0.5) -> Dict[str, DetectorResult]:
    """Run every applicable detector on one sample's profiles."""
    available = dict(profiles)
    if extra_profiles:
        available.update(extra_profiles)
    out: Dict[str, DetectorResult] = {}
    for method, (key, use_abs) in DETECTOR_SPECS.items():
        prof = available.get(key)
        if prof is None:
            out[method] = DetectorResult(method, key, None, None, None, None,
                                         float("nan"), float("nan"),
                                         float("nan"), float("nan"),
                                         "undetermined", "profile_unavailable")
            continue
        out[method] = detect_from_profile(prof, method, key, use_abs=use_abs,
                                          threshold_fraction=threshold_fraction)
    return out


# ---------------------------------------------------------------------------
# Consensus
# ---------------------------------------------------------------------------
@dataclass
class ConsensusResult:
    critical_layer_consensus: Optional[int]
    normalised_consensus: Optional[float]
    consensus_status: str
    n_methods_available: int
    n_methods_agreeing: int
    agreeing_methods: List[str]
    per_method_layers: Dict[str, Optional[int]]
    spread: Optional[float]
    pairwise_agreement_rate: float
    shapes: Dict[str, str]
    dominant_shape: str

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


def consensus(detections: Dict[str, DetectorResult], n_layers: int, *,
              min_methods_agreeing: int = 3,
              agreement_tolerance: Optional[int] = None) -> ConsensusResult:
    """Emit a consensus layer only when the detectors actually agree.

    ``agreement_tolerance`` defaults to ``max(1, round(0.1 * n_layers))`` --
    ten percent of depth, so the criterion scales with model size rather than
    being tighter for deep models by accident.
    """
    if agreement_tolerance is None:
        agreement_tolerance = max(1, int(round(0.1 * n_layers)))

    layers = {m: d.critical_layer for m, d in detections.items()}
    shapes = {m: d.transition_shape for m, d in detections.items()}
    valid = {m: l for m, l in layers.items() if l is not None}
    n_available = len(valid)

    if n_available == 0:
        return ConsensusResult(None, None, "no_detector_produced_a_layer", 0, 0,
                               [], layers, None, float("nan"), shapes,
                               "undetermined")

    values = np.array(list(valid.values()), dtype=float)
    # Largest cluster of detectors within tolerance of a common centre.
    best_members: List[str] = []
    best_centre: Optional[float] = None
    for centre in values:
        members = [m for m, l in valid.items() if abs(l - centre) <= agreement_tolerance]
        if len(members) > len(best_members):
            best_members, best_centre = members, centre

    pairs = 0
    agreeing_pairs = 0
    keys = list(valid.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            pairs += 1
            if abs(valid[keys[i]] - valid[keys[j]]) <= agreement_tolerance:
                agreeing_pairs += 1
    pairwise_rate = agreeing_pairs / pairs if pairs else float("nan")

    shape_counts: Dict[str, int] = {}
    for s in shapes.values():
        shape_counts[s] = shape_counts.get(s, 0) + 1
    dominant = max(shape_counts.items(), key=lambda kv: kv[1])[0] if shape_counts \
        else "undetermined"

    if len(best_members) < min_methods_agreeing:
        return ConsensusResult(
            None, None,
            f"insufficient_agreement (largest cluster {len(best_members)} < "
            f"{min_methods_agreeing} within +/-{agreement_tolerance} layers)",
            n_available, len(best_members), best_members, layers,
            float(np.std(values)), pairwise_rate, shapes, dominant)

    cluster = [valid[m] for m in best_members]
    layer = int(round(float(np.median(cluster))))
    norm = float(layer / (n_layers - 1)) if n_layers > 1 else 0.0
    return ConsensusResult(layer, norm, "ok", n_available, len(best_members),
                           sorted(best_members), layers,
                           float(np.std(values)), pairwise_rate, shapes, dominant)


def agreement_matrix(per_sample: Sequence[Dict[str, DetectorResult]],
                     tolerance: int = 2) -> Dict[str, Any]:
    """How often each pair of detectors lands within ``tolerance`` layers.

    Reported as a matrix rather than a single agreement score so that a pair
    of detectors that are mathematically near-duplicates (margin growth and
    order-parameter growth share a profile) can be recognised as such rather
    than counted as independent corroboration.
    """
    methods = sorted(DETECTOR_SPECS.keys())
    n = len(methods)
    counts = np.zeros((n, n), dtype=np.float64)
    totals = np.zeros((n, n), dtype=np.float64)
    diffs: Dict[str, List[float]] = {}
    for det in per_sample:
        for i, a in enumerate(methods):
            for j, b in enumerate(methods):
                la = det.get(a).critical_layer if det.get(a) else None
                lb = det.get(b).critical_layer if det.get(b) else None
                if la is None or lb is None:
                    continue
                totals[i, j] += 1
                if abs(la - lb) <= tolerance:
                    counts[i, j] += 1
                if i < j:
                    diffs.setdefault(f"{a}|{b}", []).append(abs(la - lb))
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(totals > 0, counts / np.maximum(totals, 1), np.nan)
    return {
        "methods": methods,
        "agreement_rate": rate,
        "n_comparisons": totals,
        "tolerance": tolerance,
        "mean_abs_difference": {k: float(np.mean(v)) for k, v in diffs.items()},
        "note": ("order_parameter_growth and margin_growth derive from the "
                 "same profile and are not independent detectors; their "
                 "agreement is expected and carries no evidential weight"),
    }


# ---------------------------------------------------------------------------
# Population-level detectors
# ---------------------------------------------------------------------------
def susceptibility_detector(values_by_sample: np.ndarray,
                            label: str = "margin") -> Dict[str, Any]:
    """Across-sample variance peak (protocol section 25).

    Population-level by nature -- it has no per-sample value -- so it is kept
    out of the per-sample consensus and reported separately.
    """
    from .signals import empirical_susceptibility

    result = empirical_susceptibility(values_by_sample)
    chi = result.get("susceptibility")
    if chi is None or not np.isfinite(np.asarray(chi, float)).any():
        return {"status": "unavailable", "label": label}
    det = detect_from_profile(np.asarray(chi, float),
                              f"susceptibility_{label}", f"susceptibility_{label}")
    return {
        "status": "ok",
        "label": label,
        "profile": np.asarray(chi, float),
        "mean_profile": result.get("mean"),
        "coefficient_of_variation": result.get("coefficient_of_variation"),
        "detection": det.to_dict(),
        "n_samples": result.get("n_samples"),
        "caveat": ("empirical susceptibility-like measure: across-sample "
                   "variance of an order-parameter analogue. No "
                   "fluctuation-dissipation relation is claimed."),
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def summarise_detections(detections: Dict[str, DetectorResult],
                         cons: ConsensusResult, n_layers: int) -> Dict[str, Any]:
    """Flatten one sample's detection results for the master table."""
    out: Dict[str, Any] = {
        "critical_layer_consensus": cons.critical_layer_consensus,
        "critical_layer_consensus_normalised": cons.normalised_consensus,
        "consensus_status": cons.consensus_status,
        "n_detectors_available": cons.n_methods_available,
        "n_detectors_agreeing": cons.n_methods_agreeing,
        "detector_spread": cons.spread,
        "detector_pairwise_agreement": cons.pairwise_agreement_rate,
        "dominant_transition_shape": cons.dominant_shape,
        "n_layers": n_layers,
    }
    for method, det in detections.items():
        out[f"critical_layer_{method}"] = det.critical_layer
        out[f"critical_layer_{method}_normalised"] = det.normalised_layer
        out[f"transition_shape_{method}"] = det.transition_shape
        out[f"sharpness_{method}"] = det.sharpness_ratio
        out[f"width_{method}"] = det.width_half_max
        out[f"interval_start_{method}"] = det.interval_start
        out[f"interval_end_{method}"] = det.interval_end
    return out


def transition_strength(profiles: Dict[str, np.ndarray],
                        key: str = "order_margin_delta") -> float:
    """Scalar summary of how concentrated a profile's change is.

    Defined as the peak's share of the total absolute change:
    1.0 means all movement happened at one layer, 1/L means perfectly uniform.
    Explicitly an operational index, not a physical order parameter.
    """
    prof = profiles.get(key)
    if prof is None:
        return float("nan")
    x = np.abs(np.asarray(prof, dtype=np.float64))
    finite = x[np.isfinite(x)]
    if finite.size == 0 or finite.sum() <= EPS:
        return float("nan")
    return float(finite.max() / finite.sum())
