"""Phase 60 -- the signal registry, and data lineage for derived artefacts.

Every quantity the experiment computes is declared here with its formula,
tensor source, resolution and storage location. The registry is written to
``config/signal_registry.json`` at the start of a run, so a later reader can
tell exactly what ``jsd_prev_layer`` meant in *this* run without reading the
code that produced it.

The ``caveat`` field is deliberately part of the schema: a signal whose
interpretation has a known limitation carries that limitation next to its
definition rather than in a paragraph someone might not read.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .storage import save_json


@dataclass
class SignalSpec:
    signal_name: str
    definition: str
    formula: str
    tensor_source: str
    layer_resolution: str      # per_layer | per_layer_pair | scalar | population
    token_resolution: str      # per_position | answer_position | pooled | n/a
    storage_location: str
    dtype: str
    description: str
    phase: str
    caveat: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _s(**kwargs: Any) -> SignalSpec:
    return SignalSpec(**kwargs)


SIGNAL_REGISTRY: Dict[str, SignalSpec] = {}


def register(spec: SignalSpec) -> SignalSpec:
    SIGNAL_REGISTRY[spec.signal_name] = spec
    return spec


# --- Phase 4/5: logit lens and entropy --------------------------------------
register(_s(
    signal_name="entropy",
    definition="Shannon entropy of the layer-l next-token distribution",
    formula="H_l = -sum_v p_l(v) log p_l(v),  p_l = softmax(W_U FinalNorm(h_l))",
    tensor_source="residual stream after block l (hook capture)",
    layer_resolution="per_layer", token_resolution="per_position",
    storage_location="derived/entropy/<sample_id>.npz :: profiles::entropy",
    dtype="float32", phase="4/5",
    description="Uncertainty of the vocabulary distribution read off layer l.",
    caveat=("Logit-lens quantity. The final norm is applied out of "
            "distribution at early layers; compare with the no-norm control "
            "before interpreting the shape."),
))
register(_s(
    signal_name="entropy_normalised",
    definition="Entropy divided by log(vocabulary size)",
    formula="H_l / log V",
    tensor_source="same as entropy",
    layer_resolution="per_layer", token_resolution="per_position",
    storage_location="derived/entropy/<sample_id>.npz",
    dtype="float32", phase="5",
    description="Scale-free entropy; the only entropy comparable across models.",
    caveat="Comparable across models only if both use the same lens transform.",
))
register(_s(
    signal_name="entropy_delta",
    definition="First derivative of entropy across layers",
    formula="dH/dl via central differences (np.gradient)",
    tensor_source="entropy profile",
    layer_resolution="per_layer", token_resolution="per_position",
    storage_location="derived/entropy/<sample_id>.npz",
    dtype="float32", phase="5",
    description="Rate of uncertainty change; a detector input.",
    caveat="Boundary entries use one-sided differences and are excluded from "
           "peak detection.",
))
register(_s(
    signal_name="entropy_curvature",
    definition="Second derivative of entropy across layers",
    formula="d2H/dl2",
    tensor_source="entropy profile",
    layer_resolution="per_layer", token_resolution="per_position",
    storage_location="derived/entropy/<sample_id>.npz",
    dtype="float32", phase="5", description="Acceleration of entropy change.",
))
register(_s(
    signal_name="top1_prob",
    definition="Probability of the most likely token at layer l",
    formula="max_v p_l(v)",
    tensor_source="logit lens",
    layer_resolution="per_layer", token_resolution="per_position",
    storage_location="derived/entropy/<sample_id>.npz",
    dtype="float32", phase="4", description="Layer-wise confidence.",
))

# --- Phase 8: distributional dynamics ---------------------------------------
register(_s(
    signal_name="jsd_prev_layer",
    definition="Jensen-Shannon divergence between consecutive layers",
    formula="JSD(p_l || p_{l+1}) = 0.5 KL(p||m) + 0.5 KL(q||m), m = (p+q)/2",
    tensor_source="logit lens",
    layer_resolution="per_layer_pair", token_resolution="per_position",
    storage_location="derived/jsd/<sample_id>.npz",
    dtype="float32", phase="8",
    description="Distributional movement from one layer to the next.",
    caveat="Bounded by log 2 (nats). Index l holds JSD(l-1, l); index 0 is NaN.",
))
register(_s(
    signal_name="kl_prev_layer",
    definition="KL divergence between consecutive layers",
    formula="KL(p_{l-1} || p_l)",
    tensor_source="logit lens",
    layer_resolution="per_layer_pair", token_resolution="per_position",
    storage_location="derived/jsd/<sample_id>.npz",
    dtype="float32", phase="8", description="Asymmetric distributional movement.",
    caveat="Can be infinite when supports differ; JSD is the primary measure "
           "and KL is reported for completeness only.",
))
register(_s(
    signal_name="jsd_final_layer",
    definition="Divergence from the final layer's distribution",
    formula="JSD(p_l || p_L)",
    tensor_source="logit lens",
    layer_resolution="per_layer", token_resolution="per_position",
    storage_location="derived/jsd/<sample_id>.npz",
    dtype="float32", phase="8",
    description="How far layer l still is from the model's output.",
))

# --- Phase 10/13: order parameter -------------------------------------------
register(_s(
    signal_name="order_margin",
    definition="Answer order parameter: correct minus best-wrong probability",
    formula=("closed set: m_l = q_l(correct) - max_{w != correct} q_l(w), "
             "q = candidate probs renormalised. "
             "open vocab: m_l = p_l(correct) - max_{v != correct} p_l(v)"),
    tensor_source="logit lens candidate probabilities",
    layer_resolution="per_layer", token_resolution="answer_position",
    storage_location="derived/critical_layers/<sample_id>.npz",
    dtype="float32", phase="10/13",
    description="Primary order-parameter analogue.",
    caveat=("The closed-set and open-vocab definitions have different scales "
            "and are never pooled. Undefined for ambiguous samples."),
))
register(_s(
    signal_name="order_symmetry_breaking_index",
    definition="Concentration of the candidate distribution",
    formula="SB_l = 1 - H(q_l)/log K",
    tensor_source="candidate probabilities",
    layer_resolution="per_layer", token_resolution="answer_position",
    storage_location="derived/critical_layers/<sample_id>.npz",
    dtype="float32", phase="24",
    description="Moves from 0 (all candidates tied) to 1 (one dominates).",
    caveat=("Operational measure only. Not a physical order parameter and no "
            "symmetry group is being broken in any formal sense."),
))
register(_s(
    signal_name="order_rank",
    definition="Rank of the correct token in the layer-l distribution",
    formula="1 + |{v : p_l(v) > p_l(correct)}|",
    tensor_source="logit lens",
    layer_resolution="per_layer", token_resolution="answer_position",
    storage_location="derived/critical_layers/<sample_id>.npz",
    dtype="float32", phase="13", description="Rank-based order parameter.",
))

# --- Phase 9: trajectory ----------------------------------------------------
register(_s(
    signal_name="traj_velocity",
    definition="Euclidean step length between consecutive layers",
    formula="v_l = ||h_l - h_{l-1}||_2",
    tensor_source="residual stream",
    layer_resolution="per_layer_pair", token_resolution="answer_position",
    storage_location="derived/dynamics/<sample_id>.npz",
    dtype="float32", phase="9", description="Raw representation speed.",
    caveat=("Residual-stream norm grows with depth in decoder-only models, so "
            "raw velocity trends upward for architectural reasons. Use "
            "traj_velocity_normalised for transition detection."),
))
register(_s(
    signal_name="traj_velocity_normalised",
    definition="Step length divided by the local residual norm",
    formula="v_l / (0.5 (||h_{l-1}|| + ||h_l||))",
    tensor_source="residual stream",
    layer_resolution="per_layer_pair", token_resolution="answer_position",
    storage_location="derived/dynamics/<sample_id>.npz",
    dtype="float32", phase="9",
    description="Scale-free representation speed; the detector input.",
))
register(_s(
    signal_name="traj_curvature",
    definition="Turning angle per unit path length",
    formula="k_l = angle(h_l - h_{l-1}, h_{l+1} - h_l) / mean step length",
    tensor_source="residual stream",
    layer_resolution="per_layer", token_resolution="answer_position",
    storage_location="derived/dynamics/<sample_id>.npz",
    dtype="float32", phase="9",
    description="Direction change of the layer-wise trajectory.",
    caveat="In high dimensions random vectors are near-orthogonal, so angles "
           "cluster near pi/2; compare against the random-walk null.",
))
register(_s(
    signal_name="traj_turning_angle",
    definition="Angle between consecutive displacement vectors",
    formula="arccos( <d_{l-1}, d_l> / (||d_{l-1}|| ||d_l||) )",
    tensor_source="residual stream",
    layer_resolution="per_layer", token_resolution="answer_position",
    storage_location="derived/dynamics/<sample_id>.npz",
    dtype="float32", phase="9", description="Scale-free direction change.",
))

# --- Phase 6/17: attention --------------------------------------------------
register(_s(
    signal_name="attn_entropy_mean",
    definition="Attention entropy averaged over heads and positions",
    formula="mean_{h,t} [ -sum_k a_{l,h,t,k} log a_{l,h,t,k} ]",
    tensor_source="attention probabilities (eager attention required)",
    layer_resolution="per_layer", token_resolution="pooled",
    storage_location="raw/attention/<sample_id>.npz",
    dtype="float32", phase="6",
    description="How diffuse attention is at layer l.",
    caveat="Computed over the causally visible support only.",
))
register(_s(
    signal_name="attn_restructuring_frobenius_delta",
    definition="Frobenius norm of the layer-to-layer attention change",
    formula="||A_l - A_{l-1}||_F over query rows at analysed positions",
    tensor_source="attention probabilities",
    layer_resolution="per_layer_pair", token_resolution="pooled",
    storage_location="raw/attention/<sample_id>.npz",
    dtype="float32", phase="17",
    description="Magnitude of attention pattern reorganisation.",
    caveat=("Heads are matched by index across layers, which is a convention "
            "and not a correspondence. Attention weights are not attribution."),
))

# --- Phase 7/18-20: geometry ------------------------------------------------
register(_s(
    signal_name="effective_rank",
    definition="Effective rank of the layer's representation cloud",
    formula="exp(-sum_i p_i log p_i), p_i = s_i / sum_j s_j (singular values)",
    tensor_source="hidden states, population at layer l",
    layer_resolution="per_layer", token_resolution="answer_position",
    storage_location="derived/geometry/layer_geometry.json",
    dtype="float64", phase="18/20",
    description="Continuous count of directions carrying variance.",
    caveat=("Population-level: needs many samples at the same layer. "
            "Dimensional change is not evidence of reasoning on its own."),
))
register(_s(
    signal_name="neighbourhood_reorganisation",
    definition="1 minus the kNN-set Jaccard overlap between adjacent layers",
    formula="1 - mean_i |N_k^{l}(i) ∩ N_k^{l+1}(i)| / |N_k^{l}(i) ∪ N_k^{l+1}(i)|",
    tensor_source="hidden states, population",
    layer_resolution="per_layer_pair", token_resolution="answer_position",
    storage_location="derived/geometry/layer_geometry.json",
    dtype="float64", phase="19",
    description="Manifold reordering, insensitive to translation and rescaling.",
))
register(_s(
    signal_name="twonn_intrinsic_dimension",
    definition="TwoNN intrinsic dimension estimate",
    formula="MLE on the ratio of 2nd to 1st nearest-neighbour distances",
    tensor_source="hidden states, population",
    layer_resolution="per_layer", token_resolution="answer_position",
    storage_location="derived/geometry/layer_geometry.json",
    dtype="float64", phase="18",
    description="Local manifold dimension, scale-invariant.",
    caveat="Underestimates at high true dimension with few hundred points.",
))

# --- Phase 22/23: J-space ---------------------------------------------------
register(_s(
    signal_name="jspace_amplification",
    definition="Mean directional perturbation amplification at layer l",
    formula="s_i(l) = ||F_l(h_l + eps v_i) - F_l(h_l)|| / ||eps v_i||; mean over i",
    tensor_source="residual stream under hooked perturbation",
    layer_resolution="per_layer", token_resolution="answer_position",
    storage_location="derived/j_space/<sample_id>.npz",
    dtype="float64", phase="22/23",
    description="Local sensitivity of the layer transformation.",
    caveat=("Finite-difference estimate, not an exact JVP. Norm ratio only: "
            "blind to rotation. Includes the effect on later positions through "
            "attention, so it is not an isolated block Jacobian."),
))
register(_s(
    signal_name="jspace_descriptor",
    definition="The k-vector of directional sensitivities at layer l",
    formula="J_l(x) = [s_1(l), ..., s_k(l)]",
    tensor_source="same as jspace_amplification",
    layer_resolution="per_layer", token_resolution="answer_position",
    storage_location="derived/j_space/<sample_id>.npz",
    dtype="float64", phase="23",
    description="Point in J-space; probe directions are shared across samples.",
    caveat="Comparable across samples only because the probe seed is fixed.",
))

# --- Phase 25/26: susceptibility and detection ------------------------------
register(_s(
    signal_name="susceptibility_margin",
    definition="Across-sample variance of the order parameter at layer l",
    formula="chi_l = Var_x[ m_l(x) ]",
    tensor_source="order_margin across the sample population",
    layer_resolution="per_layer", token_resolution="answer_position",
    storage_location="statistics/susceptibility.json",
    dtype="float64", phase="25",
    description="Empirical susceptibility-like measure.",
    caveat=("NOT a physical susceptibility. No conjugate field and no "
            "fluctuation-dissipation relation is established. A peak is a "
            "descriptive observation about across-prompt variance."),
))
register(_s(
    signal_name="critical_layer_consensus",
    definition="Median layer of the largest agreeing detector cluster",
    formula="median{ l_m : |l_m - centre| <= tol }, emitted only if >= 3 agree",
    tensor_source="per-sample detector outputs",
    layer_resolution="scalar", token_resolution="answer_position",
    storage_location="derived/critical_layers/sample_summary.parquet",
    dtype="int", phase="26",
    description="Consensus critical layer, or null when detectors disagree.",
    caveat="Null is a real and common outcome and must not be imputed.",
))
register(_s(
    signal_name="transition_strength",
    definition="Concentration of layer-wise change at its peak",
    formula="max_l |d_l| / sum_l |d_l|",
    tensor_source="order_margin_delta",
    layer_resolution="scalar", token_resolution="answer_position",
    storage_location="derived/critical_layers/sample_summary.parquet",
    dtype="float64", phase="27",
    description="1 = all change at one layer; 1/L = perfectly uniform.",
    caveat="Operational index. A high value does not by itself establish a "
           "phase-transition-like phenomenon.",
))

# --- Phase 31: causal -------------------------------------------------------
register(_s(
    signal_name="jsd_output",
    definition="Divergence of the output distribution under perturbation",
    formula="JSD(p_clean || p_perturbed) at the measured token position",
    tensor_source="model logits under hooked residual edit",
    layer_resolution="scalar", token_resolution="answer_position",
    storage_location="interventions/<sample_id>.json",
    dtype="float64", phase="31",
    description="Causal effect size of a layer-l perturbation.",
    caveat=("Only interpretable relative to the matched random-layer control "
            "at the same epsilon and perturbation kind."),
))


def registry_as_list() -> List[Dict[str, Any]]:
    return [spec.to_dict() for spec in SIGNAL_REGISTRY.values()]


def save_registry(path: str | Path) -> Path:
    return save_json(path, {
        "n_signals": len(SIGNAL_REGISTRY),
        "signals": registry_as_list(),
        "note": ("Signals not listed here are derived quantities computed "
                 "inline; the primary measurement set is fully enumerated."),
    })


def describe(signal_name: str) -> Optional[Dict[str, Any]]:
    spec = SIGNAL_REGISTRY.get(signal_name)
    return spec.to_dict() if spec else None


# ---------------------------------------------------------------------------
# Data lineage (protocol section 61)
# ---------------------------------------------------------------------------
def write_lineage(path: str | Path, *, artefact: str, produced_by: str,
                  config_hash: str, sample_ids: Optional[List[str]] = None,
                  signals: Optional[List[str]] = None,
                  parameters: Optional[Dict[str, Any]] = None,
                  source_files: Optional[List[str]] = None,
                  git_commit: Optional[str] = None,
                  model_name: Optional[str] = None,
                  extra: Optional[Dict[str, Any]] = None) -> Path:
    """Write ``<artefact>.json`` beside a figure or table.

    Records everything needed to regenerate the artefact: which samples, which
    signal definitions, which configuration and which code version.
    """
    import time

    payload: Dict[str, Any] = {
        "artefact": artefact,
        "produced_by": produced_by,
        "created_at": time.time(),
        "created_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_hash": config_hash,
        "git_commit": git_commit,
        "model_name": model_name,
        "n_samples": len(sample_ids) if sample_ids is not None else None,
        "sample_ids": sample_ids,
        "signals": signals,
        "signal_definitions": {s: describe(s) for s in (signals or [])
                               if describe(s)},
        "parameters": parameters or {},
        "source_files": source_files or [],
    }
    if extra:
        payload.update(extra)
    return save_json(path, payload)
