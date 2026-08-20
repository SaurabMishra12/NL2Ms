"""Phases 13, 31, 32 -- causal intervention and its control battery.

The correlational phases can, at best, show that something changes sharply
around some layer. They cannot show that the layer matters. This phase
perturbs the residual stream and measures whether the model's answer changes.

Every intervention is paired with controls, because "perturbing layer 14
changed the answer" is uninformative on its own -- perturbing *any* layer
with enough magnitude changes the answer. What would be informative is a
*layer-specific* effect that survives:

``random_layer``
    Same perturbation kind and magnitude at a layer drawn uniformly at random.
    Controls for "any perturbation does this".
``norm_matched``
    Every perturbation is scaled to the same fraction of ``||h_l||``, so a
    late-layer effect is not just late-layer residual norm growth.
``random_direction``
    Isotropic direction, controlling for the answer-aligned direction being
    special.
``cross_sample``
    A direction taken from a *different* sample's answer geometry, which has
    the same statistical character but the wrong content.
``along`` vs ``orthogonal_to_answer_dir``
    If only the along-direction perturbation matters, the effect is about the
    answer representation; if orthogonal works equally well, it is generic
    disruption.

Nothing here writes to the original checkpoints; every intervention produces
its own artefact under ``interventions/``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .hooks import PerturbationSpec, orthogonal_component, random_directions
from .signals import EPS, js_divergence, shannon_entropy, softmax


@dataclass
class InterventionOutcome:
    sample_id: str
    layer: int
    normalised_layer: float
    layer_role: str            # critical / pre / post / early / late / random
    perturbation_kind: str
    epsilon: float
    delta_norm: float
    residual_norm: float
    relative_magnitude: float
    baseline_prediction: Optional[str]
    perturbed_prediction: Optional[str]
    answer_changed: Optional[bool]
    baseline_correct: Optional[bool]
    perturbed_correct: Optional[bool]
    margin_baseline: float
    margin_perturbed: float
    margin_delta: float
    entropy_baseline: float
    entropy_perturbed: float
    entropy_delta: float
    jsd_output: float
    top1_changed: Optional[bool]
    decoded_baseline: str
    decoded_perturbed: str
    seed: int
    status: str = "ok"

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


def answer_direction(wrapper: Any, answer_spec: Any) -> Optional[Any]:
    """Unembedding row for the correct answer token, unit-normalised.

    This is the direction that, added to the residual stream, most directly
    increases the correct token's logit. Perturbing along it is the strongest
    available "push toward the right answer"; perturbing orthogonally to it is
    the matched control.
    """
    import torch

    if not answer_spec.usable or answer_spec.correct_index is None:
        return None
    head = wrapper.model.get_output_embeddings()
    if head is None:
        return None
    tid = int(answer_spec.first_token_ids[answer_spec.correct_index])
    if tid >= head.weight.shape[0]:
        return None
    row = head.weight[tid].detach().to(torch.float32)
    return row / row.norm().clamp(min=EPS)


def pca_direction(hidden_bank: Optional[np.ndarray], layer: int,
                  device: Any, component: int = 0) -> Optional[Any]:
    """Top principal direction of the population at this layer.

    ``hidden_bank``: ``(n_samples, n_depth, d)`` from earlier extraction.
    Perturbing along a high-variance population direction tests whether the
    effect depends on moving within the manifold the model actually uses.
    """
    import torch

    if hidden_bank is None or layer >= hidden_bank.shape[1]:
        return None
    X = np.asarray(hidden_bank[:, layer, :], dtype=np.float64)
    if X.shape[0] < 3:
        return None
    Xc = X - X.mean(axis=0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if component >= vt.shape[0]:
        return None
    v = torch.tensor(vt[component], dtype=torch.float32, device=device)
    return v / v.norm().clamp(min=EPS)


def build_perturbation(kind: str, *, dim: int, device: Any, seed: int,
                       answer_dir: Optional[Any] = None,
                       pca_dir: Optional[Any] = None,
                       cross_dir: Optional[Any] = None) -> Tuple[Optional[Any], str]:
    """Return a unit-norm direction of the requested family."""
    import torch

    rand = random_directions(1, dim, seed, device=device, dtype=torch.float32)[0]
    if kind == "gaussian":
        return rand, "isotropic gaussian, unit norm"
    if kind == "along_answer_dir":
        if answer_dir is None:
            return None, "answer direction unavailable"
        return answer_dir, "unembedding row of the correct answer token"
    if kind == "orthogonal_to_answer_dir":
        if answer_dir is None:
            return None, "answer direction unavailable"
        return orthogonal_component(rand, answer_dir), \
            "random direction with the answer component removed"
    if kind == "pca_direction":
        if pca_dir is None:
            return None, "pca direction unavailable"
        return pca_dir, "top principal direction of the layer population"
    if kind == "cross_sample":
        if cross_dir is None:
            return None, "cross-sample direction unavailable"
        return cross_dir, "answer direction of a different sample"
    return None, f"unknown perturbation kind {kind}"


def _measure_output(wrapper: Any, logits_last: Any, answer_spec: Any
                    ) -> Dict[str, Any]:
    """Reduce a final-position logit vector to the outcome measures."""
    import torch

    p = torch.softmax(logits_last.to(torch.float32), dim=-1)
    pnp = p.detach().cpu().numpy().astype(np.float64)
    out: Dict[str, Any] = {
        "entropy": float(shannon_entropy(pnp)),
        "top1_id": int(np.argmax(pnp)),
        "top1_prob": float(np.max(pnp)),
        "probs": pnp,
    }
    if answer_spec.usable and answer_spec.correct_index is not None:
        ids = [int(t) for t in answer_spec.first_token_ids]
        cand = pnp[ids]
        total = cand.sum()
        q = cand / total if total > EPS else np.full_like(cand, 1.0 / len(cand))
        ci = answer_spec.correct_index
        wrong = np.delete(q, ci)
        out["margin"] = float(q[ci] - (wrong.max() if wrong.size else 0.0))
        out["candidate_mass"] = float(total)
        out["predicted_label"] = answer_spec.labels[int(np.argmax(cand))]
    else:
        out["margin"] = float("nan")
        out["predicted_label"] = None
    return out


def run_interventions(wrapper: Any, sample: Any, generation: Any,
                      answer_spec: Any, cfg: Any, *,
                      critical_layer: Optional[int],
                      token_position: int,
                      hidden_bank: Optional[np.ndarray] = None,
                      cross_sample_dir: Optional[Any] = None,
                      seed: int = 0) -> Dict[str, Any]:
    """Sweep perturbations across layers, kinds and magnitudes for one sample.

    Efficiency note: for each (kind, epsilon) the layers are swept in one
    batched forward pass -- the same trick as the J-space sweep -- with row 0
    left clean as the baseline. Answers are read from the next-token
    distribution at ``token_position`` rather than by re-generating, because
    re-generating for every cell would multiply the cost by
    ``max_new_tokens``. A separate, smaller re-generation check is run for the
    critical layer only, so the cheap proxy can be validated against the
    behaviour it stands in for.
    """
    import torch

    icfg = cfg.interventions
    n_layers = wrapper.arch.n_layers
    d = wrapper.arch.hidden_size
    device = wrapper.device

    full_ids = list(generation.input_ids) + list(generation.generated_ids)
    if len(full_ids) < 2:
        return {"status": "sequence_too_short", "sample_id": sample.sample_id}
    ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    T = ids.shape[1]
    pos = int(token_position) % T

    # Baseline: residual norms per layer + clean output distribution.
    from .hooks import residual_edits
    with torch.inference_mode():
        with residual_edits(wrapper, [], capture_layers=list(range(n_layers)),
                            capture_positions=[pos]) as cap:
            base_out = wrapper.model(input_ids=ids, attention_mask=mask,
                                     use_cache=False)
    residual_norms = {l: float(cap[l][0, 0, :].to(torch.float32).norm().item())
                      for l in cap}
    baseline = _measure_output(wrapper, base_out.logits[0, pos, :], answer_spec)
    del base_out, cap

    # Layer plan: offsets relative to the critical layer, plus absolute
    # fractions of depth, plus random control layers.
    rng = np.random.default_rng(seed)
    layer_roles: Dict[int, str] = {}

    def add(layer: int, role: str) -> None:
        layer = int(np.clip(layer, 0, n_layers - 1))
        layer_roles.setdefault(layer, role)

    if critical_layer is not None:
        for off in icfg.layer_offsets:
            role = ("critical" if off == 0 else
                    ("pre_critical" if off < 0 else "post_critical"))
            add(critical_layer + off, role)
    for frac in icfg.include_absolute_layers:
        add(int(round(frac * (n_layers - 1))), f"absolute_{frac:g}")
    control_pool = [l for l in range(n_layers) if l not in layer_roles]
    if control_pool:
        n_ctrl = min(icfg.control_random_layers, len(control_pool))
        for l in rng.choice(control_pool, size=n_ctrl, replace=False):
            add(int(l), "random_control")

    layers = sorted(layer_roles.keys())

    answer_dir = answer_direction(wrapper, answer_spec)
    outcomes: List[InterventionOutcome] = []
    direction_notes: Dict[str, str] = {}

    for kind in icfg.perturbation_kinds:
        for eps in icfg.epsilons:
            specs: List[PerturbationSpec] = []
            meta: Dict[int, Dict[str, Any]] = {}
            for row, l in enumerate(layers, start=1):
                pdir = pca_direction(hidden_bank, l, device) \
                    if kind == "pca_direction" else None
                vec, note = build_perturbation(
                    kind, dim=d, device=device,
                    seed=seed * 1000 + l * 17 + int(eps * 100),
                    answer_dir=answer_dir, pca_dir=pdir,
                    cross_dir=cross_sample_dir)
                direction_notes[kind] = note
                if vec is None:
                    continue
                # Norm matching: every perturbation is eps * ||h_l||, so
                # magnitude is comparable across layers despite residual-norm
                # growth with depth.
                scale = float(eps) * residual_norms.get(l, 1.0)
                delta = (vec * scale).to(torch.float32)
                specs.append(PerturbationSpec(layer=l, delta=delta,
                                              token_positions=[pos],
                                              batch_rows=[row], mode="add"))
                meta[l] = {"row": row, "delta_norm": float(delta.norm().item()),
                           "residual_norm": residual_norms.get(l, float("nan")),
                           "note": note}
            if not specs:
                continue

            B = len(layers) + 1
            bids = ids.expand(B, -1).contiguous()
            bmask = mask.expand(B, -1).contiguous()
            with torch.inference_mode():
                with residual_edits(wrapper, specs):
                    out = wrapper.model(input_ids=bids, attention_mask=bmask,
                                        use_cache=False)
            logits = out.logits[:, pos, :]
            for l in layers:
                if l not in meta:
                    continue
                row = meta[l]["row"]
                pert = _measure_output(wrapper, logits[row], answer_spec)
                jsd = float(js_divergence(baseline["probs"], pert["probs"]))
                outcomes.append(InterventionOutcome(
                    sample_id=sample.sample_id, layer=int(l),
                    normalised_layer=float(l / max(1, n_layers - 1)),
                    layer_role=layer_roles[l], perturbation_kind=kind,
                    epsilon=float(eps),
                    delta_norm=meta[l]["delta_norm"],
                    residual_norm=meta[l]["residual_norm"],
                    relative_magnitude=float(eps),
                    baseline_prediction=baseline.get("predicted_label"),
                    perturbed_prediction=pert.get("predicted_label"),
                    answer_changed=(None if baseline.get("predicted_label") is None
                                    else baseline.get("predicted_label") !=
                                    pert.get("predicted_label")),
                    baseline_correct=_is_correct(baseline.get("predicted_label"),
                                                 sample.ground_truth),
                    perturbed_correct=_is_correct(pert.get("predicted_label"),
                                                  sample.ground_truth),
                    margin_baseline=baseline["margin"],
                    margin_perturbed=pert["margin"],
                    margin_delta=pert["margin"] - baseline["margin"],
                    entropy_baseline=baseline["entropy"],
                    entropy_perturbed=pert["entropy"],
                    entropy_delta=pert["entropy"] - baseline["entropy"],
                    jsd_output=jsd,
                    top1_changed=bool(pert["top1_id"] != baseline["top1_id"]),
                    decoded_baseline="", decoded_perturbed="",
                    seed=seed,
                ))
            del out, logits

    return {
        "status": "ok",
        "sample_id": sample.sample_id,
        "critical_layer": critical_layer,
        "token_position": pos,
        "n_layers": n_layers,
        "layers": layers,
        "layer_roles": {str(k): v for k, v in layer_roles.items()},
        "baseline": {k: v for k, v in baseline.items() if k != "probs"},
        "residual_norms": residual_norms,
        "direction_notes": direction_notes,
        "outcomes": [o.to_dict() for o in outcomes],
        "n_outcomes": len(outcomes),
        "answer_direction_available": answer_dir is not None,
    }


def _is_correct(prediction: Optional[str], ground_truth: Optional[str]
                ) -> Optional[bool]:
    if prediction is None or ground_truth is None:
        return None
    return str(prediction).strip().upper() == str(ground_truth).strip().upper()


def verify_with_regeneration(wrapper: Any, sample: Any, generation: Any,
                             answer_spec: Any, cfg: Any, *, layer: int,
                             epsilon: float, kind: str = "gaussian",
                             seed: int = 0) -> Dict[str, Any]:
    """Re-generate under one perturbation, to validate the cheap proxy.

    The sweep reads the answer from a single next-token distribution. This
    function actually decodes, so we can check that "the next-token argmax
    changed" tracks "the model produced a different answer". If the two
    disagree systematically, the sweep's conclusions do not transfer to
    behaviour and the report says so.
    """
    import torch
    from .hooks import residual_edits
    from .modeling import parse_prediction, trim_generated

    device = wrapper.device
    prompt_ids = torch.tensor([list(generation.input_ids)], dtype=torch.long,
                              device=device)
    mask = torch.ones_like(prompt_ids)
    pos = prompt_ids.shape[1] - 1

    with torch.inference_mode():
        with residual_edits(wrapper, [], capture_layers=[layer],
                            capture_positions=[pos]) as cap:
            wrapper.model(input_ids=prompt_ids, attention_mask=mask, use_cache=False)
    norm = float(cap[layer][0, 0, :].to(torch.float32).norm().item())

    answer_dir = answer_direction(wrapper, answer_spec)
    vec, note = build_perturbation(kind, dim=wrapper.arch.hidden_size,
                                   device=device, seed=seed,
                                   answer_dir=answer_dir)
    if vec is None:
        return {"status": "direction_unavailable", "note": note}
    delta = (vec * (epsilon * norm)).to(torch.float32)

    def decode(specs: Sequence[PerturbationSpec]) -> Tuple[str, Optional[str]]:
        with residual_edits(wrapper, specs):
            with torch.inference_mode():
                out = wrapper.model.generate(
                    input_ids=prompt_ids, attention_mask=mask,
                    max_new_tokens=cfg.interventions.intervention_max_new_tokens,
                    do_sample=False, pad_token_id=wrapper.tokenizer.pad_token_id,
                    return_dict_in_generate=True)
        gen = out.sequences[0, prompt_ids.shape[1]:].tolist()
        trimmed = trim_generated(gen, wrapper.tokenizer)
        text = wrapper.tokenizer.decode(trimmed, skip_special_tokens=True)
        pred, _ = parse_prediction(sample, text)
        return text, pred

    # Perturbing only the last prompt position leaves earlier positions clean,
    # but generation continues past it; the hook stays installed for the whole
    # decode so newly generated positions are unaffected by construction (the
    # spec targets an absolute index inside the prompt).
    base_text, base_pred = decode([])
    pert_text, pert_pred = decode([PerturbationSpec(
        layer=layer, delta=delta, token_positions=[pos], batch_rows=[0])])

    return {
        "status": "ok", "layer": int(layer), "epsilon": float(epsilon),
        "kind": kind, "direction_note": note,
        "baseline_text": base_text, "perturbed_text": pert_text,
        "baseline_prediction": base_pred, "perturbed_prediction": pert_pred,
        "prediction_changed": (None if base_pred is None and pert_pred is None
                               else base_pred != pert_pred),
        "ground_truth": sample.ground_truth,
        "delta_norm": float(delta.norm().item()),
        "residual_norm": norm,
    }


# ---------------------------------------------------------------------------
# Aggregation across samples
# ---------------------------------------------------------------------------
def aggregate_interventions(results: Sequence[Dict[str, Any]]) -> Any:
    """Flatten every outcome into a tidy DataFrame for the statistics phase."""
    import pandas as pd

    rows: List[Dict[str, Any]] = []
    for r in results:
        if r.get("status") != "ok":
            continue
        for o in r.get("outcomes", []):
            row = dict(o)
            row["critical_layer"] = r.get("critical_layer")
            cl = r.get("critical_layer")
            row["offset_from_critical"] = (None if cl is None
                                           else int(o["layer"]) - int(cl))
            rows.append(row)
    return pd.DataFrame(rows)


def sensitivity_by_layer_role(df: Any, metric: str = "jsd_output"
                              ) -> Dict[str, Any]:
    """Effect size by layer role, with the random-layer control alongside.

    The comparison that matters is ``critical`` vs ``random_control`` at
    matched epsilon and kind. A large absolute effect at the critical layer
    means nothing if the random control shows the same.
    """
    import pandas as pd

    if df is None or len(df) == 0:
        return {"status": "no_data"}
    out: Dict[str, Any] = {"status": "ok", "metric": metric}
    grouped = df.groupby(["layer_role", "perturbation_kind", "epsilon"])[metric]
    summary = grouped.agg(["count", "mean", "std", "median"]).reset_index()
    out["summary"] = summary.to_dict(orient="records")

    crit = df[df["layer_role"] == "critical"]
    ctrl = df[df["layer_role"] == "random_control"]
    if len(crit) and len(ctrl):
        out["critical_mean"] = float(crit[metric].mean())
        out["random_control_mean"] = float(ctrl[metric].mean())
        pooled = np.sqrt(0.5 * (crit[metric].var(ddof=1) + ctrl[metric].var(ddof=1)))
        out["cohens_d_critical_vs_random"] = (
            float((crit[metric].mean() - ctrl[metric].mean()) / pooled)
            if np.isfinite(pooled) and pooled > EPS else float("nan"))
        out["n_critical"] = int(len(crit))
        out["n_random_control"] = int(len(ctrl))
    else:
        out["comparison_status"] = "missing_critical_or_control_rows"
    return out
