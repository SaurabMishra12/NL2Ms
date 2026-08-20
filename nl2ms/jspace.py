"""Local sensitivity descriptors ("J-space") -- protocol sections 22 and 23.

Definition
----------
Let ``F_l`` be the map the model applies from the residual stream at layer *l*
to the residual stream at layer *l+1*, holding the rest of the context fixed:

    h_{l+1} = F_l(h_l)

The full Jacobian ``J_l = dF_l/dh_l`` is ``d x d`` (about 4096^2 = 17M entries
per layer for a 7B model) and is never materialised. Instead we probe it with
a fixed set of *k* unit directions ``v_1..v_k`` and record the directional
amplification

    s_i(l) = || F_l(h_l + eps*v_i) - F_l(h_l) ||  /  || eps*v_i ||

which is a finite-difference estimate of ``||J_l v_i||``. The descriptor

    J_l(x) = [s_1(l), ..., s_k(l)]

is what this module calls a point in **J-space**: a k-dimensional summary of
*how the layer transforms perturbations*, as opposed to the hidden state,
which summarises *what the layer represents*. The two are stored separately
and the descriptors are directly comparable across samples because the probe
directions are shared (seeded once per experiment, not per sample).

The research question this enables: do correct and incorrect trajectories
separate more strongly in J-space than in hidden-state space? A positive
answer would indicate that the computation's local sensitivity carries
information that its representation does not.

Caveats recorded here so they travel with the numbers
-----------------------------------------------------
* This is a **finite-difference** estimate, not an exact JVP. With fp16
  weights the estimate is unreliable below eps ~ 1e-2 relative; several eps
  values are therefore swept and the eps-dependence is reported rather than
  hidden behind a single number.
* Perturbing at one token position changes the keys and values that *later*
  positions attend to. The measured amplification is therefore the response
  of the full causal computation, not of an isolated MLP block. That is the
  quantity we want, but it is not the textbook layer Jacobian.
* ``s_i`` is a norm ratio and is blind to direction. A layer that rotates a
  perturbation without amplifying it registers as ``s = 1``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .hooks import PerturbationSpec, random_directions, residual_edits
from .signals import EPS, derivative


def _nanreduce(fn: Any, matrix: np.ndarray) -> np.ndarray:
    """Row-wise reduction over finite entries; all-NaN rows yield NaN quietly.

    Avoids ``np.nanmean``/``np.nanmax`` warnings on rows that are empty by
    construction (the final layer, which has no l+1 to measure into).
    """
    out = np.full(matrix.shape[0], np.nan, dtype=np.float64)
    for i in range(matrix.shape[0]):
        row = matrix[i]
        finite = row[np.isfinite(row)]
        if finite.size:
            out[i] = float(fn(finite))
    return out


def jacobian_descriptors(wrapper: Any, input_ids: Any, attention_mask: Any, *,
                         token_position: int,
                         n_probes: int = 8,
                         epsilons: Sequence[float] = (1e-2,),
                         seed: int = 7,
                         relative: bool = True,
                         layers: Optional[Sequence[int]] = None,
                         layer_chunk: int = 8) -> Dict[str, Any]:
    """Sweep directional sensitivity across layers for one sequence.

    Batching strategy: a perturbation applied at layer *l* only affects layers
    ``> l``, so *different layers cannot share a batch row* -- but they can
    share a *forward pass* if each is given its own row. We replicate the
    sequence ``chunk+1`` times, perturb layer ``l_j`` in row ``j``, and keep
    row 0 clean as the reference. That turns ``n_layers`` separate forward
    passes into ``ceil(n_layers / chunk)`` per (probe, eps) pair.

    Returns descriptors of shape ``(n_layers, n_probes)`` per epsilon.
    """
    import torch

    arch = wrapper.arch
    n_layers = arch.n_layers
    layer_list = list(layers) if layers is not None else list(range(n_layers))
    device = input_ids.device
    d = arch.hidden_size

    probes = random_directions(n_probes, d, seed, device=device,
                               dtype=torch.float32)

    # Clean reference pass: residual stream at every layer, at the probe position.
    with torch.inference_mode():
        with residual_edits(wrapper, [], capture_layers=list(range(n_layers)),
                            capture_positions=[token_position]) as clean_cap:
            wrapper.model(input_ids=input_ids, attention_mask=attention_mask,
                          use_cache=False)
    clean = {l: clean_cap[l][:, 0, :].to(torch.float32)
             for l in clean_cap}  # (1, d) each
    if len(clean) < n_layers:
        return {"status": "incomplete_capture",
                "captured": len(clean), "expected": n_layers}

    clean_norms = {l: float(clean[l].norm().item()) for l in clean}

    results: Dict[float, np.ndarray] = {}
    downstream: Dict[float, np.ndarray] = {}
    per_eps_meta: Dict[str, Any] = {}

    for eps in epsilons:
        S = np.full((n_layers, n_probes), np.nan, dtype=np.float64)
        S_final = np.full((n_layers, n_probes), np.nan, dtype=np.float64)
        for p_i in range(n_probes):
            for start in range(0, len(layer_list), layer_chunk):
                chunk = layer_list[start:start + layer_chunk]
                B = len(chunk) + 1  # row 0 stays clean
                ids = input_ids.expand(B, -1).contiguous()
                mask = (attention_mask.expand(B, -1).contiguous()
                        if attention_mask is not None else None)

                specs: List[PerturbationSpec] = []
                applied_norm: Dict[int, float] = {}
                for row, l in enumerate(chunk, start=1):
                    scale = (eps * clean_norms[l]) if relative else eps
                    delta = (probes[p_i] * scale).to(torch.float32)
                    applied_norm[l] = float(delta.norm().item())
                    specs.append(PerturbationSpec(
                        layer=l, delta=delta, token_positions=[token_position],
                        batch_rows=[row], mode="add"))

                capture = sorted({min(l + 1, n_layers - 1) for l in chunk} |
                                 {n_layers - 1})
                with torch.inference_mode():
                    with residual_edits(wrapper, specs, capture_layers=capture,
                                        capture_positions=[token_position]) as cap:
                        wrapper.model(input_ids=ids, attention_mask=mask,
                                      use_cache=False)

                for row, l in enumerate(chunk, start=1):
                    nxt = min(l + 1, n_layers - 1)
                    if nxt not in cap or l == n_layers - 1:
                        # The last layer has no l+1 to measure into.
                        continue
                    tensor = cap[nxt][:, 0, :].to(torch.float32)
                    d_next = float((tensor[row] - tensor[0]).norm().item())
                    denom = max(applied_norm[l], 1e-12)
                    S[l, p_i] = d_next / denom
                    if (n_layers - 1) in cap:
                        fin = cap[n_layers - 1][:, 0, :].to(torch.float32)
                        S_final[l, p_i] = float((fin[row] - fin[0]).norm().item()) / denom
                del cap
        results[float(eps)] = S
        downstream[float(eps)] = S_final
        per_eps_meta[str(eps)] = {
            "n_finite": int(np.sum(np.isfinite(S))),
            "median_amplification": float(np.nanmedian(S)) if np.isfinite(S).any() else None,
        }

    primary_eps = float(epsilons[len(epsilons) // 2])
    primary = results[primary_eps]
    # The final layer has no l+1 to measure into, so its row is all-NaN by
    # construction; the resulting warnings are expected, not a data problem.
    profile = _nanreduce(np.mean, primary)
    profile_max = _nanreduce(np.max, primary)
    spread = _nanreduce(np.std, primary)

    return {
        "status": "ok",
        "descriptors": {str(k): v for k, v in results.items()},
        "descriptors_to_final": {str(k): v for k, v in downstream.items()},
        "epsilons": [float(e) for e in epsilons],
        "primary_epsilon": primary_eps,
        "n_probes": n_probes,
        "probe_seed": seed,
        "relative": relative,
        "token_position": int(token_position),
        "amplification_mean": profile,
        "amplification_max": profile_max,
        "amplification_spread": spread,
        "amplification_delta": derivative(profile),
        "clean_norms": np.array([clean_norms[l] for l in range(n_layers)]),
        "eps_meta": per_eps_meta,
    }


def epsilon_consistency(result: Dict[str, Any]) -> Dict[str, Any]:
    """Is the amplification profile stable across perturbation magnitudes?

    If the shape of ``s(l)`` changes with eps, the finite-difference estimate
    is in a nonlinear regime and any "sensitivity peak" may be an artefact of
    the chosen step size. This check is what makes the eps sweep worth doing;
    its result is reported alongside every J-space claim.
    """
    desc = result.get("descriptors", {})
    if len(desc) < 2:
        return {"status": "single_epsilon", "n_epsilons": len(desc)}
    keys = sorted(desc.keys(), key=float)
    profiles = [_nanreduce(np.mean, desc[k]) for k in keys]
    correlations = {}
    peaks = {}
    for i in range(len(keys)):
        finite = np.isfinite(profiles[i])
        peaks[keys[i]] = int(np.nanargmax(np.where(finite, profiles[i], -np.inf))) \
            if finite.any() else None
        for j in range(i + 1, len(keys)):
            both = np.isfinite(profiles[i]) & np.isfinite(profiles[j])
            if both.sum() >= 3:
                c = float(np.corrcoef(profiles[i][both], profiles[j][both])[0, 1])
            else:
                c = float("nan")
            correlations[f"{keys[i]}_vs_{keys[j]}"] = c
    peak_values = [v for v in peaks.values() if v is not None]
    return {
        "status": "ok",
        "n_epsilons": len(keys),
        "profile_correlations": correlations,
        "peak_layer_per_epsilon": peaks,
        "peak_layer_agreement": (len(set(peak_values)) == 1) if peak_values else None,
        "peak_layer_spread": (int(max(peak_values) - min(peak_values))
                              if peak_values else None),
        "min_correlation": (float(np.nanmin(list(correlations.values())))
                            if correlations else float("nan")),
    }


# ---------------------------------------------------------------------------
# J-space as a representation space
# ---------------------------------------------------------------------------
def jspace_matrix(results: Sequence[Dict[str, Any]], layer: int,
                  epsilon: Optional[float] = None) -> Optional[np.ndarray]:
    """Assemble ``(n_samples, n_probes)`` descriptors at one layer."""
    rows: List[np.ndarray] = []
    for r in results:
        if r.get("status") != "ok":
            continue
        eps_key = str(epsilon) if epsilon is not None else str(r["primary_epsilon"])
        desc = r.get("descriptors", {}).get(eps_key)
        if desc is None or layer >= desc.shape[0]:
            continue
        rows.append(desc[layer])
    if not rows:
        return None
    return np.stack(rows, axis=0)


def compare_separability(jspace: Optional[np.ndarray],
                         hidden: Optional[np.ndarray],
                         labels: np.ndarray, *, seed: int = 0,
                         n_splits: int = 5) -> Dict[str, Any]:
    """Cross-validated separability of two groups in J-space vs hidden space.

    Uses a linear probe with standardisation, and reports balanced accuracy so
    that an unbalanced correct/incorrect split cannot masquerade as
    separability. The *difference* between the two spaces is the quantity of
    interest; both are computed on the same samples and folds.

    A dimensionality caveat: J-space is ``k``-dimensional (typically 8) while
    hidden space is thousands. A linear probe in the larger space overfits
    more readily, so hidden space is PCA-reduced to the same width, and the
    unreduced result is reported too.
    """
    out: Dict[str, Any] = {"n_samples": int(len(labels))}
    labels = np.asarray(labels)
    valid = np.array([l is not None and not _isnan(l) for l in labels])
    y = labels[valid].astype(int)
    if len(np.unique(y)) < 2 or y.size < 2 * n_splits:
        return {**out, "status": "insufficient_labelled_samples",
                "n_valid": int(valid.sum())}

    def probe(X: Optional[np.ndarray], tag: str, n_components: Optional[int] = None
              ) -> None:
        if X is None:
            out[f"{tag}_status"] = "unavailable"
            return
        Xv = np.asarray(X, dtype=np.float64)[valid]
        finite = np.all(np.isfinite(Xv), axis=1)
        if finite.sum() < 2 * n_splits or len(np.unique(y[finite])) < 2:
            out[f"{tag}_status"] = "insufficient_finite_rows"
            return
        Xv, yv = Xv[finite], y[finite]
        if n_components and Xv.shape[1] > n_components:
            from sklearn.decomposition import PCA
            Xv = PCA(n_components=min(n_components, Xv.shape[0] - 1,
                                      Xv.shape[1]),
                     random_state=seed).fit_transform(Xv)
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, random_state=seed))
        counts = np.bincount(yv)
        splits = int(min(n_splits, counts[counts > 0].min()))
        if splits < 2:
            out[f"{tag}_status"] = "too_few_per_class"
            return
        cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
        scores = cross_val_score(clf, Xv, yv, cv=cv, scoring="balanced_accuracy")
        out[f"{tag}_balanced_accuracy"] = float(np.mean(scores))
        out[f"{tag}_balanced_accuracy_std"] = float(np.std(scores))
        out[f"{tag}_n"] = int(len(yv))
        out[f"{tag}_n_features"] = int(Xv.shape[1])
        out[f"{tag}_status"] = "ok"

    k = jspace.shape[1] if jspace is not None else None
    probe(jspace, "jspace")
    probe(hidden, "hidden_full")
    if k:
        probe(hidden, "hidden_matched_dim", n_components=k)

    if out.get("jspace_status") == "ok" and out.get("hidden_matched_dim_status") == "ok":
        out["jspace_minus_hidden_matched"] = (out["jspace_balanced_accuracy"] -
                                              out["hidden_matched_dim_balanced_accuracy"])
    out["chance_level"] = 0.5
    out["note"] = ("balanced accuracy; chance is 0.5 by construction. "
                   "A difference between spaces is descriptive and needs the "
                   "permutation test in the statistics phase before it counts "
                   "as evidence.")
    return out


def _isnan(x: Any) -> bool:
    try:
        return bool(np.isnan(x))
    except (TypeError, ValueError):
        return False


def summarise_jspace(result: Dict[str, Any]) -> Dict[str, Any]:
    """Scalar summary of one sample's J-space sweep, for the master table."""
    if result.get("status") != "ok":
        return {"jspace_status": result.get("status", "missing")}
    prof = np.asarray(result["amplification_mean"], dtype=np.float64)
    finite = np.isfinite(prof)
    if not finite.any():
        return {"jspace_status": "all_nan"}
    peak = int(np.nanargmax(np.where(finite, prof, -np.inf)))
    return {
        "jspace_status": "ok",
        "jspace_peak_layer": peak,
        "jspace_peak_value": float(prof[peak]),
        "jspace_mean_amplification": float(np.nanmean(prof)),
        "jspace_max_amplification": float(np.nanmax(prof)),
        "jspace_amplification_range": float(np.nanmax(prof) - np.nanmin(prof)),
        "jspace_probe_spread_at_peak": float(result["amplification_spread"][peak]),
    }
