"""Phase 4 -- layer-wise logit-lens analysis.

The transformation applied is recorded explicitly rather than assumed:

    p_l(v) = softmax( W_U * FinalNorm(h_l) )[v]

where ``h_l`` is the residual stream **after block l** (captured by hook, so
never double-normalised), ``FinalNorm`` is the backbone's own final
normalisation module and ``W_U`` its output embedding. A ``no_norm`` variant
omitting ``FinalNorm`` is computed at a subset of layers as a control.

What the lens is and is not
---------------------------
Applying the final norm and unembedding at layer *l* is a *projection of an
intermediate state into vocabulary space*, not a claim that the model
computes a distribution there. The learned norm scale is fitted to layer-L
statistics; at layer 3 it is being used out of distribution. Consequences
that matter for this experiment:

* Apparent "sharpening" in late layers is partly a property of the
  unembedding geometry, which is why the no-norm control and the shuffled-
  unembedding null exist.
* Entropy computed this way is comparable *across layers within a model* but
  not across models with different vocabulary sizes, unless normalised by
  ``log V``.

Everything is reduced on-device. A full ``(n_layers, n_positions, V)``
probability tensor for a 7B model with a 152k vocabulary is roughly 1 GB in
fp32 -- it is computed one layer at a time and never materialised in full,
and never written to disk unless ``SAVE_FULL_VOCAB_LOGITS`` is set.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .signals import EPS

# Scalar metrics stored per (layer, position). Order defines the array layout.
LENS_SCALARS = [
    "entropy",
    "entropy_normalised",
    "top1_prob",
    "top1_minus_top2",
    "topk_mass",
    "jsd_prev_layer",
    "kl_prev_layer",
    "jsd_final_layer",
    "jsd_first_layer",
    "logit_max",
    "logit_mean",
    "logit_std",
]


def _entropy_torch(logp: Any, p: Any) -> Any:
    return -(p * logp).sum(dim=-1)


def _jsd_torch(logp: Any, p: Any, logq: Any, q: Any) -> Any:
    """JSD in torch, computed from log-probs for stability."""
    import torch
    m = 0.5 * (p + q)
    logm = torch.log(m.clamp(min=EPS))
    kl_pm = (p * (logp - logm)).sum(dim=-1)
    kl_qm = (q * (logq - logm)).sum(dim=-1)
    return (0.5 * kl_pm + 0.5 * kl_qm).clamp(min=0.0)


def run_logit_lens(wrapper: Any, residual_by_layer: Sequence[Any], *,
                   positions: Sequence[int], top_k: int = 20,
                   answer_token_ids: Optional[Sequence[int]] = None,
                   correct_token_id: Optional[int] = None,
                   apply_norm: bool = True,
                   no_norm_control_layers: Optional[Sequence[int]] = None,
                   return_full_probs: bool = False) -> Dict[str, Any]:
    """Project every layer's residual stream to vocabulary space.

    ``residual_by_layer``: sequence of ``(1, T, d)`` tensors, index 0 being the
    embedding output and index ``l`` the stream after block ``l-1``, so the
    returned arrays have ``n_layers + 1`` rows and the depth axis includes the
    embedding.

    Returns arrays indexed ``[layer, position]``.
    """
    import torch

    n_depth = len(residual_by_layer)
    P = len(positions)
    if n_depth == 0 or P == 0:
        return {"status": "empty"}

    device = wrapper.device
    pos_idx = None

    scalars = np.full((n_depth, P, len(LENS_SCALARS)), np.nan, dtype=np.float32)
    topk_ids = np.zeros((n_depth, P, top_k), dtype=np.int32)
    topk_probs = np.zeros((n_depth, P, top_k), dtype=np.float32)

    n_cand = len(answer_token_ids) if answer_token_ids else 0
    cand_probs = (np.full((n_depth, P, n_cand), np.nan, dtype=np.float64)
                  if n_cand else None)
    correct_prob = np.full((n_depth, P), np.nan, dtype=np.float64)
    correct_rank = np.full((n_depth, P), np.nan, dtype=np.float64)

    full_probs = [] if return_full_probs else None

    cand_tensor = (torch.tensor(list(answer_token_ids), device=device,
                                dtype=torch.long) if n_cand else None)

    prev_p = prev_logp = None
    first_p = first_logp = None
    stored: List[Tuple[Any, Any]] = []   # kept only to compute jsd_final

    # Pass 1: compute per-layer distributions, streaming metrics that only need
    # the previous layer. The final-layer comparison needs the last
    # distribution, so we retain a fp16 copy per layer -- (n_depth, P, V) in
    # fp16 is ~450 MB for a 7B/152k model at P=48, which fits, whereas fp32
    # would not.
    for l, h in enumerate(residual_by_layer):
        if h is None:
            continue
        if pos_idx is None:
            T = h.shape[1]
            pos_idx = torch.tensor([int(p) % T for p in positions],
                                   device=h.device, dtype=torch.long)
        hp = h.index_select(1, pos_idx.to(h.device))[0]        # (P, d)
        logits = wrapper.logit_lens(hp, apply_norm=apply_norm).to(torch.float32)
        logp = torch.log_softmax(logits, dim=-1)
        p = logp.exp()

        scalars[l, :, LENS_SCALARS.index("entropy")] = \
            _entropy_torch(logp, p).cpu().numpy()
        V = logits.shape[-1]
        scalars[l, :, LENS_SCALARS.index("entropy_normalised")] = \
            (_entropy_torch(logp, p) / float(np.log(V))).cpu().numpy()
        scalars[l, :, LENS_SCALARS.index("logit_max")] = \
            logits.max(dim=-1).values.cpu().numpy()
        scalars[l, :, LENS_SCALARS.index("logit_mean")] = \
            logits.mean(dim=-1).cpu().numpy()
        scalars[l, :, LENS_SCALARS.index("logit_std")] = \
            logits.std(dim=-1).cpu().numpy()

        k = min(top_k, V)
        tk = torch.topk(p, k=k, dim=-1)
        topk_ids[l, :, :k] = tk.indices.cpu().numpy().astype(np.int32)
        topk_probs[l, :, :k] = tk.values.cpu().numpy().astype(np.float32)
        scalars[l, :, LENS_SCALARS.index("top1_prob")] = topk_probs[l, :, 0]
        if k > 1:
            scalars[l, :, LENS_SCALARS.index("top1_minus_top2")] = \
                topk_probs[l, :, 0] - topk_probs[l, :, 1]
        scalars[l, :, LENS_SCALARS.index("topk_mass")] = \
            tk.values.sum(dim=-1).cpu().numpy()

        if cand_tensor is not None:
            cand_probs[l] = p.index_select(1, cand_tensor).double().cpu().numpy()

        if correct_token_id is not None and 0 <= correct_token_id < V:
            cp = p[:, correct_token_id]
            correct_prob[l] = cp.double().cpu().numpy()
            # Rank = 1 + number of tokens with strictly greater probability.
            correct_rank[l] = (p > cp.unsqueeze(-1)).sum(dim=-1).double().cpu().numpy() + 1

        if prev_p is not None:
            scalars[l, :, LENS_SCALARS.index("jsd_prev_layer")] = \
                _jsd_torch(logp, p, prev_logp, prev_p).cpu().numpy()
            kl = (prev_p * (prev_logp - logp)).sum(dim=-1)
            scalars[l, :, LENS_SCALARS.index("kl_prev_layer")] = kl.cpu().numpy()
        if first_p is None:
            first_p, first_logp = p.clone(), logp.clone()
        else:
            scalars[l, :, LENS_SCALARS.index("jsd_first_layer")] = \
                _jsd_torch(logp, p, first_logp, first_p).cpu().numpy()

        stored.append((p.to(torch.float16), logp.to(torch.float16)))
        if return_full_probs:
            full_probs.append(p.to(torch.float16).cpu().numpy())

        prev_p, prev_logp = p, logp
        del logits

    # Pass 2: distance from the final layer's distribution.
    if stored:
        final_p = stored[-1][0].to(torch.float32)
        final_logp = torch.log(final_p.clamp(min=EPS))
        for l, (pl, _) in enumerate(stored):
            pf = pl.to(torch.float32)
            lf = torch.log(pf.clamp(min=EPS))
            scalars[l, :, LENS_SCALARS.index("jsd_final_layer")] = \
                _jsd_torch(lf, pf, final_logp, final_p).cpu().numpy()
    del stored, prev_p, prev_logp, first_p, first_logp

    out: Dict[str, Any] = {
        "status": "ok",
        "scalars": scalars,
        "scalar_names": list(LENS_SCALARS),
        "topk_ids": topk_ids,
        "topk_probs": topk_probs,
        "positions": np.array(list(positions), dtype=np.int32),
        "n_depth": n_depth,
        "top_k": top_k,
        "apply_norm": bool(apply_norm),
        "transform": wrapper.arch.logit_lens_transform if apply_norm
                     else "lm_head(h_l)  [no-norm control]",
    }
    if cand_probs is not None:
        out["candidate_probs"] = cand_probs
        out["candidate_token_ids"] = np.array(list(answer_token_ids), dtype=np.int64)
    if correct_token_id is not None:
        out["correct_prob"] = correct_prob
        out["correct_rank"] = correct_rank
        out["correct_token_id"] = int(correct_token_id)
    if return_full_probs:
        out["full_probs"] = np.stack(full_probs, axis=0)

    if no_norm_control_layers:
        out["no_norm_control"] = _no_norm_control(
            wrapper, residual_by_layer, positions, no_norm_control_layers,
            correct_token_id)
    return out


def _no_norm_control(wrapper: Any, residual_by_layer: Sequence[Any],
                     positions: Sequence[int], layers: Sequence[int],
                     correct_token_id: Optional[int]) -> Dict[str, Any]:
    """Entropy/confidence without the final norm, at selected layers.

    If the with-norm and without-norm profiles have the same shape, the
    apparent transition is not an artefact of the normalisation. If they
    differ sharply, the normalisation is doing the work -- a result that
    counts as evidence *against* a representational interpretation.
    """
    import torch

    sel = [l for l in layers if 0 <= l < len(residual_by_layer)
           and residual_by_layer[l] is not None]
    if not sel:
        return {"status": "no_layers"}
    ent = np.full((len(sel), len(positions)), np.nan, dtype=np.float32)
    top1 = np.full((len(sel), len(positions)), np.nan, dtype=np.float32)
    cprob = np.full((len(sel), len(positions)), np.nan, dtype=np.float64)
    for i, l in enumerate(sel):
        h = residual_by_layer[l]
        T = h.shape[1]
        idx = torch.tensor([int(p) % T for p in positions], device=h.device,
                           dtype=torch.long)
        hp = h.index_select(1, idx)[0]
        logits = wrapper.logit_lens(hp, apply_norm=False).to(torch.float32)
        logp = torch.log_softmax(logits, dim=-1)
        p = logp.exp()
        ent[i] = _entropy_torch(logp, p).cpu().numpy()
        top1[i] = p.max(dim=-1).values.cpu().numpy()
        if correct_token_id is not None and correct_token_id < logits.shape[-1]:
            cprob[i] = p[:, correct_token_id].double().cpu().numpy()
        del logits, logp, p
    return {"status": "ok", "layers": np.array(sel, dtype=np.int32),
            "entropy": ent, "top1_prob": top1, "correct_prob": cprob}


# ---------------------------------------------------------------------------
# Per-position extraction of layer profiles
# ---------------------------------------------------------------------------
def scalar_profile(lens: Dict[str, Any], name: str, position_index: int
                   ) -> np.ndarray:
    """``(n_depth,)`` profile of one scalar at one token position."""
    i = LENS_SCALARS.index(name)
    return lens["scalars"][:, position_index, i].astype(np.float64)


def all_profiles_at(lens: Dict[str, Any], position_index: int
                    ) -> Dict[str, np.ndarray]:
    return {name: scalar_profile(lens, name, position_index)
            for name in LENS_SCALARS}


def shuffled_unembedding_null(wrapper: Any, residual_by_layer: Sequence[Any],
                              positions: Sequence[int], seed: int = 0
                              ) -> Dict[str, Any]:
    """Null model: permute the unembedding rows and recompute entropy.

    Row permutation preserves the unembedding's norm distribution and its
    overall geometry while destroying the token-identity mapping. Any
    layer-wise entropy structure that survives is attributable to the
    *geometry* of the readout rather than to what the model has computed
    about specific tokens (protocol section 34).
    """
    import torch

    head = wrapper.model.get_output_embeddings()
    if head is None:
        return {"status": "no_lm_head"}
    original = head.weight.data
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    perm = torch.randperm(original.shape[0], generator=gen).to(original.device)
    ent = np.full((len(residual_by_layer), len(positions)), np.nan, dtype=np.float32)
    try:
        head.weight.data = original[perm]
        for l, h in enumerate(residual_by_layer):
            if h is None:
                continue
            T = h.shape[1]
            idx = torch.tensor([int(p) % T for p in positions],
                               device=h.device, dtype=torch.long)
            hp = h.index_select(1, idx)[0]
            logits = wrapper.logit_lens(hp, apply_norm=True).to(torch.float32)
            logp = torch.log_softmax(logits, dim=-1)
            ent[l] = _entropy_torch(logp, logp.exp()).cpu().numpy()
            del logits, logp
    finally:
        # Restoring the real weights is not optional: leaving a permuted head
        # in place would corrupt every subsequent sample in the run.
        head.weight.data = original
    return {"status": "ok", "entropy": ent, "seed": seed,
            "note": "unembedding rows permuted; token identity destroyed, "
                    "readout geometry preserved"}
