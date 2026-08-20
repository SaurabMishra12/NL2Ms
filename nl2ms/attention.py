"""Attention analysis: per-head summaries and layer-to-layer restructuring.

Full attention tensors are ``L x H x T x T`` and cannot be stored for a whole
dataset: at 28 layers, 28 heads and 512 tokens that is ~11 GB per sample in
fp16. The strategy is therefore to reduce on the GPU to a fixed set of
per-``(layer, head, query-position)`` statistics, and to keep full matrices
only for a handful of named samples so that the summaries can be audited
against the thing they summarise.

An important caveat carried through this module: attention weights are not
attribution. A head attending strongly to a token does not establish that the
token's content drove the output. Restructuring metrics here describe *what
the attention pattern does*, not what it means.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .signals import EPS, derivative

# The fixed statistic set. Order matters: it defines the last axis of the
# summary tensor and is recorded in the shard metadata.
ATTENTION_STATISTICS = [
    "entropy",              # H over the attended distribution (nats)
    "entropy_normalised",   # H / log(n_visible)
    "max_attention",        # top-1 weight
    "top1_minus_top2",
    "concentration_gini",
    "effective_support",    # exp(H): how many positions are effectively used
    "sparsity_frac_below",  # fraction of positions below 1/n_visible
    "mean_distance",        # E[query_pos - key_pos] in tokens
    "distance_std",
    "attention_to_prompt",  # mass on prompt positions
    "attention_to_generated",
    "attention_to_self",    # weight on the query position itself
]


def summarise_attention(attn: Any, *, prompt_length: int,
                        query_positions: Sequence[int],
                        valid_mask: Optional[Any] = None) -> np.ndarray:
    """Reduce one layer's attention to ``(n_heads, n_positions, n_statistics)``.

    ``attn``: torch tensor ``(n_heads, T, T)`` for a single sequence, already
    softmax-normalised over the key axis.

    Reduction happens in torch (on device) before anything reaches NumPy,
    because moving the full ``H x T x T`` tensor to CPU per layer is what makes
    naive attention analysis unusable at scale.
    """
    import torch

    A = attn
    if A.dim() == 4:  # (batch, heads, T, T) -- take the single sequence
        A = A[0]
    n_heads, T, _ = A.shape
    qpos = [int(p) for p in query_positions if 0 <= int(p) < T]
    if not qpos:
        return np.zeros((n_heads, 0, len(ATTENTION_STATISTICS)), dtype=np.float32)

    idx = torch.tensor(qpos, device=A.device, dtype=torch.long)
    P = A.index_select(1, idx).to(torch.float32)   # (heads, n_pos, T)

    # Causal masking means only keys <= query are visible; positions beyond
    # carry exact zeros. Counting them as "low attention" would inflate every
    # sparsity and entropy-normalisation figure, so visibility is explicit.
    key_index = torch.arange(T, device=A.device).view(1, 1, T)
    q_index = idx.view(1, -1, 1)
    visible = (key_index <= q_index)
    if valid_mask is not None:
        vm = valid_mask.to(A.device).view(1, 1, T).bool()
        visible = visible & vm
    n_visible = visible.sum(dim=-1).clamp(min=1).to(torch.float32)  # (1, n_pos)

    P = P * visible.to(P.dtype)
    P = P / P.sum(dim=-1, keepdim=True).clamp(min=EPS)

    logP = torch.log(P.clamp(min=EPS))
    entropy = -(P * logP).sum(dim=-1)                                # (heads, n_pos)
    entropy_norm = entropy / torch.log(n_visible.clamp(min=2.0))

    top2 = torch.topk(P, k=min(2, T), dim=-1).values
    max_attn = top2[..., 0]
    second = top2[..., 1] if top2.shape[-1] > 1 else torch.zeros_like(max_attn)

    effective_support = torch.exp(entropy)

    uniform = (1.0 / n_visible).unsqueeze(-1)          # (1, n_pos, 1)
    sparsity = ((P < uniform) & visible).sum(dim=-1).to(torch.float32) / n_visible

    # Gini restricted to the visible support. The masked entries sort to the
    # front as exact zeros; the closed-form correction below removes their
    # contribution so the result equals the Gini of the m visible weights and
    # is therefore comparable across query positions with different m.
    sortedP, _ = torch.sort(P, dim=-1)
    ranks = torch.arange(1, T + 1, device=A.device, dtype=torch.float32).view(1, 1, T)
    raw_gini = ((2 * ranks - float(T) - 1) * sortedP).sum(dim=-1)
    m = n_visible                                       # (1, n_pos)
    gini = (raw_gini - (float(T) - m)) / m.clamp(min=1.0)

    distances = (q_index.to(torch.float32) - key_index.to(torch.float32))
    mean_dist = (P * distances).sum(dim=-1)
    var_dist = (P * (distances - mean_dist.unsqueeze(-1)) ** 2).sum(dim=-1)
    std_dist = torch.sqrt(var_dist.clamp(min=0.0))

    prompt_sel = (key_index < prompt_length) & visible
    gen_sel = (key_index >= prompt_length) & visible
    to_prompt = (P * prompt_sel.to(P.dtype)).sum(dim=-1)
    to_generated = (P * gen_sel.to(P.dtype)).sum(dim=-1)
    self_sel = (key_index == q_index)
    to_self = (P * self_sel.to(P.dtype)).sum(dim=-1)

    stacked = torch.stack([
        entropy, entropy_norm, max_attn, max_attn - second, gini,
        effective_support, sparsity, mean_dist, std_dist,
        to_prompt, to_generated, to_self,
    ], dim=-1)
    return stacked.detach().cpu().numpy().astype(np.float32)


def summarise_all_layers(attentions: Sequence[Any], *, prompt_length: int,
                         query_positions: Sequence[int],
                         valid_mask: Optional[Any] = None) -> np.ndarray:
    """Stack per-layer summaries: ``(n_layers, n_heads, n_positions, n_stats)``."""
    out = [summarise_attention(a, prompt_length=prompt_length,
                               query_positions=query_positions,
                               valid_mask=valid_mask)
           for a in attentions]
    if not out:
        return np.zeros((0, 0, 0, len(ATTENTION_STATISTICS)), dtype=np.float32)
    return np.stack(out, axis=0)


def statistic_index(name: str) -> int:
    return ATTENTION_STATISTICS.index(name)


def extract_statistic(summary: np.ndarray, name: str) -> np.ndarray:
    """``(n_layers, n_heads, n_positions)`` slice for one named statistic."""
    return summary[..., statistic_index(name)]


# ---------------------------------------------------------------------------
# Layer-to-layer restructuring (protocol section 17)
# ---------------------------------------------------------------------------
def attention_restructuring(attentions: Sequence[Any], *,
                            query_positions: Sequence[int],
                            valid_mask: Optional[Any] = None) -> Dict[str, np.ndarray]:
    """Change in attention pattern between consecutive layers.

    ``||A_l - A_{l-1}||`` and cosine similarity are computed head-wise on the
    attention rows at the query positions. Heads are matched by index across
    layers, which is a convention, not a claim: head *h* in layer *l* has no
    privileged correspondence to head *h* in layer *l+1*. The head-index
    matching is why these numbers are reported as a population summary and why
    per-head results below are labelled as candidates only.
    """
    import torch

    if len(attentions) < 2:
        n = len(attentions)
        return {"frobenius_delta": np.zeros(n), "cosine_similarity": np.ones(n),
                "jsd_delta": np.zeros(n), "per_head_frobenius": np.zeros((n, 0))}

    rows: List[Any] = []
    for A in attentions:
        a = A[0] if A.dim() == 4 else A
        T = a.shape[-1]
        qpos = [int(p) for p in query_positions if 0 <= int(p) < T]
        idx = torch.tensor(qpos, device=a.device, dtype=torch.long)
        r = a.index_select(1, idx).to(torch.float32)      # (heads, n_pos, T)
        if valid_mask is not None:
            r = r * valid_mask.to(a.device).view(1, 1, -1).to(r.dtype)
        r = r / r.sum(dim=-1, keepdim=True).clamp(min=EPS)
        rows.append(r)

    n_layers = len(rows)
    n_heads = rows[0].shape[0]
    frob = np.zeros(n_layers, dtype=np.float64)
    cosim = np.ones(n_layers, dtype=np.float64)
    jsd = np.zeros(n_layers, dtype=np.float64)
    per_head = np.zeros((n_layers, n_heads), dtype=np.float64)
    per_head_cos = np.ones((n_layers, n_heads), dtype=np.float64)

    for l in range(1, n_layers):
        prev, cur = rows[l - 1], rows[l]
        if prev.shape != cur.shape:
            frob[l] = np.nan
            cosim[l] = np.nan
            continue
        diff = (cur - prev).flatten(start_dim=1)          # (heads, n_pos*T)
        frob[l] = float(torch.linalg.vector_norm(diff).item())
        per_head[l] = torch.linalg.vector_norm(diff, dim=1).cpu().numpy()

        a = prev.flatten(start_dim=1)
        b = cur.flatten(start_dim=1)
        head_cos = torch.nn.functional.cosine_similarity(a, b, dim=1)
        per_head_cos[l] = head_cos.cpu().numpy()
        cosim[l] = float(head_cos.mean().item())

        # JSD between the attention rows treated as distributions -- bounded,
        # so a near-disjoint pattern change does not blow up.
        m = 0.5 * (prev + cur)
        kl_pm = (prev * (torch.log(prev.clamp(min=EPS)) -
                         torch.log(m.clamp(min=EPS)))).sum(dim=-1)
        kl_qm = (cur * (torch.log(cur.clamp(min=EPS)) -
                        torch.log(m.clamp(min=EPS)))).sum(dim=-1)
        jsd[l] = float((0.5 * kl_pm + 0.5 * kl_qm).mean().item())

    return {
        "frobenius_delta": frob,
        "cosine_similarity": cosim,
        "jsd_delta": jsd,
        "per_head_frobenius": per_head,
        "per_head_cosine": per_head_cos,
        "restructuring_delta": derivative(frob),
    }


def abrupt_heads(per_head_frobenius: np.ndarray, *, z_threshold: float = 2.5
                 ) -> List[Dict[str, Any]]:
    """Heads whose layer-to-layer change is an outlier within their layer.

    Deliberately *not* called "reasoning heads": this identifies statistical
    outliers in a pattern-change metric. Establishing a functional role would
    require ablation evidence, which this function does not provide.
    """
    F = np.asarray(per_head_frobenius, dtype=np.float64)
    if F.ndim != 2 or F.shape[0] < 2:
        return []
    out: List[Dict[str, Any]] = []
    for l in range(1, F.shape[0]):
        row = F[l]
        finite = np.isfinite(row)
        if finite.sum() < 3:
            continue
        mu, sd = np.mean(row[finite]), np.std(row[finite])
        if sd < EPS:
            continue
        z = (row - mu) / sd
        for h in np.where(z > z_threshold)[0]:
            out.append({
                "layer": int(l), "head": int(h),
                "frobenius_delta": float(row[h]), "z_score": float(z[h]),
                "note": "outlier in layer-to-layer attention change; "
                        "functional role not established",
            })
    out.sort(key=lambda d: -d["z_score"])
    return out


def head_similarity_matrix(attentions: Sequence[Any], layer: int,
                           query_positions: Sequence[int]) -> np.ndarray:
    """Head-to-head cosine similarity within one layer."""
    import torch
    A = attentions[layer]
    a = A[0] if A.dim() == 4 else A
    T = a.shape[-1]
    qpos = [int(p) for p in query_positions if 0 <= int(p) < T]
    idx = torch.tensor(qpos, device=a.device, dtype=torch.long)
    r = a.index_select(1, idx).to(torch.float32).flatten(start_dim=1)
    r = torch.nn.functional.normalize(r, dim=1)
    return (r @ r.T).cpu().numpy()


# ---------------------------------------------------------------------------
# Per-sample aggregation
# ---------------------------------------------------------------------------
def aggregate_summary(summary: np.ndarray) -> Dict[str, np.ndarray]:
    """Collapse ``(L, H, P, S)`` to per-layer profiles for the main analysis.

    Head-mean and head-max are both kept: a single restructuring head is
    invisible in the mean, and a diffuse shift is invisible in the max.
    """
    if summary.size == 0:
        return {}
    out: Dict[str, np.ndarray] = {}
    for i, name in enumerate(ATTENTION_STATISTICS):
        stat = summary[..., i]                     # (L, H, P)
        with np.errstate(invalid="ignore"):
            out[f"attn_{name}_mean"] = np.nanmean(stat, axis=(1, 2))
            out[f"attn_{name}_max"] = np.nanmax(stat, axis=(1, 2))
            out[f"attn_{name}_head_std"] = np.nanstd(np.nanmean(stat, axis=2), axis=1)
    return out


def save_full_attention(attentions: Sequence[Any], path: str,
                        sample_id: str, *, max_tokens: int = 256) -> Dict[str, Any]:
    """Persist full ``L x H x T x T`` matrices for one audited sample.

    Truncated to the last ``max_tokens`` positions if the sequence is long,
    which keeps the file bounded while retaining the generated region where
    the analysis actually looks.
    """
    import torch
    from .storage import save_npz, file_checksum

    if not attentions:
        return {"status": "no_attention"}
    stacked = []
    for A in attentions:
        a = A[0] if A.dim() == 4 else A
        T = a.shape[-1]
        if T > max_tokens:
            a = a[:, -max_tokens:, -max_tokens:]
        stacked.append(a.to(torch.float16).cpu().numpy())
    arr = np.stack(stacked, axis=0)
    save_npz(path, {"attention": arr,
                    "sample_id": np.array([sample_id]),
                    "truncated": np.array([arr.shape[-1] < attentions[0].shape[-1]])})
    return {
        "status": "ok", "path": str(path), "shape": list(arr.shape),
        "dtype": str(arr.dtype), "checksum": file_checksum(path),
        "n_layers": arr.shape[0], "n_heads": arr.shape[1],
        "n_tokens": arr.shape[-1],
    }
