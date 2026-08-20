"""Residual-stream write hooks shared by the sensitivity and causal phases.

Both J-space (section 23) and causal intervention (section 31) need the same
primitive: replace the residual stream at layer *l*, token position *t*, for a
chosen subset of batch rows, during an otherwise-normal forward pass.

Doing this with hooks rather than by calling block modules directly matters
for correctness. A decoder block needs its attention mask, rotary position
embeddings and cache state, and those calling conventions change between
transformers releases; running the real forward pass and editing the tensor
in flight is both version-robust and guaranteed to be the computation the
model actually performs.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class PerturbationSpec:
    """One edit to apply to the residual stream.

    ``batch_rows`` restricts the edit to specific sequences in the batch,
    which is what lets many independent perturbations share a single forward
    pass. ``token_positions`` are absolute indices into the sequence.
    """

    layer: int
    delta: Any                       # torch tensor, broadcastable to (d,) or (P, d)
    token_positions: Sequence[int]
    batch_rows: Optional[Sequence[int]] = None
    mode: str = "add"                # "add" | "replace"


def _apply_to_tensor(t: Any, spec: PerturbationSpec) -> Any:
    import torch

    rows = (list(spec.batch_rows) if spec.batch_rows is not None
            else list(range(t.shape[0])))
    if not rows:
        return t
    positions = [int(p) for p in spec.token_positions if -t.shape[1] <= int(p) < t.shape[1]]
    if not positions:
        return t

    delta = spec.delta.to(t.device).to(t.dtype)
    row_idx = torch.tensor(rows, device=t.device, dtype=torch.long)
    pos_idx = torch.tensor(positions, device=t.device, dtype=torch.long)

    # Clone before writing: the incoming tensor may be a view that other
    # modules still reference, and in-place edits there corrupt the pass.
    out = t.clone()
    grid_r = row_idx.view(-1, 1).expand(len(rows), len(positions))
    grid_p = pos_idx.view(1, -1).expand(len(rows), len(positions))
    if delta.dim() == 1:
        payload = delta.view(1, 1, -1).expand(len(rows), len(positions), -1)
    elif delta.dim() == 2:
        payload = delta.unsqueeze(0).expand(len(rows), -1, -1)
    else:
        payload = delta
    if spec.mode == "add":
        out[grid_r, grid_p] = out[grid_r, grid_p] + payload
    else:
        out[grid_r, grid_p] = payload
    return out


@contextmanager
def residual_edits(wrapper: Any, specs: Sequence[PerturbationSpec],
                   capture_layers: Optional[Sequence[int]] = None,
                   capture_positions: Optional[Sequence[int]] = None
                   ) -> Iterator[Dict[str, Any]]:
    """Install edit + capture hooks for the duration of the block.

    Yields a dict that is populated during the forward pass with
    ``captured[layer] -> tensor`` for the requested layers and positions.
    Hooks are always removed, including on exception, because a leaked hook
    silently corrupts every later forward pass in the session.
    """
    import torch

    by_layer: Dict[int, List[PerturbationSpec]] = {}
    for spec in specs:
        by_layer.setdefault(int(spec.layer), []).append(spec)

    captured: Dict[int, Any] = {}
    want_capture = set(int(l) for l in (capture_layers or []))
    handles: List[Any] = []

    def make_hook(layer_index: int):
        layer_specs = by_layer.get(layer_index, [])

        def hook(_module, _inputs, output):
            is_tuple = isinstance(output, tuple)
            tensor = output[0] if is_tuple else output
            for spec in layer_specs:
                tensor = _apply_to_tensor(tensor, spec)
            if layer_index in want_capture:
                if capture_positions is not None:
                    # index_select rejects negative indices, so -1 ("last real
                    # token") is resolved against this tensor's length here.
                    T = tensor.shape[1]
                    idx = torch.tensor([int(p) % T for p in capture_positions],
                                       device=tensor.device, dtype=torch.long)
                    captured[layer_index] = tensor.index_select(1, idx).detach()
                else:
                    captured[layer_index] = tensor.detach()
            if not layer_specs:
                return output
            return (tensor,) + tuple(output[1:]) if is_tuple else tensor

        return hook

    try:
        touched = sorted(set(by_layer.keys()) | want_capture)
        for layer_index in touched:
            module = wrapper.layer_module(layer_index)
            handles.append(module.register_forward_hook(make_hook(layer_index)))
        yield captured
    finally:
        for h in handles:
            h.remove()


def random_directions(n: int, dim: int, seed: int, *, device: Any = None,
                      dtype: Any = None) -> Any:
    """Unit-norm isotropic probe directions, reproducible from ``seed``.

    Unit norm is what makes amplification ratios comparable across layers:
    otherwise the ratio would mix the probe's own scale with the model's
    response to it.
    """
    import torch

    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    v = torch.randn(n, dim, generator=gen, dtype=torch.float32)
    v = v / v.norm(dim=1, keepdim=True).clamp(min=1e-12)
    if device is not None:
        v = v.to(device)
    if dtype is not None:
        v = v.to(dtype)
    return v


def orthogonal_component(v: Any, reference: Any) -> Any:
    """Remove the component of ``v`` along ``reference``, renormalising."""
    import torch
    r = reference / reference.norm().clamp(min=1e-12)
    proj = (v @ r) if v.dim() == 1 else (v @ r).unsqueeze(-1)
    out = v - proj * r
    norm = out.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return out / norm
