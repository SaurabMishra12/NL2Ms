"""Phases 2, 3, 5, 6, 8, 9, 10 -- the per-sample measurement pipeline.

One sample at a time:

1. Generate an answer (Phase 2), save it immediately.
2. Re-run the *full* sequence (prompt + generated) in a single forward pass,
   capturing the residual stream at every layer and optionally attention
   (Phases 3, 6).
3. Project every layer to vocabulary space and reduce to scalars (Phase 4).
4. Derive entropy, JSD, trajectory and symmetry-breaking profiles
   (Phases 5, 8, 9, 10).
5. Persist: hidden states to shards, metrics to a per-sample ``.npz``,
   summaries to the manifest.

Why re-run instead of capturing during generation: incremental decoding uses
a KV cache and produces one position per step, so the residual stream for the
prompt is only available at step 0 and attention rows are ragged. A single
teacher-forced pass over the completed sequence gives every layer at every
position under identical conditions, which is what the layer-wise comparisons
require. The cost is one extra forward pass per sample.

Token positions analysed (protocol section 36):

* ``last_input``   -- the final prompt token, which predicts the first
  generated token. This is where "the answer is decided" for single-token
  answers.
* ``generated_k``  -- each generated position, capped by configuration.
* ``answer``       -- the position whose next-token prediction *is* the
  answer token, located by search rather than assumed.
* ``final``        -- the last generated position.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import attention as attn_mod
from . import logit_lens as lens_mod
from . import signals as sig
from .datasets_build import (ANSWER_CLOSED_SET, ANSWER_OPEN_VOCAB,
                             ANSWER_UNDEFINED, Sample)
from .modeling import (AnswerSpec, GenerationResult, build_answer_spec,
                       finish_reason, parse_prediction, render_prompt,
                       score_prediction, trim_generated)
from .storage import save_npz


# ---------------------------------------------------------------------------
# Token position selection
# ---------------------------------------------------------------------------
@dataclass
class PositionPlan:
    """Which sequence positions get analysed, and what each one means."""

    positions: List[int]
    roles: List[str]
    last_input_index: int              # index into ``positions``
    answer_index: Optional[int]
    final_index: int
    answer_position_source: str
    prompt_length: int
    total_length: int

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


def plan_positions(prompt_length: int, generated_ids: Sequence[int],
                   answer_spec: AnswerSpec, *, modes: Sequence[str],
                   max_generated: int,
                   explicit_positions: Optional[Sequence[int]] = None
                   ) -> PositionPlan:
    """Choose analysis positions and locate the answer-deciding position.

    The answer position is *found*, not assumed: we look for the first
    generated token that is one of the candidate first tokens, and take the
    position immediately before it (that position's next-token distribution is
    the one that selects the answer). If no candidate token appears in the
    generation, we fall back to the last prompt position and say so in
    ``answer_position_source`` -- a downstream analysis can then exclude these
    rather than silently treating an arbitrary token as the answer.
    """
    total = prompt_length + len(generated_ids)
    positions: List[int] = []
    roles: List[str] = []

    last_input = prompt_length - 1
    positions.append(last_input)
    roles.append("last_input")

    gen_positions: List[int] = []
    if "C" in modes and len(generated_ids) > 0:
        n = min(len(generated_ids), max_generated)
        # Keep a contiguous prefix: the interesting dynamics are at answer
        # formation, which is early in the generation for these tasks.
        gen_positions = [prompt_length + i for i in range(n)]
    elif "A" in modes and len(generated_ids) > 0:
        gen_positions = [total - 1]
    for p in gen_positions:
        if p not in positions:
            positions.append(p)
            roles.append("generated")

    if "E" in modes and explicit_positions:
        for p in explicit_positions:
            q = int(p) % total
            if q not in positions:
                positions.append(q)
                roles.append("explicit")

    # Locate the answer-deciding position.
    answer_pos: Optional[int] = None
    source = "not_found"
    if answer_spec.usable and answer_spec.first_token_ids:
        cand = set(int(t) for t in answer_spec.first_token_ids)
        for i, tid in enumerate(generated_ids):
            if int(tid) in cand:
                answer_pos = prompt_length + i - 1
                source = "found_candidate_token_in_generation"
                break
    if answer_pos is None:
        answer_pos = last_input
        source = ("fallback_last_input" if answer_spec.usable
                  else "no_usable_answer_spec")
    if answer_pos not in positions:
        positions.append(answer_pos)
        roles.append("answer")
    answer_index = positions.index(answer_pos)

    final_pos = total - 1
    if final_pos not in positions:
        positions.append(final_pos)
        roles.append("final")
    final_index = positions.index(final_pos)

    order = np.argsort(positions)
    positions = [positions[i] for i in order]
    roles = [roles[i] for i in order]
    return PositionPlan(
        positions=positions, roles=roles,
        last_input_index=positions.index(last_input),
        answer_index=answer_index if answer_pos in positions else None,
        final_index=positions.index(final_pos),
        answer_position_source=source,
        prompt_length=prompt_length, total_length=total,
    )


# ---------------------------------------------------------------------------
# Generation (Phase 2)
# ---------------------------------------------------------------------------
def generate_batch(wrapper: Any, samples: Sequence[Sample], cfg: Any
                   ) -> List[GenerationResult]:
    """Greedy (or sampled) generation for a batch, with parsing and scoring."""
    gen_cfg = cfg.generation
    prompts = [render_prompt(s, cfg.prompt_template_id, wrapper.tokenizer)
               for s in samples]
    out = wrapper.generate(prompts, max_new_tokens=gen_cfg.max_new_tokens,
                           temperature=gen_cfg.temperature,
                           top_p=gen_cfg.top_p, seed=gen_cfg.seed)
    results: List[GenerationResult] = []
    per_sample_time = out["runtime_seconds"] / max(1, len(samples))
    for i, sample in enumerate(samples):
        raw_gen = out["generated_ids"][i].tolist()
        trimmed = trim_generated(raw_gen, wrapper.tokenizer)
        decoded = wrapper.tokenizer.decode(trimmed, skip_special_tokens=True)
        prediction, parse_status = parse_prediction(sample, decoded)
        correct = score_prediction(sample, prediction)
        # The prompt is left-padded in the batch; strip pads so the recorded
        # input_ids match what a single-sequence rerun would produce.
        ids = out["input_ids"][i].tolist()
        mask = out["attention_mask"][i].tolist()
        real_ids = [t for t, m in zip(ids, mask) if m == 1]
        results.append(GenerationResult(
            sample_id=sample.sample_id,
            prompt=prompts[i],
            input_ids=real_ids,
            generated_ids=trimmed,
            decoded_output=decoded,
            prediction=prediction,
            ground_truth=sample.ground_truth,
            correct=correct,
            generation_length=len(trimmed),
            prompt_length=len(real_ids),
            finish_reason=finish_reason(raw_gen, wrapper.tokenizer,
                                        gen_cfg.max_new_tokens),
            model_name=wrapper.model_cfg.name,
            seed=gen_cfg.seed,
            temperature=gen_cfg.temperature,
            runtime_seconds=per_sample_time,
            parse_status=parse_status,
        ))
    return results


# ---------------------------------------------------------------------------
# Full-sequence analysis (Phases 3-10)
# ---------------------------------------------------------------------------
@dataclass
class SampleAnalysis:
    sample_id: str
    plan: PositionPlan
    profiles: Dict[str, np.ndarray]            # per-depth profiles at answer pos
    per_position: Dict[str, np.ndarray]        # (n_depth, n_positions) arrays
    hidden: Optional[np.ndarray]               # (n_depth, n_saved_pos, d) fp16
    hidden_positions: List[int]
    attention_summary: Optional[np.ndarray]
    attention_restructuring: Optional[Dict[str, np.ndarray]]
    answer_spec: AnswerSpec
    diagnostics: Dict[str, Any]
    runtime_seconds: float


def analyse_sample(wrapper: Any, sample: Sample, generation: GenerationResult,
                   cfg: Any, *, save_full_attention_path: Optional[Path] = None
                   ) -> SampleAnalysis:
    """Run the full measurement suite for a single sample."""
    import torch

    start = time.time()
    ext = cfg.extraction
    flags = cfg.effective_flags()
    tok = wrapper.tokenizer

    input_ids = generation.input_ids
    gen_ids = generation.generated_ids
    full_ids = list(input_ids) + list(gen_ids)
    if len(full_ids) < 2:
        raise ValueError(f"sequence too short to analyse: {len(full_ids)} tokens")

    ids = torch.tensor([full_ids], dtype=torch.long, device=wrapper.device)
    mask = torch.ones_like(ids)

    answer_spec = build_answer_spec(sample, tok)
    plan = plan_positions(len(input_ids), gen_ids, answer_spec,
                          modes=ext.modes, max_generated=ext.max_generated_positions,
                          explicit_positions=ext.explicit_positions)

    need_attention = bool(flags["save_attention_summaries"] or
                          save_full_attention_path is not None)
    captured = wrapper.capture_residual_stream(ids, mask,
                                               need_attention=need_attention)

    # Depth axis: index 0 = embedding output, index l = after block l-1.
    residual: List[Any] = [captured.get("embed")]
    residual.extend(captured["resid_post"])
    n_depth = len(residual)

    correct_token_id = None
    cand_ids: Optional[List[int]] = None
    if answer_spec.usable and answer_spec.first_token_ids:
        cand_ids = list(answer_spec.first_token_ids)
        if answer_spec.correct_index is not None:
            correct_token_id = int(cand_ids[answer_spec.correct_index])

    no_norm_layers = sorted({0, n_depth // 4, n_depth // 2,
                             3 * n_depth // 4, n_depth - 1})
    lens = lens_mod.run_logit_lens(
        wrapper, residual, positions=plan.positions, top_k=ext.logit_lens_top_k,
        answer_token_ids=cand_ids, correct_token_id=correct_token_id,
        apply_norm=True, no_norm_control_layers=no_norm_layers,
        return_full_probs=flags["save_full_vocab_logits"],
    )

    # ---- profiles at the answer-deciding position -------------------
    ai = plan.answer_index if plan.answer_index is not None else plan.last_input_index
    profiles: Dict[str, np.ndarray] = {}
    for name in lens_mod.LENS_SCALARS:
        profiles[name] = lens_mod.scalar_profile(lens, name, ai)

    # Entropy derivatives (Phase 5).
    H = profiles["entropy"]
    profiles["entropy_delta"] = sig.derivative(H)
    profiles["entropy_curvature"] = sig.curvature_1d(H)

    # ---- order parameter / symmetry breaking (Phases 10, 13, 24) ----
    order_metrics: Dict[str, np.ndarray] = {}
    order_status = "unavailable"
    if answer_spec.spec_type == ANSWER_CLOSED_SET and answer_spec.usable \
            and "candidate_probs" in lens:
        cp = lens["candidate_probs"][:, ai, :]
        order_metrics = sig.order_parameter_closed_set(cp, answer_spec.correct_index)
        order_metrics["pairwise_candidate_distance"] = \
            sig.pairwise_candidate_distances(cp)
        order_status = "closed_set"
    elif answer_spec.spec_type == ANSWER_OPEN_VOCAB and correct_token_id is not None \
            and "correct_prob" in lens:
        # Open-vocabulary margin from the stored top-k: the best competitor is
        # the top-1 token unless that token *is* the correct one, in which case
        # it is the top-2. This is exact without keeping the full vocabulary.
        cprob = lens["correct_prob"][:, ai]
        top1 = lens["topk_probs"][:, ai, 0].astype(np.float64)
        top1_id = lens["topk_ids"][:, ai, 0]
        top2 = lens["topk_probs"][:, ai, 1].astype(np.float64) \
            if lens["topk_probs"].shape[-1] > 1 else np.zeros_like(top1)
        best_other = np.where(top1_id == correct_token_id, top2, top1)
        margin = cprob - best_other
        with np.errstate(divide="ignore", invalid="ignore"):
            log_odds = np.log(np.maximum(cprob, sig.EPS)) - \
                np.log(np.maximum(1.0 - cprob, sig.EPS))
        order_metrics = {
            "margin": margin, "correct_prob": cprob,
            "best_wrong_prob": best_other, "log_odds": log_odds,
            "rank": lens["correct_rank"][:, ai],
            "confidence": top1,
            "margin_delta": sig.derivative(margin),
            "margin_curvature": sig.curvature_1d(margin),
        }
        order_status = "open_vocab"
    for key, value in order_metrics.items():
        profiles[f"order_{key}"] = np.asarray(value, dtype=np.float64)

    # ---- trajectory dynamics (Phase 9) -------------------------------
    Hmat = np.stack([
        residual[l][0, plan.positions[ai] % residual[l].shape[1], :]
        .to(torch.float32).cpu().numpy()
        for l in range(n_depth) if residual[l] is not None
    ], axis=0).astype(np.float64)
    traj = sig.trajectory_metrics(Hmat)
    for key, value in traj.items():
        profiles[f"traj_{key}"] = value

    # ---- attention (Phases 6, 17) ------------------------------------
    attn_summary = None
    attn_restruct = None
    full_attention_meta = None
    attentions = captured.get("attentions") or []
    if attentions and flags["save_attention_summaries"]:
        attn_summary = attn_mod.summarise_all_layers(
            attentions, prompt_length=len(input_ids),
            query_positions=plan.positions)
        attn_restruct = attn_mod.attention_restructuring(
            attentions, query_positions=plan.positions)
        for key, value in attn_mod.aggregate_summary(attn_summary).items():
            # Attention has n_layers entries; pad to the n_depth axis (which
            # includes the embedding) so every profile shares one index space.
            profiles[key] = np.concatenate([[np.nan], np.asarray(value, float)])
        for key in ["frobenius_delta", "cosine_similarity", "jsd_delta"]:
            profiles[f"attn_restructuring_{key}"] = np.concatenate(
                [[np.nan], np.asarray(attn_restruct[key], float)])
    if attentions and save_full_attention_path is not None:
        full_attention_meta = attn_mod.save_full_attention(
            attentions, str(save_full_attention_path), sample.sample_id)

    # ---- hidden state persistence ------------------------------------
    hidden_array = None
    hidden_positions: List[int] = []
    if flags["save_hidden_states"]:
        if flags["save_all_token_positions"]:
            keep_idx = list(range(len(plan.positions)))
        else:
            # Only the positions the analysis actually revisits. Storing every
            # generated position for every sample is what turns a 5 GB run into
            # a 50 GB one, and the population-level geometry only needs these.
            keep_idx = sorted({plan.last_input_index, ai, plan.final_index})
        hidden_positions = [plan.positions[i] for i in keep_idx]
        pieces = []
        for l in range(n_depth):
            if residual[l] is None:
                continue
            T = residual[l].shape[1]
            idx = torch.tensor([plan.positions[i] % T for i in keep_idx],
                               device=residual[l].device, dtype=torch.long)
            pieces.append(residual[l].index_select(1, idx)[0]
                          .to(torch.float16).cpu().numpy())
        hidden_array = np.stack(pieces, axis=0)   # (n_depth, n_keep, d)

    per_position: Dict[str, np.ndarray] = {
        "lens_scalars": lens["scalars"],
        "topk_ids": lens["topk_ids"],
        "topk_probs": lens["topk_probs"],
    }
    if "candidate_probs" in lens:
        per_position["candidate_probs"] = lens["candidate_probs"].astype(np.float32)
    if "correct_prob" in lens:
        per_position["correct_prob"] = lens["correct_prob"].astype(np.float32)
        per_position["correct_rank"] = lens["correct_rank"].astype(np.float32)
    if attn_summary is not None:
        per_position["attention_summary"] = attn_summary

    diagnostics = {
        "n_depth": n_depth,
        "n_positions": len(plan.positions),
        "answer_position_source": plan.answer_position_source,
        "order_parameter_status": order_status,
        "answer_spec_usable": answer_spec.usable,
        "answer_spec_warnings": answer_spec.warnings,
        "lens_transform": lens.get("transform"),
        "no_norm_control": _summarise_no_norm(lens.get("no_norm_control"),
                                              profiles["entropy"], ai),
        "full_attention": full_attention_meta,
        "n_nan_in_profiles": int(sum(int(np.sum(~np.isfinite(v)))
                                     for v in profiles.values())),
        "prompt_length": len(input_ids),
        "generation_length": len(gen_ids),
        "sequence_length": len(full_ids),
    }

    del captured, residual, attentions
    return SampleAnalysis(
        sample_id=sample.sample_id, plan=plan, profiles=profiles,
        per_position=per_position, hidden=hidden_array,
        hidden_positions=hidden_positions, attention_summary=attn_summary,
        attention_restructuring=attn_restruct, answer_spec=answer_spec,
        diagnostics=diagnostics, runtime_seconds=time.time() - start,
    )


def _summarise_no_norm(control: Optional[Dict[str, Any]],
                       entropy_with_norm: np.ndarray,
                       position_index: int) -> Dict[str, Any]:
    """Compare with-norm and without-norm entropy profiles at the same layers.

    A high correlation means the final normalisation is not what creates the
    entropy structure; a low one means it largely is.
    """
    if not control or control.get("status") != "ok":
        return {"status": (control or {}).get("status", "missing")}
    layers = control["layers"]
    with_norm = entropy_with_norm[layers]
    without = control["entropy"][:, position_index].astype(np.float64)
    both = np.isfinite(with_norm) & np.isfinite(without)
    corr = float(np.corrcoef(with_norm[both], without[both])[0, 1]) \
        if both.sum() >= 3 else float("nan")
    return {
        "status": "ok",
        "layers": layers.tolist(),
        "entropy_with_norm": with_norm.tolist(),
        "entropy_without_norm": without.tolist(),
        "correlation": corr,
        "mean_abs_difference": float(np.nanmean(np.abs(with_norm - without))),
        "interpretation": ("high correlation => the layer-wise entropy shape "
                           "is not produced by the final normalisation"),
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_analysis(analysis: SampleAnalysis, path: str | Path) -> Path:
    """Write one sample's derived metrics as a self-describing ``.npz``."""
    arrays: Dict[str, np.ndarray] = {}
    for name, value in analysis.profiles.items():
        arrays[f"profiles::{name}"] = np.asarray(value, dtype=np.float32)
    for name, value in analysis.per_position.items():
        arrays[f"perpos::{name}"] = np.asarray(value)
    if analysis.attention_restructuring:
        for name, value in analysis.attention_restructuring.items():
            arrays[f"attn::{name}"] = np.asarray(value, dtype=np.float32)
    arrays["meta::positions"] = np.asarray(analysis.plan.positions, dtype=np.int32)
    arrays["meta::roles"] = np.asarray(analysis.plan.roles)
    arrays["meta::answer_index"] = np.asarray([analysis.plan.answer_index
                                               if analysis.plan.answer_index
                                               is not None else -1], dtype=np.int32)
    arrays["meta::last_input_index"] = np.asarray([analysis.plan.last_input_index],
                                                  dtype=np.int32)
    arrays["meta::final_index"] = np.asarray([analysis.plan.final_index], dtype=np.int32)
    arrays["meta::sample_id"] = np.asarray([analysis.sample_id])
    return save_npz(path, arrays, compressed=True)


def load_analysis(path: str | Path) -> Dict[str, Any]:
    """Read back a saved analysis into ``profiles`` / ``perpos`` / ``meta``."""
    from .storage import load_npz
    raw = load_npz(path)
    out: Dict[str, Any] = {"profiles": {}, "perpos": {}, "attn": {}, "meta": {}}
    for key, value in raw.items():
        section, _, name = key.partition("::")
        if section in out:
            out[section][name] = value
    out["profiles"] = {k: v.astype(np.float64) for k, v in out["profiles"].items()}
    return out


def validate_analysis_file(path: str | Path) -> bool:
    """Checkpoint validator: the file must parse and carry required keys."""
    try:
        data = load_analysis(path)
    except Exception:
        return False
    if not data["profiles"] or "entropy" not in data["profiles"]:
        return False
    if "sample_id" not in data["meta"]:
        return False
    return True
