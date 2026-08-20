"""Architecture-agnostic model wrapper, residual-stream capture, generation.

Nothing in this module hard-codes a layer count, hidden size, head count,
tokenizer or vocabulary. Everything is discovered from ``AutoConfig`` and by
walking the module tree, so swapping ``MODEL_NAME`` between Qwen2.5-7B,
Mistral-7B and Llama-3.1-8B changes no analysis code.

Two correctness concerns are handled explicitly here because getting either
wrong would silently invalidate the whole experiment:

**The last ``output_hidden_states`` entry is already normalised.**
    In the Llama/Mistral/Qwen family, ``hidden_states[-1]`` is
    ``final_norm(block_{L-1}(...))`` while entries ``0..L-1`` are raw residual
    stream. Applying the final norm to all of them uniformly double-normalises
    the last one and manufactures an artefactual jump at the final layer --
    exactly the kind of "abrupt transition" this experiment is meant to
    detect. We therefore capture the residual stream with explicit hooks and
    verify the layout at load time.

**The logit lens is an approximation, not an identity.**
    The final norm's learned scale is fitted to layer-L activation statistics.
    Applying it to layer 3 is a *defensible projection*, not the model's own
    computation. We record the exact transformation used, and provide a
    no-norm variant so the norm's contribution can be measured rather than
    assumed (protocol section 35).
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .datasets_build import (ANSWER_CLOSED_SET, ANSWER_OPEN_VOCAB,
                             ANSWER_UNDEFINED, Sample)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
PROMPT_TEMPLATES: Dict[str, str] = {
    # Deliberately minimal: no chain-of-thought instruction, because the
    # primary experiment analyses internal representations during ordinary
    # generation rather than elicited reasoning traces (protocol section 6).
    "plain_qa_v1": "Question: {question}\nAnswer:",
    "instruct_v1": ("Answer the question as directly as possible.\n\n"
                    "Question: {question}\nAnswer:"),
}


def render_prompt(sample: Sample, template_id: str, tokenizer: Any,
                  use_chat_template: bool = True) -> str:
    """Render a sample into the model's expected input string.

    Instruction-tuned models behave very differently inside a chat template
    versus raw text, so we use the tokenizer's own template when it exposes
    one; that choice is recorded per sample.
    """
    body = PROMPT_TEMPLATES[template_id].format(question=sample.question)
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": body}]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return body
    return body


# ---------------------------------------------------------------------------
# Architecture introspection
# ---------------------------------------------------------------------------
@dataclass
class ArchitectureInfo:
    """Everything downstream code needs to know about the model's shape."""

    model_type: str
    architectures: List[str]
    n_layers: int
    hidden_size: int
    n_heads: int
    n_kv_heads: Optional[int]
    head_dim: int
    vocab_size: int
    max_position_embeddings: Optional[int]
    tie_word_embeddings: bool
    layer_module_path: str
    final_norm_path: Optional[str]
    final_norm_type: Optional[str]
    embedding_path: Optional[str]
    lm_head_path: Optional[str]
    logit_lens_valid: bool
    logit_lens_transform: str
    logit_lens_notes: List[str] = field(default_factory=list)
    hidden_states_last_is_normed: Optional[bool] = None
    device_map: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


def _find_layer_list(model: Any, n_layers: int) -> Tuple[str, Any]:
    """Locate the ``ModuleList`` of transformer blocks by length, not by name.

    Name-based lookup breaks the moment a new architecture uses a different
    attribute; matching on ``len(module) == config.num_hidden_layers`` works
    across llama/gpt2/neox/opt/falcon without a lookup table.
    """
    import torch.nn as nn

    candidates: List[Tuple[str, Any]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) == n_layers:
            candidates.append((name, module))
    if not candidates:
        raise RuntimeError(
            f"could not locate a ModuleList of {n_layers} transformer blocks; "
            "this architecture needs explicit support")
    # Prefer the shallowest match (the top-level block list, not something
    # nested inside a block).
    candidates.sort(key=lambda kv: kv[0].count("."))
    return candidates[0]


def _find_final_norm(model: Any, layer_path: str) -> Tuple[Optional[str], Optional[Any]]:
    """Find the backbone's final normalisation, excluding per-block norms."""
    import torch.nn as nn

    backbone_prefix = layer_path.rsplit(".", 1)[0] if "." in layer_path else ""
    norm_types = (nn.LayerNorm,)
    extra = []
    for attr in ("RMSNorm", "LlamaRMSNorm", "Qwen2RMSNorm", "MistralRMSNorm"):
        pass  # concrete classes vary by version; we match duck-typed below

    best: Optional[Tuple[str, Any]] = None
    for name, module in model.named_modules():
        if name.startswith(layer_path + "."):
            continue  # inside a transformer block
        cls_name = type(module).__name__
        is_norm = isinstance(module, norm_types) or "norm" in cls_name.lower()
        if not is_norm:
            continue
        # Must live directly under the backbone that owns the block list.
        if backbone_prefix and not name.startswith(backbone_prefix):
            continue
        depth = name.count(".")
        if best is None or depth < best[0].count("."):
            best = (name, module)
    if best is None:
        return None, None
    return best


def _module_by_path(model: Any, path: str) -> Any:
    obj = model
    for part in path.split("."):
        obj = getattr(obj, part) if not part.isdigit() else obj[int(part)]
    return obj


def _named_path(model: Any, target: Any) -> Optional[str]:
    for name, module in model.named_modules():
        if module is target:
            return name
    return None


def introspect_architecture(model: Any, config: Any) -> ArchitectureInfo:
    """Discover everything structural about the loaded model."""
    n_layers = int(getattr(config, "num_hidden_layers",
                           getattr(config, "n_layer", 0)))
    hidden_size = int(getattr(config, "hidden_size",
                              getattr(config, "n_embd", 0)))
    n_heads = int(getattr(config, "num_attention_heads",
                          getattr(config, "n_head", 0)))
    n_kv_heads = getattr(config, "num_key_value_heads", None)
    head_dim = int(getattr(config, "head_dim", 0) or
                   (hidden_size // n_heads if n_heads else 0))
    vocab_size = int(getattr(config, "vocab_size", 0))

    layer_path, _layers = _find_layer_list(model, n_layers)
    norm_path, norm_module = _find_final_norm(model, layer_path)

    embed = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()
    embed_path = _named_path(model, embed) if embed is not None else None
    lm_head_path = _named_path(model, lm_head) if lm_head is not None else None

    notes: List[str] = []
    valid = True
    if lm_head is None:
        valid = False
        notes.append("no output embedding module exposed; logit lens unavailable")
    if norm_module is None:
        notes.append("no backbone final normalisation found; logit lens will "
                     "apply the unembedding directly to the residual stream")
        transform = "lm_head(h_l)"
    else:
        transform = f"lm_head({type(norm_module).__name__}(h_l))"
        notes.append(
            "The final norm's learned scale is fitted to final-layer statistics; "
            "applying it at earlier layers is a projection, not the model's own "
            "computation. A no-norm variant is computed alongside for contrast.")

    return ArchitectureInfo(
        model_type=getattr(config, "model_type", "unknown"),
        architectures=list(getattr(config, "architectures", []) or []),
        n_layers=n_layers,
        hidden_size=hidden_size,
        n_heads=n_heads,
        n_kv_heads=int(n_kv_heads) if n_kv_heads else None,
        head_dim=head_dim,
        vocab_size=vocab_size,
        max_position_embeddings=getattr(config, "max_position_embeddings", None),
        tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        layer_module_path=layer_path,
        final_norm_path=norm_path,
        final_norm_type=type(norm_module).__name__ if norm_module else None,
        embedding_path=embed_path,
        lm_head_path=lm_head_path,
        logit_lens_valid=valid,
        logit_lens_transform=transform,
        logit_lens_notes=notes,
    )


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------
@dataclass
class GenerationResult:
    sample_id: str
    prompt: str
    input_ids: List[int]
    generated_ids: List[int]
    decoded_output: str
    prediction: Optional[str]
    ground_truth: Optional[str]
    correct: Optional[bool]
    generation_length: int
    prompt_length: int
    finish_reason: str
    model_name: str
    seed: int
    temperature: float
    runtime_seconds: float
    parse_status: str

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


class ModelWrapper:
    """Owns the single in-memory copy of the model and all access to it.

    Deliberately a single object: the protocol forbids holding the model
    twice, and every helper that needs weights takes this wrapper rather than
    loading its own copy.
    """

    def __init__(self, model: Any, tokenizer: Any, config: Any,
                 arch: ArchitectureInfo, model_cfg: Any,
                 load_info: Dict[str, Any]) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.hf_config = config
        self.arch = arch
        self.model_cfg = model_cfg
        self.load_info = load_info
        self._layers = _module_by_path(model, arch.layer_module_path)
        self._final_norm = (_module_by_path(model, arch.final_norm_path)
                            if arch.final_norm_path else None)
        self._lm_head = model.get_output_embeddings()

    # -- basic properties ----------------------------------------------
    @property
    def n_layers(self) -> int:
        return self.arch.n_layers

    @property
    def device(self) -> Any:
        import torch
        try:
            return next(self.model.parameters()).device
        except StopIteration:  # pragma: no cover
            return torch.device("cpu")

    def layer_module(self, index: int) -> Any:
        return self._layers[index]

    # -- residual stream capture ---------------------------------------
    def capture_residual_stream(self, input_ids: Any, attention_mask: Any,
                                *, need_attention: bool = False
                                ) -> Dict[str, Any]:
        """Run one forward pass, capturing explicitly-labelled tensors.

        Returns a dict with:

        ``embed``        -- output of the input embedding (pre block 0)
        ``resid_post``   -- list of length L: residual stream after each block
        ``final_norm``   -- output of the backbone final norm
        ``logits``       -- the model's own logits (ground truth for the lens)
        ``attentions``   -- per-layer attention probabilities if requested

        Using hooks rather than ``output_hidden_states`` removes the
        pre/post-final-norm ambiguity described in the module docstring.
        """
        import torch

        captured: Dict[str, Any] = {"resid_post": [None] * self.n_layers}
        handles = []

        def _embed_hook(module, args, kwargs=None):
            # The depth-0 row must be the residual stream *entering block 0*,
            # not the token-embedding table's output. In GPT-2-style models
            # positional embeddings (and dropout) are added between the two,
            # so hooking the embedding module would omit them; in Llama-style
            # models the two coincide. Taking block 0's input is correct for
            # both without a per-architecture special case.
            if args:
                tensor = _first_tensor(args[0]) if not isinstance(args[0], (int, float)) \
                    else None
                if tensor is not None:
                    captured["embed"] = tensor.detach()
            return None

        def _make_layer_hook(idx: int):
            def hook(_m, _i, out):
                captured["resid_post"][idx] = _first_tensor(out).detach()
            return hook

        def _norm_hook(_m, _i, out):
            captured["final_norm"] = _first_tensor(out).detach()

        handles.append(self._layers[0].register_forward_pre_hook(_embed_hook))
        for i in range(self.n_layers):
            handles.append(self._layers[i].register_forward_hook(_make_layer_hook(i)))
        if self._final_norm is not None:
            handles.append(self._final_norm.register_forward_hook(_norm_hook))

        try:
            with torch.inference_mode():
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=bool(need_attention),
                    output_hidden_states=False,
                    use_cache=False,
                )
            captured["logits"] = out.logits.detach()
            if need_attention:
                captured["attentions"] = [a.detach() for a in (out.attentions or [])]
        finally:
            for h in handles:
                h.remove()
        return captured

    def verify_capture(self, tolerance: float = 5e-2) -> Dict[str, Any]:
        """Confirm the captured pathway reproduces the model's own logits.

        If ``lm_head(final_norm_out)`` does not match ``model.logits``, the
        discovered pathway is wrong and every logit-lens number would be
        meaningless. This check runs once at load time and its result is
        recorded in the manifest.
        """
        import torch

        text = "The capital of France is"
        enc = self.tokenizer(text, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc.get("attention_mask")
        attention_mask = (attention_mask.to(self.device)
                          if attention_mask is not None else None)

        cap = self.capture_residual_stream(input_ids, attention_mask)
        report: Dict[str, Any] = {
            "n_layers_captured": sum(1 for x in cap["resid_post"] if x is not None),
            "expected_layers": self.n_layers,
            "embed_captured": cap.get("embed") is not None,
            "final_norm_captured": cap.get("final_norm") is not None,
        }

        if cap.get("final_norm") is not None and self._lm_head is not None:
            with torch.inference_mode():
                head_in = cap["final_norm"].to(self._lm_head.weight.dtype)
                head_in = head_in.to(self._lm_head.weight.device)
                recomputed = self._lm_head(head_in)
            ref = cap["logits"].to(recomputed.dtype).to(recomputed.device)
            diff = (recomputed - ref).abs().max().item()
            scale = ref.abs().max().item() or 1.0
            report["max_abs_logit_diff"] = float(diff)
            report["relative_logit_diff"] = float(diff / scale)
            report["pathway_verified"] = bool(diff / scale < tolerance)
        else:
            report["pathway_verified"] = False
            report["reason"] = "final norm or lm_head unavailable"

        # Determine whether output_hidden_states[-1] is already normalised, so
        # anyone reading that field elsewhere knows which convention holds.
        with torch.inference_mode():
            out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             output_hidden_states=True, use_cache=False)
        hs_last = out.hidden_states[-1]
        report["n_hidden_states"] = len(out.hidden_states)
        if cap.get("final_norm") is not None:
            same_as_norm = torch.allclose(
                hs_last.float().cpu(), cap["final_norm"].float().cpu(),
                atol=1e-3, rtol=1e-2)
            report["hidden_states_last_is_normed"] = bool(same_as_norm)
            self.arch.hidden_states_last_is_normed = bool(same_as_norm)
        last_block = cap["resid_post"][self.n_layers - 1]
        if last_block is not None:
            report["hidden_states_last_equals_last_block"] = bool(torch.allclose(
                hs_last.float().cpu(), last_block.float().cpu(),
                atol=1e-3, rtol=1e-2))
        del out, cap
        return report

    # -- logit lens -----------------------------------------------------
    def logit_lens(self, hidden: Any, *, apply_norm: bool = True) -> Any:
        """Project a residual-stream tensor to vocabulary logits.

        ``apply_norm=False`` gives the no-norm control used to test whether an
        apparent transition is manufactured by the final normalisation
        (protocol section 35).
        """
        import torch

        if self._lm_head is None:
            raise RuntimeError("model exposes no output embedding; logit lens "
                               "is not available for this architecture")
        with torch.inference_mode():
            h = hidden
            if apply_norm and self._final_norm is not None:
                norm_device = next(self._final_norm.parameters(), None)
                if norm_device is not None:
                    h = h.to(norm_device.device).to(norm_device.dtype)
                h = self._final_norm(h)
            w = self._lm_head.weight
            h = h.to(w.device).to(w.dtype)
            return self._lm_head(h)

    # -- generation -----------------------------------------------------
    def generate(self, prompts: Sequence[str], *, max_new_tokens: int,
                 temperature: float = 0.0, top_p: float = 1.0,
                 seed: int = 0) -> Dict[str, Any]:
        """Batched generation with left padding.

        Left padding is required: with right padding the last real token is
        not at position -1, and every "final token" representation we extract
        would come from a pad slot.
        """
        import torch

        tok = self.tokenizer
        prev_side = tok.padding_side
        tok.padding_side = "left"
        try:
            enc = tok(list(prompts), return_tensors="pt", padding=True,
                      truncation=True,
                      max_length=self.arch.max_position_embeddings or 4096)
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            do_sample = temperature is not None and temperature > 0.0
            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": tok.pad_token_id,
                "return_dict_in_generate": True,
            }
            if do_sample:
                gen_kwargs.update({"temperature": float(temperature),
                                   "top_p": float(top_p)})
                torch.manual_seed(seed)
            start = time.time()
            with torch.inference_mode():
                out = self.model.generate(input_ids=input_ids,
                                          attention_mask=attention_mask,
                                          **gen_kwargs)
            runtime = time.time() - start
            sequences = out.sequences
            prompt_len = input_ids.shape[1]
            generated = sequences[:, prompt_len:]
            return {
                "input_ids": input_ids.cpu(),
                "attention_mask": attention_mask.cpu(),
                "sequences": sequences.cpu(),
                "generated_ids": generated.cpu(),
                "prompt_length": int(prompt_len),
                "runtime_seconds": runtime,
            }
        finally:
            tok.padding_side = prev_side

    def free(self) -> None:
        del self.model
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def _first_tensor(obj: Any) -> Any:
    """Layer hooks receive tensors or tuples depending on architecture."""
    import torch
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, (tuple, list)):
        for item in obj:
            if isinstance(item, torch.Tensor):
                return item
    if hasattr(obj, "last_hidden_state"):
        return obj.last_hidden_state
    raise TypeError(f"cannot extract tensor from hook output of type {type(obj)}")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _dtype_kwarg_name() -> str:
    """``dtype`` on transformers >= 5, ``torch_dtype`` before that."""
    try:
        import transformers
        major = int(str(transformers.__version__).split(".")[0])
    except Exception:
        return "torch_dtype"
    return "dtype" if major >= 5 else "torch_dtype"


def build_quantization_config(kind: Optional[str], compute_dtype: Any) -> Any:
    """4-bit NF4 config when bitsandbytes is present, else ``None``.

    A 7B model in fp16 is ~14 GB, which does not leave room for activations on
    a 15 GB T4. NF4 brings weights to ~4 GB and makes single-GPU operation
    comfortable; if bitsandbytes is missing we fall back to sharding across
    GPUs rather than failing.
    """
    if not kind:
        return None
    try:
        from transformers import BitsAndBytesConfig
        import bitsandbytes  # noqa: F401
    except ImportError:
        print("bitsandbytes unavailable -- proceeding without quantization.")
        return None
    if kind == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    if kind == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def load_model(model_cfg: Any, *, max_memory: Optional[Dict[str, str]] = None,
               verbose: bool = True) -> ModelWrapper:
    """Load tokenizer + model and introspect it. One copy, inference only."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from .env import resolve_dtype

    dtype = resolve_dtype(model_cfg.dtype)
    tok_name = model_cfg.resolved_tokenizer_name()
    tok_rev = model_cfg.resolved_tokenizer_revision()

    tokenizer = AutoTokenizer.from_pretrained(
        tok_name, revision=tok_rev,
        trust_remote_code=model_cfg.trust_remote_code)
    if tokenizer.pad_token_id is None:
        # Padding with EOS is standard for decoder-only models; the attention
        # mask keeps pads out of the computation.
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(
        model_cfg.name, revision=model_cfg.revision,
        trust_remote_code=model_cfg.trust_remote_code)

    quant_config = build_quantization_config(model_cfg.quantization, dtype)

    load_kwargs: Dict[str, Any] = {
        "revision": model_cfg.revision,
        "trust_remote_code": model_cfg.trust_remote_code,
        "attn_implementation": model_cfg.attn_implementation,
        "low_cpu_mem_usage": True,
    }
    # ``torch_dtype`` was renamed to ``dtype`` in transformers 5.x. Both go
    # through ``**kwargs``, so the signature cannot be inspected; the version
    # is the only reliable discriminator. Kaggle images have shipped both.
    load_kwargs[_dtype_kwarg_name()] = dtype

    if quant_config is not None:
        load_kwargs["quantization_config"] = quant_config
    if model_cfg.device_map:
        load_kwargs["device_map"] = model_cfg.device_map
        if max_memory:
            load_kwargs["max_memory"] = max_memory

    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_cfg.name, **load_kwargs)
    load_seconds = time.time() - start
    model.eval()
    # Inference only: no phase of this experiment trains anything, and leaving
    # requires_grad on would silently build graphs during extraction.
    for p in model.parameters():
        p.requires_grad_(False)

    if not model_cfg.device_map and torch.cuda.is_available():
        model = model.to("cuda")

    arch = introspect_architecture(model, config)
    arch.device_map = getattr(model, "hf_device_map", None)

    load_info = {
        "model_name": model_cfg.name,
        "model_revision": model_cfg.revision,
        "tokenizer_name": tok_name,
        "tokenizer_revision": tok_rev,
        "dtype_requested": model_cfg.dtype,
        "dtype_effective": str(dtype),
        "quantization_requested": model_cfg.quantization,
        "quantization_effective": (type(quant_config).__name__
                                   if quant_config else None),
        "device_map": arch.device_map,
        "attn_implementation": model_cfg.attn_implementation,
        "load_seconds": load_seconds,
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "commit_hash": getattr(config, "_commit_hash", None),
    }

    wrapper = ModelWrapper(model, tokenizer, config, arch, model_cfg, load_info)

    if verbose:
        print(f"loaded {model_cfg.name} in {load_seconds:.1f}s")
        print(f"  layers={arch.n_layers} hidden={arch.hidden_size} "
              f"heads={arch.n_heads} kv_heads={arch.n_kv_heads} "
              f"vocab={arch.vocab_size}")
        print(f"  block list  : {arch.layer_module_path}")
        print(f"  final norm  : {arch.final_norm_path} ({arch.final_norm_type})")
        print(f"  lm_head     : {arch.lm_head_path} "
              f"(tied={arch.tie_word_embeddings})")
        print(f"  logit lens  : {arch.logit_lens_transform}")
    return wrapper


# ---------------------------------------------------------------------------
# Answer candidates in token space
# ---------------------------------------------------------------------------
@dataclass
class AnswerSpec:
    """Token-level realisation of a sample's answer specification.

    ``first_token_ids`` holds one token id per candidate: the first token the
    model would emit for that candidate under this tokenizer. Restricting the
    distribution to these ids is what makes ``m_l`` a comparison between the
    answers the task actually offers.
    """

    spec_type: str
    labels: List[str]
    texts: List[str]
    first_token_ids: List[int]
    correct_index: Optional[int]
    full_token_ids: List[List[int]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    usable: bool = True

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


def _leading_variants(text: str) -> List[str]:
    """Surface forms a model might emit, with and without a leading space."""
    text = str(text)
    return [text, " " + text, text.strip(), " " + text.strip()]


def build_answer_spec(sample: Sample, tokenizer: Any) -> AnswerSpec:
    """Map a sample's candidates onto tokenizer ids.

    Collisions matter: if two candidates share a first token, the closed-set
    margin between them is not measurable and the spec is marked unusable
    rather than reporting a meaningless zero.
    """
    warnings: List[str] = []

    if sample.answer_spec_type == ANSWER_UNDEFINED:
        return AnswerSpec(ANSWER_UNDEFINED, [], [], [], None,
                          warnings=["sample has no defined correct answer; "
                                    "margin analyses are skipped"],
                          usable=False)

    if sample.answer_spec_type == ANSWER_CLOSED_SET:
        # The prompt asks for a letter, so the candidate surface form is the
        # option letter, not the option text.
        labels = [c.label for c in sample.candidates]
        texts = [c.label for c in sample.candidates]
        correct_index = next((i for i, c in enumerate(sample.candidates)
                              if c.is_correct), None)
    elif sample.answer_spec_type == ANSWER_OPEN_VOCAB:
        gold = sample.candidates[0].text if sample.candidates else sample.ground_truth
        labels, texts, correct_index = ["gold"], [str(gold)], 0
    else:
        return AnswerSpec(sample.answer_spec_type, [], [], [], None,
                          warnings=[f"unknown answer spec "
                                    f"{sample.answer_spec_type}"], usable=False)

    first_ids: List[int] = []
    full_ids: List[List[int]] = []
    for text in texts:
        chosen: Optional[List[int]] = None
        for variant in _leading_variants(text):
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if ids:
                chosen = ids
                break
        if not chosen:
            warnings.append(f"candidate {text!r} does not tokenize; unusable")
            return AnswerSpec(sample.answer_spec_type, labels, texts, [],
                              correct_index, warnings=warnings, usable=False)
        first_ids.append(int(chosen[0]))
        full_ids.append([int(x) for x in chosen])

    usable = True
    if sample.answer_spec_type == ANSWER_CLOSED_SET:
        if len(set(first_ids)) != len(first_ids):
            warnings.append(
                "candidate first tokens collide under this tokenizer; the "
                "closed-set margin is not identifiable for this sample")
            usable = False
    if correct_index is None:
        warnings.append("no candidate marked correct")
        usable = False

    return AnswerSpec(sample.answer_spec_type, labels, texts, first_ids,
                      correct_index, full_token_ids=full_ids,
                      warnings=warnings, usable=usable)


# ---------------------------------------------------------------------------
# Prediction parsing
# ---------------------------------------------------------------------------
_NUMBER_RE = None


def parse_prediction(sample: Sample, decoded: str) -> Tuple[Optional[str], str]:
    """Extract a prediction from generated text.

    Returns ``(prediction, parse_status)``. ``parse_status`` is recorded for
    every sample so that "model was wrong" is never confused with "we could
    not read the model's answer" -- an unparsed generation is *not* counted as
    incorrect anywhere downstream.
    """
    import re
    global _NUMBER_RE
    if _NUMBER_RE is None:
        _NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")

    text = decoded.strip()
    if not text:
        return None, "empty_generation"

    if sample.answer_spec_type == ANSWER_UNDEFINED:
        return None, "no_ground_truth"

    if sample.answer_spec_type == ANSWER_CLOSED_SET:
        letters = [c.label for c in sample.candidates]
        # Prefer an explicit letter token near the start of the answer.
        m = re.search(r"\b([" + "".join(letters) + r"])\b", text[:64])
        if m:
            return m.group(1), "ok"
        # Fall back to matching the option text verbatim.
        lowered = text.lower()
        hits = [c.label for c in sample.candidates
                if c.text and c.text.lower() in lowered]
        if len(hits) == 1:
            return hits[0], "ok_text_match"
        if len(hits) > 1:
            return None, "ambiguous_multiple_options_mentioned"
        return None, "no_option_found"

    # open_vocab / numeric
    numbers = _NUMBER_RE.findall(text)
    if not numbers:
        return None, "no_number_found"
    return numbers[-1].replace(",", "").rstrip("."), "ok"


def score_prediction(sample: Sample, prediction: Optional[str]
                     ) -> Optional[bool]:
    """Grade a parsed prediction, or return ``None`` when ungradable."""
    if prediction is None or sample.ground_truth is None:
        return None
    if sample.answer_spec_type == ANSWER_CLOSED_SET:
        return prediction.strip().upper() == sample.ground_truth.strip().upper()
    if sample.answer_spec_type == ANSWER_OPEN_VOCAB:
        try:
            return abs(float(prediction) - float(sample.ground_truth)) < 1e-6
        except ValueError:
            return prediction.strip() == sample.ground_truth.strip()
    return None


def finish_reason(generated_ids: Sequence[int], tokenizer: Any,
                  max_new_tokens: int) -> str:
    eos = tokenizer.eos_token_id
    ids = list(generated_ids)
    if eos is not None and eos in ids:
        return "eos"
    if len(ids) >= max_new_tokens:
        return "length"
    return "unknown"


def trim_generated(generated_ids: Sequence[int], tokenizer: Any) -> List[int]:
    """Drop everything from the first EOS/pad onward."""
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    out: List[int] = []
    for tid in generated_ids:
        tid = int(tid)
        if eos is not None and tid == eos:
            break
        if pad is not None and tid == pad and eos is not None and pad == eos:
            break
        out.append(tid)
    return out
