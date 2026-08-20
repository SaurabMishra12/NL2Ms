"""Offline tiny-model fixtures for smoke-testing the pipeline.

These build a real ``LlamaForCausalLM`` / ``GPT2LMHeadModel`` with random
weights and a real fast tokenizer, saved to a local directory, so
``load_model`` exercises exactly the same code path it would on Kaggle --
including ``AutoConfig``/``AutoTokenizer`` resolution -- without touching the
network.

The weights are random, so nothing measured on these models says anything
about reasoning. They exist to validate shapes, plumbing, checkpointing and
numerical stability only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


BASE_VOCAB = [
    "<unk>", "<s>", "</s>", "<pad>",
    "A", "B", "C", "D", "Answer", "Question", "What", "is", "the", "of", "a",
    "and", "or", "if", "then", "not", "yes", "no", "cannot", "be", "determined",
    "correct", "option", "letter", "with", "Start", "Multiply", "subtract",
    "result", "before", "after", "last", "first", "who", "finished", "broke",
    "most", "likely", "cause", "container", "holds", "items", "some", "removed",
    "how", "many", "remain", "give", "final", "numeric", "answer", "fill",
    "in", "blank", "sentence", "run", "running", "switch", "on", "door", "open",
    "sensor", "active", "alarm", "light", "fan", "pump", "cat", "wind", "child",
    "window", "vase", "glass", "outside", "whole", "time", "could", "reach",
    "x", "capital", "France", "Paris", "\n", ")", "?", ".", ",", "+", "-", "*", "=",
]
BASE_VOCAB += [str(i) for i in range(0, 300)]
BASE_VOCAB += [f"tok{i}" for i in range(40)]


def build_tokenizer(directory: Path) -> Any:
    """A real WordLevel fast tokenizer saved in HF format."""
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    from transformers import PreTrainedTokenizerFast

    vocab: Dict[str, int] = {}
    for token in BASE_VOCAB:
        if token not in vocab:
            vocab[token] = len(vocab)

    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Punctuation(),
        pre_tokenizers.WhitespaceSplit(),
    ])
    tk.decoder = decoders.WordPiece(prefix="")

    directory.mkdir(parents=True, exist_ok=True)
    tk_path = directory / "tokenizer.json"
    tk.save(str(tk_path))

    fast = PreTrainedTokenizerFast(
        tokenizer_file=str(tk_path),
        unk_token="<unk>", bos_token="<s>", eos_token="</s>", pad_token="<pad>",
    )
    fast.save_pretrained(str(directory))
    return fast


def build_tiny_llama(directory: str | Path, *, n_layers: int = 4,
                     hidden: int = 32, heads: int = 4, kv_heads: int = 2,
                     seed: int = 0) -> str:
    """Create and save a tiny Llama-family model + tokenizer. Returns the path."""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tok = build_tokenizer(directory)

    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=len(tok),
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=n_layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        max_position_embeddings=512,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config)
    model.save_pretrained(str(directory))
    return str(directory)


def build_tiny_gpt2(directory: str | Path, *, n_layers: int = 3,
                    hidden: int = 32, heads: int = 4, seed: int = 0) -> str:
    """A second architecture family, to prove nothing is llama-specific."""
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tok = build_tokenizer(directory)

    torch.manual_seed(seed)
    config = GPT2Config(
        vocab_size=len(tok), n_embd=hidden, n_layer=n_layers, n_head=heads,
        n_positions=512, bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    model = GPT2LMHeadModel(config)
    model.save_pretrained(str(directory))
    return str(directory)


def tiny_model_config(path: str, output_root: str):
    """A :class:`ExperimentConfig` pointed at a tiny local model."""
    from nl2ms.config import smoke_config
    cfg = smoke_config(path, output_root)
    return cfg
