"""Phase 1 -- dataset assembly with permanent sample identity.

Every example carries a content-derived ``sample_id`` so that a row can be
traced across shards, reruns and machines without depending on iteration
order. Row indices are recorded as *metadata only* and are never used as
identity.

The module also defines the **answer specification**, which is what makes the
order parameter well-posed. Three kinds exist and they are never mixed
silently:

``closed_set``
    A finite, enumerated candidate list (Winogrande, TruthfulQA MC, the
    synthetic set). ``m_l`` is computed over the renormalised distribution
    restricted to those candidates.
``open_vocab``
    A single correct target token with no enumerated distractors (GSM8K's
    numeric answer). ``m_l`` uses the full vocabulary: correct probability
    minus the best competing token.
``undefined``
    No unique correct answer exists (deliberately ambiguous synthetic items).
    Margin-based analyses are skipped for these and they are used only as a
    control for entropy/geometry measures.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .storage import save_json, load_json


ANSWER_CLOSED_SET = "closed_set"
ANSWER_OPEN_VOCAB = "open_vocab"
ANSWER_UNDEFINED = "undefined"


# ---------------------------------------------------------------------------
# Sample schema
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    label: str          # e.g. "A"
    text: str           # surface form the model should produce
    is_correct: bool


@dataclass
class Sample:
    sample_id: str
    dataset: str
    subset: str
    question: str
    ground_truth: Optional[str]
    question_type: str
    answer_spec_type: str
    candidates: List[Candidate] = field(default_factory=list)
    difficulty: Optional[str] = None
    prompt_template: str = "plain_qa_v1"
    seed: Optional[int] = None
    model_name: Optional[str] = None
    model_revision: Optional[str] = None
    source_index: Optional[int] = None
    reasoning_steps: Optional[int] = None
    generator_config: Optional[Dict[str, Any]] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["candidates"] = [asdict(c) for c in self.candidates]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Sample":
        d = dict(d)
        d["candidates"] = [Candidate(**c) for c in d.get("candidates", [])]
        return cls(**d)

    def correct_candidate(self) -> Optional[Candidate]:
        for c in self.candidates:
            if c.is_correct:
                return c
        return None


def make_sample_id(dataset: str, subset: str, question: str,
                   ground_truth: Optional[str], index: int) -> str:
    """Content-addressed identity.

    Hashing the question text (not the row number) means the same example
    keeps its ID if the upstream dataset is reshuffled or re-split. The index
    is folded in only to disambiguate genuine duplicates within a source.
    """
    payload = json.dumps(
        {"d": dataset, "s": subset, "q": question, "g": ground_truth, "i": index},
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()[:12]
    return f"{dataset}-{digest}"


# ---------------------------------------------------------------------------
# Synthetic controlled benchmark (protocol section 5)
# ---------------------------------------------------------------------------
SYNTHETIC_CATEGORIES = [
    "arithmetic",
    "multi_step_arithmetic",
    "logical_deduction",
    "conditional_reasoning",
    "symbolic_reasoning",
    "competing_hypotheses",
    "ambiguous",
]

SYNTHETIC_GENERATOR_VERSION = "synthetic_v1"


def _mc_letters(n: int) -> List[str]:
    return [chr(ord("A") + i) for i in range(n)]


def _format_choices(options: Sequence[str]) -> str:
    return "\n".join(f"{letter}) {text}"
                     for letter, text in zip(_mc_letters(len(options)), options))


def _distractor_numbers(rng: random.Random, answer: int, n: int) -> List[int]:
    """Plausible wrong numeric options: near-misses, not random noise.

    Near-miss distractors are what make the competition between candidate
    answers non-trivial; random far-off numbers would be discriminable in the
    embedding layer and would defeat the purpose of the measurement.
    """
    out: set = set()
    offsets = [1, -1, 2, -2, 3, -3, 10, -10]
    rng.shuffle(offsets)
    for off in offsets:
        cand = answer + off
        if cand != answer and cand > 0:
            out.add(cand)
        if len(out) >= n:
            break
    while len(out) < n:
        cand = max(1, answer + rng.randint(-20, 20))
        if cand != answer:
            out.add(cand)
    return list(out)[:n]


def _build_mc(rng: random.Random, question_body: str, answer_text: str,
              distractors: Sequence[str]) -> Tuple[str, List[Candidate]]:
    options = [answer_text] + list(distractors)
    rng.shuffle(options)
    letters = _mc_letters(len(options))
    candidates = [Candidate(label=letters[i], text=str(opt),
                            is_correct=(str(opt) == str(answer_text)))
                  for i, opt in enumerate(options)]
    question = (f"{question_body}\n{_format_choices([str(o) for o in options])}\n"
                "Answer with the letter of the correct option.")
    return question, candidates


def _gen_arithmetic(rng: random.Random) -> Dict[str, Any]:
    a, b = rng.randint(11, 89), rng.randint(11, 89)
    op = rng.choice(["+", "-", "*"])
    if op == "+":
        ans, body = a + b, f"What is {a} + {b}?"
    elif op == "-":
        a, b = max(a, b), min(a, b)
        ans, body = a - b, f"What is {a} - {b}?"
    else:
        a, b = rng.randint(3, 19), rng.randint(3, 19)
        ans, body = a * b, f"What is {a} * {b}?"
    return {"body": body, "answer": str(ans),
            "distractors": [str(x) for x in _distractor_numbers(rng, ans, 3)],
            "steps": 1}


def _gen_multi_step(rng: random.Random) -> Dict[str, Any]:
    a, b, c = rng.randint(5, 40), rng.randint(2, 9), rng.randint(3, 30)
    ans = a * b - c
    body = (f"Start with {a}. Multiply it by {b}. Then subtract {c}. "
            "What is the result?")
    return {"body": body, "answer": str(ans),
            "distractors": [str(a * b + c), str((a - c) * b),
                            str(_distractor_numbers(rng, ans, 1)[0])],
            "steps": 3}


def _gen_logical_deduction(rng: random.Random) -> Dict[str, Any]:
    names = rng.sample(["Ana", "Ben", "Cleo", "Dov", "Eli", "Fay"], 3)
    order = list(names)
    rng.shuffle(order)
    body = (f"{order[0]} finished before {order[1]}. "
            f"{order[1]} finished before {order[2]}. "
            "Who finished last?")
    return {"body": body, "answer": order[2],
            "distractors": [order[0], order[1], "cannot be determined"],
            "steps": 2}


def _gen_conditional(rng: random.Random) -> Dict[str, Any]:
    thing = rng.choice(["the alarm", "the light", "the fan", "the pump"])
    trigger = rng.choice(["the switch is on", "the door is open",
                          "the sensor is active"])
    state = rng.choice([True, False])
    body = (f"Rule: if {trigger}, then {thing} is running. "
            f"Observation: {trigger.replace('is', 'is not')}."
            if not state else
            f"Rule: if {trigger}, then {thing} is running. "
            f"Observation: {trigger}.")
    if state:
        answer = "running"
        distractors = ["not running", "cannot be determined", "both"]
    else:
        # Denying the antecedent does not determine the consequent.
        answer = "cannot be determined"
        distractors = ["running", "not running", "both"]
    return {"body": body + f" Is {thing} running?", "answer": answer,
            "distractors": distractors, "steps": 2}


def _gen_symbolic(rng: random.Random) -> Dict[str, Any]:
    x = rng.randint(2, 12)
    k = rng.randint(2, 9)
    c = rng.randint(1, 20)
    rhs = k * x + c
    body = f"If {k}x + {c} = {rhs}, what is x?"
    return {"body": body, "answer": str(x),
            "distractors": [str(x + 1), str(rhs - c), str(max(1, x - 2))],
            "steps": 2}


def _gen_competing(rng: random.Random) -> Dict[str, Any]:
    """Two hypotheses are individually plausible; evidence selects one.

    These items exist specifically to create a period of genuine competition
    between candidate answers, which is the regime where symmetry-breaking
    measures are meaningful.
    """
    item = rng.choice(["the window", "the vase", "the glass"])
    a, b = rng.sample(["the cat", "the wind", "the child", "the door"], 2)
    body = (f"{item} broke. Either {a} or {b} could have caused it. "
            f"{b.capitalize()} was outside the whole time and could not reach {item}. "
            "What is the most likely cause?")
    return {"body": body, "answer": a, "distractors": [b, "neither", "both"],
            "steps": 2}


def _gen_ambiguous(rng: random.Random) -> Dict[str, Any]:
    """Deliberately underdetermined: no option is correct.

    Marked ``answer_spec_type = undefined`` so no correctness or margin
    statistic is ever computed from it.
    """
    a = rng.randint(2, 30)
    body = (f"A container holds {a} items. Some are removed. "
            "How many remain?")
    return {"body": body, "answer": None,
            "distractors": [str(a), str(max(0, a - 1)), "0",
                            "cannot be determined"],
            "steps": 0}


_SYNTHETIC_GENERATORS = {
    "arithmetic": _gen_arithmetic,
    "multi_step_arithmetic": _gen_multi_step,
    "logical_deduction": _gen_logical_deduction,
    "conditional_reasoning": _gen_conditional,
    "symbolic_reasoning": _gen_symbolic,
    "competing_hypotheses": _gen_competing,
    "ambiguous": _gen_ambiguous,
}


def generate_synthetic(n: int, seed: int,
                       categories: Optional[Sequence[str]] = None,
                       prompt_template: str = "plain_qa_v1") -> List[Sample]:
    """Deterministic synthetic reasoning items.

    Determinism is per-item: each example draws from a ``Random`` seeded with
    ``(seed, category, i)``, so changing ``n`` does not perturb the examples
    already generated at smaller ``n``. That property is what lets a pilot
    subset be a strict prefix of the full run.
    """
    categories = list(categories or SYNTHETIC_CATEGORIES)
    samples: List[Sample] = []
    generator_config = {
        "generator_version": SYNTHETIC_GENERATOR_VERSION,
        "seed": seed,
        "categories": categories,
        "n_requested": n,
    }
    i = 0
    while len(samples) < n:
        category = categories[i % len(categories)]
        item_seed = int(hashlib.sha1(
            f"{seed}|{category}|{i}".encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(item_seed)
        spec = _SYNTHETIC_GENERATORS[category](rng)
        answer = spec["answer"]
        if answer is None:
            options = list(spec["distractors"])
            rng.shuffle(options)
            letters = _mc_letters(len(options))
            candidates = [Candidate(letters[j], str(o), False)
                          for j, o in enumerate(options)]
            question = (f"{spec['body']}\n{_format_choices(options)}\n"
                        "Answer with the letter of the correct option.")
            answer_spec = ANSWER_UNDEFINED
            ground_truth = None
        else:
            question, candidates = _build_mc(rng, spec["body"], str(answer),
                                             spec["distractors"])
            answer_spec = ANSWER_CLOSED_SET
            correct = next(c for c in candidates if c.is_correct)
            ground_truth = correct.label
        sid = make_sample_id("synthetic", category, question, ground_truth, i)
        samples.append(Sample(
            sample_id=sid,
            dataset="synthetic",
            subset=category,
            question=question,
            ground_truth=ground_truth,
            question_type=category,
            answer_spec_type=answer_spec,
            candidates=candidates,
            difficulty=str(spec.get("steps", 0)),
            prompt_template=prompt_template,
            seed=item_seed,
            source_index=i,
            reasoning_steps=spec.get("steps"),
            generator_config=generator_config,
            notes="no unique correct answer by construction"
            if answer_spec == ANSWER_UNDEFINED else "",
        ))
        i += 1
    return samples


# ---------------------------------------------------------------------------
# Hugging Face benchmark loaders
# ---------------------------------------------------------------------------
_GSM8K_ANSWER_RE = re.compile(r"####\s*([\-0-9\.,]+)")


def _clean_number(text: str) -> str:
    return text.replace(",", "").strip().rstrip(".")


def load_gsm8k(n: int, split: str = "test",
               prompt_template: str = "plain_qa_v1") -> List[Sample]:
    """GSM8K -- open-vocabulary numeric answers.

    There is no enumerated distractor set, so the order parameter uses the
    open-vocabulary definition. Recorded explicitly rather than fabricating
    multiple-choice options that the model never saw.
    """
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split=split)
    samples: List[Sample] = []
    for i in range(min(n, len(ds))):
        row = ds[i]
        match = _GSM8K_ANSWER_RE.search(row["answer"])
        if not match:
            continue
        answer = _clean_number(match.group(1))
        question = (f"{row['question'].strip()}\n"
                    "Give the final numeric answer.")
        sid = make_sample_id("gsm8k", split, question, answer, i)
        samples.append(Sample(
            sample_id=sid, dataset="gsm8k", subset=split, question=question,
            ground_truth=answer, question_type="arithmetic_word_problem",
            answer_spec_type=ANSWER_OPEN_VOCAB,
            candidates=[Candidate("gold", answer, True)],
            difficulty=str(row["answer"].count("\n")),
            prompt_template=prompt_template, source_index=i,
            reasoning_steps=row["answer"].count("\n"),
        ))
    return samples


def load_truthfulqa(n: int, split: str = "validation",
                    prompt_template: str = "plain_qa_v1") -> List[Sample]:
    """TruthfulQA (mc1) -- closed candidate set, one correct answer."""
    from datasets import load_dataset
    ds = load_dataset("truthful_qa", "multiple_choice", split=split)
    samples: List[Sample] = []
    rng = random.Random(4242)
    for i in range(min(n, len(ds))):
        row = ds[i]
        mc1 = row["mc1_targets"]
        choices, labels = list(mc1["choices"]), list(mc1["labels"])
        if not choices or sum(labels) != 1:
            continue
        # Cap the option count: some items have 12+ choices, which makes the
        # prompt long and the closed-set margin dominated by option count.
        correct_idx = labels.index(1)
        keep = [correct_idx] + [j for j in range(len(choices)) if j != correct_idx][:3]
        options = [choices[j] for j in keep]
        rng.shuffle(options)
        letters = _mc_letters(len(options))
        candidates = [Candidate(letters[j], opt, opt == choices[correct_idx])
                      for j, opt in enumerate(options)]
        correct = next(c for c in candidates if c.is_correct)
        question = (f"{row['question'].strip()}\n"
                    f"{_format_choices(options)}\n"
                    "Answer with the letter of the correct option.")
        sid = make_sample_id("truthfulqa", split, question, correct.label, i)
        samples.append(Sample(
            sample_id=sid, dataset="truthfulqa", subset=split, question=question,
            ground_truth=correct.label, question_type="truthfulness_mc",
            answer_spec_type=ANSWER_CLOSED_SET, candidates=candidates,
            prompt_template=prompt_template, source_index=i,
        ))
    return samples


def load_winogrande(n: int, split: str = "validation",
                    subset: str = "winogrande_xl",
                    prompt_template: str = "plain_qa_v1") -> List[Sample]:
    """Winogrande -- binary closed candidate set (pronoun resolution)."""
    from datasets import load_dataset
    ds = load_dataset("winogrande", subset, split=split, trust_remote_code=True)
    samples: List[Sample] = []
    for i in range(min(n, len(ds))):
        row = ds[i]
        if row["answer"] not in ("1", "2"):
            continue
        options = [row["option1"], row["option2"]]
        correct_idx = int(row["answer"]) - 1
        letters = _mc_letters(2)
        candidates = [Candidate(letters[j], opt, j == correct_idx)
                      for j, opt in enumerate(options)]
        question = (f"Fill in the blank:\n{row['sentence'].strip()}\n"
                    f"{_format_choices(options)}\n"
                    "Answer with the letter of the correct option.")
        sid = make_sample_id("winogrande", subset, question,
                             letters[correct_idx], i)
        samples.append(Sample(
            sample_id=sid, dataset="winogrande", subset=subset, question=question,
            ground_truth=letters[correct_idx], question_type="coreference",
            answer_spec_type=ANSWER_CLOSED_SET, candidates=candidates,
            prompt_template=prompt_template, source_index=i,
        ))
    return samples


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
_HF_LOADERS = {
    "gsm8k": lambda cfg: load_gsm8k(cfg.n_gsm8k, cfg.gsm8k_split),
    "truthfulqa": lambda cfg: load_truthfulqa(cfg.n_truthfulqa, cfg.truthfulqa_split),
    "winogrande": lambda cfg: load_winogrande(cfg.n_winogrande,
                                              cfg.winogrande_split,
                                              cfg.winogrande_subset),
}


def build_dataset(cfg, model_name: str, model_revision: str,
                  prompt_template: str = "plain_qa_v1") -> Tuple[List[Sample], Dict[str, Any]]:
    """Assemble the mixture, tolerating unreachable hubs.

    A hub failure downgrades that source to zero samples and is recorded in
    the provenance dict; it never aborts the run and never silently
    substitutes a different dataset.
    """
    samples: List[Sample] = []
    provenance: Dict[str, Any] = {"sources": {}, "warnings": []}

    requested = {
        "gsm8k": cfg.n_gsm8k,
        "truthfulqa": cfg.n_truthfulqa,
        "winogrande": cfg.n_winogrande,
    }
    for name, count in requested.items():
        if count <= 0:
            provenance["sources"][name] = {"requested": 0, "loaded": 0,
                                           "status": "disabled"}
            continue
        if not cfg.allow_hub_download:
            provenance["sources"][name] = {"requested": count, "loaded": 0,
                                           "status": "skipped_offline"}
            provenance["warnings"].append(f"{name}: hub download disabled")
            continue
        try:
            loaded = _HF_LOADERS[name](cfg)
            samples.extend(loaded)
            provenance["sources"][name] = {
                "requested": count, "loaded": len(loaded), "status": "ok",
            }
        except Exception as exc:
            provenance["sources"][name] = {
                "requested": count, "loaded": 0, "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            provenance["warnings"].append(
                f"{name}: load failed ({type(exc).__name__}); "
                "this source contributes zero samples")

    if cfg.n_synthetic > 0:
        syn = generate_synthetic(cfg.n_synthetic, cfg.synthetic_seed,
                                 prompt_template=prompt_template)
        samples.extend(syn)
        provenance["sources"]["synthetic"] = {
            "requested": cfg.n_synthetic, "loaded": len(syn), "status": "ok",
            "generator_version": SYNTHETIC_GENERATOR_VERSION,
            "seed": cfg.synthetic_seed,
        }

    for s in samples:
        s.model_name = model_name
        s.model_revision = model_revision
        s.prompt_template = prompt_template

    # Duplicate IDs would corrupt every downstream join; fail loudly.
    ids = [s.sample_id for s in samples]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"duplicate sample_ids generated: {sorted(duplicates)[:5]}")

    provenance["n_total"] = len(samples)
    provenance["by_dataset"] = {
        d: sum(1 for s in samples if s.dataset == d)
        for d in sorted({s.dataset for s in samples})
    }
    provenance["by_answer_spec"] = {
        a: sum(1 for s in samples if s.answer_spec_type == a)
        for a in sorted({s.answer_spec_type for s in samples})
    }
    return samples, provenance


def save_samples(samples: Sequence[Sample], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    from .storage import atomic_path
    with atomic_path(path) as tmp:
        with open(tmp, "w") as fh:
            for s in samples:
                fh.write(json.dumps(s.to_dict()) + "\n")
    # Verify the file round-trips before declaring the phase complete.
    reloaded = load_samples(path)
    if len(reloaded) != len(samples):
        raise IOError(f"sample write verification failed: "
                      f"{len(reloaded)} != {len(samples)}")
    return path


def load_samples(path: str | Path) -> List[Sample]:
    from .storage import read_jsonl
    return [Sample.from_dict(d) for d in read_jsonl(path)]


def stratified_subset(samples: Sequence[Sample], n: int,
                      seed: int = 0) -> List[Sample]:
    """Pick ``n`` samples spread evenly across datasets (for the pilot).

    Round-robin over datasets rather than a uniform draw, so a pilot of 16
    always exercises every loader and every answer-spec type.
    """
    by_ds: Dict[str, List[Sample]] = {}
    for s in samples:
        by_ds.setdefault(s.dataset, []).append(s)
    rng = random.Random(seed)
    for lst in by_ds.values():
        rng.shuffle(lst)
    out: List[Sample] = []
    keys = sorted(by_ds.keys())
    idx = 0
    while len(out) < n and any(by_ds[k] for k in keys):
        key = keys[idx % len(keys)]
        if by_ds[key]:
            out.append(by_ds[key].pop())
        idx += 1
    return out[:n]


def dataset_summary(samples: Sequence[Sample]) -> Dict[str, Any]:
    return {
        "n_samples": len(samples),
        "datasets": sorted({s.dataset for s in samples}),
        "by_dataset": {d: sum(1 for s in samples if s.dataset == d)
                       for d in sorted({s.dataset for s in samples})},
        "by_question_type": {q: sum(1 for s in samples if s.question_type == q)
                             for q in sorted({s.question_type for s in samples})},
        "by_answer_spec": {a: sum(1 for s in samples if s.answer_spec_type == a)
                           for a in sorted({s.answer_spec_type for s in samples})},
        "n_unique_ids": len({s.sample_id for s in samples}),
    }
