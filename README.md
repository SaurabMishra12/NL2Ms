# NL2Ms — Critical transitions in transformer reasoning dynamics

A reproducible measurement instrument for investigating whether transformer
language models exhibit **localized critical transitions in representation
dynamics** during reasoning.

> **Research question.** When a language model "figures something out", does a
> measurable critical transition occur in its hidden-state, logit, attention,
> latent-space or dynamical trajectory *before* the final answer is produced?

This repository is designed to answer that question **either way**. It is not
a demonstration of the hypothesis, and it contains no code path that can
conclude the hypothesis is true.

---

## Table of contents

- [What this is](#what-this-is)
- [Quick start on Kaggle](#quick-start-on-kaggle)
- [Repository layout](#repository-layout)
- [Configuration](#configuration)
- [What gets measured](#what-gets-measured)
- [Scientific safeguards](#scientific-safeguards)
- [Resumability](#resumability)
- [Storage and runtime](#storage-and-runtime)
- [Output directory](#output-directory)
- [Testing](#testing)
- [Known limitations](#known-limitations)

---

## What this is

An instrument, in the measurement sense. It:

1. runs a decoder-only LM over a mixture of reasoning benchmarks,
2. records the residual stream, logit-lens distributions, attention, latent
   geometry and local sensitivity at every layer,
3. locates candidate critical layers with **eight independent detectors**,
4. tests whether any such layer is **causally** privileged,
5. compares everything against **null models**, and
6. writes a report that enumerates the evidence *against* the hypothesis.

The design commitment is that all of these outcomes stay reachable: no
transition, gradual transition, abrupt transition, multiple transitions,
task-specific transitions, model-specific transitions, token-specific
transitions, an artefact of unembedding geometry, a correlation without causal
relevance, or a genuine causal critical region.

---

## Quick start on Kaggle

### 1. Create the notebook

New Notebook → **Settings**:

| setting | value |
|---|---|
| Accelerator | **GPU T4 x2** (T4 x1 also works) |
| Internet | **On** (needed to download the model and datasets) |
| Persistence | Files only — `/kaggle/working` survives between sessions |

### 2. Get the code into the session

Either clone it in the first cell:

```python
!git clone -q https://github.com/SaurabMishra12/NL2Ms.git /kaggle/working/NL2Ms
```

or attach this repository as a Kaggle Dataset. The notebook searches
`/kaggle/working/NL2Ms`, `/kaggle/input/nl2ms` and the working directory
automatically.

### 3. Run the notebook

Open `notebooks/phase_transitions.ipynb` and **Run All**. The cell sequence is:

| # | cell | purpose |
|---|---|---|
| 1 | dependency check | installs only what is actually missing |
| 2 | path setup | makes `nl2ms` importable |
| 3 | Phase 0 | environment capture + seeding |
| 4 | configuration | `MODEL_NAME`, sample counts, storage level |
| 5 | signal registry | prints the measurement definitions |
| 6 | Phase 1 | dataset preview |
| 7 | model load | architecture discovery + **pathway verification** |
| 8 | GPU report | per-device placement and memory |
| 9 | planning | storage estimate; refuses to start if it won't fit |
| 10 | **pilot** | exercises every code path on ~16 samples |
| 11 | throughput | measured runtime estimate for the full run |
| 12 | **resume test** | proves resumability; prints `RESUME TEST PASSED` |
| 13 | **full run** | all phases; safe to re-run to continue |
| 14+ | results | tables, nulls, causal comparison, figures, report |

Steps 10 and 12 are not optional. The pilot catches failures in two minutes
that would otherwise appear six hours into extraction, and the resume test
verifies that a lost session costs only the work in flight.

### 4. If the session ends early

You will see:

```
SAFE STOP: checkpoint complete. Resume notebook to continue.
```

Start a new session and **Run All** again. Completed samples are skipped and
the run continues from the next incomplete one.

---

## Repository layout

```
nl2ms/
├── config.py           experiment configuration (single source of settings)
├── env.py              Phase 0: environment, seeding, GPU discovery
├── storage.py          atomic IO, shards, manifests, checkpoint recovery
├── runtime.py          wall-clock budget, throughput, heartbeat
├── datasets_build.py   Phase 1: benchmark mixture + synthetic generator
├── modeling.py         architecture-agnostic wrapper, answer specs
├── hooks.py            residual-stream read/write primitives
├── extraction.py       Phases 2-10: per-sample measurement pipeline
├── logit_lens.py       Phase 4: layer-wise vocabulary projection
├── signals.py          pure-NumPy metric definitions
├── geometry.py         Phases 7/18-20: latent-space geometry
├── attention.py        Phases 6/17: attention summaries, restructuring
├── jspace.py           Phases 22/23: local sensitivity descriptors
├── critical.py         Phases 11/26/27: critical-layer / -region detection
├── interventions.py    Phases 13/31/32: causal perturbation + controls
├── stats.py            Phases 14/33-35: effect sizes, nulls, confounds
├── plots.py            Phase 15: figures with lineage sidecars
├── registry.py         Phase 60: signal registry, data lineage
├── report.py           Phases 16/49/50/64: manifest, integrity, report
├── pipeline.py         phase sequencing with checkpoint/resume
├── analysis.py         aggregation, master tables, statistics
└── run.py              entry points: pilot, full run, resume test

notebooks/phase_transitions.ipynb    the Kaggle driver
tests/test_nl2ms.py                  58 tests
build_notebook.py                    regenerates the notebook from source
```

Every phase is an importable function, so the analysis can be moved to a
script or a cluster without touching the notebook.

---

## Configuration

```python
from nl2ms.config import ExperimentConfig, ModelConfig

cfg = ExperimentConfig(output_root="/kaggle/working/experiment",
                       storage_level=3, seed=20260820)
cfg.model = ModelConfig(name="Qwen/Qwen2.5-7B-Instruct",
                        dtype="float16", quantization="4bit",
                        device_map="auto", attn_implementation="eager")
cfg.datasets.n_gsm8k = 150   # etc.
```

Supported out of the box (nothing else changes):

- `Qwen/Qwen2.5-7B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`
- `meta-llama/Llama-3.1-8B-Instruct` (gated; needs a HF token)

Layer count, hidden size, head count, tokenizer and vocabulary size are all
discovered from the loaded model. GPT-2-family models work too, and are used
in the test suite to keep the code honest about architecture-independence.

> **`attn_implementation="eager"` is required.** The SDPA and FlashAttention
> backends do not return attention probabilities, so Phase 6 would silently
> produce nothing.

### Storage levels

| level | contents |
|---|---|
| 0 | metrics only |
| 1 | metrics + hidden states for a selected subset |
| 2 | hidden states at analysed positions, all samples |
| **3** | level 2 + attention summaries — **recommended** |
| 4 | full research dump (all positions, full-vocabulary logits) |

---

## What gets measured

| phase | measurement |
|---|---|
| 0 | environment, seeds, hardware |
| 1 | dataset with content-addressed sample IDs |
| 2 | baseline generation (saved immediately) |
| 3 | residual stream at every layer, by hook |
| 4 | logit lens: top-k, entropy, correct-token rank, margin |
| 5 | entropy, its derivative and curvature across layers |
| 6 | attention: 12 statistics per (layer, head, position) |
| 7 | latent geometry: effective rank, anisotropy, intrinsic dimension |
| 8 | JSD / KL between consecutive layers, cumulative movement |
| 9 | trajectory: velocity, acceleration, turning angle, curvature |
| 10 | candidate competition, symmetry-breaking index |
| 11 | eight critical-layer detectors + consensus + region |
| 12 | correct vs incorrect, on every measure |
| 13 | causal intervention with five control families |
| 14 | bootstrap CIs, permutation tests, FDR correction |
| 15 | 23 figures, each with a lineage sidecar |
| 16 | manifest, integrity check, `FINAL_REPORT.md` |

### The order parameter

Three answer specifications, never pooled:

- **`closed_set`** — enumerated candidates.
  `m_l = q_l(correct) − max_{w≠correct} q_l(w)` on the renormalised
  within-candidate distribution.
- **`open_vocab`** — a single correct token, no distractors (GSM8K).
  `m_l = p_l(correct) − max_{v≠correct} p_l(v)` over the whole vocabulary.
- **`undefined`** — deliberately ambiguous synthetic items with no correct
  answer. Used only as an entropy/geometry control.

### J-space

For probe directions `v_1..v_k` and the layer map `h_{l+1} = F_l(h_l)`:

```
s_i(l) = || F_l(h_l + ε v_i) − F_l(h_l) || / || ε v_i ||
J_l(x) = [ s_1(l), …, s_k(l) ]
```

A finite-difference estimate of `||J_l v_i||` — the full `d × d` Jacobian is
never materialised. Probe directions are seeded once per experiment, so
descriptors are comparable across samples. The ε-dependence is swept and
reported rather than hidden behind one number.

---

## Scientific safeguards

These are the parts that make the result trustworthy in either direction.

**Correct residual-stream capture.** In Llama/Mistral/Qwen models,
`hidden_states[-1]` is *already normalised* while the rest are not. Applying
the final norm to all of them uniformly manufactures an artefactual jump at
the final layer — precisely where this experiment looks for transitions. We
hook the blocks directly and verify at load time that
`lm_head(FinalNorm(h_L))` reproduces the model's own logits.

**Eight independent detectors, honest consensus.** A consensus layer is
emitted only when ≥3 detectors agree within 10% of depth. Otherwise the field
is `None` — a real and common outcome that is never imputed. The report notes
that two of the detectors share a source profile and are therefore not
independent corroboration.

**Null models.** Structureless AR(1) curves measure how often the detection
machinery invents a transition where none exists. At realistic depth the
false-`sharp` rate is 0%; at very shallow depth it is ~50%, and the report
prints that floor next to any observed fraction.

**Causal test with matched controls.** Every perturbation is norm-matched to
`ε·‖h_l‖` and paired with: a random layer, a random direction, a direction
orthogonal to the answer direction, a PCA direction, and a direction taken
from a *different* sample. A large effect at the critical layer means nothing
if the random-layer control matches it.

**Confound analysis.** Prompt length, generation length, baseline confidence,
answer-token identity, EOS behaviour and candidate mass are correlated against
the critical-layer location, and the group effect is recomputed on a
confound-matched subsample.

**Unparsed generations are not wrong answers.** Every generation carries a
`parse_status`; ungraded samples are excluded from correct-vs-incorrect
comparisons rather than counted as incorrect.

**Constrained language.** The report generator can emit "consistent with",
"associated with" and "candidate critical transition". It cannot emit "a phase
transition was observed" — there is a test asserting the phrase does not exist
in the source.

---

## Resumability

Every `(sample, phase)` unit of work is recorded in `manifests/manifest.jsonl`.
Before processing, the manifest is consulted:

| state | action |
|---|---|
| complete, file present, checksum matches | `SKIP` |
| absent or incomplete | `RESUME` |
| file missing or checksum mismatch | `MARK_CORRUPTED_AND_RECOMPUTE` |

Nothing is written directly to its final path: data goes to `<path>.tmp`, is
fsynced, then atomically renamed. A crash mid-write leaves the previous valid
file untouched.

`logs/heartbeat.json` is refreshed every couple of minutes with the current
phase, sample, shard, elapsed time, GPU/CPU memory and disk usage — so a
crashed session can be diagnosed from its last heartbeat.

The `resume_test()` function proves this end to end and prints
`RESUME TEST PASSED` only if completed work is skipped, the run continues from
the next incomplete sample, and no existing checkpoint was overwritten.

---

## Storage and runtime

`experiment.plan(n)` estimates disk usage before extraction and **refuses to
start** a run that does not fit, printing the per-component breakdown.

Runtime estimates come from the pilot's measured throughput. `RuntimeController`
checks before every batch whether it can finish inside the budget minus a
finalisation reserve; if not, it stops cleanly.

Rough figures for a 7B model at storage level 3 (verify with your own pilot —
these are extrapolations from the estimator, not measured on a T4):

| samples | hidden states | metrics + attention | total |
|---|---|---|---|
| 100 | ~0.08 GB | ~0.3 GB | ~0.4 GB |
| 500 | ~0.4 GB | ~1.5 GB | ~2 GB |
| 1000 | ~0.8 GB | ~3 GB | ~4 GB |

Storing only the three analysed positions per sample (rather than every
generated token) is what keeps this bounded; level 4 raises it by roughly the
number of generated positions.

---

## Output directory

```
experiment/
├── config/               experiment_config.json, environment.json,
│                         signal_registry.json, model_verification.json
├── checkpoints/          per-phase completion markers (config-hash guarded)
├── manifests/            manifest.jsonl, shard_manifest.json
├── data/                 samples.jsonl, provenance.json
├── raw/
│   ├── generations/      generations.jsonl (written as they are produced)
│   ├── hidden_states/    sharded .npz + per-shard metadata + checksums
│   └── attention/        full matrices for a few audited samples
├── derived/
│   ├── entropy/          per-sample profiles (.npz)
│   ├── geometry/         layer_geometry.json
│   ├── j_space/          per-sample descriptors + separability.json
│   └── critical_layers/  agreement.json, sample_summary.parquet
├── interventions/        per-sample results + intervention_outcomes.parquet
├── statistics/           correct_vs_incorrect, nulls, confounds,
│                         susceptibility, signal_correlation_matrix
├── figures/              23 figures (PNG + PDF) each with a .json sidecar
├── examples/             individual traces + per-example JSON reports
├── reports/              FINAL_REPORT.md, experiment_summary.json,
│                         integrity_report.json
├── logs/                 heartbeat.json, errors.jsonl, plan.json,
│                         runtime_report.json, resume_test.json
├── experiment_manifest.json
├── sample_summary.parquet     one row per sample (~110 columns)
└── layer_summary.parquet      one row per (dataset, group, layer)
```

A compressed backup of everything except the raw tensors is written at the end
and its path is printed. **Download it before the session ends.**

---

## Testing

```bash
python tests/test_nl2ms.py          # or: python -m pytest tests/ -v
```

58 tests covering:

- **Metric correctness** against closed forms and scipy (entropy, KL, JSD,
  Cliff's delta against brute force, Benjamini–Hochberg against hand
  calculation).
- **Discrimination** — the null models must return `p ≈ uniform` on
  structureless data and `p < 0.05` on a planted peak; the shape classifier
  must call a flat profile flat.
- **Storage safety** — atomic writes preserve the previous file on a simulated
  crash; a torn JSONL line does not corrupt the manifest; corrupted checkpoints
  are detected by checksum.
- **Refusal to fabricate** — comparisons below the minimum group size report
  `status`, not a number; tests that could not run keep `significant=None`
  rather than `False`; the report generator cannot emit hypothesis-asserting
  phrases.

The suite runs on CPU in about a minute and needs no model download.

---

## Known limitations

**Validated on CPU with tiny models only.** The development environment had no
GPU and no access to the HuggingFace hub. Everything below was tested with
randomly-initialised 3–4 layer Llama and GPT-2 models built locally:
architecture discovery, logit-lens pathway verification, hook semantics,
all metric math, sharding, checkpoint/resume, interventions, figures, report
and integrity checks — the full pipeline end to end, on both architectures.

**Not yet validated:** 4-bit quantized loading (`bitsandbytes` was
unavailable), multi-GPU `device_map="auto"` sharding, the GSM8K / TruthfulQA /
Winogrande loaders (the hub was unreachable), and real-model throughput. The
runtime and storage figures above are extrapolations from the estimator, not
measurements on a T4. **Run the pilot before trusting any of them** — that is
what it is for.

**What the experiment cannot establish**, even when it runs perfectly:

- That any transition constitutes a phase transition in the statistical-physics
  sense. No critical exponents, no finite-size scaling, no order-parameter
  conjugate field.
- That a model "understands" or experiences anything.
- Causality beyond the specific interventions performed — a null result on
  perturbation is informative, but a positive one shows only that the layer
  participates, not that it is where reasoning happens.
- Universality from one or two models.
- Anything about models outside the tested size and family range.

---

## Citation

If this instrument contributes to published work, please cite the repository
and record the `config_hash` and `git_commit` from
`experiment_manifest.json` — they identify the exact measurement definitions
used.
