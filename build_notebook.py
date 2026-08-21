"""Build ``notebooks/phase_transitions.ipynb``.

The notebook is generated from this script so that it stays in sync with the
package and so the cell structure is reviewable as source. Run:

    python build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str):
    return nbf.v4.new_code_cell(text.strip("\n"))


cells = []

# ===========================================================================
cells.append(md(r"""
# Critical transitions in transformer reasoning dynamics

**A measurement instrument, not a demonstration.**

---

## Research question

> When a language model "figures something out", does a measurable critical
> transition occur in its hidden-state, logit, attention, latent-space or
> dynamical trajectory *before* the final answer is produced?

## Hypothesis under test

> Successful reasoning in transformer language models may contain localized
> critical transitions in representation dynamics, analogous to
> phase-transition-like behaviour.

## What this notebook is designed to do

This notebook is built to **discover whether the phenomenon exists**, not to
show that it does. Every one of these outcomes must remain reachable:

| | outcome |
|---|---|
| A | no transition |
| B | gradual transition |
| C | abrupt transition |
| D | multiple transitions |
| E | task-dependent transitions |
| F | model-dependent transitions |
| G | token-dependent transitions |
| H | apparent transition caused by unembedding / logit geometry |
| I | transition correlated but not causally relevant |
| J | genuine causal critical region |

Concretely, the design commits to the following in advance:

1. **Eight independent critical-layer detectors.** A consensus layer is
   emitted only when at least three agree; otherwise the field is `None`.
   Picking whichever heuristic gives the cleanest picture is the central
   failure mode of this kind of work, and the code cannot do it.
2. **Null models are first-class.** A structureless AR(1) baseline measures
   how often the detection machinery invents a transition where none exists.
   Every claim is read against that floor.
3. **Causal intervention with matched controls.** Correlational sharpness
   without causal privilege is a negative result, and the report says so.
4. **A mandatory closing section**, *"What would invalidate the
   phase-transition hypothesis?"*, generated from the run's own numbers.

## Interpretation caveats carried throughout

- The **logit lens** applies the final normalisation and unembedding at layers
  where they are out of distribution. It is a defensible projection, not the
  model's own computation. A no-norm control and a shuffled-unembedding null
  are computed alongside.
- **Residual-stream norm grows with depth** for architectural reasons. Raw
  velocity and distance measures inherit that trend; the scale-free
  counterparts are the ones that carry information.
- **"Susceptibility"** here is the across-sample variance of an
  order-parameter analogue. No fluctuation-dissipation relation is
  established and no critical exponent is claimed.
- **Attention weights are not attribution.**
- Averaging gradual transitions that occur at *different* layers produces a
  sharp population curve. Individual trajectories are plotted next to every
  mean for this reason.
"""))

# ===========================================================================
cells.append(md(r"""
## Phase 0 — Environment, dependencies, reproducibility

Checks the runtime, records package versions and hardware, and seeds every
RNG. The record is written to `config/environment.json` *before* any
measurement, so a run can be reproduced from its own artefacts.
"""))

cells.append(code(r"""
# Kaggle images ship most of this. Only install what is genuinely missing so
# a flaky network cannot break a session that was already usable.
import importlib, subprocess, sys

REQUIRED = ["torch", "transformers", "numpy", "scipy", "pandas", "sklearn",
            "matplotlib"]
OPTIONAL = {"datasets": "datasets", "accelerate": "accelerate",
            "bitsandbytes": "bitsandbytes", "pyarrow": "pyarrow"}

missing_required = [m for m in REQUIRED if importlib.util.find_spec(m) is None]
missing_optional = [pkg for mod, pkg in OPTIONAL.items()
                    if importlib.util.find_spec(mod) is None]

print("missing required:", missing_required or "none")
print("missing optional:", missing_optional or "none")

if missing_required or missing_optional:
    to_install = missing_required + missing_optional
    print(f"installing: {to_install}")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *to_install],
                   check=False)
    importlib.invalidate_caches()
"""))

cells.append(code(r"""
# Make the nl2ms package importable.
#
# On Kaggle, attach this repository as a dataset/utility script, or clone it
# into the working directory. The loop below covers the usual locations.
import sys
from pathlib import Path

CANDIDATES = [
    Path.cwd(),
    Path.cwd().parent,
    Path("/kaggle/working/NL2Ms"),
    Path("/kaggle/input/nl2ms"),
    Path("/kaggle/usr/lib/nl2ms"),
]
for c in CANDIDATES:
    if (c / "nl2ms" / "__init__.py").exists():
        sys.path.insert(0, str(c))
        print(f"nl2ms found at: {c}")
        break
else:
    raise ImportError(
        "Could not locate the nl2ms package. Clone the repository into "
        "/kaggle/working, or attach it as a Kaggle dataset, then re-run."
    )

import nl2ms
print("nl2ms version:", nl2ms.__version__)
"""))

cells.append(code(r"""
from nl2ms.env import (capture_environment, summarise_environment,
                       detect_hardware, default_output_root, supports_bfloat16)

environment = capture_environment(seed=20260820, repo_dir=".")
print(summarise_environment(environment))
print()
print("bfloat16 usable here:", supports_bfloat16(),
      "(T4 is sm_75 -> float16 is the right choice)")
print("default output root:", default_output_root())
"""))

# ===========================================================================
cells.append(md(r"""
## Configuration

Everything that changes *what is measured* lives in `EXPERIMENT_CONFIG`. It is
serialised to `config/experiment_config.json` before anything runs, and its
hash guards the checkpoints: resuming with different measurement settings
recomputes rather than silently reusing results produced under other
definitions.

### Choosing the model

`MODEL_NAME` is the only model-specific setting. Layer count, hidden size,
head count, tokenizer and vocabulary are all discovered from the loaded
config — no analysis code below refers to any of them.

### Storage levels

| level | contents |
|---|---|
| 0 | metrics only |
| 1 | metrics + hidden states for a selected subset |
| 2 | hidden states at the analysed positions, all samples |
| 3 | level 2 + attention summaries *(recommended)* |
| 4 | full research dump (all positions, full vocab logits) |
"""))

cells.append(code(r"""
from nl2ms.config import ExperimentConfig, ModelConfig
from nl2ms.env import default_output_root

# --- the one setting that selects the model --------------------------------
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
# Alternatives, unchanged elsewhere in the notebook:
#   MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
#   MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"   # gated; needs a token

MODEL_REVISION = "main"     # pin to a commit hash for a citable experiment

EXPERIMENT_CONFIG = ExperimentConfig(
    experiment_name="phase_transitions_v1",
    output_root=default_output_root(),
    seed=20260820,
    storage_level=3,
    discovery_mode=True,
    pilot_n_samples=16,
)

EXPERIMENT_CONFIG.model = ModelConfig(
    name=MODEL_NAME,
    revision=MODEL_REVISION,
    dtype="float16",          # T4 has no usable bfloat16
    quantization="4bit",      # ~4 GB of weights instead of ~14 GB
    device_map="auto",        # shards across both T4s when present
    attn_implementation="eager",   # REQUIRED: sdpa cannot return attentions
)

# Pilot sample counts. Raise these for the full run (see the scaling cell).
EXPERIMENT_CONFIG.datasets.n_gsm8k = 150
EXPERIMENT_CONFIG.datasets.n_truthfulqa = 150
EXPERIMENT_CONFIG.datasets.n_winogrande = 150
EXPERIMENT_CONFIG.datasets.n_synthetic = 150

EXPERIMENT_CONFIG.generation.max_new_tokens = 96
EXPERIMENT_CONFIG.generation.temperature = 0.0    # deterministic primary run
EXPERIMENT_CONFIG.generation.batch_size = 4

EXPERIMENT_CONFIG.extraction.shard_size = 16
EXPERIMENT_CONFIG.extraction.save_full_vocab_logits = False   # storage guard
EXPERIMENT_CONFIG.extraction.max_generated_positions = 48

EXPERIMENT_CONFIG.runtime.max_runtime_hours = 11.2   # Kaggle allows ~12

print("config hash:", EXPERIMENT_CONFIG.config_hash())
print("samples requested:", EXPERIMENT_CONFIG.total_requested_samples())
print("storage flags:")
for k, v in EXPERIMENT_CONFIG.effective_flags().items():
    print(f"  {k:<32} {v}")
"""))

# ===========================================================================
cells.append(md(r"""
## Signal registry

Every measured quantity is declared with its formula, source tensor,
resolution and known caveat. The registry is written to
`config/signal_registry.json`, so a later reader can tell exactly what
`jsd_prev_layer` meant in *this* run without reading the code that made it.
"""))

cells.append(code(r"""
from nl2ms.registry import SIGNAL_REGISTRY, describe

print(f"{len(SIGNAL_REGISTRY)} signals registered\n")
for name, spec in list(SIGNAL_REGISTRY.items())[:6]:
    print(f"{name}")
    print(f"    formula : {spec.formula}")
    print(f"    caveat  : {spec.caveat or '(none recorded)'}")
    print()

# Any single signal can be inspected in full:
import json
print(json.dumps(describe("order_margin"), indent=2))
"""))

# ===========================================================================
cells.append(md(r"""
## Phase 1 — Dataset

Four sources with different reasoning demands, so that a transition found in
one can be checked against the others.

| source | kind | answer specification |
|---|---|---|
| GSM8K | multi-step arithmetic word problems | `open_vocab` (numeric target) |
| TruthfulQA | truthfulness, multiple choice | `closed_set` |
| Winogrande | pronoun resolution, binary | `closed_set` |
| synthetic | controlled, seven categories | `closed_set` / `undefined` |

### The answer specification matters

The order parameter is only well posed when the candidate answers are. Three
kinds are tracked and **never mixed**:

- **`closed_set`** — enumerated candidates. `m_l` is computed on the
  distribution restricted to them and renormalised.
- **`open_vocab`** — one correct target token, no distractors (GSM8K).
  `m_l = p(correct) − max_{v≠correct} p(v)`, on a different scale entirely.
- **`undefined`** — the deliberately ambiguous synthetic items. No correct
  answer exists, so no margin or accuracy statistic is ever computed from
  them; they serve as an entropy/geometry control.

Sample identity is content-addressed (a hash of the question), never a row
index, so an example keeps its ID across reruns and reshuffles.
"""))

cells.append(code(r"""
from nl2ms.datasets_build import generate_synthetic, dataset_summary

# The synthetic set is deterministic from its seed and needs no network,
# so it always works even when the hub is unreachable.
demo = generate_synthetic(14, seed=EXPERIMENT_CONFIG.datasets.synthetic_seed)
print(dataset_summary(demo))
print()
for s in demo[:3]:
    print(f"[{s.subset}]  id={s.sample_id}")
    print(f"  {s.question.splitlines()[0]}")
    print(f"  candidates : {[(c.label, c.text, c.is_correct) for c in s.candidates]}")
    print(f"  spec       : {s.answer_spec_type}   ground truth: {s.ground_truth}")
    print()

ambiguous = [s for s in demo if s.answer_spec_type == "undefined"]
if ambiguous:
    a = ambiguous[0]
    print("Deliberately ambiguous control item (no correct answer by construction):")
    print(" ", a.question.replace("\n", "\n  "))
"""))

# ===========================================================================
cells.append(md(r"""
## Model loading and pathway verification

Two things are verified before any measurement is trusted.

### 1. The residual stream is captured by hook, not by `output_hidden_states`

In the Llama/Mistral/Qwen family, `hidden_states[-1]` has **already had the
final normalisation applied**, while entries `0..L-1` are raw residual
stream. Applying the final norm uniformly to all of them double-normalises
the last one and manufactures an artefactual jump at exactly the place this
experiment is looking for a transition. We therefore hook the blocks
directly, and the layout is verified at load time.

The depth-0 row is taken from **block 0's input**, not the embedding table's
output: GPT-2-style models add positional embeddings in between, Llama-style
models do not, and block-0 input is correct for both.

### 2. The logit lens reproduces the model's own logits

We check that

$$\mathrm{lm\_head}\big(\mathrm{FinalNorm}(h_L)\big) = \text{model logits}$$

to floating-point tolerance. If this fails, the discovered pathway is wrong
and every lens number would be meaningless — the run continues but flags the
result loudly.
"""))

cells.append(code(r"""
from nl2ms.pipeline import Experiment

experiment = Experiment(EXPERIMENT_CONFIG)
environment = experiment.phase0_environment()
"""))

cells.append(code(r"""
# Loads the model once. Nothing else in this notebook loads a second copy.
model = experiment.load()

print()
print("Discovered architecture (nothing here is hard-coded):")
arch = model.arch
for field in ["model_type", "n_layers", "hidden_size", "n_heads", "n_kv_heads",
              "head_dim", "vocab_size", "tie_word_embeddings",
              "layer_module_path", "final_norm_path", "final_norm_type",
              "lm_head_path", "logit_lens_transform"]:
    print(f"  {field:<24} {getattr(arch, field)}")

print()
print("Logit-lens caveats recorded with the architecture:")
for note in arch.logit_lens_notes:
    print(f"  - {note}")
"""))

cells.append(code(r"""
# Multi-GPU placement report (protocol section 56).
from nl2ms.env import print_gpu_report
print_gpu_report()
print()
if getattr(model.model, "hf_device_map", None):
    from collections import Counter
    placement = Counter(str(v) for v in model.model.hf_device_map.values())
    print("module placement:", dict(placement))
else:
    print("single-device placement:", model.device)
"""))

# ===========================================================================
cells.append(md(r"""
## Storage and runtime planning

Before extraction begins, the expected disk usage is estimated and compared
against what is actually free. If the run cannot fit, it stops **here**
rather than filling the disk halfway through and losing the session.

The runtime table is empty on a cold start (nothing has been measured yet)
and fills in from real throughput once the pilot has run — the estimates are
measured, never guessed.
"""))

cells.append(code(r"""
plan = experiment.plan(EXPERIMENT_CONFIG.total_requested_samples())
"""))

# ===========================================================================
cells.append(md(r"""
## Pilot run — mandatory

A small run that exercises **every** code path: generation, hidden-state
extraction, logit lens, entropy, attention, geometry, JSD, trajectory
metrics, J-space, interventions, checkpointing, figures and the report.

A failure in, say, attention summarisation should surface after a couple of
minutes, not after six hours of extraction.
"""))

cells.append(code(r"""
from nl2ms.run import run_pilot, validate_pilot

pilot = run_pilot(EXPERIMENT_CONFIG)
pilot_validation = validate_pilot(pilot)

assert pilot_validation["passed"], (
    "Pilot components failed to validate; investigate before the full run. "
    f"Failed: {pilot_validation['failed']}"
)
"""))

cells.append(code(r"""
# Measured throughput from the pilot -> a real estimate for the full run.
from nl2ms.runtime import _fmt_duration

pilot_exp = pilot["experiment"]
status = pilot_exp.controller.status()
n_full = EXPERIMENT_CONFIG.total_requested_samples()

print("measured throughput (from the pilot, not assumed):")
total = 0.0
for phase, stats in sorted(status["throughput"].items()):
    spu = stats.get("seconds_per_unit")
    if not spu:
        continue
    scale = {"generation": n_full, "analysis": n_full,
             "jspace": min(n_full, EXPERIMENT_CONFIG.jspace.max_samples),
             "intervention": min(n_full, EXPERIMENT_CONFIG.interventions.max_samples)}
    n = scale.get(phase, n_full)
    est = spu * n
    total += est
    print(f"  {phase:<14} {spu:7.2f} s/sample  x {n:>5} = {_fmt_duration(est)}")
print(f"\n  estimated total for {n_full} samples: {_fmt_duration(total)}")
print(f"  Kaggle budget: {EXPERIMENT_CONFIG.runtime.max_runtime_hours} h")
if total > EXPERIMENT_CONFIG.runtime.max_runtime_hours * 3600 * 0.8:
    print("\n  WARNING: this will not finish in one session. That is safe -- "
          "the run checkpoints and resumes -- but plan on 2+ sessions, or "
          "reduce the sample counts.")
"""))

# ===========================================================================
cells.append(md(r"""
## Resume test — mandatory

Kaggle sessions die. The whole design assumes it. This cell proves
resumability rather than asserting it:

1. process a few samples and checkpoint,
2. destroy the experiment object and rebuild it from disk,
3. verify the completed samples are **skipped**,
4. verify work continues from the next incomplete sample,
5. verify no completed checkpoint was overwritten.

`RESUME TEST PASSED` prints only if all four hold.
"""))

cells.append(code(r"""
from nl2ms.run import resume_test
import copy

resume_cfg = copy.deepcopy(EXPERIMENT_CONFIG)
resume_cfg.output_root = str(Path(EXPERIMENT_CONFIG.output_root).parent /
                             "experiment_resume_test")
resume_cfg.pilot_n_samples = 8
resume_cfg.generation.max_new_tokens = 24

resume_result = resume_test(resume_cfg, n_first=3)
assert resume_result["passed"], "Resumability is broken; do not start a long run."
"""))

# ===========================================================================
cells.append(md(r"""
## Full experiment

Runs every phase in order. **Safe to re-run**: each phase consults the
manifest first, so calling this again after an interrupted session picks up
where it stopped.

If the wall-clock budget runs out, the run stops cleanly with

```
SAFE STOP: checkpoint complete. Resume notebook to continue.
```

and re-running this cell in a new session continues from that point.

### Phase order

| phases | what happens |
|---|---|
| 0–1 | environment, dataset |
| 2 | baseline generation, saved immediately |
| 3–10 | residual stream, logit lens, entropy, attention, JSD, trajectory, symmetry breaking |
| 22–23 | J-space local sensitivity |
| 11, 26–27 | critical-layer and critical-region detection |
| 7, 18–20 | population latent geometry |
| 13, 31–32 | causal intervention + control battery |
| 47–48 | master tables |
| 12, 14, 29, 34–35 | statistics, nulls, confounds |
| 15 | figures |
| 16, 49, 50, 64 | manifest, integrity check, final report |
"""))

cells.append(code(r"""
from nl2ms.run import run_experiment

result = run_experiment(EXPERIMENT_CONFIG)

if result.get("stopped") == "budget":
    print("\nThe run stopped to protect the finalisation window.")
    print("Re-run this cell in a fresh session to continue from the checkpoint.")
"""))

# ===========================================================================
cells.append(md(r"""
## Results

Everything below reads from the artefacts on disk, so these cells work in a
fresh session without re-running the experiment.
"""))

cells.append(code(r"""
import pandas as pd

df = result.get("sample_summary")
if df is None:
    df = pd.read_parquet(Path(EXPERIMENT_CONFIG.output_root) / "sample_summary.parquet")

print(f"sample_summary: {len(df)} rows x {len(df.columns)} columns")
print()
graded = df["correct"].dropna()
print(f"graded: {len(graded)}   accuracy: "
      f"{graded.mean():.1%}" if len(graded) else "no graded samples")
print(f"ungraded (unparsed or no ground truth): {df['correct'].isna().sum()}")
print()
print("accuracy by dataset:")
print(df.groupby("dataset")["correct"].agg(["count", "mean"]))
"""))

cells.append(code(r"""
# Critical-layer location, and how often the detectors actually agreed.
cons_col = "critical_layer_consensus_normalised"
n_total = len(df)
n_cons = int(df[cons_col].notna().sum()) if cons_col in df else 0

print(f"samples with a detector consensus : {n_cons} / {n_total}")
print(f"samples WITHOUT a consensus       : {n_total - n_cons}")
print()
if n_cons:
    vals = df[cons_col].dropna()
    print(f"consensus critical layer (l/L): mean {vals.mean():.3f}  "
          f"median {vals.median():.3f}  SD {vals.std():.3f}")
print()
print("dominant transition shape across samples:")
if "dominant_transition_shape" in df:
    print(df["dominant_transition_shape"].value_counts())
"""))

cells.append(code(r"""
# The calibration that makes the shape counts interpretable.
from nl2ms.stats import detector_false_positive_rate

n_layers = model.arch.n_layers + 1
fp = detector_false_positive_rate(max(20, len(df)), n_layers, seed=0, n_repeats=15)
print("On structureless AR(1) curves of the same depth, the same classifier "
      "labels:")
for shape, n in sorted(fp["shape_counts"].items(), key=lambda kv: -kv[1]):
    print(f"  {shape:<14} {n / fp['n_curves']:.1%}")
print()
print(f"false 'sharp' rate = {fp['false_sharp_rate']:.1%}")
print("Any observed sharp fraction must be read against this floor.")
"""))

cells.append(code(r"""
# The decisive comparison: is the critical layer causally privileged?
import json
from pathlib import Path

stats_dir = Path(EXPERIMENT_CONFIG.output_root) / "statistics"
summary_path = Path(EXPERIMENT_CONFIG.output_root) / "reports" / "experiment_summary.json"
summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

causal = summary.get("causal", {})
if causal.get("status") != "ok":
    print("causal intervention:", causal.get("status", "not run"))
elif causal.get("cohens_d_critical_vs_random") is None:
    # Happens when no layer was left over to serve as a random control --
    # only possible on very shallow models. Reported, never imputed.
    print("The critical-vs-random comparison could not be formed.")
    print("  reason:", causal.get("comparison_status", "unknown"))
    print(f"  outcomes recorded: {causal.get('n_outcomes')} across "
          f"{causal.get('n_samples')} samples")
    print("\nWithout this comparison the run supports no causal claim.")
else:
    print(f"mean output JSD, critical layer      : "
          f"{causal['critical_mean']:.6f}")
    print(f"mean output JSD, random control layer: "
          f"{causal['random_control_mean']:.6f}")
    print(f"effect size (critical vs random)     : "
          f"{causal['cohens_d_critical_vs_random']:.4f}")
    print(f"p (corrected)                        : "
          f"{causal.get('p_value_corrected')}")
    print(f"n critical / n control               : "
          f"{causal.get('n_critical')} / {causal.get('n_random_control')}")
    print()
    d = causal["cohens_d_critical_vs_random"]
    if abs(d) < 0.2:
        print("The candidate critical layer is NOT more perturbation-sensitive "
              "than a random layer at matched magnitude. That is evidence "
              "against a causally privileged critical layer, whatever the "
              "correlational profiles show.")
    else:
        print("The critical layer shows a larger perturbation response than a "
              "matched random layer. Read this together with the "
              "perturbation-kind breakdown in figure 14: a generic "
              "disruption effect and an answer-direction-specific effect "
              "look identical in this single number.")
"""))

cells.append(code(r"""
# Null-model outcomes: does the signal beat a structureless baseline?
nulls = summary.get("null_models", {})
for name, res in sorted(nulls.items()):
    if not isinstance(res, dict) or res.get("status") != "ok":
        continue
    if name == "detector_false_positive":
        continue
    print(f"{name}")
    print(f"    observed {res.get('observed', res.get('observed_sharpness'))}"
          f"  null p95 {res.get('null_p95')}"
          f"  p {res.get('p_value')}"
          f"  exceeds null: {res.get('exceeds_null')}")
"""))

cells.append(code(r"""
# Evidence AGAINST the hypothesis, generated from this run's own numbers.
for item in summary.get("evidence_against", []) or ["(none recorded)"]:
    print("-", item)
"""))

cells.append(code(r"""
# Display the generated figures inline.
from IPython.display import Image, display, Markdown

figdir = Path(EXPERIMENT_CONFIG.output_root) / "figures"
for png in sorted(figdir.glob("figure_*.png")):
    display(Markdown(f"### {png.stem}"))
    display(Image(filename=str(png)))
"""))

cells.append(code(r"""
# The final report.
from IPython.display import Markdown, display

report_path = Path(EXPERIMENT_CONFIG.output_root) / "reports" / "FINAL_REPORT.md"
display(Markdown(report_path.read_text()))
"""))

cells.append(code(r"""
# Integrity check summary.
integrity = result.get("integrity") or json.loads(
    (Path(EXPERIMENT_CONFIG.output_root) / "reports" / "integrity_report.json").read_text())

print(f"integrity: {'PASSED' if integrity['passed'] else 'PROBLEMS FOUND'}")
print(f"  problems : {len(integrity['problems'])}")
print(f"  warnings : {len(integrity['warnings'])}")
print()
for name, chk in integrity["checks"].items():
    print(f"  [{'OK ' if chk['ok'] else 'BAD'}] {name}")
print()
print(f"completed {integrity['n_completed']}, failed {integrity['n_failed']}, "
      f"not reached {integrity['n_skipped']}")
print(f"storage: {integrity['storage_gb']:.2f} GB")
"""))

# ===========================================================================
cells.append(md(r"""
## Scaling up

The pilot settings above use 150 examples per source. To scale:

```python
EXPERIMENT_CONFIG.datasets.n_gsm8k = 500       # etc.
```

Before scaling, re-run `experiment.plan(...)` — it will refuse to start a run
that does not fit on disk, and the runtime table (now populated with measured
throughput) shows whether it fits in the session budget.

**Sessions are cheap; lost work is not.** Because every phase checkpoints, the
right way to run 1000+ samples is simply to run this notebook repeatedly: each
session continues from the last checkpoint until the run completes.

### Adding a second model

Point `MODEL_NAME` at a different model and set `output_root` to a *sibling*
directory under the same parent:

```python
EXPERIMENT_CONFIG.model.name = "mistralai/Mistral-7B-Instruct-v0.3"
EXPERIMENT_CONFIG.output_root = "/kaggle/working/experiment_mistral"
```

Figure 16 then compares the models on normalised depth `l/L` automatically —
the only depth coordinate that is meaningful across models of different sizes.
"""))

cells.append(code(r"""
# Optional: secondary stochastic experiment (protocol section 6).
# The primary run is greedy and deterministic; this measures how much of the
# structure survives sampling.
#
# import copy
# stoch = copy.deepcopy(EXPERIMENT_CONFIG)
# stoch.experiment_name += "_temp07"
# stoch.output_root = str(Path(EXPERIMENT_CONFIG.output_root).parent /
#                         "experiment_temp07")
# stoch.generation.temperature = 0.7
# stoch.generation.do_sample = True
# stochastic_result = run_experiment(stoch)
print("stochastic replication is disabled by default; uncomment to run")
"""))

cells.append(code(r"""
# Where everything ended up.
root = Path(EXPERIMENT_CONFIG.output_root)
print(f"experiment root : {root}")
print(f"final report    : {root / 'reports' / 'FINAL_REPORT.md'}")
print(f"manifest        : {root / 'experiment_manifest.json'}")
print(f"master table    : {root / 'sample_summary.parquet'}")
print(f"layer table     : {root / 'layer_summary.parquet'}")
print(f"backup archive  : {result.get('backup_path')}")
print()
print("Download the backup archive before the session ends -- it holds the "
      "configuration, manifests, derived metrics, figures and reports "
      "(raw tensors are excluded to keep it small).")
"""))

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
    "nl2ms": {
        "purpose": "measurement instrument for phase-transition-like behaviour "
                   "in transformer reasoning dynamics",
        "designed_for": "Kaggle, 1-2x NVIDIA T4, ~12h session limit",
    },
}

out = Path(__file__).parent / "notebooks" / "phase_transitions.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(out))
print(f"wrote {out} ({len(cells)} cells: "
      f"{sum(1 for c in cells if c.cell_type == 'markdown')} markdown, "
      f"{sum(1 for c in cells if c.cell_type == 'code')} code)")
