# Validated Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deliver reproducible Kaggle and CPU notebooks covering the audit's repairs and new experiments.
**Architecture:** Isolated v2 package, atomic per-unit storage, supervised worker with 11-hour wall-clock ceiling, mounted prior-session indexes, and CPU-only analysis.
**Tech Stack:** Python, PyTorch, Transformers, Hugging Face datasets/hub, NumPy/SciPy/pandas, nbformat, pytest.
**Spec:** docs/superpowers/specs/2026-10-21-validated-experiments-design.md

## Global Constraints
- No original raw-output changes, fabricated results or synthetic benchmark fallback.
- Long run must refuse invalid preflight; deadline and disk stops remain incomplete.
- Save raw failed tensors and provenance without sanitizing nonfinite values.
- A benchmark source/model revision/source-code change must invalidate reuse.
- Kaggle GPU 7B tests cannot be claimed from CPU tiny-model tests.

## Review Focus
- Kill or interruption during array writes must leave no falsely completed unit.
- Stale mounted config/code/dataset/model must never count as completed current work.
- A single long generation must be bounded by parent supervisor independently of CUDA callbacks.
- Token alignment must use final-answer spans and causal predecessor, not incidental digits.
- Behavioral interventions must share baseline context, direction, norm and random seed; count samples rather than rows.

### Task 1: Storage, configuration and measurement contracts
Files: nl2ms/v2/{config,store,protocol}.py; tests/test_v2.py.
- [ ] Write and run failing tests for atomic recovery, hashes, budgets, strict parsing, coordinate roles, nonfinite rejection.
- [ ] Implement smallest interfaces: StudyConfig, EvidenceStore, Budget, parse_final_answer, select_positions, layer_plan.
- [ ] Run `python -m pytest tests/test_v2.py -q`; require pass.

### Task 2: GPU collection and experiments
Files: nl2ms/v2/{data,model,experiments,runner}.py; tests/test_v2_model.py.
- [ ] Write failing tests for real tiny-model capture, score identities, no-op edits, incremental resume and preflight refusal.
- [ ] Implement pinned real loaders, raw capture, gated run scheduling, matched full-decoding interventions and J-space units.
- [ ] Execute CPU tiny-architecture suite and separately a real pretrained-model / real-dataset smoke if downloads available.

### Task 3: Local analysis and self-contained notebooks
Files: nl2ms/v2/analysis.py; build_validated_notebooks.py; notebooks/*v2.ipynb; configs/v2*.json; docs/VALIDATED_EXPERIMENTS.md.
- [ ] Write failing analytic/reference and notebook portability tests.
- [ ] Implement stratified summaries/nulls/paired inference, diagrams and exports; embed source archive and generate separate GPU/local notebooks.
- [ ] Run notebook validation, execute local notebook against recorded test fixture evidence explicitly labeled engineering-only, test short watchdog termination with a real subprocess.

### Task 4: Verification and review
- [ ] Run full repository pytest suite; record failures/skips accurately.
- [ ] Fresh reviewer checks spec, code, notebooks and failure modes.
- [ ] Fix substantive findings with regression tests; regenerate payload after every source change.
- [ ] Deliver notebooks, source bundle, commands, exact test evidence and hardware limitations.
