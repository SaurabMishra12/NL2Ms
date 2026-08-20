# Example configurations

```python
from nl2ms.config import ExperimentConfig
cfg = ExperimentConfig.load("configs/example_config.json")
```

`example_config.json` is the pilot-scale primary run: 150 examples from each
of four sources (600 total), deterministic greedy decoding, storage level 3.

## Scaling up

Change only the sample counts; nothing else needs to move.

| scale | per source | total | notes |
|---|---|---|---|
| pilot | 150 | 600 | fits comfortably in one session |
| medium | 500 | 2000 | plan on 2–3 sessions; it resumes |
| large | 1000 | 4000 | 4+ sessions; consider `storage_level = 2` |

Always re-run `experiment.plan(n)` after changing counts. It refuses to start
a run that will not fit on disk, and once the pilot has measured throughput,
the runtime table tells you whether it fits in the session budget.

## Common variations

**Second model for the cross-model comparison.** Set a *sibling* output root
so Figure 16 finds both runs:

```python
cfg.model.name = "mistralai/Mistral-7B-Instruct-v0.3"
cfg.output_root = "/kaggle/working/experiment_mistral"
```

**No bitsandbytes available.** Set `quantization: null`. With `device_map:
"auto"` and two T4s the 7B weights shard across both GPUs in fp16; on a single
T4 you will need a smaller model.

**Single T4.** Keep `quantization: "4bit"` and lower
`generation.batch_size` to 2 if generation runs out of memory.

**Storage pressure.** Drop to `storage_level: 2` (loses attention summaries,
so the `attention_restructuring` detector goes unavailable and is reported as
such) or reduce `extraction.max_generated_positions`.

**Faster iteration while developing.** Lower `stats.n_bootstrap` and
`stats.n_permutation` to 200, and `jspace.max_samples` /
`interventions.max_samples` to 8. Restore them before any run you intend to
report — the permutation p-value floor is `1/(n_permutation+1)`.

**Stochastic replication.** Set `generation.temperature: 0.7`,
`generation.do_sample: true` and a distinct `output_root`. The primary run
should stay deterministic.

## What must not change mid-run

The configuration hash guards the checkpoints. Changing anything that affects
*what is measured* — model, datasets, generation, extraction, geometry,
jspace, interventions, stats, seed, storage level — invalidates the completed
phases and they recompute.

Changing `runtime` settings or `notes` does **not** change the hash, so you can
adjust the wall-clock budget between sessions and keep every checkpoint. That
asymmetry is deliberate: session length is not a measurement decision.
