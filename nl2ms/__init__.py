"""nl2ms -- an instrument for measuring whether transformer reasoning contains
localized critical transitions in representation dynamics.

The package is deliberately structured so that every phase of the protocol is
an importable, testable function rather than a notebook cell. The notebook in
``notebooks/`` is a thin driver over this code.

Module map
----------
``config``          experiment configuration (the single source of settings)
``env``             Phase 0: environment capture, seeding, GPU discovery
``storage``         atomic IO, shards, manifests, checkpoint recovery
``runtime``         wall-clock budget, throughput, heartbeats
``datasets_build``  Phase 1: benchmark mixture + controlled synthetic set
``modeling``        architecture-agnostic model wrapper and answer specs
``hooks``           residual-stream read/write primitives
``extraction``      Phases 2-10: the per-sample measurement pipeline
``logit_lens``      Phase 4: layer-wise vocabulary projection
``signals``         pure-NumPy metric definitions (entropy, JSD, dynamics)
``geometry``        Phases 7/18-20: latent-space geometry
``attention``       Phases 6/17: attention summaries and restructuring
``jspace``          Phases 22/23: local sensitivity descriptors
``critical``        Phases 11/26/27: critical-layer and -region detection
``interventions``   Phases 13/31/32: causal perturbation and controls
``stats``           Phases 14/33-35: effect sizes, nulls, confounds
``plots``           Phase 15: figures with lineage sidecars
``registry``        Phase 60: signal registry and data lineage
``report``          Phases 16/49/50/64: manifest, integrity, final report
``pipeline``        phase sequencing with checkpoint/resume
``analysis``        aggregation, master tables, statistics assembly
``run``             entry points: pilot, full run, resume test
"""

__version__ = "1.0.0"

from .config import (ExperimentConfig, ModelConfig, DatasetConfig,
                     GenerationConfig, ExtractionConfig, GeometryConfig,
                     JSpaceConfig, InterventionConfig, StatsConfig,
                     RuntimeConfig, pilot_config, smoke_config,
                     STORAGE_LEVELS, EXTRACTION_MODES)

__all__ = [
    "ExperimentConfig", "ModelConfig", "DatasetConfig", "GenerationConfig",
    "ExtractionConfig", "GeometryConfig", "JSpaceConfig", "InterventionConfig",
    "StatsConfig", "RuntimeConfig", "pilot_config", "smoke_config",
    "STORAGE_LEVELS", "EXTRACTION_MODES", "run_experiment", "run_pilot",
    "resume_test", "validate_pilot", "Experiment",
]


def __getattr__(name: str):
    """Lazy re-exports so ``import nl2ms`` does not pull in torch."""
    if name in ("run_experiment", "run_pilot", "resume_test", "validate_pilot"):
        from . import run
        return getattr(run, name)
    if name == "Experiment":
        from .pipeline import Experiment
        return Experiment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
