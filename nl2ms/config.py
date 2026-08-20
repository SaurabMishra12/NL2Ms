"""Central experiment configuration.

Every knob that changes what the experiment measures lives here so that a
single JSON dump (``config/experiment_config.json``) fully describes a run.

Design rule: nothing downstream may read a hard-coded constant that is not
reachable from :class:`ExperimentConfig`.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Storage levels
# ---------------------------------------------------------------------------
# 0: metrics only (no raw hidden states persisted)
# 1: metrics + hidden states for a small selected subset of samples
# 2: metrics + hidden states for the configured token positions, all samples
# 3: level 2 + attention summary statistics (+ full attention for a few samples)
# 4: full research dump (all token positions, full attention for selected)
STORAGE_LEVELS = {
    0: "metrics_only",
    1: "metrics_plus_selected_hidden",
    2: "hidden_selected_positions_all_samples",
    3: "hidden_plus_attention_summaries",
    4: "full_research_dump",
}

# Token-position extraction modes (section 10 of the protocol).
EXTRACTION_MODES = {
    "A": "final_generated_token",
    "B": "last_input_token",
    "C": "every_generated_token",
    "D": "question_token_pooled",
    "E": "explicit_positions",
}


@dataclass
class ModelConfig:
    """Identity + placement of the decoder-only model under study.

    Nothing here describes architecture (layers/heads/dims): those are
    discovered from ``AutoConfig`` at load time. Recording ``revision``
    matters because HF main branches move.
    """

    name: str = "Qwen/Qwen2.5-7B-Instruct"
    revision: str = "main"
    tokenizer_name: Optional[str] = None  # defaults to ``name``
    tokenizer_revision: Optional[str] = None  # defaults to ``revision``
    dtype: str = "float16"  # float16 | bfloat16 | float32
    quantization: Optional[str] = "4bit"  # None | "4bit" | "8bit"
    device_map: str = "auto"
    trust_remote_code: bool = False
    attn_implementation: str = "eager"  # eager required for attention outputs
    max_memory: Optional[Dict[str, str]] = None  # e.g. {"0": "13GiB", "1": "13GiB"}

    def resolved_tokenizer_name(self) -> str:
        return self.tokenizer_name or self.name

    def resolved_tokenizer_revision(self) -> str:
        return self.tokenizer_revision or self.revision


@dataclass
class DatasetConfig:
    """How many examples to draw from each source, and from where."""

    n_gsm8k: int = 150
    n_truthfulqa: int = 150
    n_winogrande: int = 150
    n_synthetic: int = 150
    gsm8k_split: str = "test"
    truthfulqa_split: str = "validation"
    winogrande_split: str = "validation"
    winogrande_subset: str = "winogrande_xl"
    synthetic_seed: int = 20260820
    # If HF hub datasets are unreachable (Kaggle offline), fall back to the
    # bundled synthetic generator only and record that fact in the manifest.
    allow_hub_download: bool = True
    local_dataset_dir: Optional[str] = None


@dataclass
class GenerationConfig:
    """Primary (deterministic) generation settings + secondary stochastic run."""

    max_new_tokens: int = 96
    temperature: float = 0.0  # 0.0 => greedy, deterministic primary experiment
    top_p: float = 1.0
    do_sample: bool = False
    seed: int = 1234
    # Secondary stochastic experiment (Phase 2b). Empty list disables it.
    stochastic_temperatures: List[float] = field(default_factory=lambda: [0.7])
    n_stochastic_samples: int = 0  # per example; 0 disables
    batch_size: int = 4


@dataclass
class ExtractionConfig:
    """What internal state gets read out, and how it is sharded to disk."""

    # Default research mode: final input position + every generated token
    # (which includes the final answer position).
    modes: List[str] = field(default_factory=lambda: ["B", "C"])
    explicit_positions: List[int] = field(default_factory=list)  # for mode E
    max_generated_positions: int = 48  # cap tokens kept per sample
    hidden_dtype: str = "float16"  # raw storage dtype
    metric_dtype: str = "float32"  # dtype used for numerically sensitive maths
    shard_size: int = 16  # samples per hidden-state shard
    save_hidden_states: bool = True
    save_attention: bool = True
    save_full_attention_for_n_samples: int = 4
    save_full_vocab_logits: bool = False
    logit_lens_top_k: int = 20
    # Residual-stream decomposition (Phase 3b). Costs an extra set of hooks.
    capture_residual_decomposition: bool = True


@dataclass
class GeometryConfig:
    """Latent-geometry and local-geometry parameters (Phases 7 / 19 / 20)."""

    max_samples_for_pairwise: int = 256
    knn_k: int = 10
    pca_max_components: int = 64
    effective_rank_eps: float = 1e-12
    local_cov_neighbours: int = 16
    intrinsic_dim_enabled: bool = True


@dataclass
class JSpaceConfig:
    """Local-sensitivity (Jacobian-vector-product) descriptor settings."""

    enabled: bool = True
    n_probe_directions: int = 8
    probe_seed: int = 7
    epsilons: List[float] = field(default_factory=lambda: [1e-3, 1e-2, 1e-1])
    max_samples: int = 64  # J-space is expensive; run on a subset
    relative_perturbation: bool = True  # scale eps by ||h||


@dataclass
class InterventionConfig:
    """Causal-intervention settings (Phases 13 / 31 / 32)."""

    enabled: bool = True
    max_samples: int = 48
    # Layer selection is relative to the detected critical layer per sample.
    layer_offsets: List[int] = field(default_factory=lambda: [-4, -2, -1, 0, 1, 2, 4])
    include_absolute_layers: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    epsilons: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    # Perturbation families; each is applied with matched norm.
    perturbation_kinds: List[str] = field(
        default_factory=lambda: [
            "gaussian",
            "orthogonal_to_answer_dir",
            "along_answer_dir",
            "pca_direction",
            "cross_sample",
        ]
    )
    control_random_layers: int = 2
    intervention_max_new_tokens: int = 48


@dataclass
class StatsConfig:
    n_bootstrap: int = 2000
    n_permutation: int = 2000
    alpha: float = 0.05
    multiple_comparison_method: str = "fdr_bh"
    bootstrap_seed: int = 99
    min_group_size: int = 5


@dataclass
class RuntimeConfig:
    max_runtime_hours: float = 11.2
    heartbeat_seconds: float = 120.0
    reserve_minutes_for_finalisation: float = 25.0
    min_free_disk_gb: float = 3.0
    oom_retry_attempts: int = 2
    max_sample_failures: int = 3


@dataclass
class ExperimentConfig:
    """Top-level configuration object; serialised before anything runs."""

    experiment_name: str = "phase_transitions_v1"
    output_root: str = "/kaggle/working/experiment"
    seed: int = 20260820
    storage_level: int = 3
    discovery_mode: bool = True
    pilot_n_samples: int = 16
    run_pilot_first: bool = True
    run_resume_test: bool = True

    model: ModelConfig = field(default_factory=ModelConfig)
    datasets: DatasetConfig = field(default_factory=DatasetConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    jspace: JSpaceConfig = field(default_factory=JSpaceConfig)
    interventions: InterventionConfig = field(default_factory=InterventionConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    prompt_template_id: str = "plain_qa_v1"
    notes: str = ""

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def config_hash(self) -> str:
        """Stable hash of the configuration, used to detect config drift.

        Runtime-only fields are excluded: changing the wall-clock budget or
        heartbeat cadence does not change what is being measured, so it must
        not invalidate existing checkpoints.
        """
        payload = self.to_dict()
        payload.pop("runtime", None)
        payload.pop("notes", None)
        payload.pop("output_root", None)
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        payload["_config_hash"] = self.config_hash()
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str))
        tmp.replace(path)
        assert path.exists() and path.stat().st_size > 0, f"config write failed: {path}"
        return path

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ExperimentConfig":
        payload = copy.deepcopy(payload)
        # Underscore-prefixed keys are metadata, not settings: the saved
        # ``_config_hash`` and any ``_comment`` block a human added to a
        # hand-written config file.
        payload = {k: v for k, v in payload.items() if not k.startswith("_")}
        nested = {
            "model": ModelConfig,
            "datasets": DatasetConfig,
            "generation": GenerationConfig,
            "extraction": ExtractionConfig,
            "geometry": GeometryConfig,
            "jspace": JSpaceConfig,
            "interventions": InterventionConfig,
            "stats": StatsConfig,
            "runtime": RuntimeConfig,
        }
        kwargs: Dict[str, Any] = {}
        for key, value in payload.items():
            if key in nested:
                kwargs[key] = nested[key](**value)
            else:
                kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    # ------------------------------------------------------------------
    def effective_flags(self) -> Dict[str, bool]:
        """Resolve storage level into concrete save flags.

        The storage level is the coarse dial the operator turns; these flags
        are what the extraction code actually reads.
        """
        lvl = self.storage_level
        return {
            "save_hidden_states": bool(self.extraction.save_hidden_states and lvl >= 1),
            "hidden_all_samples": lvl >= 2,
            "save_attention_summaries": bool(self.extraction.save_attention and lvl >= 3),
            "save_full_attention": bool(self.extraction.save_attention and lvl >= 3),
            "save_all_token_positions": lvl >= 4,
            "save_full_vocab_logits": bool(self.extraction.save_full_vocab_logits and lvl >= 4),
        }

    def total_requested_samples(self) -> int:
        d = self.datasets
        return d.n_gsm8k + d.n_truthfulqa + d.n_winogrande + d.n_synthetic


def pilot_config(base: Optional[ExperimentConfig] = None) -> ExperimentConfig:
    """A cheap configuration used for the mandatory pilot pass.

    Small counts, short generations, everything enabled so that each code
    path is exercised before the expensive run starts.
    """
    cfg = copy.deepcopy(base) if base is not None else ExperimentConfig()
    cfg.experiment_name = f"{cfg.experiment_name}_pilot"
    cfg.datasets.n_gsm8k = 5
    cfg.datasets.n_truthfulqa = 5
    cfg.datasets.n_winogrande = 5
    # Population geometry (covariance, effective rank, kNN structure) is
    # undefined below a handful of points, so the pilot must carry enough
    # samples to exercise that code path -- otherwise the pilot silently skips
    # the very phase most likely to break at scale.
    cfg.datasets.n_synthetic = max(12, cfg.datasets.n_synthetic if
                                   cfg.datasets.n_synthetic < 12 else 12)
    cfg.pilot_n_samples = max(12, cfg.pilot_n_samples)
    cfg.generation.max_new_tokens = 32
    cfg.extraction.shard_size = 4
    cfg.extraction.max_generated_positions = 16
    cfg.extraction.save_full_attention_for_n_samples = 2
    cfg.jspace.max_samples = 8
    cfg.interventions.max_samples = 4
    cfg.interventions.epsilons = [1.0]
    cfg.stats.n_bootstrap = 200
    cfg.stats.n_permutation = 200
    cfg.storage_level = 3
    cfg.run_pilot_first = False
    return cfg


def smoke_config(model_name: str, output_root: str) -> ExperimentConfig:
    """Minimal configuration for CPU smoke tests against a tiny model."""
    cfg = ExperimentConfig(
        experiment_name="smoke",
        output_root=output_root,
        storage_level=3,
    )
    cfg.model = ModelConfig(
        name=model_name,
        dtype="float32",
        quantization=None,
        device_map=None,
        attn_implementation="eager",
    )
    cfg.datasets = DatasetConfig(
        n_gsm8k=0, n_truthfulqa=0, n_winogrande=0, n_synthetic=6,
        allow_hub_download=False,
    )
    cfg.generation = GenerationConfig(max_new_tokens=8, batch_size=2)
    cfg.extraction = ExtractionConfig(
        shard_size=3,
        max_generated_positions=8,
        save_full_attention_for_n_samples=1,
    )
    cfg.geometry = GeometryConfig(max_samples_for_pairwise=32, knn_k=3, pca_max_components=8,
                                  local_cov_neighbours=4)
    cfg.jspace = JSpaceConfig(n_probe_directions=3, max_samples=4, epsilons=[1e-2])
    cfg.interventions = InterventionConfig(
        max_samples=3, layer_offsets=[-1, 0, 1], include_absolute_layers=[0.5],
        epsilons=[1.0], control_random_layers=1, intervention_max_new_tokens=6,
    )
    cfg.stats = StatsConfig(n_bootstrap=100, n_permutation=100, min_group_size=2)
    cfg.runtime = RuntimeConfig(max_runtime_hours=1.0, heartbeat_seconds=5.0,
                                min_free_disk_gb=0.05)
    cfg.pilot_n_samples = 4
    cfg.run_pilot_first = False
    return cfg
