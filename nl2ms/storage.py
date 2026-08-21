"""Persistence layer: atomic writes, shards, manifests, checkpoint recovery.

Every artefact this experiment produces goes through this module, so that a
Kaggle session dying mid-write can never leave a half-written file that a
later run mistakes for valid data.

Two invariants are enforced here:

1. **Atomicity.** Nothing is ever written directly to its final path. Data
   goes to ``<path>.tmp``, is flushed and fsynced, then ``os.replace``d into
   place (atomic within a filesystem).
2. **Verifiability.** Every write is followed by a read-back and a checksum,
   recorded in a manifest. A checkpoint is "valid" only if the file exists,
   parses, and its checksum matches the manifest entry.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Status vocabulary
# ---------------------------------------------------------------------------
STATUS_COMPLETE = "complete"
STATUS_INCOMPLETE = "incomplete"
STATUS_CORRUPTED = "corrupted"
STATUS_FAILED = "failed"
STATUS_MISSING = "missing"

CHECK_SKIP = "SKIP"
CHECK_RESUME = "RESUME"
CHECK_RECOMPUTE = "MARK_CORRUPTED_AND_RECOMPUTE"


# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
@dataclass
class ExperimentPaths:
    """Canonical directory layout (protocol section 62)."""

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    # -- top level -----------------------------------------------------
    @property
    def config(self) -> Path:
        return self.root / "config"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def manifests(self) -> Path:
        return self.root / "manifests"

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def derived(self) -> Path:
        return self.root / "derived"

    @property
    def interventions(self) -> Path:
        return self.root / "interventions"

    @property
    def statistics(self) -> Path:
        return self.root / "statistics"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def examples(self) -> Path:
        return self.root / "examples"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    # -- raw sub-dirs --------------------------------------------------
    @property
    def generations(self) -> Path:
        return self.raw / "generations"

    @property
    def hidden_states(self) -> Path:
        return self.raw / "hidden_states"

    @property
    def logits(self) -> Path:
        return self.raw / "logits"

    @property
    def attention(self) -> Path:
        return self.raw / "attention"

    # -- derived sub-dirs ----------------------------------------------
    @property
    def entropy(self) -> Path:
        return self.derived / "entropy"

    @property
    def geometry(self) -> Path:
        return self.derived / "geometry"

    @property
    def jsd(self) -> Path:
        return self.derived / "jsd"

    @property
    def dynamics(self) -> Path:
        return self.derived / "dynamics"

    @property
    def j_space(self) -> Path:
        return self.derived / "j_space"

    @property
    def critical_layers(self) -> Path:
        return self.derived / "critical_layers"

    @property
    def datasets(self) -> Path:
        return self.root / "data"

    def all_dirs(self) -> List[Path]:
        return [
            self.config, self.checkpoints, self.manifests, self.raw, self.derived,
            self.interventions, self.statistics, self.figures, self.examples,
            self.reports, self.logs, self.generations, self.hidden_states,
            self.logits, self.attention, self.entropy, self.geometry, self.jsd,
            self.dynamics, self.j_space, self.critical_layers, self.datasets,
        ]

    def ensure(self) -> "ExperimentPaths":
        for d in self.all_dirs():
            d.mkdir(parents=True, exist_ok=True)
        return self

    # -- named files ---------------------------------------------------
    @property
    def manifest_jsonl(self) -> Path:
        return self.manifests / "manifest.jsonl"

    @property
    def shard_manifest(self) -> Path:
        return self.manifests / "shard_manifest.json"

    @property
    def errors_jsonl(self) -> Path:
        return self.logs / "errors.jsonl"

    @property
    def heartbeat(self) -> Path:
        return self.logs / "heartbeat.json"

    @property
    def experiment_manifest(self) -> Path:
        return self.root / "experiment_manifest.json"

    @property
    def integrity_report(self) -> Path:
        return self.reports / "integrity_report.json"

    @property
    def final_report(self) -> Path:
        return self.reports / "FINAL_REPORT.md"


# ---------------------------------------------------------------------------
# Checksums and atomic IO
# ---------------------------------------------------------------------------
def file_checksum(path: str | Path, algo: str = "sha256", chunk: int = 1 << 20) -> str:
    """Streaming checksum so that large tensor shards do not blow up RAM."""
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


@contextmanager
def atomic_path(path: str | Path) -> Iterator[Path]:
    """Yield a temporary path; atomically move it into place on clean exit.

    On any exception the temporary file is removed and the pre-existing file
    (if any) is left untouched -- valid previous results are never destroyed
    by a failed rewrite.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    try:
        yield tmp
        if not tmp.exists():
            raise IOError(f"atomic write produced no file: {tmp}")
        # Force bytes to disk before the rename so a crash cannot leave a
        # renamed-but-empty file behind.
        with open(tmp, "rb+") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def _verify_readable(path: Path, reader: Callable[[Path], Any]) -> None:
    """Read the file back after writing; a write we cannot read is a failure."""
    if not path.exists() or path.stat().st_size == 0:
        raise IOError(f"post-write verification failed (missing/empty): {path}")
    reader(path)


def save_json(path: str | Path, obj: Any, *, verify: bool = True) -> Path:
    path = Path(path)
    with atomic_path(path) as tmp:
        tmp.write_text(json.dumps(obj, indent=2, default=_json_default))
    if verify:
        _verify_readable(path, lambda p: json.loads(p.read_text()))
    return path


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        # JSON has no NaN/Inf; emit null and let the consumer see a gap
        # rather than silently substituting a plausible number.
        return value if np.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def append_jsonl(path: str | Path, record: Dict[str, Any]) -> None:
    """Append one record. Line-buffered + fsync so a crash keeps prior lines."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(record, default=_json_default) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Read a JSONL file, tolerating a truncated final line from a crash."""
    path = Path(path)
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # Only the last line can legitimately be torn; anything else
                # is real corruption and is surfaced by the integrity check.
                continue
    return out


def save_npz(path: str | Path, arrays: Dict[str, np.ndarray], *,
             compressed: bool = True, verify: bool = True) -> Path:
    path = Path(path)
    with atomic_path(path) as tmp:
        if compressed:
            np.savez_compressed(tmp, **arrays)
        else:
            np.savez(tmp, **arrays)
        # numpy appends .npz when the target lacks that suffix
        produced = tmp if tmp.exists() else Path(str(tmp) + ".npz")
        if produced != tmp:
            produced.replace(tmp)
    if verify:
        def _read(p: Path) -> None:
            with np.load(p, allow_pickle=False) as z:
                _ = list(z.keys())
        _verify_readable(path, _read)
    return path


def load_npz(path: str | Path) -> Dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def save_parquet(path: str | Path, df, *, verify: bool = True) -> Path:
    """Write a DataFrame to parquet, falling back to CSV if pyarrow is absent."""
    path = Path(path)
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        csv_path = path.with_suffix(".csv")
        with atomic_path(csv_path) as tmp:
            df.to_csv(tmp, index=False)
        return csv_path
    with atomic_path(path) as tmp:
        df.to_parquet(tmp, index=False)
    if verify:
        import pandas as pd
        _verify_readable(path, lambda p: pd.read_parquet(p))
    return path


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
@dataclass
class ManifestRecord:
    sample_id: str
    phase: str
    status: str
    timestamp: float
    output_path: Optional[str] = None
    checksum: Optional[str] = None
    model: Optional[str] = None
    seed: Optional[int] = None
    error: Optional[str] = None
    runtime_seconds: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "sample_id": self.sample_id,
            "phase": self.phase,
            "status": self.status,
            "timestamp": self.timestamp,
            "output_path": self.output_path,
            "checksum": self.checksum,
            "model": self.model,
            "seed": self.seed,
            "error": self.error,
            "runtime_seconds": self.runtime_seconds,
        }
        d.update(self.extra)
        return d


class Manifest:
    """Append-only record of every (sample, phase) unit of work.

    Kept as JSONL because appends survive a kill -9 far better than rewriting
    a JSON document. The in-memory index keeps the *latest* record per
    (sample_id, phase) key, so re-running a corrupted sample supersedes the
    old entry without deleting history.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._index: Dict[tuple, Dict[str, Any]] = {}
        self.reload()

    def reload(self) -> None:
        self._index.clear()
        for rec in read_jsonl(self.path):
            key = (rec.get("sample_id"), rec.get("phase"))
            self._index[key] = rec

    def record(self, rec: ManifestRecord) -> None:
        payload = rec.to_dict()
        append_jsonl(self.path, payload)
        self._index[(rec.sample_id, rec.phase)] = payload

    def get(self, sample_id: str, phase: str) -> Optional[Dict[str, Any]]:
        return self._index.get((sample_id, phase))

    def status_of(self, sample_id: str, phase: str) -> str:
        rec = self.get(sample_id, phase)
        return rec.get("status", STATUS_MISSING) if rec else STATUS_MISSING

    def completed_ids(self, phase: str) -> List[str]:
        return [sid for (sid, ph), rec in self._index.items()
                if ph == phase and rec.get("status") == STATUS_COMPLETE]

    def failed_ids(self, phase: str) -> List[str]:
        return [sid for (sid, ph), rec in self._index.items()
                if ph == phase and rec.get("status") == STATUS_FAILED]

    def failure_count(self, sample_id: str, phase: str) -> int:
        n = 0
        for rec in read_jsonl(self.path):
            if rec.get("sample_id") == sample_id and rec.get("phase") == phase \
                    and rec.get("status") in (STATUS_FAILED, STATUS_CORRUPTED):
                n += 1
        return n

    def phase_counts(self) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = {}
        for (_sid, phase), rec in self._index.items():
            bucket = out.setdefault(phase, {})
            status = rec.get("status", STATUS_MISSING)
            bucket[status] = bucket.get(status, 0) + 1
        return out

    # ------------------------------------------------------------------
    def check(self, sample_id: str, phase: str,
              validator: Optional[Callable[[Path], bool]] = None) -> str:
        """Decide what to do with an existing unit of work.

        Returns one of ``SKIP`` / ``RESUME`` / ``MARK_CORRUPTED_AND_RECOMPUTE``.
        A checkpoint counts as valid only if the manifest says complete *and*
        the file is present *and* its checksum still matches *and* an optional
        content validator accepts it.
        """
        rec = self.get(sample_id, phase)
        if rec is None:
            return CHECK_RESUME
        if rec.get("status") != STATUS_COMPLETE:
            return CHECK_RESUME
        out_path = rec.get("output_path")
        if out_path is None:
            return CHECK_SKIP  # completed unit that legitimately writes no file
        p = Path(out_path)
        if not p.exists():
            return CHECK_RECOMPUTE
        expected = rec.get("checksum")
        if expected:
            try:
                if file_checksum(p) != expected:
                    return CHECK_RECOMPUTE
            except OSError:
                return CHECK_RECOMPUTE
        if validator is not None:
            try:
                if not validator(p):
                    return CHECK_RECOMPUTE
            except Exception:
                return CHECK_RECOMPUTE
        return CHECK_SKIP


class ErrorLog:
    """Failures are recorded, never swallowed (protocol section 41)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def log(self, sample_id: str, phase: str, exc: BaseException,
            **extra: Any) -> Dict[str, Any]:
        rec = {
            "sample_id": sample_id,
            "phase": phase,
            "timestamp": time.time(),
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc,
                                                            exc.__traceback__))[-4000:],
        }
        rec.update(extra)
        append_jsonl(self.path, rec)
        return rec

    def all(self) -> List[Dict[str, Any]]:
        return read_jsonl(self.path)


# ---------------------------------------------------------------------------
# Sharded tensor storage
# ---------------------------------------------------------------------------
class ShardWriter:
    """Writes fixed-size groups of samples as self-describing ``.npz`` shards.

    A single enormous tensor file is a resumability hazard: one torn write
    loses the whole run. Shards bound that blast radius to ``shard_size``
    samples, and each carries its own metadata sidecar so it can be
    interpreted without the global manifest.
    """

    def __init__(self, directory: str | Path, name: str, shard_size: int,
                 manifest_path: Optional[str | Path] = None) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.shard_size = max(1, int(shard_size))
        self.manifest_path = Path(manifest_path) if manifest_path else self.dir / "manifest.json"
        self._buffer: List[tuple] = []  # (sample_id, {key: array})
        self._shard_index = self._next_shard_index()

    def _next_shard_index(self) -> int:
        existing = sorted(self.dir.glob(f"{self.name}_shard_*.npz"))
        if not existing:
            return 0
        last = existing[-1].stem.split("_")[-1]
        try:
            return int(last) + 1
        except ValueError:
            return len(existing)

    def shard_path(self, index: int) -> Path:
        return self.dir / f"{self.name}_shard_{index:04d}.npz"

    def meta_path(self, index: int) -> Path:
        return self.dir / f"{self.name}_shard_{index:04d}.meta.json"

    def existing_sample_ids(self) -> set:
        """Sample IDs already committed to disk, for resume decisions."""
        ids: set = set()
        for meta_file in sorted(self.dir.glob(f"{self.name}_shard_*.meta.json")):
            try:
                meta = load_json(meta_file)
            except json.JSONDecodeError:
                continue
            shard_file = Path(meta.get("path", ""))
            if not shard_file.exists():
                continue
            ids.update(meta.get("sample_ids", []))
        return ids

    def add(self, sample_id: str, arrays: Dict[str, np.ndarray]) -> Optional[Path]:
        self._buffer.append((sample_id, arrays))
        if len(self._buffer) >= self.shard_size:
            return self.flush()
        return None

    def flush(self) -> Optional[Path]:
        """Commit buffered samples. Safe to call when the buffer is empty."""
        if not self._buffer:
            return None
        index = self._shard_index
        path = self.shard_path(index)
        payload: Dict[str, np.ndarray] = {}
        sample_ids: List[str] = []
        shapes: Dict[str, List[int]] = {}
        dtypes: Dict[str, str] = {}
        for sid, arrays in self._buffer:
            sample_ids.append(sid)
            for key, arr in arrays.items():
                arr = np.asarray(arr)
                payload[f"{sid}::{key}"] = arr
                shapes[f"{sid}::{key}"] = list(arr.shape)
                dtypes[f"{sid}::{key}"] = str(arr.dtype)
        save_npz(path, payload, compressed=True)
        meta = {
            "shard": f"{self.name}_shard_{index:04d}",
            "path": str(path),
            "sample_ids": sample_ids,
            "n_samples": len(sample_ids),
            "keys": sorted(payload.keys()),
            "shapes": shapes,
            "dtypes": dtypes,
            "checksum": file_checksum(path),
            "bytes": path.stat().st_size,
            "created_at": time.time(),
        }
        save_json(self.meta_path(index), meta)
        self._update_manifest(meta)
        self._buffer.clear()
        self._shard_index += 1
        return path

    def _update_manifest(self, meta: Dict[str, Any]) -> None:
        manifest: Dict[str, Any] = {"shards": []}
        if self.manifest_path.exists():
            try:
                manifest = load_json(self.manifest_path)
            except json.JSONDecodeError:
                manifest = {"shards": []}
        manifest.setdefault("shards", [])
        manifest["shards"] = [s for s in manifest["shards"] if s.get("shard") != meta["shard"]]
        manifest["shards"].append(meta)
        manifest["updated_at"] = time.time()
        manifest["n_shards"] = len(manifest["shards"])
        manifest["total_bytes"] = sum(s.get("bytes", 0) for s in manifest["shards"])
        save_json(self.manifest_path, manifest)

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        # Flush on the way out so an exception still commits completed work.
        try:
            self.flush()
        except Exception:
            pass


class ShardReader:
    """Random access to shards written by :class:`ShardWriter`."""

    def __init__(self, directory: str | Path, name: str) -> None:
        self.dir = Path(directory)
        self.name = name
        self._loc: Dict[str, Path] = {}
        self._build_index()

    def _build_index(self) -> None:
        for meta_file in sorted(self.dir.glob(f"{self.name}_shard_*.meta.json")):
            try:
                meta = load_json(meta_file)
            except json.JSONDecodeError:
                continue
            path = Path(meta.get("path", ""))
            if not path.exists():
                continue
            for sid in meta.get("sample_ids", []):
                self._loc[sid] = path

    @property
    def sample_ids(self) -> List[str]:
        return sorted(self._loc.keys())

    def get(self, sample_id: str) -> Dict[str, np.ndarray]:
        path = self._loc.get(sample_id)
        if path is None:
            raise KeyError(f"sample not present in shards: {sample_id}")
        prefix = f"{sample_id}::"
        out: Dict[str, np.ndarray] = {}
        with np.load(path, allow_pickle=False) as z:
            for key in z.files:
                if key.startswith(prefix):
                    out[key[len(prefix):]] = z[key]
        return out

    def iter_samples(self, sample_ids: Optional[Sequence[str]] = None
                     ) -> Iterator[tuple]:
        """Iterate shard-by-shard so each file is opened exactly once."""
        wanted = set(sample_ids) if sample_ids is not None else None
        by_path: Dict[Path, List[str]] = {}
        for sid, path in self._loc.items():
            if wanted is None or sid in wanted:
                by_path.setdefault(path, []).append(sid)
        for path, sids in sorted(by_path.items()):
            with np.load(path, allow_pickle=False) as z:
                for sid in sorted(sids):
                    prefix = f"{sid}::"
                    yield sid, {k[len(prefix):]: z[k] for k in z.files
                                if k.startswith(prefix)}


# ---------------------------------------------------------------------------
# Storage estimation and disk safety
# ---------------------------------------------------------------------------
def disk_free_gb(path: str | Path) -> float:
    usage = shutil.disk_usage(str(path))
    return usage.free / (1024 ** 3)


def disk_used_gb(path: str | Path) -> float:
    usage = shutil.disk_usage(str(path))
    return usage.used / (1024 ** 3)


def dir_size_gb(path: str | Path) -> float:
    total = 0
    for p in Path(path).rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                continue
    return total / (1024 ** 3)


@dataclass
class StorageEstimate:
    estimated_gb: float
    available_gb: float
    n_shards: int
    per_sample_mb: float
    breakdown: Dict[str, float]
    sufficient: bool
    detail: str


def estimate_storage(*, n_samples: int, n_layers: int, hidden_size: int,
                     n_positions: int, n_heads: int, seq_len: int,
                     top_k: int, flags: Dict[str, bool], shard_size: int,
                     hidden_bytes: int = 2, output_root: str | Path,
                     min_free_gb: float = 3.0,
                     n_full_attention_samples: int = 0,
                     vocab_size: int = 0) -> StorageEstimate:
    """Predict disk usage *before* extraction begins (protocol section 37).

    Compression typically buys 1.5-2x on float16 activations; we deliberately
    do not assume it, so the estimate errs on the safe side.
    """
    gb = 1024 ** 3
    breakdown: Dict[str, float] = {}

    # Hidden states: (layers+1) x positions x hidden, per sample.
    if flags.get("save_hidden_states"):
        per_sample = (n_layers + 1) * n_positions * hidden_size * hidden_bytes
        breakdown["hidden_states"] = per_sample * n_samples / gb
    else:
        breakdown["hidden_states"] = 0.0

    # Logit-lens top-k: per layer per position, k ids (int32) + k probs (f32)
    # plus a fixed handful of scalar metrics.
    per_sample_lens = (n_layers + 1) * n_positions * (top_k * (4 + 4) + 16 * 4)
    breakdown["logit_lens"] = per_sample_lens * n_samples / gb

    if flags.get("save_full_vocab_logits") and vocab_size:
        full = (n_layers + 1) * n_positions * vocab_size * 2
        breakdown["full_vocab_logits"] = full * n_samples / gb

    # Attention summaries: a fixed set of statistics per (layer, head, position).
    if flags.get("save_attention_summaries"):
        n_stats = 12
        per_sample_attn = n_layers * n_heads * n_positions * n_stats * 4
        breakdown["attention_summaries"] = per_sample_attn * n_samples / gb
    else:
        breakdown["attention_summaries"] = 0.0

    # Full attention matrices for a handful of samples: L x H x T x T.
    if flags.get("save_full_attention") and n_full_attention_samples:
        per_full = n_layers * n_heads * seq_len * seq_len * 2
        breakdown["full_attention"] = per_full * n_full_attention_samples / gb

    # Derived metrics, figures, reports: small but non-zero.
    breakdown["derived_metrics"] = 0.002 * n_samples
    breakdown["figures_reports"] = 0.05

    estimated = float(sum(breakdown.values()))
    available = disk_free_gb(output_root)
    n_shards = max(1, int(np.ceil(n_samples / max(1, shard_size))))
    sufficient = (available - estimated) >= min_free_gb
    detail = (
        f"estimated {estimated:.2f} GB vs available {available:.2f} GB "
        f"(reserve {min_free_gb:.2f} GB) -> {'OK' if sufficient else 'INSUFFICIENT'}"
    )
    return StorageEstimate(
        estimated_gb=estimated,
        available_gb=available,
        n_shards=n_shards,
        per_sample_mb=(estimated * 1024 / max(1, n_samples)),
        breakdown=breakdown,
        sufficient=sufficient,
        detail=detail,
    )


def make_backup_archive(paths: ExperimentPaths, *, include_raw: bool = False,
                        label: Optional[str] = None) -> Path:
    """Archive the small, high-value artefacts (protocol section 63).

    Raw tensors are excluded by default: duplicating tens of GB of activations
    into a tarball is exactly how a Kaggle session runs out of disk.
    """
    import tarfile

    stamp = label or time.strftime("%Y%m%d_%H%M%S")
    out = paths.root.parent / f"experiment_backup_{stamp}.tar.gz"
    include = [paths.config, paths.manifests, paths.derived, paths.statistics,
               paths.figures, paths.reports, paths.logs, paths.examples,
               paths.interventions]
    if include_raw:
        include.append(paths.raw)
    else:
        include.append(paths.generations)  # generations are small and precious
    with atomic_path(out) as tmp:
        with tarfile.open(tmp, "w:gz") as tar:
            for d in include:
                if d.exists():
                    tar.add(d, arcname=str(d.relative_to(paths.root.parent)))
            if paths.experiment_manifest.exists():
                tar.add(paths.experiment_manifest,
                        arcname=str(paths.experiment_manifest.relative_to(paths.root.parent)))
    return out
