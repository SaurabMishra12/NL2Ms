"""Runtime budget control, throughput measurement, and heartbeats.

Kaggle kills a session at ~12 hours with no warning and no chance to flush
state. The controller here answers one question repeatedly: *is there enough
time left to start the next unit of work and still finalise cleanly?* If the
answer is no, the run stops on its own terms with every checkpoint intact.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .storage import ExperimentPaths, disk_free_gb, dir_size_gb, save_json


SAFE_STOP_MESSAGE = "SAFE STOP: checkpoint complete. Resume notebook to continue."


class BudgetExceeded(Exception):
    """Raised when a phase must stop to protect the finalisation window."""


@dataclass
class ThroughputTracker:
    """Rolling seconds-per-unit estimate.

    Uses a trailing window because the first few samples are dominated by
    warm-up (CUDA graph capture, kernel autotuning, page-cache misses) and
    would otherwise inflate every downstream estimate.
    """

    window: int = 20
    durations: List[float] = field(default_factory=list)
    warmup_skip: int = 2

    def record(self, seconds: float) -> None:
        self.durations.append(float(seconds))

    @property
    def n(self) -> int:
        return len(self.durations)

    def seconds_per_unit(self) -> Optional[float]:
        usable = self.durations[self.warmup_skip:] if self.n > self.warmup_skip \
            else self.durations
        if not usable:
            return None
        return float(np.mean(usable[-self.window:]))

    def units_per_hour(self) -> Optional[float]:
        spu = self.seconds_per_unit()
        if not spu or spu <= 0:
            return None
        return 3600.0 / spu

    def estimate_seconds(self, n_units: int) -> Optional[float]:
        spu = self.seconds_per_unit()
        return None if spu is None else spu * n_units

    def summary(self) -> Dict[str, Any]:
        return {
            "n_measurements": self.n,
            "seconds_per_unit": self.seconds_per_unit(),
            "units_per_hour": self.units_per_hour(),
            "p50_seconds": float(np.median(self.durations)) if self.durations else None,
            "p90_seconds": float(np.percentile(self.durations, 90)) if self.durations else None,
            "max_seconds": float(np.max(self.durations)) if self.durations else None,
        }


class RuntimeController:
    """Wall-clock guard for the whole session."""

    def __init__(self, max_runtime_hours: float,
                 reserve_minutes: float = 25.0,
                 start_time: Optional[float] = None) -> None:
        self.start_time = start_time if start_time is not None else time.time()
        self.max_seconds = max_runtime_hours * 3600.0
        self.reserve_seconds = reserve_minutes * 60.0
        self.trackers: Dict[str, ThroughputTracker] = {}

    # -- time accounting ------------------------------------------------
    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def remaining(self) -> float:
        return self.max_seconds - self.elapsed

    @property
    def usable_remaining(self) -> float:
        """Time left after protecting the finalisation reserve."""
        return self.remaining - self.reserve_seconds

    def expired(self) -> bool:
        return self.usable_remaining <= 0

    # -- throughput -----------------------------------------------------
    def tracker(self, phase: str) -> ThroughputTracker:
        return self.trackers.setdefault(phase, ThroughputTracker())

    def record(self, phase: str, seconds: float) -> None:
        self.tracker(phase).record(seconds)

    def can_afford(self, phase: str, n_units: int, *,
                   safety_factor: float = 1.25) -> bool:
        """Can ``n_units`` of ``phase`` finish inside the usable window?

        With no measurements yet we return True: refusing to start before any
        timing data exists would deadlock the very first shard.
        """
        est = self.tracker(phase).estimate_seconds(n_units)
        if est is None:
            return self.usable_remaining > 0
        return est * safety_factor <= self.usable_remaining

    def affordable_units(self, phase: str, *, safety_factor: float = 1.25
                         ) -> Optional[int]:
        spu = self.tracker(phase).seconds_per_unit()
        if spu is None or spu <= 0:
            return None
        return max(0, int(self.usable_remaining / (spu * safety_factor)))

    def check(self, phase: str, n_units: int = 1, *,
              safety_factor: float = 1.25) -> None:
        if not self.can_afford(phase, n_units, safety_factor=safety_factor):
            raise BudgetExceeded(
                f"{SAFE_STOP_MESSAGE} (phase={phase}, "
                f"usable_remaining={self.usable_remaining / 60:.1f} min)"
            )

    # -- reporting ------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        return {
            "start_time": self.start_time,
            "elapsed_seconds": self.elapsed,
            "elapsed_hours": self.elapsed / 3600.0,
            "remaining_seconds": self.remaining,
            "remaining_hours": self.remaining / 3600.0,
            "usable_remaining_seconds": self.usable_remaining,
            "reserve_seconds": self.reserve_seconds,
            "throughput": {k: v.summary() for k, v in self.trackers.items()},
        }

    def planning_table(self, plan: List[Dict[str, Any]]) -> str:
        """Render the pre-flight plan (protocol section 58)."""
        header = f"{'phase':<28}{'samples':>9}{'est runtime':>16}{'storage GB':>13}{'GPU GB':>10}"
        lines = [header, "-" * len(header)]
        for row in plan:
            est = row.get("estimated_seconds")
            est_str = "unmeasured" if est is None else _fmt_duration(est)
            storage = row.get("estimated_storage_gb")
            storage_str = "-" if storage is None else f"{storage:.2f}"
            gpu = row.get("estimated_gpu_gb")
            gpu_str = "-" if gpu is None else f"{gpu:.1f}"
            lines.append(f"{row.get('phase', '?'):<28}{row.get('samples', 0):>9}"
                         f"{est_str:>16}{storage_str:>13}{gpu_str:>10}")
        total = sum(r.get("estimated_seconds") or 0.0 for r in plan)
        lines.append("-" * len(header))
        lines.append(f"{'TOTAL (measured phases)':<28}{'':>9}{_fmt_duration(total):>16}")
        lines.append(f"usable budget remaining: {_fmt_duration(self.usable_remaining)}")
        return "\n".join(lines)


def _fmt_duration(seconds: float) -> str:
    if seconds is None:
        return "unknown"
    seconds = float(seconds)
    if seconds < 90:
        return f"{seconds:.1f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.2f}h"


class Heartbeat:
    """Periodic liveness file (protocol section 39).

    Written at most every ``interval`` seconds so a tight sample loop does not
    turn into a disk-write loop; ``force=True`` bypasses the throttle at phase
    boundaries where the record genuinely matters.
    """

    def __init__(self, path: str | Path, interval: float = 120.0,
                 controller: Optional[RuntimeController] = None,
                 experiment_root: Optional[str | Path] = None) -> None:
        self.path = Path(path)
        self.interval = float(interval)
        self.controller = controller
        self.experiment_root = Path(experiment_root) if experiment_root else self.path.parent
        self._last_write = 0.0
        self.state: Dict[str, Any] = {
            "current_phase": None,
            "current_sample": None,
            "current_shard": None,
            "completed_samples": 0,
            "remaining_samples": 0,
            "last_successful_checkpoint": None,
        }

    def update(self, **kwargs: Any) -> None:
        self.state.update(kwargs)

    def beat(self, force: bool = False, **kwargs: Any) -> Optional[Path]:
        self.state.update(kwargs)
        now = time.time()
        if not force and (now - self._last_write) < self.interval:
            return None
        payload = dict(self.state)
        payload["timestamp"] = now
        payload["timestamp_iso"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))
        if self.controller is not None:
            status = self.controller.status()
            payload["elapsed_time"] = status["elapsed_seconds"]
            payload["elapsed_hours"] = status["elapsed_hours"]
            payload["estimated_remaining_time"] = status["usable_remaining_seconds"]
            phase = self.state.get("current_phase")
            if phase:
                spu = self.controller.tracker(phase).seconds_per_unit()
                remaining = self.state.get("remaining_samples") or 0
                payload["estimated_phase_seconds"] = None if spu is None else spu * remaining
        payload["gpu_memory"] = _gpu_memory()
        payload["cpu_memory"] = _cpu_memory()
        payload["disk"] = {
            "free_gb": disk_free_gb(self.experiment_root),
            "experiment_gb": _safe_dir_size(self.experiment_root),
        }
        save_json(self.path, payload, verify=False)
        self._last_write = now
        return self.path

    def read(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {}
        import json
        try:
            return json.loads(self.path.read_text())
        except Exception:
            return {}


def _safe_dir_size(path: str | Path) -> Optional[float]:
    try:
        return dir_size_gb(path)
    except OSError:
        return None


def _gpu_memory() -> List[Dict[str, Any]]:
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    out = []
    for i in range(torch.cuda.device_count()):
        out.append({
            "gpu": i,
            "allocated_gb": torch.cuda.memory_allocated(i) / (1024 ** 3),
            "reserved_gb": torch.cuda.memory_reserved(i) / (1024 ** 3),
            "max_allocated_gb": torch.cuda.max_memory_allocated(i) / (1024 ** 3),
        })
    return out


def _cpu_memory() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss is KB on Linux, bytes on macOS.
        factor = 1024 if os.uname().sysname == "Linux" else 1
        out["max_rss_gb"] = usage.ru_maxrss * factor / (1024 ** 3)
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as fh:
            info = {}
            for line in fh:
                key, _, rest = line.partition(":")
                info[key] = rest.strip()
        avail = info.get("MemAvailable", "0 kB").split()[0]
        out["available_gb"] = float(avail) / (1024 ** 2)
    except Exception:
        pass
    return out


def free_gpu_memory() -> None:
    """Release cached blocks. Call at phase boundaries, not inside hot loops."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def with_oom_retry(fn: Callable[..., Any], *args: Any,
                   attempts: int = 2, on_oom: Optional[Callable[[int], None]] = None,
                   **kwargs: Any) -> Any:
    """Run ``fn``, retrying after clearing CUDA cache on out-of-memory.

    Retries only on OOM; every other exception propagates immediately so real
    bugs are not masked as capacity problems.
    """
    import torch

    last_exc: Optional[BaseException] = None
    for attempt in range(attempts + 1):
        try:
            return fn(*args, **kwargs)
        except torch.cuda.OutOfMemoryError as exc:  # type: ignore[attr-defined]
            last_exc = exc
            free_gpu_memory()
            if on_oom is not None:
                on_oom(attempt)
            if attempt == attempts:
                raise
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            last_exc = exc
            free_gpu_memory()
            if on_oom is not None:
                on_oom(attempt)
            if attempt == attempts:
                raise
    if last_exc:
        raise last_exc


def benchmark_plan(controller: RuntimeController, phases: List[Dict[str, Any]]
                   ) -> List[Dict[str, Any]]:
    """Attach measured runtime estimates to a list of planned phases."""
    plan = []
    for row in phases:
        row = dict(row)
        phase = row["phase"]
        n = int(row.get("samples", 0))
        row["estimated_seconds"] = controller.tracker(phase).estimate_seconds(n)
        plan.append(row)
    return plan


def save_runtime_report(controller: RuntimeController, paths: ExperimentPaths,
                        extra: Optional[Dict[str, Any]] = None) -> Path:
    payload = controller.status()
    if extra:
        payload.update(extra)
    return save_json(paths.logs / "runtime_report.json", payload)
