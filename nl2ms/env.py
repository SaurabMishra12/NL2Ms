"""Phase 0 -- environment capture, seeding, and GPU discovery.

Two jobs:

* Make the run reproducible (seeds set *and recorded*).
* Make the run auditable (versions, hardware, git commit written to
  ``config/environment.json`` before any measurement happens).

Nothing here assumes a GPU exists; the same code path runs on a CPU-only
machine so the analysis functions can be smoke-tested off-accelerator.
"""

from __future__ import annotations

import importlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .storage import save_json

# Packages we report on. Missing ones are recorded as ``None`` rather than
# raising, so an incomplete Kaggle image still yields a usable environment
# record and a clear message about what is absent.
TRACKED_PACKAGES = [
    "torch", "transformers", "accelerate", "bitsandbytes", "numpy", "scipy",
    "pandas", "sklearn", "matplotlib", "seaborn", "joblib", "h5py", "zarr",
    "datasets", "pyarrow", "safetensors", "tqdm",
]

# Modules the pipeline cannot run without.
REQUIRED_PACKAGES = ["torch", "transformers", "numpy", "scipy", "pandas",
                     "sklearn", "matplotlib"]


def package_version(name: str) -> Optional[str]:
    try:
        mod = importlib.import_module(name)
    except Exception:
        return None
    for attr in ("__version__", "VERSION", "version"):
        v = getattr(mod, attr, None)
        if isinstance(v, str):
            return v
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        return "unknown"


def check_packages() -> Dict[str, Any]:
    versions = {name: package_version(name) for name in TRACKED_PACKAGES}
    missing_required = [n for n in REQUIRED_PACKAGES if versions.get(n) is None]
    missing_optional = [n for n in TRACKED_PACKAGES
                        if versions.get(n) is None and n not in REQUIRED_PACKAGES]
    return {
        "versions": versions,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "ok": not missing_required,
    }


@dataclass
class GPUInfo:
    index: int
    name: str
    total_memory_gb: float
    capability: str


@dataclass
class HardwareInfo:
    cuda_available: bool
    n_gpus: int
    gpus: List[GPUInfo] = field(default_factory=list)
    cuda_version: Optional[str] = None
    cpu_count: int = 0
    total_ram_gb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cuda_available": self.cuda_available,
            "n_gpus": self.n_gpus,
            "gpus": [g.__dict__ for g in self.gpus],
            "cuda_version": self.cuda_version,
            "cpu_count": self.cpu_count,
            "total_ram_gb": self.total_ram_gb,
        }


def detect_hardware() -> HardwareInfo:
    cpu_count = os.cpu_count() or 0
    total_ram_gb = 0.0
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        total_ram_gb = pages * page_size / (1024 ** 3)
    except (ValueError, OSError, AttributeError):
        pass

    try:
        import torch
    except ImportError:
        return HardwareInfo(False, 0, [], None, cpu_count, total_ram_gb)

    if not torch.cuda.is_available():
        return HardwareInfo(False, 0, [], getattr(torch.version, "cuda", None),
                            cpu_count, total_ram_gb)

    gpus: List[GPUInfo] = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gpus.append(GPUInfo(
            index=i,
            name=props.name,
            total_memory_gb=props.total_memory / (1024 ** 3),
            capability=f"{props.major}.{props.minor}",
        ))
    return HardwareInfo(True, len(gpus), gpus, torch.version.cuda, cpu_count,
                        total_ram_gb)


def gpu_memory_report() -> List[Dict[str, Any]]:
    """Per-GPU allocated/reserved memory, printed after model placement."""
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    out = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        out.append({
            "gpu": i,
            "name": props.name,
            "allocated_gb": torch.cuda.memory_allocated(i) / (1024 ** 3),
            "reserved_gb": torch.cuda.memory_reserved(i) / (1024 ** 3),
            "total_gb": props.total_memory / (1024 ** 3),
        })
    return out


def print_gpu_report() -> None:
    report = gpu_memory_report()
    if not report:
        print("No CUDA GPUs visible -- running on CPU.")
        return
    for r in report:
        print(f"GPU {r['gpu']} ({r['name']}): "
              f"allocated {r['allocated_gb']:.2f} GB / "
              f"reserved {r['reserved_gb']:.2f} GB / "
              f"total {r['total_gb']:.2f} GB")


def build_max_memory(hardware: HardwareInfo, headroom_gb: float = 1.6
                     ) -> Optional[Dict[str, str]]:
    """Per-device memory caps for ``device_map="auto"``.

    Leaving headroom matters on 15 GB T4s: activations, the KV cache and the
    attention buffers all land outside the weight budget, and an OOM during
    generation costs far more than a slightly smaller shard on GPU 0.
    """
    if not hardware.cuda_available or hardware.n_gpus == 0:
        return None
    max_memory: Dict[str, str] = {}
    for gpu in hardware.gpus:
        budget = max(1.0, gpu.total_memory_gb - headroom_gb)
        max_memory[str(gpu.index)] = f"{budget:.1f}GiB"
    max_memory["cpu"] = "8GiB"
    return max_memory


def set_all_seeds(seed: int) -> Dict[str, Any]:
    """Seed every RNG the pipeline touches and report what was set."""
    record: Dict[str, Any] = {"seed": seed}
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    record["python_random"] = True
    record["numpy"] = True
    try:
        import torch
        torch.manual_seed(seed)
        record["torch"] = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            record["torch_cuda"] = True
        else:
            record["torch_cuda"] = False
    except ImportError:
        record["torch"] = False
        record["torch_cuda"] = False
    return record


def enable_determinism(strict: bool = False) -> Dict[str, Any]:
    """Bias cuDNN toward reproducible kernels.

    ``strict`` additionally requests deterministic algorithms, which can raise
    for ops lacking a deterministic implementation. It is off by default so a
    single unsupported kernel cannot abort a 10-hour run; the choice is
    recorded either way.
    """
    out: Dict[str, Any] = {"strict": strict}
    try:
        import torch
    except ImportError:
        return {**out, "applied": False}
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if strict:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            out["deterministic_algorithms"] = True
        except Exception as exc:  # pragma: no cover - backend dependent
            out["deterministic_algorithms"] = False
            out["deterministic_error"] = str(exc)
    out["applied"] = True
    return out


def git_commit(repo_dir: str | Path = ".") -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_dir),
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def detect_platform() -> str:
    if os.path.isdir("/kaggle"):
        return "kaggle"
    if "COLAB_GPU" in os.environ or os.path.isdir("/content"):
        return "colab"
    return "local"


def default_output_root() -> str:
    """Kaggle only persists ``/kaggle/working``; elsewhere use a local dir."""
    if detect_platform() == "kaggle":
        return "/kaggle/working/experiment"
    return str(Path.cwd() / "experiment")


def capture_environment(seed: int, *, repo_dir: str | Path = ".",
                        strict_determinism: bool = False) -> Dict[str, Any]:
    """Assemble the full ``environment.json`` payload."""
    hardware = detect_hardware()
    packages = check_packages()
    env = {
        "captured_at": time.time(),
        "captured_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": detect_platform(),
        "python_version": sys.version,
        "python_version_short": platform.python_version(),
        "os": platform.platform(),
        "machine": platform.machine(),
        "packages": packages["versions"],
        "missing_required_packages": packages["missing_required"],
        "missing_optional_packages": packages["missing_optional"],
        "hardware": hardware.to_dict(),
        "seeds": set_all_seeds(seed),
        "determinism": enable_determinism(strict_determinism),
        "git_commit": git_commit(repo_dir),
        "env_vars": {
            k: os.environ.get(k)
            for k in ["HF_HOME", "TRANSFORMERS_CACHE", "HF_DATASETS_OFFLINE",
                      "TRANSFORMERS_OFFLINE", "CUDA_VISIBLE_DEVICES",
                      "KAGGLE_KERNEL_RUN_TYPE"]
        },
    }
    return env


def save_environment(env: Dict[str, Any], path: str | Path) -> Path:
    return save_json(path, env)


def summarise_environment(env: Dict[str, Any]) -> str:
    hw = env["hardware"]
    lines = [
        f"platform            : {env['platform']}",
        f"python              : {env['python_version_short']}",
        f"torch               : {env['packages'].get('torch')}",
        f"transformers        : {env['packages'].get('transformers')}",
        f"cuda available      : {hw['cuda_available']} "
        f"(version {hw['cuda_version']})",
        f"gpus                : {hw['n_gpus']}",
    ]
    for g in hw["gpus"]:
        lines.append(f"  - GPU {g['index']}: {g['name']} "
                     f"({g['total_memory_gb']:.1f} GB, sm_{g['capability']})")
    lines.append(f"cpu cores           : {hw['cpu_count']}")
    lines.append(f"system RAM          : {hw['total_ram_gb']:.1f} GB")
    if env["missing_required_packages"]:
        lines.append(f"MISSING REQUIRED    : {env['missing_required_packages']}")
    if env["missing_optional_packages"]:
        lines.append(f"missing optional    : {env['missing_optional_packages']}")
    lines.append(f"git commit          : {env['git_commit']}")
    return "\n".join(lines)


def supports_bfloat16() -> bool:
    """T4s are sm_75: bf16 is emulated and slow, so fp16 is the right default."""
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_bf16_supported())
    except Exception:
        return False


def resolve_dtype(requested: str) -> Any:
    """Map a config dtype string to a torch dtype, downgrading bf16 on T4."""
    import torch
    mapping = {"float16": torch.float16, "bfloat16": torch.bfloat16,
               "float32": torch.float32}
    dtype = mapping.get(requested, torch.float16)
    if dtype is torch.bfloat16 and not supports_bfloat16():
        print("bfloat16 unsupported on this device -- falling back to float16.")
        return torch.float16
    return dtype
