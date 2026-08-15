"""Collect the hardware, software and reproducibility metadata for a run.

Two things here are security-relevant rather than merely tidy:

* Environment variables are **allow-listed by name**, never collected wholesale.
  A benchmark record is exported, shared and committed; sweeping up ``os.environ``
  would put API keys into it.
* ``git`` is invoked with a fixed argument vector and no shell, so a repository
  path can never become a command.
"""
from __future__ import annotations

import platform
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from app.core.config import REPO_ROOT
from app.core.logging import get_logger
from app.instrumentation.probes.gpu import NvmlProbe
from app.instrumentation.probes.system import cpu_static_info
from app.schemas.environment import (
    GpuDescriptor,
    HardwareInfo,
    Reproducibility,
    SoftwareEnvironment,
)

log = get_logger("instrumentation.environment")

#: Variables that genuinely change execution. Anything not named here is excluded,
#: which is what keeps secrets out of exported benchmark records.
_ALLOWED_ENV_VARS = (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_LAUNCH_BLOCKING",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "ORT_TENSORRT_FP16_ENABLE",
    "ORT_DISABLE_ALL_OPTIMIZATION",
    "PYTORCH_CUDA_ALLOC_CONF",
    "TOKENIZERS_PARALLELISM",
    "IL_DEFAULT_MODEL_ID",
    "IL_DEFAULT_INPUT_SIZE",
)

#: Packages whose version changes measured performance.
_TRACKED_PACKAGES = (
    "onnxruntime", "onnxruntime-gpu", "numpy", "torch", "torchvision",
    "opencv-python-headless", "opencv-python", "psutil", "pillow",
    "transformers", "tokenizers", "openvino", "tensorrt",
)


def _run_git(*args: str) -> str | None:
    """Run a git command in the repo. Returns None if git or the repo is absent."""
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("git_unavailable", error=str(exc))
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _commit_from_git_dir() -> str | None:
    """Resolve HEAD by reading ``.git`` directly, without the git binary.

    The deploy verifies that the process answering /health is running the commit it
    just checked out — the check that catches a stale server surviving a restart.
    On the VPS the backend runs as SYSTEM, where ``git`` is off PATH and the
    repository is owned by another account, so ``_run_git`` returns None and that
    verification silently degrades to a warning. Reading the files needs neither a
    binary on PATH nor ownership of the worktree.
    """
    git_dir = REPO_ROOT / ".git"
    try:
        # A worktree or submodule has a `.git` FILE pointing at the real directory.
        if git_dir.is_file():
            pointer = git_dir.read_text(encoding="utf-8").strip()
            if not pointer.startswith("gitdir:"):
                return None
            git_dir = Path(pointer.split(":", 1)[1].strip())
            if not git_dir.is_absolute():
                git_dir = (REPO_ROOT / git_dir).resolve()

        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref:"):
            return head or None  # detached HEAD holds the sha itself

        ref = head.split(":", 1)[1].strip()
        loose = git_dir / ref
        if loose.is_file():
            return loose.read_text(encoding="utf-8").strip() or None

        # A ref that has been packed away has no file of its own.
        packed = git_dir / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line.startswith(("#", "^")):
                    continue
                sha, _, name = line.partition(" ")
                if name.strip() == ref:
                    return sha.strip() or None
    except (OSError, ValueError, IndexError) as exc:
        log.warning("git_dir_unreadable", error=str(exc))
    return None


def git_commit() -> str | None:
    """The deployed commit, from git when it is usable and from ``.git`` otherwise."""
    return _run_git("rev-parse", "HEAD") or _commit_from_git_dir()


def git_dirty() -> bool | None:
    """True when the working tree has uncommitted changes.

    A dirty run is not reproducible from its commit alone, so this is recorded and
    surfaced as a warning rather than quietly ignored.
    """
    status = _run_git("status", "--porcelain")
    return None if status is None else bool(status)


@lru_cache(maxsize=1)
def _package_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    found: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            continue
        except Exception as exc:  # noqa: BLE001
            log.warning("version_lookup_failed", package=name, error=str(exc))
    return found


def _kernel_version() -> str | None:
    try:
        return platform.release()
    except Exception:  # noqa: BLE001
        return None


def _cuda_versions(gpu: NvmlProbe) -> tuple[str | None, str | None]:
    """CUDA and cuDNN versions, from whichever source can answer.

    torch is asked first because it reports the versions it was *built* against,
    which is what actually governs its kernels. NVML's driver-level CUDA version is
    the fallback.
    """
    cuda = cudnn = None
    try:
        import torch

        cuda = getattr(torch.version, "cuda", None)
        if torch.backends.cudnn.is_available():
            raw = torch.backends.cudnn.version()
            if raw:
                # cuDNN encodes as MAJOR*10000 + MINOR*100 + PATCH (e.g. 90100 -> 9.1.0)
                cudnn = f"{raw // 10000}.{(raw % 10000) // 100}.{raw % 100}"
    except Exception:  # noqa: BLE001
        pass

    if cuda is None and gpu.available:
        try:
            import pynvml

            raw = pynvml.nvmlSystemGetCudaDriverVersion()
            cuda = f"{raw // 1000}.{(raw % 1000) // 10}"
        except Exception:  # noqa: BLE001
            pass
    return cuda, cudnn


def collect_hardware(gpu: NvmlProbe | None = None) -> HardwareInfo:
    """Probe the machine. Never raises; absent probes become empty or None fields."""
    owns_probe = gpu is None
    probe = gpu if gpu is not None else NvmlProbe()
    try:
        cpu = cpu_static_info()
        import psutil

        gpus = [
            GpuDescriptor(
                index=g.index,
                name=g.name,
                memory_total_mb=g.memory_total_mb,
                driver_version=g.driver_version,
                compute_capability=g.compute_capability,
                power_limit_w=g.power_limit_w,
            )
            for g in probe.static_info()
        ]
        cuda, cudnn = _cuda_versions(probe)

        return HardwareInfo(
            cpu_model=cpu.model,
            cpu_cores_physical=cpu.cores_physical,
            cpu_cores_logical=cpu.cores_logical,
            cpu_instruction_sets=cpu.instruction_sets,
            cpu_max_freq_mhz=cpu.max_freq_mhz,
            ram_total_mb=int(psutil.virtual_memory().total / (1024 * 1024)),
            gpus=gpus,
            gpu_count=len(gpus),
            cuda_version=cuda,
            cudnn_version=cudnn,
            nvml_available=probe.available,
        )
    finally:
        if owns_probe:
            probe.shutdown()


def collect_software() -> SoftwareEnvironment:
    import os

    return SoftwareEnvironment(
        os=platform.system(),
        os_version=platform.version(),
        kernel_version=_kernel_version(),
        python_version=platform.python_version(),
        node_version=None,
        package_versions=_package_versions(),
        relevant_env_vars={k: os.environ[k] for k in _ALLOWED_ENV_VARS if k in os.environ},
    )


def collect_reproducibility(
    seed: int | None = None,
    deterministic: bool = False,
    dataset_revision: str | None = None,
    reproduction_command: str | None = None,
) -> Reproducibility:
    return Reproducibility(
        git_commit=git_commit(),
        git_dirty=git_dirty(),
        random_seed=seed,
        deterministic_mode=deterministic,
        dataset_revision=dataset_revision,
        reproduction_command=reproduction_command,
    )


def python_executable() -> str:
    return sys.executable


def repo_root() -> Path:
    return REPO_ROOT
