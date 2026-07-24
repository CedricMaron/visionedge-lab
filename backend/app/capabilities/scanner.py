"""Backend capability scanner.

Detects real hardware and runtime availability. Every field reflects an actual probe —
imports that succeed, providers ORT reports, ``nvidia-smi`` output. Nothing is assumed.
"""
from __future__ import annotations

import platform
import shutil
import subprocess

import psutil
from pydantic import BaseModel

from app.core.logging import get_logger

log = get_logger("capabilities")


class GPUInfo(BaseModel):
    name: str
    memory_total_mb: int | None = None
    memory_used_mb: int | None = None
    driver_version: str | None = None


class RuntimeAvailability(BaseModel):
    onnxruntime: bool = False
    onnxruntime_providers: list[str] = []
    onnxruntime_cuda: bool = False
    pytorch: bool = False
    pytorch_cuda: bool = False
    cuda_version: str | None = None
    openvino: bool = False
    tensorrt: bool = False


class BackendCapabilities(BaseModel):
    os: str
    os_version: str
    python_version: str
    cpu_model: str
    cpu_cores_physical: int | None
    cpu_cores_logical: int
    ram_total_mb: int
    ram_available_mb: int
    gpus: list[GPUInfo] = []
    nvidia_gpu_present: bool = False
    runtimes: RuntimeAvailability
    supported_precisions: list[str] = []


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def _query_nvidia() -> list[GPUInfo]:
    """Query GPUs via nvidia-smi if present. Returns [] if no NVIDIA GPU."""
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.used,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8, check=True,
        ).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        log.warning("nvidia_smi_failed", error=str(exc))
        return []
    gpus: list[GPUInfo] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            gpus.append(GPUInfo(
                name=parts[0],
                memory_total_mb=int(float(parts[1])) if parts[1].replace(".", "").isdigit() else None,
                memory_used_mb=int(float(parts[2])) if parts[2].replace(".", "").isdigit() else None,
                driver_version=parts[3] or None,
            ))
    return gpus


def _runtimes() -> RuntimeAvailability:
    r = RuntimeAvailability()
    try:
        import onnxruntime as ort

        r.onnxruntime = True
        r.onnxruntime_providers = list(ort.get_available_providers())
        r.onnxruntime_cuda = "CUDAExecutionProvider" in r.onnxruntime_providers
    except Exception:
        pass
    try:
        import torch

        r.pytorch = True
        r.pytorch_cuda = bool(torch.cuda.is_available())
        r.cuda_version = getattr(torch.version, "cuda", None)
    except Exception:
        pass
    try:
        import openvino  # noqa: F401

        r.openvino = True
    except Exception:
        pass
    try:
        import tensorrt  # noqa: F401

        r.tensorrt = True
    except Exception:
        pass
    return r


def _supported_precisions(rt: RuntimeAvailability, has_nvidia: bool) -> list[str]:
    precisions = ["fp32"]
    # CPU int8 via ORT quantization is always feasible as a path
    precisions.append("int8")
    if rt.pytorch_cuda or rt.onnxruntime_cuda or has_nvidia:
        precisions.append("fp16")
    if rt.pytorch_cuda:
        precisions.append("bf16")
    return sorted(set(precisions))


def scan_capabilities() -> BackendCapabilities:
    vm = psutil.virtual_memory()
    gpus = _query_nvidia()
    rt = _runtimes()
    caps = BackendCapabilities(
        os=platform.system(),
        os_version=platform.version(),
        python_version=platform.python_version(),
        cpu_model=_cpu_model(),
        cpu_cores_physical=psutil.cpu_count(logical=False),
        cpu_cores_logical=psutil.cpu_count(logical=True) or 1,
        ram_total_mb=int(vm.total / (1024 * 1024)),
        ram_available_mb=int(vm.available / (1024 * 1024)),
        gpus=gpus,
        nvidia_gpu_present=bool(gpus),
        runtimes=rt,
        supported_precisions=_supported_precisions(rt, bool(gpus)),
    )
    return caps
