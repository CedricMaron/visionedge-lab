"""Hardware, software and reproducibility metadata attached to every run.

The environment fingerprint (§13) is the mechanism that lets results be grouped:
two runs with the same fingerprint were produced on equivalent configurations and
may be aggregated; two runs with different fingerprints may be *compared* but never
pooled.
"""
from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, Field

from app.schemas.enums import BenchmarkMode, DeviceKind, Precision


class GpuDescriptor(BaseModel):
    index: int
    name: str
    memory_total_mb: int | None = None
    driver_version: str | None = None
    compute_capability: str | None = None
    power_limit_w: float | None = None


class HardwareInfo(BaseModel):
    cpu_model: str
    cpu_cores_physical: int | None = None
    cpu_cores_logical: int
    cpu_instruction_sets: list[str] = Field(
        default_factory=list,
        description="Detected ISA extensions (avx2, avx512f, …). Empty when undetectable.",
    )
    cpu_max_freq_mhz: float | None = None
    ram_total_mb: int
    gpus: list[GpuDescriptor] = Field(default_factory=list)
    gpu_count: int = 0
    cuda_version: str | None = None
    cudnn_version: str | None = None
    nvml_available: bool = False


class SoftwareEnvironment(BaseModel):
    os: str
    os_version: str
    kernel_version: str | None = None
    python_version: str
    node_version: str | None = None
    package_versions: dict[str, str] = Field(
        default_factory=dict,
        description="Versions of packages that affect execution: onnxruntime, torch, numpy, …",
    )
    relevant_env_vars: dict[str, str] = Field(
        default_factory=dict,
        description="Execution-affecting variables only (OMP_NUM_THREADS, CUDA_VISIBLE_DEVICES, …). "
                    "Never contains secrets — the collector allow-lists names.",
    )


class ModelReference(BaseModel):
    model_id: str
    display_name: str = ""
    revision: str | None = Field(default=None, description="Upstream commit or model revision.")
    weights_checksum_sha256: str | None = None
    parameters_millions: float | None = None
    file_size_bytes: int | None = None


class RuntimeReference(BaseModel):
    runtime_id: str
    runtime_version: str | None = None
    execution_provider: str | None = None
    device: DeviceKind
    device_index: int = 0
    precision: Precision
    quantization: str | None = None
    compilation: dict[str, str] = Field(default_factory=dict)
    optimization_level: str | None = None
    thread_config: dict[str, int] = Field(default_factory=dict)
    backend_options: dict[str, str] = Field(default_factory=dict)


class Reproducibility(BaseModel):
    """Everything needed to re-run this benchmark and expect the same answer."""

    git_commit: str | None = None
    git_dirty: bool | None = Field(
        default=None,
        description="True when the working tree had uncommitted changes. A dirty run is "
                    "not reproducible from the commit alone and is flagged as such.",
    )
    random_seed: int | None = None
    deterministic_mode: bool = False
    dataset_revision: str | None = None
    reproduction_command: str | None = Field(
        default=None,
        description="A runnable `inference-lab benchmark run …` invocation reproducing this run.",
    )


class EnvironmentFingerprint(BaseModel):
    """Stable hash over the execution-relevant parts of the environment.

    Deliberately excludes timestamps, run ids, thermal state and anything else that
    varies between two otherwise-identical runs — those belong in the result, not in
    the identity of the configuration.
    """

    digest: str
    components: dict[str, str]

    @classmethod
    def compute(
        cls,
        hardware: HardwareInfo,
        software: SoftwareEnvironment,
        model: ModelReference,
        runtime: RuntimeReference,
        mode: BenchmarkMode,
    ) -> EnvironmentFingerprint:
        components = {
            "cpu_model": hardware.cpu_model,
            "cpu_cores_logical": str(hardware.cpu_cores_logical),
            "ram_total_mb": str(hardware.ram_total_mb),
            "gpus": ",".join(sorted(g.name for g in hardware.gpus)) or "none",
            "cuda_version": hardware.cuda_version or "none",
            "os": f"{software.os} {software.os_version}",
            "python": software.python_version,
            "packages": json.dumps(software.package_versions, sort_keys=True),
            "model_id": model.model_id,
            "model_revision": model.revision or "none",
            "weights_checksum": model.weights_checksum_sha256 or "none",
            "runtime": runtime.runtime_id,
            "runtime_version": runtime.runtime_version or "none",
            "execution_provider": runtime.execution_provider or "none",
            "device": runtime.device.value,
            "precision": runtime.precision.value,
            "quantization": runtime.quantization or "none",
            "benchmark_mode": mode.value,
        }
        payload = json.dumps(components, sort_keys=True).encode()
        return cls(digest=hashlib.sha256(payload).hexdigest()[:16], components=components)


class ThermalAndLoadState(BaseModel):
    """Conditions that invalidate a comparison if they differ, recorded so they can be checked."""

    gpu_temperature_start_c: float | None = None
    gpu_temperature_end_c: float | None = None
    thermal_throttling_detected: bool | None = Field(
        default=None,
        description="From NVML clock-throttle reasons. None when the probe is unavailable.",
    )
    system_load_average_1m: float | None = None
    concurrent_workload_detected: bool = Field(
        default=False,
        description="True when non-benchmark CPU/GPU activity was observed during the run. "
                    "Such results are still stored, but the UI warns before comparing them.",
    )
    concurrent_workload_detail: str | None = None
