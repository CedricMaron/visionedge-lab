"""Memory snapshots taken at named points in a run's lifecycle.

The point of taking several snapshots rather than one number is that "how much
memory does this model need" has different answers depending on when you ask:
before load, after load, at peak during inference, and after unload. The delta
between the first two is the weight footprint; a gap between the first and last
is a leak indicator.

Terminology is kept strictly separate — see :mod:`app.schemas.resources` for why
allocated, reserved, process and device totals are four different quantities.
"""
from __future__ import annotations

from app.instrumentation.probes.gpu import NvmlProbe
from app.instrumentation.probes.system import SystemProbe
from app.schemas.measurement import FloatMeasurement, Measurement
from app.schemas.resources import MemoryMetrics, MemorySnapshot

_PSUTIL = "psutil.Process.memory_info"
_NVML_MEM = "NVML nvmlDeviceGetMemoryInfo"

#: Framework allocator statistics (torch.cuda.memory_allocated / memory_reserved) are
#: the only true source for these two. ONNX Runtime does not expose an equivalent, so
#: on the ORT path they are honestly unavailable rather than approximated from NVML.
_NO_ALLOCATOR_STATS = (
    "ONNX Runtime does not expose framework allocator statistics; NVML reports "
    "device-wide usage, which is a different quantity and is reported separately"
)


def _f(value: float | None, unit: str, source: str, reason: str) -> FloatMeasurement:
    if value is None:
        return Measurement[float].unavailable(reason, unit=unit, source=source)
    return Measurement[float].of(value, unit=unit, source=source)


def snapshot(
    label: str,
    system: SystemProbe,
    gpu: NvmlProbe | None = None,
    gpu_index: int = 0,
) -> MemorySnapshot:
    """Capture memory state at one named moment."""
    s = system.sample()
    gpu_sample = gpu.sample(gpu_index) if (gpu and gpu.available) else None
    gpu_reason = (
        (gpu.unavailable_reason if gpu else None) or "no GPU probe available on this host"
    )

    return MemorySnapshot(
        label=label,
        process_rss_mb=_f(s.process_rss_mb, "MB", _PSUTIL, "process memory unreadable"),
        process_vms_mb=_f(s.process_vms_mb, "MB", _PSUTIL, "process memory unreadable"),
        system_used_mb=_f(s.ram_used_mb, "MB", "psutil.virtual_memory", "system memory unreadable"),
        system_available_mb=_f(
            s.ram_available_mb, "MB", "psutil.virtual_memory", "system memory unreadable"
        ),
        gpu_allocated_mb=Measurement[float].unavailable(_NO_ALLOCATOR_STATS, "MB", "framework allocator"),
        gpu_reserved_mb=Measurement[float].unavailable(_NO_ALLOCATOR_STATS, "MB", "framework allocator"),
        gpu_process_used_mb=_f(
            gpu_sample.process_memory_mb if gpu_sample else None, "MB",
            "NVML nvmlDeviceGetComputeRunningProcesses",
            gpu_reason if not gpu_sample else
            "NVML could not attribute device memory to this process",
        ),
        gpu_device_used_mb=_f(
            gpu_sample.memory_used_mb if gpu_sample else None, "MB", _NVML_MEM, gpu_reason
        ),
        gpu_device_total_mb=_f(
            gpu_sample.memory_total_mb if gpu_sample else None, "MB", _NVML_MEM, gpu_reason
        ),
    )


def build_memory_metrics(
    snapshots: list[MemorySnapshot],
    peak_rss_mb: float | None,
    kv_cache_mb: float | None = None,
) -> MemoryMetrics:
    """Assemble run-level memory metrics from ordered snapshots.

    Expects snapshots labelled ``before_load``, ``after_load`` and ``after_run``;
    any that is missing turns the metric that depends on it into an explicit
    unavailable rather than a silent zero.
    """
    by_label = {s.label: s for s in snapshots}

    def rss(label: str) -> float | None:
        snap = by_label.get(label)
        return snap.process_rss_mb.value if snap and snap.process_rss_mb.available else None

    def device_used(label: str) -> float | None:
        snap = by_label.get(label)
        return snap.gpu_device_used_mb.value if snap and snap.gpu_device_used_mb.available else None

    before, after, end = rss("before_load"), rss("after_load"), rss("after_run")

    # Prefer the GPU delta when the model is on the device; fall back to RSS on CPU.
    gpu_before, gpu_after = device_used("before_load"), device_used("after_load")
    if gpu_before is not None and gpu_after is not None:
        weights = Measurement[float].derived(
            gpu_after - gpu_before, "MB", _NVML_MEM,
            note="device memory growth across model load; includes the CUDA context, "
                 "which is charged to the first model loaded on the device",
        )
    elif before is not None and after is not None:
        weights = Measurement[float].derived(
            after - before, "MB", _PSUTIL,
            note="process RSS growth across model load",
        )
    else:
        weights = Measurement[float].unavailable(
            "before_load and after_load snapshots are required to attribute weight memory", "MB"
        )

    if before is not None and end is not None:
        leak = Measurement[float].derived(
            end - before, "MB", _PSUTIL,
            note="process RSS after the run minus before load. Non-zero is expected "
                 "(the model is still resident); growth across repeated identical runs "
                 "is the signal worth investigating",
        )
    else:
        leak = Measurement[float].unavailable(
            "before_load and after_run snapshots are required to compute a leak indicator", "MB"
        )

    return MemoryMetrics(
        snapshots=snapshots,
        peak_process_rss_mb=_f(peak_rss_mb, "MB", _PSUTIL, "no samples captured during the run"),
        peak_gpu_allocated_mb=Measurement[float].unavailable(_NO_ALLOCATOR_STATS, "MB"),
        peak_gpu_reserved_mb=Measurement[float].unavailable(_NO_ALLOCATOR_STATS, "MB"),
        model_weights_mb=weights,
        kv_cache_mb=(
            Measurement[float].of(kv_cache_mb, "MB", "runtime allocator")
            if kv_cache_mb is not None
            else Measurement[float].unavailable(
                "this workload is not generative, so it has no KV cache", "MB"
            )
        ),
        leak_indicator_mb=leak,
    )
