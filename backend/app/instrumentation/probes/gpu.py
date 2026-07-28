"""NVIDIA GPU probing through NVML.

NVML is used in preference to shelling out to ``nvidia-smi``: a subprocess costs
several milliseconds per call, which is far too expensive to sample at 100 ms
intervals during a benchmark, and it cannot report per-process memory cleanly.

Every method degrades to ``None`` rather than raising. A CPU-only host (the
production VPS) must run the whole platform with this probe reporting unavailable,
and a benchmark must not fail because a temperature sensor is missing.

A note on the memory fields, because this is where GPU benchmark reports usually
mislead: NVML reports *device* memory — everything every process on the card is
using, including the display server. It is not this process's allocation, and it
is not the framework allocator's arena. Those three numbers are kept in separate
fields with separate names.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from app.core.logging import get_logger

log = get_logger("instrumentation.gpu")


@dataclass(slots=True)
class GpuSample:
    """One instantaneous read of a GPU's state. Fields are None when unsupported."""

    index: int
    utilization_percent: float | None = None
    memory_utilization_percent: float | None = None
    memory_used_mb: float | None = None
    memory_total_mb: float | None = None
    process_memory_mb: float | None = None
    graphics_clock_mhz: float | None = None
    memory_clock_mhz: float | None = None
    temperature_c: float | None = None
    power_w: float | None = None
    power_limit_w: float | None = None
    encoder_percent: float | None = None
    decoder_percent: float | None = None
    throttle_reasons: list[str] | None = None


@dataclass(slots=True)
class GpuStaticInfo:
    index: int
    name: str
    memory_total_mb: int | None
    driver_version: str | None
    compute_capability: str | None
    power_limit_w: float | None


# NVML clock-throttle reason bits worth surfacing. A run that thermally throttled
# produced valid measurements of a *throttled* device, which is a different thing
# from the same device unthrottled — the result carries a warning.
_THROTTLE_BITS: tuple[tuple[str, int], ...] = (
    ("gpu_idle", 1),
    ("applications_clocks_setting", 2),
    ("sw_power_cap", 4),
    ("hw_slowdown", 8),
    ("sync_boost", 16),
    ("sw_thermal_slowdown", 32),
    ("hw_thermal_slowdown", 64),
    ("hw_power_brake_slowdown", 128),
    ("display_clock_setting", 256),
)


class NvmlProbe:
    """Wraps pynvml. Construct once and reuse; init/shutdown are not free."""

    def __init__(self) -> None:
        self._nvml: Any | None = None
        self._handles: list[Any] = []
        self._unavailable_reason: str | None = None
        self._pid = os.getpid()
        self._init()

    # --- lifecycle -------------------------------------------------------

    def _init(self) -> None:
        try:
            import pynvml
        except Exception as exc:  # noqa: BLE001
            self._unavailable_reason = (
                f"pynvml (nvidia-ml-py) is not importable: {type(exc).__name__}: {exc}"
            )
            return
        try:
            pynvml.nvmlInit()
        except Exception as exc:  # noqa: BLE001
            # The usual causes: no NVIDIA driver, or a container without the device.
            self._unavailable_reason = f"nvmlInit() failed: {type(exc).__name__}: {exc}"
            return

        self._nvml = pynvml
        try:
            count = pynvml.nvmlDeviceGetCount()
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
        except Exception as exc:  # noqa: BLE001
            self._unavailable_reason = f"could not enumerate devices: {exc}"
            self._nvml = None
            return

        if not self._handles:
            self._unavailable_reason = "NVML initialized but reported zero devices"
            self._nvml = None

    @property
    def available(self) -> bool:
        return self._nvml is not None and bool(self._handles)

    @property
    def unavailable_reason(self) -> str | None:
        return self._unavailable_reason

    @property
    def device_count(self) -> int:
        return len(self._handles)

    def shutdown(self) -> None:
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception as exc:  # noqa: BLE001
                log.warning("nvml_shutdown_failed", error=str(exc))
            self._nvml = None
            self._handles = []

    # --- helpers ---------------------------------------------------------

    @staticmethod
    def _try(fn, *args):
        """Call an NVML function, returning None when the device does not support it.

        Support varies by card: a GeForce reports power draw but not always encoder
        utilization, and NVML raises NVMLError_NotSupported for the missing ones.
        An unsupported metric is legitimately absent, not an error condition.
        """
        try:
            return fn(*args)
        except Exception:  # noqa: BLE001
            return None

    def _decode(self, value: Any) -> str | None:
        if value is None:
            return None
        return value.decode() if isinstance(value, bytes) else str(value)

    # --- reads -----------------------------------------------------------

    def static_info(self) -> list[GpuStaticInfo]:
        """Per-device properties that do not change during a run."""
        if not self.available:
            return []
        n = self._nvml
        out: list[GpuStaticInfo] = []
        driver = self._decode(self._try(n.nvmlSystemGetDriverVersion))
        for i, handle in enumerate(self._handles):
            mem = self._try(n.nvmlDeviceGetMemoryInfo, handle)
            cap = self._try(n.nvmlDeviceGetCudaComputeCapability, handle)
            limit = self._try(n.nvmlDeviceGetPowerManagementLimit, handle)
            out.append(
                GpuStaticInfo(
                    index=i,
                    name=self._decode(self._try(n.nvmlDeviceGetName, handle)) or f"gpu{i}",
                    memory_total_mb=int(mem.total / (1024 * 1024)) if mem else None,
                    driver_version=driver,
                    compute_capability=f"{cap[0]}.{cap[1]}" if cap else None,
                    power_limit_w=(limit / 1000.0) if limit else None,
                )
            )
        return out

    def sample(
        self,
        index: int = 0,
        *,
        include_clocks: bool = True,
        include_codec: bool = False,
        include_throttle: bool = False,
        include_process_memory: bool = True,
    ) -> GpuSample | None:
        """Read the current state of one device. None when NVML is unavailable.

        The flags exist because NVML calls differ in cost by two orders of
        magnitude, and a sampler that pays for all of them cannot tick quickly
        without distorting the workload it is measuring. Measured on an RTX 2060:

            throttle reasons     15.0 ms   <- excluded from per-tick sampling
            clocks (graphics+mem) 2.7 ms
            power                 1.4 ms
            encoder + decoder     1.1 ms   <- video workloads only
            utilization           0.7 ms
            process memory        0.4 ms
            memory info           0.15 ms
            temperature           0.13 ms

        Throttle state changes on a thermal timescale (seconds), so it is captured
        once at the start and end of a run via :meth:`is_throttling` rather than on
        every tick. Codec utilization is meaningless outside video workloads.
        """
        if not self.available or index >= len(self._handles):
            return None
        n = self._nvml
        handle = self._handles[index]

        util = self._try(n.nvmlDeviceGetUtilizationRates, handle)
        mem = self._try(n.nvmlDeviceGetMemoryInfo, handle)
        power = self._try(n.nvmlDeviceGetPowerUsage, handle)
        temp = self._try(n.nvmlDeviceGetTemperature, handle, 0)  # 0 = NVML_TEMPERATURE_GPU

        gclk = mclk = None
        if include_clocks:
            gclk = self._try(n.nvmlDeviceGetClockInfo, handle, 0)  # 0 = graphics
            mclk = self._try(n.nvmlDeviceGetClockInfo, handle, 2)  # 2 = memory

        enc = dec = None
        if include_codec:
            enc = self._try(n.nvmlDeviceGetEncoderUtilization, handle)
            dec = self._try(n.nvmlDeviceGetDecoderUtilization, handle)

        return GpuSample(
            index=index,
            utilization_percent=float(util.gpu) if util else None,
            memory_utilization_percent=float(util.memory) if util else None,
            memory_used_mb=(mem.used / (1024 * 1024)) if mem else None,
            memory_total_mb=(mem.total / (1024 * 1024)) if mem else None,
            process_memory_mb=self._process_memory_mb(handle) if include_process_memory else None,
            graphics_clock_mhz=float(gclk) if gclk is not None else None,
            memory_clock_mhz=float(mclk) if mclk is not None else None,
            temperature_c=float(temp) if temp is not None else None,
            power_w=(power / 1000.0) if power is not None else None,
            power_limit_w=self.power_limit_w(index),
            encoder_percent=float(enc[0]) if enc else None,
            decoder_percent=float(dec[0]) if dec else None,
            throttle_reasons=self._throttle_reasons(handle) if include_throttle else None,
        )

    @lru_cache(maxsize=8)  # noqa: B019 - bounded by device count; probe lives for the process
    def power_limit_w(self, index: int = 0) -> float | None:
        """Configured power cap. Constant for a run, so it is read once and cached."""
        if not self.available or index >= len(self._handles):
            return None
        raw = self._try(self._nvml.nvmlDeviceGetPowerManagementLimit, self._handles[index])
        return (raw / 1000.0) if raw is not None else None

    def _process_memory_mb(self, handle: Any) -> float | None:
        """VRAM attributed to *this* process, as opposed to the whole device.

        Reported separately from device-wide usage so a shared GPU cannot inflate
        what a benchmark appears to have consumed.
        """
        n = self._nvml
        procs = self._try(n.nvmlDeviceGetComputeRunningProcesses_v3, handle)
        if procs is None:
            procs = self._try(n.nvmlDeviceGetComputeRunningProcesses, handle)
        if not procs:
            return None
        for p in procs:
            if getattr(p, "pid", None) == self._pid:
                used = getattr(p, "usedGpuMemory", None)
                # NVML returns None for used memory when it cannot attribute it.
                return (used / (1024 * 1024)) if used else None
        return None

    def _throttle_reasons(self, handle: Any) -> list[str] | None:
        bits = self._try(self._nvml.nvmlDeviceGetCurrentClocksThrottleReasons, handle)
        if bits is None:
            return None
        return [name for name, bit in _THROTTLE_BITS if bits & bit]

    def is_throttling(self, index: int = 0) -> bool | None:
        """True when the device reports a thermal or power throttle right now.

        Costs ~15 ms, so this is called at run boundaries rather than per tick.
        """
        if not self.available or index >= len(self._handles):
            return None
        reasons = self._throttle_reasons(self._handles[index])
        if reasons is None:
            return None
        serious = {"sw_power_cap", "hw_slowdown", "sw_thermal_slowdown",
                   "hw_thermal_slowdown", "hw_power_brake_slowdown"}
        return bool(serious.intersection(reasons))

    def temperature_c(self, index: int = 0) -> float | None:
        """Cheap standalone temperature read, for run-start/end thermal state."""
        if not self.available or index >= len(self._handles):
            return None
        raw = self._try(self._nvml.nvmlDeviceGetTemperature, self._handles[index], 0)
        return float(raw) if raw is not None else None
