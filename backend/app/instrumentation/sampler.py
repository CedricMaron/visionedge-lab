"""Background hardware sampler.

Runs on its own thread and ticks at a fixed interval, collecting a merged
CPU/RAM/GPU sample each time. Three design choices are worth stating, because §27
of the brief requires the benchmarking platform not to distort what it measures:

**The sampler measures its own cost.** Every tick times itself, and the aggregate
is reported as ``sampler_overhead_ms``. If sampling is expensive relative to the
interval, the number says so instead of the distortion being invisible.

**It never blocks the workload.** Sampling happens on a separate thread and writes
into a plain list; the benchmark thread never waits on it. The GIL means sampling
still competes for interpreter time, which is exactly what the measured overhead
quantifies.

**Drift is corrected against a fixed origin.** Sleeping for a constant interval
accumulates error, so each tick's deadline is computed from the start time rather
than from the previous wake-up. A 100 ms sampler stays at 100 ms rather than
sliding to 105 ms.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from app.core.logging import get_logger
from app.instrumentation.probes.gpu import NvmlProbe
from app.instrumentation.probes.system import SystemProbe
from app.schemas.measurement import Measurement
from app.schemas.resources import UtilizationSample, UtilizationSeries

log = get_logger("instrumentation.sampler")

#: Sampling cadence per benchmark mode.
#:
#: Standard is 250 ms rather than 1 s. At 1 s a typical short run collected only two
#: power samples, below the minimum needed to integrate a power series, so energy was
#: never available in the default mode. At 250 ms, with lean detail measured at
#: ~5 ms per tick under load, the sampler costs about 2% of one thread — small
#: against a workload saturating several — and a run of one second upwards yields a
#: usable power series.
INTERVAL_MS_BY_MODE = {
    "standard": 250.0,
    "detailed": 100.0,
    "profiler": 100.0,
}


@dataclass(frozen=True, slots=True)
class SamplingDetail:
    """Which probes each tick pays for.

    Cost is not uniform across probes — see the measured table in
    :meth:`NvmlProbe.sample`. Sampling everything took 28.9 ms per tick on the
    reference machine, which at a 100 ms interval is 29% overhead and would
    materially distort the workload being measured. These presets keep a tick
    cheap enough that the series reflects the workload rather than the observer.
    """

    per_core_cpu: bool = False
    cpu_freq: bool = False
    disk_and_network_io: bool = False
    gpu_clocks: bool = False
    gpu_codec: bool = False
    gpu_process_memory: bool = False

    @classmethod
    def lean(cls) -> SamplingDetail:
        """Standard mode: utilization, memory, power, temperature. ~2.5 ms/tick."""
        return cls()

    @classmethod
    def full(cls) -> SamplingDetail:
        """Detailed mode: adds clocks, per-core CPU, frequency and I/O. ~6 ms/tick.

        Still excludes throttle reasons (15 ms), which are captured at run
        boundaries instead, and codec utilization, which only video workloads need.
        """
        return cls(
            per_core_cpu=True,
            cpu_freq=True,
            disk_and_network_io=True,
            gpu_clocks=True,
            gpu_process_memory=True,
        )

    @classmethod
    def for_mode(cls, mode: str) -> SamplingDetail:
        return cls.full() if mode in ("detailed", "profiler") else cls.lean()


@dataclass(slots=True)
class SamplerStats:
    ticks: int
    total_overhead_ms: float
    max_tick_ms: float
    missed_deadlines: int

    @property
    def mean_overhead_ms(self) -> float:
        return self.total_overhead_ms / self.ticks if self.ticks else 0.0


class HardwareSampler:
    """Collects a utilization time series for the duration of a run."""

    def __init__(
        self,
        interval_ms: float = 1000.0,
        detail: SamplingDetail | None = None,
        gpu_index: int = 0,
        gpu_probe: NvmlProbe | None = None,
    ) -> None:
        if interval_ms <= 0:
            raise ValueError("interval_ms must be positive")
        self.interval_ms = interval_ms
        self.gpu_index = gpu_index
        self.detail = detail if detail is not None else SamplingDetail.lean()

        self._system = SystemProbe(per_core=self.detail.per_core_cpu)
        # An externally-supplied probe is not owned by this sampler and is not shut
        # down by it — several runs share one NVML init.
        self._owns_gpu_probe = gpu_probe is None
        self._gpu = gpu_probe if gpu_probe is not None else NvmlProbe()

        self._samples: list[UtilizationSample] = []
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._origin: float | None = None
        self._ticks = 0
        self._overhead_ms = 0.0
        self._max_tick_ms = 0.0
        self._missed = 0

    # --- lifecycle -------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("sampler already started")
        self._stop.clear()
        self._origin = time.perf_counter()
        self._thread = threading.Thread(
            target=self._loop, name="inferencelab-sampler", daemon=True
        )
        self._thread.start()

    def stop(self, timeout_s: float | None = None) -> None:
        """Stop sampling and join the thread.

        The default timeout scales with the sampling interval. A fixed 2 s was too
        short for a 1 s sampler whose tick had just begun: the join gave up, the
        thread kept ticking, and it went on perturbing whatever ran next — which
        showed up as a slower-than-expected result attributed to the wrong cause.
        """
        if self._thread is None:
            return
        if timeout_s is None:
            timeout_s = (self.interval_ms / 1000.0) * 2.0 + 2.0
        self._stop.set()
        self._thread.join(timeout=timeout_s)
        if self._thread.is_alive():
            log.warning("sampler_thread_did_not_stop", timeout_s=timeout_s)
        self._thread = None
        if self._owns_gpu_probe:
            self._gpu.shutdown()

    def __enter__(self) -> HardwareSampler:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # --- sampling --------------------------------------------------------

    def _loop(self) -> None:
        assert self._origin is not None
        interval_s = self.interval_ms / 1000.0
        tick = 0
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                self._samples.append(self._collect(t0))
            except Exception as exc:  # noqa: BLE001 - sampling must never kill a run
                log.warning("sample_failed", error=str(exc))
            cost = time.perf_counter() - t0
            self._ticks += 1
            self._overhead_ms += cost * 1000.0
            self._max_tick_ms = max(self._max_tick_ms, cost * 1000.0)

            # Deadline from the fixed origin, so error does not accumulate.
            tick += 1
            deadline = self._origin + tick * interval_s
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                # Sampling took longer than the interval — record it rather than
                # spinning, and resync so the series does not fall permanently behind.
                self._missed += 1
                tick = max(tick, int((time.perf_counter() - self._origin) / interval_s) + 1)
                continue
            self._stop.wait(remaining)

    def _collect(self, now: float) -> UtilizationSample:
        assert self._origin is not None
        d = self.detail
        sys_sample = self._system.sample(
            include_io=d.disk_and_network_io, include_freq=d.cpu_freq
        )
        gpu_sample = (
            self._gpu.sample(
                self.gpu_index,
                include_clocks=d.gpu_clocks,
                include_codec=d.gpu_codec,
                include_throttle=False,  # 15 ms; captured at run boundaries instead
                include_process_memory=d.gpu_process_memory,
            )
            if self._gpu.available
            else None
        )

        return UtilizationSample(
            t_offset_ms=(now - self._origin) * 1000.0,
            cpu_percent=sys_sample.cpu_percent,
            cpu_per_core_percent=sys_sample.per_core_percent,
            process_cpu_percent=sys_sample.process_cpu_percent,
            cpu_freq_mhz=sys_sample.cpu_freq_mhz,
            thread_count=sys_sample.thread_count,
            context_switches=sys_sample.context_switches,
            ram_used_mb=sys_sample.ram_used_mb,
            swap_used_mb=sys_sample.swap_used_mb,
            gpu_percent=gpu_sample.utilization_percent if gpu_sample else None,
            gpu_memory_percent=gpu_sample.memory_utilization_percent if gpu_sample else None,
            gpu_memory_used_mb=gpu_sample.memory_used_mb if gpu_sample else None,
            gpu_clock_mhz=gpu_sample.graphics_clock_mhz if gpu_sample else None,
            gpu_memory_clock_mhz=gpu_sample.memory_clock_mhz if gpu_sample else None,
            gpu_temperature_c=gpu_sample.temperature_c if gpu_sample else None,
            gpu_power_w=gpu_sample.power_w if gpu_sample else None,
            gpu_power_limit_w=gpu_sample.power_limit_w if gpu_sample else None,
            gpu_encoder_percent=gpu_sample.encoder_percent if gpu_sample else None,
            gpu_decoder_percent=gpu_sample.decoder_percent if gpu_sample else None,
            disk_read_mb_s=sys_sample.disk_read_mb_s,
            disk_write_mb_s=sys_sample.disk_write_mb_s,
            net_sent_mb_s=sys_sample.net_sent_mb_s,
            net_recv_mb_s=sys_sample.net_recv_mb_s,
        )

    # --- results ---------------------------------------------------------

    @property
    def stats(self) -> SamplerStats:
        return SamplerStats(
            ticks=self._ticks,
            total_overhead_ms=self._overhead_ms,
            max_tick_ms=self._max_tick_ms,
            missed_deadlines=self._missed,
        )

    def series(self) -> UtilizationSeries:
        """The collected series, with its provenance and any missing probes named."""
        sources: list[str] = ["psutil"]
        unavailable: dict[str, str] = {}
        if self._gpu.available:
            sources.append("NVML")
        else:
            unavailable["NVML"] = self._gpu.unavailable_reason or "GPU probe unavailable"

        stats = self.stats
        overhead = (
            Measurement[float].of(
                stats.mean_overhead_ms, "ms", "HardwareSampler self-timing",
                note=(
                    f"mean per tick over {stats.ticks} ticks; max {stats.max_tick_ms:.2f} ms"
                    + (f"; {stats.missed_deadlines} tick(s) exceeded the interval"
                       if stats.missed_deadlines else "")
                ),
            )
            if stats.ticks
            else Measurement[float].unavailable("sampler collected no ticks", "ms")
        )

        return UtilizationSeries(
            samples=list(self._samples),
            sample_interval_ms=self.interval_ms,
            sampler_overhead_ms=overhead,
            sources=sources,
            unavailable=unavailable,
        )
