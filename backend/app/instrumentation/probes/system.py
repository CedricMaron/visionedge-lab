"""CPU, memory, disk and network probing through psutil.

``psutil.cpu_percent`` has a stateful quirk that matters for correctness: called
with ``interval=None`` it reports usage *since the previous call on the same
object*, and the very first call always returns 0.0. A sampler that constructs a
fresh probe per tick would therefore record a flat zero CPU line. :class:`SystemProbe`
holds the psutil objects for the life of the run and primes them once at
construction, so the first real sample is already meaningful.
"""
from __future__ import annotations

import os
import platform
import time
from dataclasses import dataclass, field

import psutil

from app.core.logging import get_logger

log = get_logger("instrumentation.system")


@dataclass(slots=True)
class SystemSample:
    cpu_percent: float | None = None
    per_core_percent: list[float] = field(default_factory=list)
    process_cpu_percent: float | None = None
    cpu_freq_mhz: float | None = None
    thread_count: int | None = None
    context_switches: int | None = None
    ram_used_mb: float | None = None
    ram_available_mb: float | None = None
    swap_used_mb: float | None = None
    process_rss_mb: float | None = None
    process_vms_mb: float | None = None
    disk_read_mb_s: float | None = None
    disk_write_mb_s: float | None = None
    net_sent_mb_s: float | None = None
    net_recv_mb_s: float | None = None


@dataclass(slots=True)
class CpuStaticInfo:
    model: str
    cores_physical: int | None
    cores_logical: int
    max_freq_mhz: float | None
    instruction_sets: list[str]


_MB = 1024.0 * 1024.0

#: ISA extensions worth reporting, because each changes which kernels a runtime
#: selects and therefore what a benchmark is actually measuring.
_INTERESTING_FLAGS = (
    "avx", "avx2", "avx512f", "avx512_vnni", "avx_vnni", "amx_int8", "amx_bf16",
    "f16c", "fma", "sse4_1", "sse4_2", "neon", "asimd", "sve",
)


def detect_instruction_sets() -> list[str]:
    """Read ISA extensions from /proc/cpuinfo. Empty list when undetectable."""
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.lower().startswith(("flags", "features")):
                    flags = set(line.split(":", 1)[1].split())
                    return [f for f in _INTERESTING_FLAGS if f in flags]
    except Exception:  # noqa: BLE001 - absent on Windows/macOS, which is not an error
        pass
    return []


def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001
        pass
    return platform.processor() or platform.machine()


def cpu_static_info() -> CpuStaticInfo:
    freq = None
    try:
        f = psutil.cpu_freq()
        freq = f.max or f.current if f else None
    except Exception:  # noqa: BLE001 - unavailable inside many containers
        pass
    return CpuStaticInfo(
        model=cpu_model(),
        cores_physical=psutil.cpu_count(logical=False),
        cores_logical=psutil.cpu_count(logical=True) or 1,
        max_freq_mhz=float(freq) if freq else None,
        instruction_sets=detect_instruction_sets(),
    )


class SystemProbe:
    """Stateful system sampler. One instance per benchmark run."""

    def __init__(self, per_core: bool = False) -> None:
        self.per_core = per_core
        self._process = psutil.Process(os.getpid())
        self._last_disk = None
        self._last_net = None
        self._last_io_t = time.perf_counter()
        self._prime()

    def _prime(self) -> None:
        """Discard the meaningless first reading of every delta-based counter."""
        try:
            psutil.cpu_percent(interval=None)
            if self.per_core:
                psutil.cpu_percent(interval=None, percpu=True)
            self._process.cpu_percent(interval=None)
        except Exception as exc:  # noqa: BLE001
            log.warning("cpu_probe_prime_failed", error=str(exc))
        self._last_disk = self._disk_counters()
        self._last_net = self._net_counters()
        self._last_io_t = time.perf_counter()

    @staticmethod
    def _disk_counters():
        try:
            return psutil.disk_io_counters()
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _net_counters():
        try:
            return psutil.net_io_counters()
        except Exception:  # noqa: BLE001
            return None

    def sample(self, *, include_io: bool = True, include_freq: bool = True) -> SystemSample:
        """Read current system state.

        ``include_io`` and ``include_freq`` gate the two costliest psutil calls
        (``disk_io_counters`` at ~0.41 ms and ``cpu_freq`` at ~0.23 ms), so a
        high-frequency sampler can skip what a given run does not need.
        """
        s = SystemSample()

        try:
            s.cpu_percent = psutil.cpu_percent(interval=None)
            if self.per_core:
                s.per_core_percent = psutil.cpu_percent(interval=None, percpu=True)
        except Exception:  # noqa: BLE001
            pass

        try:
            # Can exceed 100% on a multi-core box; that is correct, not a bug.
            s.process_cpu_percent = self._process.cpu_percent(interval=None)
            with self._process.oneshot():
                mem = self._process.memory_info()
                s.process_rss_mb = mem.rss / _MB
                s.process_vms_mb = mem.vms / _MB
                s.thread_count = self._process.num_threads()
                ctx = self._process.num_ctx_switches()
                s.context_switches = ctx.voluntary + ctx.involuntary
        except Exception:  # noqa: BLE001
            pass

        if include_freq:
            try:
                f = psutil.cpu_freq()
                s.cpu_freq_mhz = float(f.current) if f else None
            except Exception:  # noqa: BLE001
                pass

        try:
            vm = psutil.virtual_memory()
            s.ram_used_mb = vm.used / _MB
            s.ram_available_mb = vm.available / _MB
        except Exception:  # noqa: BLE001
            pass

        if include_io:
            try:
                s.swap_used_mb = psutil.swap_memory().used / _MB
            except Exception:  # noqa: BLE001
                pass
            self._sample_io(s)
        return s

    def _sample_io(self, s: SystemSample) -> None:
        """Convert cumulative byte counters into rates over the elapsed interval."""
        now = time.perf_counter()
        dt = now - self._last_io_t
        if dt <= 0:
            return

        disk = self._disk_counters()
        if disk and self._last_disk:
            s.disk_read_mb_s = (disk.read_bytes - self._last_disk.read_bytes) / _MB / dt
            s.disk_write_mb_s = (disk.write_bytes - self._last_disk.write_bytes) / _MB / dt
        net = self._net_counters()
        if net and self._last_net:
            s.net_sent_mb_s = (net.bytes_sent - self._last_net.bytes_sent) / _MB / dt
            s.net_recv_mb_s = (net.bytes_recv - self._last_net.bytes_recv) / _MB / dt

        self._last_disk = disk
        self._last_net = net
        self._last_io_t = now

    def process_rss_mb(self) -> float | None:
        try:
            return self._process.memory_info().rss / _MB
        except Exception:  # noqa: BLE001
            return None

    def load_average_1m(self) -> float | None:
        try:
            return os.getloadavg()[0]
        except (OSError, AttributeError):
            return None  # not available on Windows

    def detect_background_load(self, threshold_percent: float = 25.0) -> tuple[bool, str | None]:
        """Whether meaningful non-benchmark CPU activity is present.

        System CPU minus this process's share. A result measured against a busy
        machine is still stored, but it is flagged so nobody compares it with one
        measured on an idle machine and concludes the model got slower.
        """
        try:
            total = psutil.cpu_percent(interval=None)
            logical = psutil.cpu_count(logical=True) or 1
            mine = self._process.cpu_percent(interval=None) / logical
            other = max(0.0, total - mine)
            if other >= threshold_percent:
                return True, f"{other:.0f}% CPU in use by other processes during the run"
            return False, None
        except Exception:  # noqa: BLE001
            return False, None
