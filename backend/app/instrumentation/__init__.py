"""Measurement: timing, hardware probes, memory, energy and environment capture.

Kept strictly separate from model and runtime adapters. Adapters execute; this
package measures. An adapter that timed itself would double-count against the
engine's spans and make two adapters' numbers incomparable.
"""
from __future__ import annotations

from app.instrumentation.energy import integrate_energy, trapezoidal_energy_j
from app.instrumentation.environment import (
    collect_hardware,
    collect_reproducibility,
    collect_software,
    git_commit,
    git_dirty,
)
from app.instrumentation.memory import build_memory_metrics, snapshot
from app.instrumentation.probes.gpu import GpuSample, NvmlProbe
from app.instrumentation.probes.system import SystemProbe, cpu_static_info
from app.instrumentation.sampler import INTERVAL_MS_BY_MODE, HardwareSampler
from app.instrumentation.timeline import Timeline, measure_ms

__all__ = [
    "INTERVAL_MS_BY_MODE",
    "GpuSample",
    "HardwareSampler",
    "NvmlProbe",
    "SystemProbe",
    "Timeline",
    "build_memory_metrics",
    "collect_hardware",
    "collect_reproducibility",
    "collect_software",
    "cpu_static_info",
    "git_commit",
    "git_dirty",
    "integrate_energy",
    "measure_ms",
    "snapshot",
    "trapezoidal_energy_j",
]
