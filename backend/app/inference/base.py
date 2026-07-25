"""Common detection-backend interface.

Every runtime (ONNX, PyTorch, OpenVINO, TensorRT, …) implements ``DetectionBackend``
so the rest of the system is runtime-agnostic. Backends return normalized
``Detection`` objects in original-image pixel coordinates.
"""
from __future__ import annotations

import time
from typing import Protocol, runtime_checkable

import numpy as np

from app.core.types import BenchmarkResult, Detection, HealthState


@runtime_checkable
class DetectionBackend(Protocol):
    """Interface every detection runtime must satisfy."""

    model_id: str
    backend_name: str

    def load(self) -> None: ...
    def warmup(self) -> None: ...
    def predict(
        self,
        image: np.ndarray,
        conf_threshold: float,
        iou_threshold: float,
        allowed_class_ids: set[int] | None = None,
    ) -> list[Detection]: ...
    def benchmark(self, runs: int = 30) -> BenchmarkResult: ...
    def health(self) -> HealthState: ...
    def close(self) -> None: ...


class BaseDetectionBackend:
    """Shared helpers: benchmarking loop, health tracking, RSS sampling.

    Subclasses implement ``load``, ``warmup``, ``predict`` and ``close``.
    """

    model_id: str = "unknown"
    backend_name: str = "base"
    precision: str = "fp32"
    device: str = "cpu"
    input_size: int = 640

    def __init__(self) -> None:
        self._health: HealthState = HealthState.UNKNOWN

    def health(self) -> HealthState:
        return self._health

    @staticmethod
    def _rss_mb() -> float | None:
        try:
            import psutil

            return psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception:
            return None

    def benchmark_frame(self) -> np.ndarray:
        """Deterministic mid-gray frame, so results reflect the runtime path, not scene content."""
        return np.full((self.input_size, self.input_size, 3), 128, dtype=np.uint8)

    def result_from_latencies(self, latencies: list[float], notes: str = "") -> BenchmarkResult:
        """Build a BenchmarkResult from measured latencies. Values are never synthesized."""
        arr = np.array(latencies)
        return BenchmarkResult(
            backend=self.backend_name,
            model_id=self.model_id,
            input_size=self.input_size,
            precision=self.precision,
            device=self.device,
            runs=len(latencies),
            fps=float(1000.0 / arr.mean()) if arr.mean() > 0 else 0.0,
            latency_mean_ms=float(arr.mean()),
            latency_p50_ms=float(np.percentile(arr, 50)),
            latency_p95_ms=float(np.percentile(arr, 95)),
            latency_p99_ms=float(np.percentile(arr, 99)),
            memory_rss_mb=self._rss_mb(),
            provider=getattr(self, "provider", None),
            notes=notes,
        )

    def benchmark(self, runs: int = 30) -> BenchmarkResult:
        """Run ``runs`` inferences on a synthetic frame and measure latency.

        Values are always measured here, never hardcoded.
        """
        if self._health not in (HealthState.READY, HealthState.DEGRADED):
            raise RuntimeError("backend not ready for benchmarking")

        frame = self.benchmark_frame()
        # small warm set excluded from timing
        for _ in range(3):
            self.predict(frame, 0.25, 0.45, None)

        lat: list[float] = []
        for _ in range(max(1, runs)):
            t0 = time.perf_counter()
            self.predict(frame, 0.25, 0.45, None)
            lat.append((time.perf_counter() - t0) * 1000.0)

        return self.result_from_latencies(lat)
