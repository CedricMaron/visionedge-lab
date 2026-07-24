"""Metrics: Prometheus counters/histograms + an in-process rolling latency aggregator.

The rolling aggregator powers the Performance page (P50/P95/P99, FPS, dropped frames)
without needing a Prometheus server. Prometheus exposition is also provided for
production scraping via /metrics.
"""
from __future__ import annotations

import time
from collections import deque
from threading import Lock

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

# --- Prometheus registry (global default) ---
FRAMES_TOTAL = Counter("ve_frames_total", "Frames processed", ["backend"])
FRAMES_DROPPED = Counter("ve_frames_dropped_total", "Frames dropped due to backpressure")
INFERENCE_LATENCY = Histogram(
    "ve_inference_latency_ms", "Inference latency (ms)", ["backend"],
    buckets=(5, 10, 20, 40, 80, 160, 320, 640, 1280),
)
ACTIVE_SESSIONS = Gauge("ve_active_sessions", "Active WebSocket sessions")
MODEL_LOAD_SECONDS = Histogram("ve_model_load_seconds", "Model load time (s)")


def prometheus_exposition() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST


class RollingMetrics:
    """Thread-safe rolling window of latency samples + frame counters."""

    def __init__(self, window: int = 300) -> None:
        self._lat: deque[float] = deque(maxlen=window)
        self._e2e: deque[float] = deque(maxlen=window)
        self._ts: deque[float] = deque(maxlen=window)
        self._lock = Lock()
        self.processed = 0
        self.dropped = 0
        self.skipped = 0

    def record(self, inference_ms: float, end_to_end_ms: float | None = None) -> None:
        with self._lock:
            self._lat.append(inference_ms)
            if end_to_end_ms is not None:
                self._e2e.append(end_to_end_ms)
            self._ts.append(time.time())
            self.processed += 1

    def record_drop(self) -> None:
        with self._lock:
            self.dropped += 1

    @staticmethod
    def _pct(data, q: float) -> float:
        if not data:
            return 0.0
        s = sorted(data)
        k = max(0, min(len(s) - 1, int(round((q / 100.0) * (len(s) - 1)))))
        return float(s[k])

    def snapshot(self) -> dict:
        with self._lock:
            lat = list(self._lat)
            e2e = list(self._e2e)
            ts = list(self._ts)
        fps = 0.0
        if len(ts) >= 2:
            span = ts[-1] - ts[0]
            fps = (len(ts) - 1) / span if span > 0 else 0.0
        mean = sum(lat) / len(lat) if lat else 0.0
        return {
            "processed_fps": round(fps, 2),
            "inference_latency_mean_ms": round(mean, 2),
            "inference_latency_p50_ms": round(self._pct(lat, 50), 2),
            "inference_latency_p95_ms": round(self._pct(lat, 95), 2),
            "inference_latency_p99_ms": round(self._pct(lat, 99), 2),
            "end_to_end_p50_ms": round(self._pct(e2e, 50), 2),
            "end_to_end_p95_ms": round(self._pct(e2e, 95), 2),
            "processed_frames": self.processed,
            "dropped_frames": self.dropped,
            "skipped_frames": self.skipped,
            "samples": len(lat),
        }
