"""Detection backend manager: runtime model switching with rollback.

Implements the switch state machine required by the spec:
  stop accepting frames -> drain -> unload previous -> release memory ->
  load selected -> warmup -> verify health -> resume.

If the new configuration fails to load or fails its health check, the manager restores
the last known-good backend so inference continues uninterrupted.
"""
from __future__ import annotations

import threading
import time

from app.core.errors import ModelLoadError
from app.core.logging import get_logger
from app.core.types import BenchmarkResult, Detection, HealthState
from app.inference.base import DetectionBackend
from app.inference.config import InferenceConfig, validate_against_capabilities
from app.inference.factory import build_backend
from app.models.registry import ModelRegistry

log = get_logger("inference.manager")

# Runtimes whose name is a promise about *where* inference runs. A backend may silently
# fall back (ONNX Runtime loads the CPU provider when the CUDA one can't be created), so
# the manager checks what actually loaded before adopting the config — reporting
# "onnxruntime-cuda" while running on the CPU would be a lie the rest of the system
# (metrics, benchmarks, the UI) would repeat.
_REQUIRED_DEVICE: dict[str, str] = {"onnxruntime-cuda": "cuda"}

# Pause between benchmark inferences so a waiting live frame can take the lock.
# See DetectionManager.benchmark for why per-inference locking alone is not enough.
_BENCHMARK_YIELD_S = 0.002


def _verify_runtime_honored(backend: DetectionBackend, cfg: InferenceConfig) -> None:
    """Raise ModelLoadError if the loaded backend did not honor ``cfg.runtime``."""
    required = _REQUIRED_DEVICE.get(cfg.runtime)
    if required is None:
        return
    device = getattr(backend, "device", None)
    if device != required:
        provider = getattr(backend, "provider", None)
        raise ModelLoadError(
            f"runtime '{cfg.runtime}' requires device '{required}' but the backend loaded "
            f"on '{device}' (provider={provider}). The execution provider could not be "
            "created — check the CUDA/cuDNN install for this onnxruntime build."
        )


class SwitchResult:
    def __init__(self, ok: bool, config: InferenceConfig, message: str, rolled_back: bool = False) -> None:
        self.ok = ok
        self.config = config
        self.message = message
        self.rolled_back = rolled_back

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "config": self.config.model_dump(mode="json"),
            "message": self.message,
            "rolled_back": self.rolled_back,
        }


class DetectionManager:
    """Owns the active backend and serializes switches. Thread-safe for predict/switch."""

    def __init__(self, registry: ModelRegistry, capabilities) -> None:
        self._registry = registry
        self._caps = capabilities
        self._backend: DetectionBackend | None = None
        self._config: InferenceConfig | None = None
        self._lock = threading.RLock()
        self._accepting = False
        self.events: list[dict] = []  # bounded event log (fallbacks, switches)

    # --- lifecycle ---
    def _log_event(self, kind: str, **fields) -> None:
        evt = {"kind": kind, **fields}
        self.events.append(evt)
        if len(self.events) > 200:
            self.events = self.events[-200:]
        log.info("manager_event", **evt)

    @property
    def config(self) -> InferenceConfig | None:
        return self._config

    def health(self) -> HealthState:
        with self._lock:
            return self._backend.health() if self._backend else HealthState.UNKNOWN

    def initialize(self, cfg: InferenceConfig) -> SwitchResult:
        """Load the first backend. No previous config to roll back to."""
        with self._lock:
            validate_against_capabilities(cfg, self._caps)
            backend = build_backend(cfg, self._registry)
            backend.load()
            backend.warmup()
            if backend.health() != HealthState.READY:
                backend.close()
                raise ModelLoadError("backend did not reach READY after warmup")
            try:
                _verify_runtime_honored(backend, cfg)
            except ModelLoadError:
                backend.close()
                raise
            self._backend = backend
            self._config = cfg
            self._accepting = True
            self._log_event("initialized", config=cfg.model_dump(mode="json"))
            return SwitchResult(True, cfg, "initialized")

    def switch(self, cfg: InferenceConfig) -> SwitchResult:
        """Switch to a new configuration with rollback on failure."""
        with self._lock:
            prev_backend = self._backend
            prev_config = self._config
            self._accepting = False  # stop accepting new frames

            new_backend: DetectionBackend | None = None
            try:
                validate_against_capabilities(cfg, self._caps)
                new_backend = build_backend(cfg, self._registry)
                new_backend.load()
                new_backend.warmup()
                if new_backend.health() != HealthState.READY:
                    raise ModelLoadError("new backend not READY after warmup")
                _verify_runtime_honored(new_backend, cfg)
            except Exception as exc:  # noqa: BLE001 — must recover gracefully
                # rollback: release the half-built backend, keep the previous one running
                if new_backend is not None:
                    try:
                        new_backend.close()
                    except Exception as close_exc:  # noqa: BLE001
                        log.warning("new_backend_close_failed", error=str(close_exc))
                self._accepting = prev_backend is not None
                self._log_event(
                    "switch_failed_rollback",
                    requested=cfg.model_dump(mode="json"),
                    restored=prev_config.model_dump(mode="json") if prev_config else None,
                    error=str(exc),
                )
                return SwitchResult(
                    False, prev_config or cfg,
                    f"switch failed, restored previous config: {exc}",
                    rolled_back=True,
                )

            # success: unload previous, release memory, activate new
            if prev_backend is not None:
                try:
                    prev_backend.close()
                except Exception as exc:  # noqa: BLE001
                    log.warning("prev_close_failed", error=str(exc))
            self._backend = new_backend
            self._config = cfg
            self._accepting = True
            self._log_event(
                "switched",
                from_config=prev_config.model_dump(mode="json") if prev_config else None,
                to_config=cfg.model_dump(mode="json"),
            )
            return SwitchResult(True, cfg, "switched")

    def predict(self, image, conf=None, iou=None, allowed_class_ids=None) -> list[Detection]:
        with self._lock:
            if self._backend is None or not self._accepting:
                return []
            cfg = self._config
            c = conf if conf is not None else (cfg.confidence if cfg else 0.25)
            i = iou if iou is not None else (cfg.iou if cfg else 0.45)
            allowed = allowed_class_ids
            if allowed is None and cfg and cfg.allowed_class_ids is not None:
                allowed = set(cfg.allowed_class_ids)
            return self._backend.predict(image, c, i, allowed)

    def benchmark_target(self):
        """Return ``(backend, frame)`` to benchmark, captured under the lock.

        Callers hold the returned backend for the duration of a run so the result
        is attributed to the model that was actually measured, even if a switch
        replaces the active backend mid-run.
        """
        with self._lock:
            if self._backend is None:
                raise ModelLoadError("no backend loaded")
            return self._backend, self._backend.benchmark_frame()

    def benchmark_step(self, frame) -> float:
        """One timed inference under the lock. Returns milliseconds.

        The lock is taken per inference so a long benchmark does not freeze live
        streaming — frames interleave with benchmark runs.
        """
        with self._lock:
            if self._backend is None:
                raise ModelLoadError("no backend loaded")
            t0 = time.perf_counter()
            self._backend.predict(frame, 0.25, 0.45, None)
            return (time.perf_counter() - t0) * 1000.0

    def benchmark(self, runs: int = 30, notes: str = "", yield_s: float = _BENCHMARK_YIELD_S) -> BenchmarkResult:
        """Measure ``runs`` inferences, deferring to live traffic between each one.

        ``self._lock`` is an RLock and RLocks are not fair: a tight acquire/release
        loop re-acquires before a waiting thread is scheduled, so per-inference
        locking alone still starves live inference (measured: a live frame blocked
        3.3 s behind a 1.6 s benchmark). Sleeping between steps deschedules this
        thread and lets a waiter through.
        """
        backend, frame = self.benchmark_target()

        for _ in range(3):  # warm set, excluded from timing
            self.benchmark_step(frame)
            time.sleep(yield_s)

        latencies = []
        for _ in range(max(1, runs)):
            latencies.append(self.benchmark_step(frame))
            time.sleep(yield_s)
        return backend.result_from_latencies(latencies, notes=notes)

    def close(self) -> None:
        with self._lock:
            self._accepting = False
            if self._backend is not None:
                self._backend.close()
                self._backend = None
