"""Detection backend manager: runtime model switching with rollback.

Implements the switch state machine required by the spec:
  stop accepting frames -> drain -> unload previous -> release memory ->
  load selected -> warmup -> verify health -> resume.

If the new configuration fails to load or fails its health check, the manager restores
the last known-good backend so inference continues uninterrupted.
"""
from __future__ import annotations

import threading

from app.core.errors import ModelLoadError
from app.core.logging import get_logger
from app.core.types import Detection, HealthState
from app.inference.base import DetectionBackend
from app.inference.config import InferenceConfig, validate_against_capabilities
from app.inference.factory import build_backend
from app.models.registry import ModelRegistry

log = get_logger("inference.manager")


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

            try:
                validate_against_capabilities(cfg, self._caps)
                new_backend = build_backend(cfg, self._registry)
                new_backend.load()
                new_backend.warmup()
                if new_backend.health() != HealthState.READY:
                    raise ModelLoadError("new backend not READY after warmup")
            except Exception as exc:  # noqa: BLE001 — must recover gracefully
                # rollback: keep previous backend running
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

    def benchmark(self, runs: int = 30):
        with self._lock:
            if self._backend is None:
                raise ModelLoadError("no backend loaded")
            return self._backend.benchmark(runs)

    def close(self) -> None:
        with self._lock:
            self._accepting = False
            if self._backend is not None:
                self._backend.close()
                self._backend = None
