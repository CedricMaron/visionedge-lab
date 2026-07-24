"""Application state container, created once at startup.

Holds the initialized singletons (capabilities, registry, DB, detection manager,
rolling metrics, and — added in the VLM slice — the VLM manager). API handlers reach
these through ``get_state()`` which reads from ``app.state``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.capabilities.scanner import BackendCapabilities, scan_capabilities
from app.core.config import Settings, get_settings
from app.inference.config import InferenceConfig
from app.inference.manager import DetectionManager
from app.models.registry import ModelRegistry, load_registry, refresh_deployment_status
from app.monitoring.metrics import RollingMetrics
from app.storage.db import Database

if TYPE_CHECKING:
    from app.vlm.manager import VLMManager


@dataclass
class AppState:
    settings: Settings
    capabilities: BackendCapabilities
    registry: ModelRegistry
    db: Database
    detection: DetectionManager
    metrics: RollingMetrics
    vlm: VLMManager | None = None
    startup_warnings: list[str] = field(default_factory=list)


def build_state() -> AppState:
    settings = get_settings()
    caps = scan_capabilities()
    registry = refresh_deployment_status(load_registry(settings.registry_path))
    db = Database(settings.db_path)
    detection = DetectionManager(registry, caps)
    metrics = RollingMetrics()
    warnings: list[str] = []

    # Try to initialize the default detection backend so /health is meaningful.
    default = registry.detection(settings.default_model_id)
    if default and default.deployment_status == "installed":
        try:
            detection.initialize(InferenceConfig(
                model_id=settings.default_model_id,
                runtime="onnxruntime-cpu",
                input_size=settings.default_input_size,
                confidence=settings.default_confidence,
                iou=settings.default_iou,
            ))
        except Exception as exc:  # noqa: BLE001 — surfaced as a warning, not a crash
            warnings.append(f"default detection backend failed to load: {exc}")
    else:
        warnings.append(f"default model '{settings.default_model_id}' not installed")

    # Lazily attach the VLM manager (mock default) if the package is present.
    vlm = None
    try:
        from app.vlm.manager import VLMManager

        vlm = VLMManager(registry, caps, settings)
        vlm.initialize(settings.default_vlm_id)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"vlm manager unavailable: {exc}")

    return AppState(
        settings=settings, capabilities=caps, registry=registry, db=db,
        detection=detection, metrics=metrics, vlm=vlm, startup_warnings=warnings,
    )


def get_state(request) -> AppState:
    return request.app.state.appstate
