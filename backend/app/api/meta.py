"""Capabilities, models, classes and monitoring endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Request, Response

from app.core.state import get_state
from app.models.coco import CLASS_GROUPS, COCO_CLASSES
from app.monitoring.metrics import prometheus_exposition

router = APIRouter(tags=["meta"])


@router.get("/health")
async def health(request: Request):
    state = get_state(request)
    return {"status": "ok", "detection_health": state.detection.health().value,
            "warnings": state.startup_warnings}


@router.get("/ready")
async def ready(request: Request):
    state = get_state(request)
    ok = state.detection.health().value in ("ready", "degraded")
    return Response(status_code=200 if ok else 503, content='{"ready": %s}' % str(ok).lower(),
                    media_type="application/json")


@router.get("/metrics")
async def metrics():
    body, content_type = prometheus_exposition()
    return Response(content=body, media_type=content_type)


@router.get("/api/capabilities")
async def capabilities(request: Request):
    return get_state(request).capabilities.model_dump()


@router.get("/api/models")
async def models(request: Request):
    reg = get_state(request).registry
    return {"detection_models": [m.model_dump() for m in reg.detection_models],
            "vlm_models": [m.model_dump() for m in reg.vlm_models]}


@router.get("/api/model-registry")
async def model_registry(request: Request):
    return get_state(request).registry.model_dump()


@router.get("/api/classes")
async def classes():
    return {
        "classes": [{"id": i, "name": n} for i, n in enumerate(COCO_CLASSES)],
        "groups": {k: v for k, v in CLASS_GROUPS.items()},
    }


@router.get("/api/runtime-status")
async def runtime_status(request: Request):
    state = get_state(request)
    cfg = state.detection.config
    return {
        "detection": {
            "config": cfg.model_dump(mode="json") if cfg else None,
            "health": state.detection.health().value,
            "events": state.detection.events[-20:],
        },
        "vlm": (state.vlm.status() if state.vlm else {"available": False}),
        "runtimes": state.capabilities.runtimes.model_dump(),
        "metrics": state.metrics.snapshot(),
    }


@router.get("/api/benchmarks")
async def benchmarks(request: Request, limit: int = 100):
    return {"benchmarks": get_state(request).db.list_benchmarks(limit)}


@router.get("/api/sessions")
async def sessions(request: Request, limit: int = 100):
    return {"sessions": get_state(request).db.list_sessions(limit)}
