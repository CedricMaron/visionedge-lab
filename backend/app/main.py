"""InferenceLab FastAPI application.

Wires the detection foundation (and, when present, the VLM slice) behind versioned
routers. Domain errors are mapped to friendly messages; raw stack traces never reach
regular users.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.ratelimit import install_rate_limit
from app.core.config import get_settings
from app.core.errors import InferenceLabError
from app.core.logging import configure_logging, get_logger
from app.core.state import build_state
from app.core.version import APP_VERSION

log = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_json, settings.log_level)
    log.info("startup_begin")
    app.state.appstate = build_state()
    for w in app.state.appstate.startup_warnings:
        log.warning("startup_warning", detail=w)
    log.info("startup_complete",
             detection_health=app.state.appstate.detection.health().value,
             vlm=bool(app.state.appstate.vlm))
    yield
    app.state.appstate.detection.close()
    if app.state.appstate.vlm:
        app.state.appstate.vlm.close()
    log.info("shutdown_complete")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="InferenceLab", version=APP_VERSION, lifespan=lifespan)

    origins = ["*"] if settings.cors_origins.strip() == "*" else [
        o.strip() for o in settings.cors_origins.split(",") if o.strip()
    ]
    app.add_middleware(
        CORSMiddleware, allow_origins=origins, allow_credentials=True,
        allow_methods=["*"], allow_headers=["*"],
    )

    install_rate_limit(app, settings.rate_limit_per_min)

    @app.exception_handler(InferenceLabError)
    async def _domain_error(request: Request, exc: InferenceLabError):
        log.warning("domain_error", path=str(request.url.path), detail=exc.detail)
        return JSONResponse(status_code=400, content={"error": exc.user_message})

    # Routers
    from app.api import advisor, detection, lab, meta, playground

    app.include_router(meta.router)
    app.include_router(detection.router)
    app.include_router(advisor.router)
    app.include_router(lab.router)
    app.include_router(playground.router)

    # VLM router is optional — present only when the package imports cleanly.
    try:
        from app.api import vlm as vlm_api

        app.include_router(vlm_api.router)
    except Exception as exc:  # noqa: BLE001
        log.warning("vlm_router_unavailable", error=str(exc))

    return app


app = create_app()
