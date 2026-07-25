"""Detection API: REST inference, model switching, benchmark, and WebSocket transport.

Inference is CPU-bound, so predict calls run in a threadpool to avoid blocking the
event loop. The WebSocket handler implements bounded-queue frame dropping: only the
most recent frame is processed under load; older frames are dropped and counted.
"""
from __future__ import annotations

import asyncio
import time
import uuid

from fastapi import APIRouter, File, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from app.api.imaging import decode_image_bytes
from app.benchmarking.auto import BENCHMARK_JOB_KIND
from app.benchmarking.auto import DEFAULT_RUNS as AUTO_BENCHMARK_RUNS
from app.core.errors import VisionEdgeError
from app.core.logging import get_logger
from app.core.state import get_state
from app.core.types import ExecutionLocation
from app.inference.config import InferenceConfig
from app.jobs.state import TERMINAL_STATES
from app.monitoring.metrics import ACTIVE_SESSIONS, FRAMES_DROPPED, FRAMES_TOTAL, INFERENCE_LATENCY

log = get_logger("api.detection")
router = APIRouter(prefix="/api", tags=["detection"])


@router.post("/infer")
async def infer(
    request: Request,
    file: UploadFile = File(...),
    confidence: float | None = Query(None, ge=0.0, le=1.0),
    iou: float | None = Query(None, ge=0.0, le=1.0),
    classes: str | None = Query(None, description="comma-separated class ids to keep"),
):
    """Single-image detection over the currently loaded backend."""
    state = get_state(request)
    data = await file.read()
    if len(data) > state.settings.max_upload_bytes:
        return {"error": "file too large"}
    img = decode_image_bytes(data)
    allowed = {int(c) for c in classes.split(",") if c.strip().isdigit()} if classes else None

    t0 = time.perf_counter()
    dets, timings = await run_in_threadpool(
        state.detection.predict_timed, img, confidence, iou, allowed
    )
    e2e = (time.perf_counter() - t0) * 1000.0

    backend = state.detection.config.runtime if state.detection.config else "none"
    state.metrics.record(timings["inference_ms"], e2e)
    FRAMES_TOTAL.labels(backend=backend).inc()
    INFERENCE_LATENCY.labels(backend=backend).observe(timings["inference_ms"])
    return {
        "detections": [d.model_dump() for d in dets],
        "timings": {
            "preprocess_ms": round(timings["preprocess_ms"], 2),
            "inference_ms": round(timings["inference_ms"], 2),
            "postprocess_ms": round(timings["postprocess_ms"], 2),
            "end_to_end_ms": round(e2e, 2),
        },
        "backend": backend,
        "count": len(dets),
    }


def _start_auto_benchmark(state) -> None:
    """Measure the newly loaded model in the background.

    A new model invalidates any benchmark still running, since the backend it was
    measuring has been unloaded — those jobs are cancelled and store nothing.
    """
    if state.jobs is None:
        return
    for rec in state.jobs.list():
        if rec.kind == BENCHMARK_JOB_KIND and rec.state not in TERMINAL_STATES:
            try:
                state.jobs.cancel(rec.job_id)
            except ValueError:  # already terminal — nothing to cancel
                pass
    job_id = f"benchmark-{uuid.uuid4().hex[:8]}"
    state.jobs.submit(job_id, BENCHMARK_JOB_KIND, {"runs": AUTO_BENCHMARK_RUNS})
    state.jobs.start(job_id)


@router.post("/detection/switch")
async def switch_model(request: Request, cfg: InferenceConfig):
    """Switch the active detection configuration (with rollback on failure)."""
    state = get_state(request)
    if state.detection.config is None:
        await run_in_threadpool(state.detection.initialize, cfg)
        _start_auto_benchmark(state)
        return {"ok": True, "config": cfg.model_dump(mode="json"), "message": "initialized"}
    result = await run_in_threadpool(state.detection.switch, cfg)
    if result.ok:
        _start_auto_benchmark(state)
    return result.as_dict()


@router.get("/detection/status")
async def detection_status(request: Request):
    state = get_state(request)
    cfg = state.detection.config
    return {
        "config": cfg.model_dump(mode="json") if cfg else None,
        "health": state.detection.health().value,
        "events": state.detection.events[-20:],
        "metrics": state.metrics.snapshot(),
    }


@router.post("/detection/benchmark")
async def benchmark(request: Request, runs: int = Query(30, ge=1, le=500)):
    """Run an actual benchmark on the loaded backend and persist the measured result."""
    state = get_state(request)
    result = await run_in_threadpool(state.detection.benchmark, runs)
    meta = {
        "hardware": state.capabilities.cpu_model,
        "os": f"{state.capabilities.os} {state.capabilities.os_version}",
        "runtime_versions": {"onnxruntime": state.capabilities.runtimes.onnxruntime_providers},
        "config": state.detection.config.model_dump(mode="json") if state.detection.config else None,
    }
    await run_in_threadpool(state.db.insert_benchmark, result.model_dump(), meta)
    return result.model_dump()


@router.websocket("/ws/detect")
async def ws_detect(ws: WebSocket):
    """Real-time detection over WebSocket with frame dropping under backpressure.

    Protocol: client sends binary JPEG frames (optionally prefixed control JSON as text).
    Server replies with JSON: {frame_id, detections, timings, dropped}.
    """
    await ws.accept()
    state = get_state(ws)
    ACTIVE_SESSIONS.inc()
    session_id = ws.headers.get("sec-websocket-key", "ws") + str(int(time.time()))
    latest: dict = {"frame": None, "id": 0}
    stop = asyncio.Event()
    dropped = 0

    async def receiver():
        nonlocal dropped
        fid = 0
        try:
            while not stop.is_set():
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                data = msg.get("bytes")
                if data is None:
                    continue  # ignore text/control for this slice
                fid += 1
                if latest["frame"] is not None:
                    dropped += 1
                    FRAMES_DROPPED.inc()
                    state.metrics.record_drop()
                latest["frame"] = data
                latest["id"] = fid
        except WebSocketDisconnect:
            pass
        finally:
            stop.set()

    async def worker():
        nonlocal dropped
        while not stop.is_set():
            frame = latest["frame"]
            if frame is None:
                await asyncio.sleep(0.003)
                continue
            fid = latest["id"]
            latest["frame"] = None
            try:
                img = decode_image_bytes(frame)
                t0 = time.perf_counter()
                dets = await run_in_threadpool(state.detection.predict, img, None, None, None)
                dt = (time.perf_counter() - t0) * 1000.0
                backend = state.detection.config.runtime if state.detection.config else "none"
                state.metrics.record(dt, dt)
                FRAMES_TOTAL.labels(backend=backend).inc()
                INFERENCE_LATENCY.labels(backend=backend).observe(dt)
                await ws.send_json({
                    "frame_id": fid,
                    "detections": [d.model_dump() for d in dets],
                    "timings": {"inference_ms": round(dt, 2)},
                    "dropped": dropped,
                    "backend": backend,
                })
            except VisionEdgeError as exc:
                await ws.send_json({"frame_id": fid, "error": exc.user_message})
            except Exception as exc:  # noqa: BLE001
                log.warning("ws_worker_error", error=str(exc))
                await ws.send_json({"frame_id": fid, "error": "inference failed"})

    try:
        state.db.upsert_session(
            session_id, client_device=ws.headers.get("user-agent", "unknown"),
            execution_location=ExecutionLocation.PC_LOCAL.value,
            model_id=state.detection.config.model_id if state.detection.config else None,
            runtime=state.detection.config.runtime if state.detection.config else None,
        )
        await asyncio.gather(receiver(), worker())
    finally:
        stop.set()
        ACTIVE_SESSIONS.dec()
