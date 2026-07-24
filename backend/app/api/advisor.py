"""Optimization Advisor API.

Exposes the ExecutionPlanner: given the device's real capabilities plus user intent
(target FPS, privacy, battery, network), recommend where each pipeline stage should run,
with a human-readable reason per stage. Presets are also available. Recommendations are
never presented as universally optimal — each carries its reasoning, and measured
trade-offs are only surfaced when real metrics are supplied.
"""
from __future__ import annotations

from fastapi import APIRouter, Query, Request

from app.core.state import get_state
from app.orchestration.execution_planner import PRESETS, ExecutionPlanner

router = APIRouter(prefix="/api/advisor", tags=["advisor"])


def _caps_to_planner_input(caps) -> dict:
    """Map the real BackendCapabilities to the planner's expected capability keys."""
    gpu_vram = caps.gpus[0].memory_total_mb if caps.gpus else 0
    runtimes = []
    r = caps.runtimes
    if r.onnxruntime:
        runtimes.append("onnxruntime-cuda" if r.onnxruntime_cuda else "onnxruntime-cpu")
    if r.pytorch:
        runtimes.append("pytorch")
    if r.openvino:
        runtimes.append("openvino")
    if r.tensorrt:
        runtimes.append("tensorrt")
    return {
        "gpu_vram_mb": gpu_vram,
        "ram_mb": caps.ram_available_mb,
        "runtimes": runtimes,
        "nvidia_gpu": caps.nvidia_gpu_present,
    }


@router.get("/recommend")
async def recommend(
    request: Request,
    target_fps: int = Query(30, ge=1, le=240),
    privacy_mode: bool = Query(False),
    battery_mode: bool = Query(False),
    network_latency_ms: float | None = Query(None, ge=0),
    vlm_invocation_hz: float = Query(0.2, ge=0),
):
    state = get_state(request)
    caps_in = _caps_to_planner_input(state.capabilities)
    caps_in.update({
        "target_fps": target_fps,
        "privacy_mode": privacy_mode,
        "battery_mode": battery_mode,
        "network_latency_ms": network_latency_ms,
        "vlm_invocation_hz": vlm_invocation_hz,
    })
    # Surface only genuinely-measured detector latency as a trade-off.
    metrics = {}
    snap = state.metrics.snapshot()
    if snap.get("samples"):
        metrics["detector"] = (
            f"measured local detector P95 {snap['inference_latency_p95_ms']} ms "
            f"at {snap['processed_fps']} FPS on this device"
        )
    plan = ExecutionPlanner(caps_in, metrics or None).recommend()
    return {
        "device": {
            "gpu": state.capabilities.gpus[0].name if state.capabilities.gpus else None,
            "gpu_vram_mb": caps_in["gpu_vram_mb"],
            "ram_available_mb": caps_in["ram_mb"],
            "runtimes": caps_in["runtimes"],
        },
        "intent": {"target_fps": target_fps, "privacy_mode": privacy_mode,
                   "battery_mode": battery_mode, "network_latency_ms": network_latency_ms},
        "plan": plan.as_dict(),
        "notes": plan.notes,
        "disclaimer": "A recommendation with reasons, not a universal optimum. "
                      "Measured trade-offs are shown only when real metrics exist.",
    }


@router.get("/presets")
async def list_presets():
    return {"presets": list(PRESETS.keys())}


@router.get("/preset/{name}")
async def preset(name: str, request: Request):
    fn = PRESETS.get(name)
    if fn is None:
        return {"error": f"unknown preset '{name}'", "available": list(PRESETS.keys())}
    snap = get_state(request).metrics.snapshot()
    metrics = None
    if snap.get("samples"):
        metrics = {"detector": f"measured P95 {snap['inference_latency_p95_ms']} ms on this device"}
    plan = fn(metrics)
    return {"preset": name, "plan": plan.as_dict(), "notes": plan.notes}
