"""VLM API: models, image analysis, VQA, frame-sequence analysis, runtime status.

When ``ground=true`` the current detection backend is run on the image and its results
are injected into the VLM prompt as grounding context. VLM answers are always labelled as
model interpretations, not verified truth.
"""
from __future__ import annotations

from fastapi import APIRouter, File, Form, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from app.api.imaging import decode_image_bytes
from app.core.errors import VisionEdgeError
from app.core.state import get_state
from app.vlm.prompting import detections_to_grounding

router = APIRouter(prefix="/api/vlm", tags=["vlm"])


def _require_vlm(state):
    if state.vlm is None:
        raise VisionEdgeError("vlm not available", user_message="The VLM slice is not enabled in this build.")
    return state.vlm


@router.get("/models")
async def vlm_models(request: Request):
    reg = get_state(request).registry
    return {"vlm_models": [m.model_dump() for m in reg.vlm_models]}


@router.get("/runtime-status")
async def vlm_runtime_status(request: Request):
    state = get_state(request)
    return state.vlm.status() if state.vlm else {"available": False}


@router.post("/switch")
async def vlm_switch(request: Request, model_id: str = Form(...)):
    state = get_state(request)
    vlm = _require_vlm(state)
    return await run_in_threadpool(vlm.switch, model_id)


async def _grounding_for(state, img):
    """Run the current detector and build grounding context (labelled as a hint)."""
    dets = await run_in_threadpool(state.detection.predict, img, None, None, None)
    h, w = img.shape[:2]
    return detections_to_grounding(dets, w, h), dets


@router.post("/analyze-image")
async def analyze_image(
    request: Request,
    file: UploadFile = File(...),
    prompt: str | None = Form(None),
    ground: bool = Form(False),
    structured: bool = Form(False),
):
    state = get_state(request)
    vlm = _require_vlm(state)
    img = decode_image_bytes(await file.read())
    grounding = None
    disagreement = None
    if ground:
        grounding, dets = await _grounding_for(state, img)
    resp = await run_in_threadpool(vlm.describe, img, prompt, grounding, structured)
    if ground:
        from app.vlm.evaluation import detector_agreement

        disagreement = detector_agreement(resp.text, dets)
    return {"response": resp.model_dump(), "grounding": grounding, "agreement": disagreement,
            "disclaimer": "VLM output is a model-generated interpretation, not verified truth."}


@router.post("/ask")
async def ask(
    request: Request,
    file: UploadFile = File(...),
    question: str = Form(...),
    ground: bool = Form(False),
):
    state = get_state(request)
    vlm = _require_vlm(state)
    img = decode_image_bytes(await file.read())
    grounding = None
    agreement = None
    if ground:
        grounding, dets = await _grounding_for(state, img)
    resp = await run_in_threadpool(vlm.ask, img, question, grounding)
    if ground:
        from app.vlm.evaluation import detector_agreement

        agreement = detector_agreement(resp.text, dets)
    return {"response": resp.model_dump(), "grounding": grounding, "agreement": agreement,
            "disclaimer": "VLM output is a model-generated interpretation, not verified truth."}


@router.post("/analyze-frames")
async def analyze_frames(
    request: Request,
    files: list[UploadFile] = File(...),
    prompt: str = Form("What changed across these frames?"),
    ground: bool = Form(False),
):
    state = get_state(request)
    vlm = _require_vlm(state)
    frames = [decode_image_bytes(await f.read()) for f in files]
    grounding = None
    if ground and frames:
        grounding, _ = await _grounding_for(state, frames[-1])
    resp = await run_in_threadpool(vlm.analyze_frames, frames, prompt, grounding)
    return {"response": resp.model_dump(), "frame_count": len(frames),
            "disclaimer": "VLM output is a model-generated interpretation, not verified truth."}
