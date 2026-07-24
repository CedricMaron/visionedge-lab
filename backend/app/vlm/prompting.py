"""Prompt construction and detector grounding.

Builds the text prompt sent to a VLM, optionally injecting object-detection results as
grounding context. Grounding is a *hint*, never asserted as ground truth — the prompt
tells the model the detections may be incomplete or wrong.
"""
from __future__ import annotations

from app.core.types import Detection

SYSTEM_DESCRIBE = (
    "You are a concise visual scene analyst. Describe what is happening in the image in "
    "one or two sentences. Be specific but do not invent details you cannot see."
)
SYSTEM_VQA = (
    "You answer questions about an image concisely and honestly. If the image does not "
    "contain enough information, say so rather than guessing."
)


def detections_to_grounding(dets: list[Detection], image_w: int, image_h: int) -> dict:
    """Convert detections to a compact grounding dict with normalized boxes."""
    objs = []
    for d in dets:
        objs.append({
            "class": d.className,
            "confidence": round(d.confidence, 2),
            "box": [round(d.x1 / image_w, 3), round(d.y1 / image_h, 3),
                    round(d.x2 / image_w, 3), round(d.y2 / image_h, 3)],
        })
    return {"detected_objects": objs}


def format_grounding_context(grounding: dict | None) -> str:
    if not grounding or not grounding.get("detected_objects"):
        return ""
    lines = ["An object detector reported the following (it may be incomplete or wrong; "
             "use it only as a hint and trust the image over it):"]
    for o in grounding["detected_objects"]:
        lines.append(f"- {o['class']} (confidence {o.get('confidence')}, box {o.get('box')})")
    return "\n".join(lines)


def build_describe_prompt(prompt: str | None, grounding: dict | None) -> tuple[str, str]:
    """Return (system, user) prompt strings for a description request."""
    user = prompt or "Describe this scene in one sentence."
    ctx = format_grounding_context(grounding)
    if ctx:
        user = f"{ctx}\n\n{user}"
    return SYSTEM_DESCRIBE, user


def build_vqa_prompt(question: str, grounding: dict | None) -> tuple[str, str]:
    ctx = format_grounding_context(grounding)
    user = f"{ctx}\n\nQuestion: {question}" if ctx else question
    return SYSTEM_VQA, user
