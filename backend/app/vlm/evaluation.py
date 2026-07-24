"""VLM evaluation helpers.

Small, honest metrics for open-ended VLM output. Exact-match is intentionally NOT the
only metric — keyword/concept coverage and detector agreement matter more for free-form
answers. These operate on already-produced responses; they never call a model.
"""
from __future__ import annotations

import re
from collections import Counter

from app.core.types import Detection


def keyword_coverage(answer: str, expected_concepts: list[str]) -> float:
    """Fraction of expected concept keywords present in the answer (case-insensitive)."""
    if not expected_concepts:
        return 1.0
    text = answer.lower()
    hits = sum(1 for c in expected_concepts if c.lower() in text)
    return hits / len(expected_concepts)


def object_count_consistency(answer: str, detections: list[Detection], class_name: str) -> dict:
    """Compare a count mentioned in the answer with the detector count for a class."""
    detector_count = sum(1 for d in detections if d.className == class_name)
    nums = [int(n) for n in re.findall(r"\b(\d+)\b", answer)]
    mentioned = nums[0] if nums else None
    return {
        "class": class_name,
        "detector_count": detector_count,
        "vlm_mentioned_count": mentioned,
        "agree": mentioned == detector_count if mentioned is not None else None,
    }


def detector_agreement(answer: str, detections: list[Detection]) -> dict:
    """Which detected classes the VLM mentions, and classes it mentions that weren't detected.

    Heuristic and lexical only — labelled as agreement, not ground truth.
    """
    detected = Counter(d.className for d in detections)
    text = answer.lower()
    mentioned_detected = [c for c in detected if c in text]
    missed_by_vlm = [c for c in detected if c not in text]
    return {
        "detected_classes": dict(detected),
        "vlm_mentioned_detected": mentioned_detected,
        "vlm_did_not_mention": missed_by_vlm,
        "note": "lexical agreement only; neither model is assumed correct",
    }
