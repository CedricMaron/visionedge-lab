"""Structured scene-understanding schema + tolerant parsing.

When a VLM is asked for structured output, we validate against ``SceneUnderstanding``.
On parse failure we preserve the raw text, emit a warning, and (optionally) retry within
a bounded limit — we never silently fabricate missing fields.
"""
from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field, ValidationError


class PersonInfo(BaseModel):
    description: str = ""
    activity: str = ""
    position: str = ""


class ActionInfo(BaseModel):
    subject: str = ""
    verb: str = ""
    object: str = ""


class SceneUnderstanding(BaseModel):
    summary: str = ""
    environment: str = ""
    people: list[PersonInfo] = Field(default_factory=list)
    important_objects: list[str] = Field(default_factory=list)
    actions: list[ActionInfo] = Field(default_factory=list)
    possible_risks: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)


STRUCTURED_INSTRUCTION = (
    "Respond ONLY with a JSON object matching this schema: "
    '{"summary": str, "environment": str, "people": [{"description": str, "activity": str, '
    '"position": str}], "important_objects": [str], "actions": [{"subject": str, "verb": str, '
    '"object": str}], "possible_risks": [str], "uncertainties": [str]}. '
    "Include an 'uncertainties' entry when you are inferring from a single frame."
)


def _extract_json(text: str) -> str | None:
    """Find the first balanced JSON object in free text."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else None


def parse_structured(text: str) -> tuple[SceneUnderstanding | None, list[str], str]:
    """Parse VLM text into SceneUnderstanding.

    Returns ``(parsed_or_none, warnings, raw_text)``. The raw text is always preserved.
    """
    warnings: list[str] = []
    candidate = _extract_json(text)
    if candidate is None:
        warnings.append("no JSON object found in model output; preserved raw text")
        return None, warnings, text
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        warnings.append(f"JSON decode failed: {exc}; preserved raw text")
        return None, warnings, text
    try:
        return SceneUnderstanding.model_validate(data), warnings, text
    except ValidationError as exc:
        warnings.append(f"schema validation failed ({exc.error_count()} errors); preserved raw text")
        return None, warnings, text
