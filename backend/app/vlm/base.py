"""Common vision-language backend interface.

Every VLM (mock, local SmolVLM, remote OpenAI-compatible) implements
``VisionLanguageBackend`` and returns a normalized ``VLMResponse`` so the API and UI are
model-agnostic. ``ImageInput`` is a BGR uint8 numpy array (the same format the detection
pipeline uses); backends convert to whatever their model needs internally.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from app.core.types import HealthState, VLMResponse

ImageInput = np.ndarray  # BGR uint8 HxWx3


@runtime_checkable
class VisionLanguageBackend(Protocol):
    model_id: str
    runtime: str
    execution_location: str

    def load(self) -> None: ...
    def warmup(self) -> None: ...
    def describe_image(self, image: ImageInput, prompt: str | None = None,
                       grounding: dict | None = None) -> VLMResponse: ...
    def answer_question(self, image: ImageInput, question: str,
                        grounding: dict | None = None) -> VLMResponse: ...
    def analyze_video(self, frames: list[ImageInput], prompt: str,
                      grounding: dict | None = None) -> VLMResponse: ...
    def health(self) -> HealthState: ...
    def unload(self) -> None: ...


class BaseVLMBackend:
    model_id: str = "unknown"
    runtime: str = "base"
    execution_location: str = "pc_local"

    def __init__(self) -> None:
        self._health: HealthState = HealthState.UNKNOWN

    def health(self) -> HealthState:
        return self._health

    @staticmethod
    def _rss_mb() -> float | None:
        try:
            import psutil

            return psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception:
            return None
