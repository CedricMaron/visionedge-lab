"""Remote VLM backend (OpenAI-compatible chat/completions with vision).

Opt-in. Endpoint and API key come from environment (VE_VLM_REMOTE_URL,
VE_VLM_REMOTE_API_KEY). Frames are transmitted ONLY when the operator has enabled
``allow_frame_transmission`` — otherwise the backend refuses and returns a privacy warning
rather than silently uploading camera frames. Supports TLS (https URL), timeout and
bounded retries.
"""
from __future__ import annotations

import base64
import time

import cv2
import numpy as np

from app.core.errors import RuntimeUnavailableError
from app.core.logging import get_logger
from app.core.types import HealthState, VLMResponse
from app.vlm.base import BaseVLMBackend, ImageInput
from app.vlm.prompting import build_describe_prompt, build_vqa_prompt

log = get_logger("vlm.remote")


def _encode_data_url(image: ImageInput, quality: int = 80) -> str:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeUnavailableError("failed to JPEG-encode frame for remote VLM")
    b64 = base64.b64encode(buf.tobytes()).decode()
    return f"data:image/jpeg;base64,{b64}"


class RemoteVLMBackend(BaseVLMBackend):
    runtime = "remote-openai-compatible"
    execution_location = "remote_server"

    def __init__(self, model_id: str, base_url: str, api_key: str, remote_model: str,
                 timeout_s: float = 30.0, max_retries: int = 2,
                 allow_frame_transmission: bool = False, jpeg_quality: int = 80) -> None:
        super().__init__()
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.remote_model = remote_model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.allow_frame_transmission = allow_frame_transmission
        self.jpeg_quality = jpeg_quality

    def load(self) -> None:
        if not self.base_url:
            self._health = HealthState.ERROR
            raise RuntimeUnavailableError("remote VLM URL not configured (set VE_VLM_REMOTE_URL)")
        self._health = HealthState.READY

    def warmup(self) -> None:
        pass

    def _chat(self, image: ImageInput, system: str, user: str) -> VLMResponse:
        if not self.allow_frame_transmission:
            return VLMResponse(
                text="", structured_output=None, model_id=self.model_id, runtime=self.runtime,
                execution_location=self.execution_location, generation_latency_ms=0.0,
                total_latency_ms=0.0, memory_usage_mb=None,
                warnings=["frame transmission to the remote server is disabled "
                          "(set VE_ALLOW_FRAME_TRANSMISSION=true to opt in). No frame was sent."],
            )
        import httpx

        data_url = _encode_data_url(image, self.jpeg_quality)
        body = {
            "model": self.remote_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "text", "text": user},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]},
            ],
            "max_tokens": 256,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

        last_exc: Exception | None = None
        t0 = time.perf_counter()
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout_s) as client:
                    r = client.post(f"{self.base_url}/chat/completions", json=body, headers=headers)
                    r.raise_for_status()
                    data = r.json()
                dt = (time.perf_counter() - t0) * 1000.0
                choice = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                return VLMResponse(
                    text=choice if isinstance(choice, str) else str(choice),
                    structured_output=None, model_id=self.model_id, runtime=self.runtime,
                    execution_location=self.execution_location,
                    prompt_tokens=usage.get("prompt_tokens"),
                    generated_tokens=usage.get("completion_tokens"),
                    time_to_first_token_ms=None, generation_latency_ms=round(dt, 2),
                    total_latency_ms=round(dt, 2), memory_usage_mb=None, warnings=[],
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                log.warning("remote_vlm_attempt_failed", attempt=attempt, error=str(exc))
                time.sleep(min(1.0 * (attempt + 1), 3.0))
        raise RuntimeUnavailableError(f"remote VLM failed after retries: {last_exc}")

    def describe_image(self, image, prompt=None, grounding=None) -> VLMResponse:
        system, user = build_describe_prompt(prompt, grounding)
        return self._chat(image, system, user)

    def answer_question(self, image, question, grounding=None) -> VLMResponse:
        system, user = build_vqa_prompt(question, grounding)
        return self._chat(image, system, user)

    def analyze_video(self, frames, prompt, grounding=None) -> VLMResponse:
        mid = frames[len(frames) // 2] if frames else np.zeros((64, 64, 3), np.uint8)
        return self.describe_image(mid, prompt, grounding)

    def unload(self) -> None:
        self._health = HealthState.UNKNOWN
