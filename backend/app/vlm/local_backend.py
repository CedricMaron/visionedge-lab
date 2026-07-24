"""Local VLM backend via Hugging Face Transformers (opt-in).

Default target: SmolVLM-256M/500M-Instruct (Apache-2.0). This is a REAL integration but
the model is an opt-in download — nothing is fetched at import. On a machine with little
RAM (this dev box has ~1.6 GB free) loading may be slow or OOM; the backend surfaces that
as a clear warning/health error rather than crashing the app.

The heavy imports (torch, transformers) happen inside ``load`` so the module always
imports even when transformers is not installed.
"""
from __future__ import annotations

import time

import numpy as np

from app.core.errors import ModelLoadError, RuntimeUnavailableError
from app.core.logging import get_logger
from app.core.types import HealthState, VLMResponse
from app.vlm.base import BaseVLMBackend, ImageInput
from app.vlm.prompting import build_describe_prompt, build_vqa_prompt

log = get_logger("vlm.local")


def transformers_available() -> bool:
    try:
        import transformers  # noqa: F401

        return True
    except Exception:
        return False


class LocalTransformersVLM(BaseVLMBackend):
    runtime = "transformers"
    execution_location = "pc_local"

    def __init__(self, model_id: str, hf_repo: str, prefer_cuda: bool = True,
                 quantization: str | None = None, max_new_tokens: int = 128) -> None:
        super().__init__()
        self.model_id = model_id
        self.hf_repo = hf_repo
        self.prefer_cuda = prefer_cuda
        self.quantization = quantization
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None
        self._device = "cpu"

    def load(self) -> None:
        if not transformers_available():
            self._health = HealthState.ERROR
            raise RuntimeUnavailableError(
                "transformers is not installed. Install extras from requirements/vlm.txt "
                "to enable the local VLM."
            )
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self._health = HealthState.LOADING
        try:
            self._device = "cuda" if (self.prefer_cuda and torch.cuda.is_available()) else "cpu"
            dtype = torch.float16 if self._device == "cuda" else torch.float32
            self._processor = AutoProcessor.from_pretrained(self.hf_repo)
            self._model = AutoModelForVision2Seq.from_pretrained(
                self.hf_repo, torch_dtype=dtype
            ).to(self._device)
        except Exception as exc:  # noqa: BLE001
            self._health = HealthState.ERROR
            raise ModelLoadError(
                f"failed to load local VLM '{self.hf_repo}': {exc}",
                user_message="The local VLM failed to load (likely insufficient memory). "
                             "Try the mock or a remote VLM.",
            ) from exc
        self._health = HealthState.READY
        log.info("local_vlm_loaded", model_id=self.model_id, device=self._device)

    def warmup(self) -> None:  # pragma: no cover - requires the real model
        img = np.zeros((64, 64, 3), np.uint8)
        self.describe_image(img, "Describe.")

    def _generate(self, image: ImageInput, system: str, user: str) -> VLMResponse:  # pragma: no cover
        if self._model is None or self._processor is None:
            raise RuntimeUnavailableError("local VLM not loaded")
        import torch
        from PIL import Image

        rgb = Image.fromarray(image[:, :, ::-1])  # BGR->RGB
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user}]}]
        prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(text=prompt, images=[rgb], return_tensors="pt").to(self._device)
        prompt_tokens = int(inputs["input_ids"].shape[1])

        t0 = time.perf_counter()
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        dt = (time.perf_counter() - t0) * 1000.0
        gen_ids = out[0][prompt_tokens:]
        text = self._processor.decode(gen_ids, skip_special_tokens=True).strip()
        return VLMResponse(
            text=text, structured_output=None, model_id=self.model_id, runtime=self.runtime,
            execution_location=self.execution_location, prompt_tokens=prompt_tokens,
            generated_tokens=int(gen_ids.shape[0]), time_to_first_token_ms=None,
            generation_latency_ms=round(dt, 2), total_latency_ms=round(dt, 2),
            memory_usage_mb=self._rss_mb(), warnings=[],
        )

    def describe_image(self, image, prompt=None, grounding=None) -> VLMResponse:  # pragma: no cover
        system, user = build_describe_prompt(prompt, grounding)
        return self._generate(image, system, user)

    def answer_question(self, image, question, grounding=None) -> VLMResponse:  # pragma: no cover
        system, user = build_vqa_prompt(question, grounding)
        return self._generate(image, system, user)

    def analyze_video(self, frames, prompt, grounding=None) -> VLMResponse:  # pragma: no cover
        # SmolVLM-256M is image-only; analyze the middle frame and note the limitation.
        mid = frames[len(frames) // 2] if frames else np.zeros((64, 64, 3), np.uint8)
        resp = self.describe_image(mid, prompt, grounding)
        resp.warnings.append("this local model is image-only; analyzed a single representative frame")
        return resp

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._health = HealthState.UNKNOWN
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
