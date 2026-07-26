"""VLM manager: owns the active VLM backend and mediates switching.

Mock is the default. Switching to the local or remote backend is opt-in and rolls back to
the previous (working) backend on failure — the same last-known-good pattern used by the
detection manager.
"""
from __future__ import annotations

import threading

from app.core.errors import InferenceLabError, ModelNotFoundError
from app.core.logging import get_logger
from app.core.types import HealthState, VLMResponse
from app.vlm.base import VisionLanguageBackend
from app.vlm.local_backend import LocalTransformersVLM
from app.vlm.mock_backend import MockVLMBackend
from app.vlm.remote_backend import RemoteVLMBackend
from app.vlm.structured_output import STRUCTURED_INSTRUCTION

log = get_logger("vlm.manager")

_LOCAL_REPOS = {
    "smolvlm-256m": "HuggingFaceTB/SmolVLM-256M-Instruct",
    "smolvlm-500m": "HuggingFaceTB/SmolVLM-500M-Instruct",
}


class VLMManager:
    def __init__(self, registry, capabilities, settings) -> None:
        self._registry = registry
        self._caps = capabilities
        self._settings = settings
        self._backend: VisionLanguageBackend | None = None
        self._model_id: str | None = None
        self._lock = threading.RLock()
        self.events: list[dict] = []

    # --- construction ---
    def _build(self, model_id: str) -> VisionLanguageBackend:
        entry = self._registry.vlm(model_id)
        if entry is None:
            raise ModelNotFoundError(f"unknown vlm model_id '{model_id}'")
        if model_id == "mock-vlm":
            return MockVLMBackend()
        if model_id in _LOCAL_REPOS:
            return LocalTransformersVLM(
                model_id=model_id, hf_repo=_LOCAL_REPOS[model_id],
                prefer_cuda=self._caps.runtimes.pytorch_cuda,
            )
        # Anything else (e.g. qwen2.5-vl) runs remotely when a remote URL is configured.
        if self._settings.vlm_remote_url:
            return RemoteVLMBackend(
                model_id=model_id, base_url=self._settings.vlm_remote_url,
                api_key=self._settings.vlm_remote_api_key, remote_model=entry.model_source,
                timeout_s=self._settings.vlm_remote_timeout_s,
                max_retries=self._settings.vlm_remote_max_retries,
                allow_frame_transmission=self._settings.allow_frame_transmission,
            )
        raise ModelNotFoundError(
            f"vlm '{model_id}' has no local mapping and no remote URL configured"
        )

    def _log_event(self, kind: str, **fields) -> None:
        evt = {"kind": kind, **fields}
        self.events.append(evt)
        if len(self.events) > 100:
            self.events = self.events[-100:]
        log.info("vlm_event", **evt)

    def initialize(self, model_id: str) -> None:
        with self._lock:
            backend = self._build(model_id)
            backend.load()
            backend.warmup()
            self._backend = backend
            self._model_id = model_id
            self._log_event("initialized", model_id=model_id)

    def switch(self, model_id: str) -> dict:
        with self._lock:
            prev, prev_id = self._backend, self._model_id
            try:
                backend = self._build(model_id)
                backend.load()
                backend.warmup()
            except Exception as exc:  # noqa: BLE001
                self._log_event("switch_failed_rollback", requested=model_id,
                                restored=prev_id, error=str(exc))
                msg = exc.user_message if isinstance(exc, InferenceLabError) else str(exc)
                return {"ok": False, "model_id": prev_id, "rolled_back": True, "message": msg}
            if prev is not None:
                try:
                    prev.unload()
                except Exception:
                    pass
            self._backend = backend
            self._model_id = model_id
            self._log_event("switched", from_model=prev_id, to_model=model_id)
            return {"ok": True, "model_id": model_id, "rolled_back": False, "message": "switched"}

    # --- inference ---
    def _require(self) -> VisionLanguageBackend:
        if self._backend is None:
            raise ModelNotFoundError("no VLM backend loaded")
        return self._backend

    def describe(self, image, prompt=None, grounding=None, structured=False) -> VLMResponse:
        with self._lock:
            backend = self._require()
            effective = prompt or "Describe this scene in one sentence."
            if structured:
                effective = f"{effective}\n\n{STRUCTURED_INSTRUCTION}"
            resp = backend.describe_image(image, effective, grounding)
            if structured and resp.structured_output is None:
                from app.vlm.structured_output import parse_structured

                parsed, warnings, _ = parse_structured(resp.text)
                resp.structured_output = parsed.model_dump() if parsed else None
                resp.warnings.extend(warnings)
            return resp

    def ask(self, image, question, grounding=None) -> VLMResponse:
        with self._lock:
            return self._require().answer_question(image, question, grounding)

    def analyze_frames(self, frames, prompt, grounding=None) -> VLMResponse:
        with self._lock:
            return self._require().analyze_video(frames, prompt, grounding)

    # --- introspection ---
    def status(self) -> dict:
        with self._lock:
            health = self._backend.health().value if self._backend else HealthState.UNKNOWN.value
            return {
                "available": self._backend is not None,
                "model_id": self._model_id,
                "runtime": getattr(self._backend, "runtime", None) if self._backend else None,
                "execution_location": getattr(self._backend, "execution_location", None)
                if self._backend else None,
                "health": health,
                "events": self.events[-10:],
            }

    def close(self) -> None:
        with self._lock:
            if self._backend is not None:
                self._backend.unload()
                self._backend = None
