"""Domain error hierarchy.

User-facing surfaces map these to friendly messages; raw stack traces are never
returned to regular users (see api error handlers).
"""
from __future__ import annotations


class VisionEdgeError(Exception):
    """Base class for all domain errors."""

    user_message = "An internal error occurred."

    def __init__(self, detail: str, user_message: str | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        if user_message:
            self.user_message = user_message


class ConfigInvalidError(VisionEdgeError):
    user_message = "The requested configuration is invalid."


class ModelNotFoundError(VisionEdgeError):
    user_message = "The requested model is not available."


class ModelLoadError(VisionEdgeError):
    user_message = "The model failed to load. Restored the previous configuration."


class RuntimeUnavailableError(VisionEdgeError):
    user_message = "The requested runtime is not available on this device."


class InferenceError(VisionEdgeError):
    user_message = "Inference failed for this frame."


class BackendClosedError(VisionEdgeError):
    user_message = "The inference backend is not loaded."
