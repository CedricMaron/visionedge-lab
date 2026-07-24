"""Inference configuration + validation.

A configuration is validated against the registry and the device's real capabilities
before any backend is built. Invalid combinations are rejected with a clear reason
rather than failing deep inside a runtime.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from app.core.errors import ConfigInvalidError
from app.core.types import ExecutionLocation

VALID_RUNTIMES = {
    "onnxruntime-cpu", "onnxruntime-cuda", "pytorch", "openvino", "tensorrt",
}


class InferenceConfig(BaseModel):
    model_id: str
    runtime: str = "onnxruntime-cpu"
    input_size: int = 640
    confidence: float = Field(0.25, ge=0.0, le=1.0)
    iou: float = Field(0.45, ge=0.0, le=1.0)
    execution_location: ExecutionLocation = ExecutionLocation.PC_LOCAL
    allowed_class_ids: list[int] | None = None

    @field_validator("runtime")
    @classmethod
    def _runtime_known(cls, v: str) -> str:
        if v not in VALID_RUNTIMES:
            raise ValueError(f"unknown runtime '{v}'. valid: {sorted(VALID_RUNTIMES)}")
        return v

    @field_validator("input_size")
    @classmethod
    def _size_ok(cls, v: int) -> int:
        if v < 128 or v > 1280 or v % 32 != 0:
            raise ValueError("input_size must be a multiple of 32 between 128 and 1280")
        return v


def validate_against_capabilities(cfg: InferenceConfig, caps) -> None:
    """Raise ConfigInvalidError if the runtime is not actually available here."""
    rt = caps.runtimes
    if cfg.runtime == "onnxruntime-cuda" and not rt.onnxruntime_cuda:
        raise ConfigInvalidError(
            "onnxruntime CUDA provider unavailable",
            user_message="CUDA execution provider isn't available. Falling back to CPU is recommended.",
        )
    if cfg.runtime == "pytorch" and not rt.pytorch:
        raise ConfigInvalidError("pytorch not installed", user_message="PyTorch runtime isn't installed.")
    if cfg.runtime == "openvino" and not rt.openvino:
        raise ConfigInvalidError("openvino not installed", user_message="OpenVINO isn't installed.")
    if cfg.runtime == "tensorrt" and not rt.tensorrt:
        raise ConfigInvalidError("tensorrt not installed", user_message="TensorRT isn't installed.")
