"""Model adapters: everything model-specific, nothing runtime-specific."""
from __future__ import annotations

from app.adapters.base import (
    HardwareRequirement,
    InferenceOutput,
    InferenceRequest,
    LoadConfig,
    LoadResult,
    ModelAdapter,
    ModelMetadata,
    PreparedInput,
    RawOutput,
    ReferenceOutput,
)

__all__ = [
    "HardwareRequirement",
    "InferenceOutput",
    "InferenceRequest",
    "LoadConfig",
    "LoadResult",
    "ModelAdapter",
    "ModelMetadata",
    "PreparedInput",
    "RawOutput",
    "ReferenceOutput",
]
