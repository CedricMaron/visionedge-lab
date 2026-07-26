"""Runtime adapters: execution backends, independent of any model."""
from __future__ import annotations

from app.runtimes.base import (
    BaseRuntimeAdapter,
    RuntimeAdapter,
    RuntimeCapability,
    SessionConfig,
    SessionHandle,
)
from app.runtimes.onnxruntime_adapter import OnnxRuntimeAdapter
from app.runtimes.registry import (
    available_runtime_ids,
    capability_matrix,
    get_adapter,
    probe_all,
)

__all__ = [
    "BaseRuntimeAdapter",
    "OnnxRuntimeAdapter",
    "RuntimeAdapter",
    "RuntimeCapability",
    "SessionConfig",
    "SessionHandle",
    "available_runtime_ids",
    "capability_matrix",
    "get_adapter",
    "probe_all",
]
