"""Shared, normalized data contracts.

These Pydantic models are the single source of truth on the backend and are mirrored
exactly by the TypeScript types in ``frontend/src/types``. Every runtime/backend must
return these shapes so the UI is backend-agnostic.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ExecutionLocation(str, Enum):
    PC_LOCAL = "pc_local"
    PHONE_LOCAL = "phone_local"
    LOCAL_SERVER = "local_server"
    REMOTE_SERVER = "remote_server"


class Precision(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    Q8 = "q8"
    Q4 = "q4"


class Detection(BaseModel):
    """One detected object. Coordinates are absolute pixels in the source frame."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    classId: int
    className: str
    inferenceBackend: str
    modelName: str
    modelVersion: str


class InferenceTimings(BaseModel):
    """Per-stage latency breakdown in milliseconds. Missing stages stay None."""

    preprocess_ms: float | None = None
    inference_ms: float | None = None
    postprocess_ms: float | None = None
    end_to_end_ms: float | None = None


class InferenceResult(BaseModel):
    detections: list[Detection]
    timings: InferenceTimings
    backend: str
    model_id: str
    input_size: int
    frame_id: int | None = None


class BenchmarkResult(BaseModel):
    """Result of an actually-executed benchmark. Never hardcoded."""

    backend: str
    model_id: str
    input_size: int
    precision: str
    device: str
    runs: int
    fps: float
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    memory_rss_mb: float | None = None
    provider: str | None = None
    notes: str = ""


class VLMResponse(BaseModel):
    """Normalized vision-language response. Mirrors the frontend VLMResponse type."""

    text: str
    structured_output: dict | None = None
    model_id: str
    runtime: str
    execution_location: str
    prompt_tokens: int | None = None
    generated_tokens: int | None = None
    time_to_first_token_ms: float | None = None
    generation_latency_ms: float
    total_latency_ms: float
    memory_usage_mb: float | None = None
    warnings: list[str] = Field(default_factory=list)


class HealthState(str, Enum):
    UNKNOWN = "unknown"
    LOADING = "loading"
    READY = "ready"
    DEGRADED = "degraded"
    ERROR = "error"
