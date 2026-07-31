"""The model adapter contract.

A ``ModelAdapter`` owns everything model-specific: what its inputs must look like,
how they become tensors, and how raw output becomes a typed result. It owns
*nothing* runtime-specific — no session creation, no provider selection, no device
placement. Those belong to the :mod:`app.runtimes` layer, which an adapter receives
and calls.

That split is what makes the (model x runtime) matrix real rather than a claim: the
same YOLOv8 adapter runs on ONNX Runtime CPU, ONNX Runtime CUDA or OpenVINO without
knowing which, and the same ONNX runtime adapter serves detection, classification
and embedding models without knowing what they mean.

Data carriers here are dataclasses rather than Pydantic models: they hold NumPy
arrays, never cross a serialization boundary, and are constructed once per
iteration on a hot path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, Field

from app.schemas.enums import DeviceKind, Modality, Phase, Precision, Task
from app.schemas.quality import QualityMetrics


class HardwareRequirement(BaseModel):
    """The floor below which this model cannot be expected to run."""

    min_ram_mb: int | None = None
    min_vram_mb: int | None = None
    min_disk_mb: int | None = None
    requires_gpu: bool = False
    note: str | None = None


class ModelMetadata(BaseModel):
    """Everything the platform must know about a model before running it.

    Licensing is mandatory and split in two, because they genuinely differ: a model
    can have Apache-2.0 *code* and non-commercial *weights*, and only reporting the
    former would mislead someone deciding whether they may ship it.
    """

    model_id: str
    display_name: str
    family: str
    task: Task
    modality: Modality

    source_repository: str | None = None
    paper_url: str | None = None
    model_license: str = Field(description="Licence of the implementation/code.")
    weights_license: str = Field(description="Licence of the trained weights. Often different.")
    commercial_use_permitted: bool | None = Field(
        default=None,
        description="None means the licence was not analysed; the UI shows 'unreviewed' "
                    "rather than guessing.",
    )
    auto_download_permitted: bool = Field(
        default=False,
        description="Whether this project may fetch the weights automatically. False for "
                    "anything requiring manual licence acceptance.",
    )

    parameters_millions: float | None = None
    model_size_bytes: int | None = None
    revision: str | None = None
    weights_checksum_sha256: str | None = None

    supported_precisions: list[Precision] = Field(default_factory=list)
    supported_devices: list[DeviceKind] = Field(default_factory=list)
    supported_runtimes: list[str] = Field(default_factory=list)

    input_format: str = Field(description="e.g. 'RGB uint8 HWC image', 'UTF-8 text', '16 kHz mono PCM'")
    output_format: str = Field(description="e.g. 'list[Detection] in source pixel coords'")
    dynamic_input_supported: bool = False
    streaming_supported: bool = False
    batch_supported: bool = False
    supported_quantizations: list[str] = Field(default_factory=list)

    hardware_requirements: HardwareRequirement = Field(default_factory=HardwareRequirement)
    known_limitations: list[str] = Field(default_factory=list)

    is_test_adapter: bool = Field(
        default=False,
        description="True only for the deterministic mock. Filtered out of production model "
                    "listings so a fabricated result can never be mistaken for a real one.",
    )


@dataclass(slots=True)
class LoadConfig:
    """How to load a model for one particular run."""

    runtime_id: str
    device: DeviceKind = DeviceKind.CPU
    device_index: int = 0
    precision: Precision = Precision.FP32
    quantization: str | None = None
    input_size: int | None = None
    batch_size: int = 1
    thread_config: dict[str, int] = field(default_factory=dict)
    backend_options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LoadResult:
    """What actually happened during load, as opposed to what was requested.

    ``effective_*`` fields exist because a request is not an outcome: ONNX Runtime
    will happily create a session on the CPU provider when CUDA is unavailable. The
    manager compares requested against effective and refuses the load rather than
    letting the rest of the system repeat a false claim.
    """

    ok: bool
    effective_device: DeviceKind
    effective_precision: Precision
    execution_provider: str | None = None
    runtime_version: str | None = None
    load_ms: float = 0.0
    compile_ms: float | None = None
    weights_bytes: int | None = None
    message: str = ""


@dataclass(slots=True)
class InferenceRequest:
    """One unit of work, in whatever modality the adapter consumes.

    Exactly one payload field is populated; the adapter validates this and raises
    rather than silently preferring one.
    """

    images: list[np.ndarray] | None = None
    text: list[str] | None = None
    audio: np.ndarray | None = None
    audio_sample_rate: int | None = None
    video_frames: list[np.ndarray] | None = None

    confidence: float | None = None
    iou: float | None = None
    allowed_class_ids: set[int] | None = None
    generation: dict[str, Any] = field(default_factory=dict)
    request_id: str | None = None


@dataclass(slots=True)
class PreparedInput:
    """Model-ready tensors plus whatever postprocessing will need to undo."""

    tensors: dict[str, np.ndarray]
    #: Data the postprocessor needs to map outputs back to input space
    #: (letterbox padding, original dimensions, tokenizer offsets, …).
    context: dict[str, Any] = field(default_factory=dict)
    token_count: int | None = None


@dataclass(slots=True)
class RawOutput:
    """Untouched runtime output, before any model-specific interpretation."""

    tensors: list[np.ndarray]
    names: list[str] = field(default_factory=list)
    #: Non-tensor output, for runtimes that return structured data rather than
    #: arrays (a remote endpoint returning JSON, for example). Kept separate from
    #: `tensors` so a postprocessor cannot mistake one for the other.
    payload: Any = None


@dataclass(slots=True)
class InferenceOutput:
    """Typed, human-meaningful result.

    Only the field matching the adapter's task is populated. ``extra`` carries
    task-specific detail that does not warrant a first-class field.
    """

    detections: list[Any] | None = None
    classifications: list[tuple[int, str, float]] | None = None
    embeddings: np.ndarray | None = None
    text: str | None = None
    audio: np.ndarray | None = None
    images: list[np.ndarray] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ReferenceOutput:
    """Ground truth for one example, for quality evaluation."""

    detections: list[Any] | None = None
    class_id: int | None = None
    text: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ModelAdapter(Protocol):
    """The contract every model integration satisfies.

    Implementations must keep the four execution methods free of measurement code.
    The benchmark engine wraps each call in its own span, so an adapter that timed
    itself would double-count and, worse, would make two adapters' numbers
    incomparable.
    """

    metadata: ModelMetadata

    #: Which timeline phase this adapter's ``preprocess`` represents. Image models
    #: report PREPROCESSING; text models report TOKENIZATION, because "preprocessing"
    #: would hide the single most informative phase of a text pipeline. Optional —
    #: the engine defaults to PREPROCESSING when an adapter does not declare one.
    preprocess_phase: Phase

    def load(self, config: LoadConfig) -> LoadResult: ...

    def preprocess(self, request: InferenceRequest) -> PreparedInput: ...

    def infer(self, prepared: PreparedInput) -> RawOutput: ...

    def postprocess(self, raw: RawOutput, prepared: PreparedInput) -> InferenceOutput: ...

    def evaluate(
        self,
        predictions: list[InferenceOutput],
        references: list[ReferenceOutput],
    ) -> QualityMetrics: ...

    def synthetic_request(self, batch_size: int = 1) -> InferenceRequest:
        """A deterministic input for measuring the runtime path rather than scene content."""
        ...

    def unload(self) -> None: ...
