"""YOLOv8 detection adapter.

Wraps the existing, validated preprocessing and NumPy decode path in the
:class:`~app.adapters.base.ModelAdapter` contract. The decoding itself is
unchanged — ``app.inference.postprocess.decode_yolov8`` is a from-scratch
reimplementation already validated against Ultralytics — so this migration adds a
contract without putting the numerics at risk.

What the adapter deliberately does *not* do:

* It does not create sessions or choose execution providers. It is handed a
  :class:`~app.runtimes.base.RuntimeAdapter` and calls it.
* It does not time itself. The benchmark engine owns the clock; an adapter that
  also timed would double-count and make adapters incomparable.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from app.adapters.base import (
    HardwareRequirement,
    InferenceOutput,
    InferenceRequest,
    LoadConfig,
    LoadResult,
    ModelMetadata,
    PreparedInput,
    RawOutput,
    ReferenceOutput,
)
from app.core.errors import ConfigInvalidError, ModelLoadError
from app.core.logging import get_logger
from app.core.types import Detection
from app.inference.postprocess import decode_yolov8
from app.inference.preprocess import letterbox, scale_boxes, to_model_input
from app.models.coco import COCO_CLASSES
from app.runtimes.base import RuntimeAdapter, SessionConfig, SessionHandle
from app.schemas.enums import DeviceKind, Modality, Precision, Task
from app.schemas.measurement import Measurement
from app.schemas.quality import DetectionQuality, QualityMetrics

log = get_logger("adapters.yolov8")

_DEFAULT_CONFIDENCE = 0.25
_DEFAULT_IOU = 0.45


class YoloV8Adapter:
    """Detection over a YOLOv8 ONNX graph."""

    def __init__(
        self,
        model_path: str | Path,
        runtime: RuntimeAdapter,
        model_id: str = "yolov8n-onnx",
        display_name: str = "YOLOv8 Nano (ONNX)",
        version: str = "v8.0",
        input_size: int = 640,
        class_names: list[str] | None = None,
        parameters_millions: float | None = 3.2,
    ) -> None:
        self.model_path = Path(model_path)
        self.runtime = runtime
        self.input_size = input_size
        self.class_names = class_names or COCO_CLASSES
        self.version = version
        self._handle: SessionHandle | None = None

        self.metadata = ModelMetadata(
            model_id=model_id,
            display_name=display_name,
            family="yolov8",
            task=Task.OBJECT_DETECTION,
            modality=Modality.IMAGE,
            source_repository="https://github.com/ultralytics/ultralytics",
            # Ultralytics ships code and weights under AGPL-3.0. Recording it as the
            # weights licence too is the point of splitting the two fields: someone
            # deciding whether they may ship this needs the weights answer.
            model_license="AGPL-3.0",
            weights_license="AGPL-3.0",
            commercial_use_permitted=False,
            auto_download_permitted=True,
            parameters_millions=parameters_millions,
            model_size_bytes=self.model_path.stat().st_size if self.model_path.exists() else None,
            supported_precisions=[Precision.FP32, Precision.INT8],
            supported_devices=[DeviceKind.CPU, DeviceKind.CUDA],
            supported_runtimes=["onnxruntime"],
            input_format="BGR uint8 HWC image, any size (letterboxed to the model input)",
            output_format="list[Detection] with xyxy boxes in source-image pixel coordinates",
            dynamic_input_supported=False,
            streaming_supported=False,
            batch_supported=False,
            supported_quantizations=["int8-dynamic"],
            hardware_requirements=HardwareRequirement(
                min_ram_mb=512, min_disk_mb=15, requires_gpu=False,
            ),
            known_limitations=[
                "Exported with a static batch size of 1; batching requires a re-export.",
                "COCO-80 classes only.",
                "AGPL-3.0 weights: unsuitable for closed-source commercial deployment.",
            ],
        )

    # --- lifecycle -------------------------------------------------------

    def load(self, config: LoadConfig) -> LoadResult:
        if not self.model_path.exists():
            raise ModelLoadError(f"model file missing: {self.model_path}")
        if config.input_size is not None:
            self.input_size = config.input_size

        t0 = time.perf_counter()
        handle = self.runtime.create_session(
            SessionConfig(
                model_path=str(self.model_path),
                device=config.device,
                device_index=config.device_index,
                precision=config.precision,
                intra_op_threads=config.thread_config.get("intra_op"),
                inter_op_threads=config.thread_config.get("inter_op"),
                enable_profiling=bool(config.backend_options.get("enable_profiling")),
            )
        )
        load_ms = (time.perf_counter() - t0) * 1000.0

        # A session that landed on a different device than requested must not be
        # adopted: every subsequent measurement would be attributed to hardware
        # that did no work.
        if not handle.honored:
            self.runtime.release(handle)
            raise ModelLoadError(handle.mismatch_message())

        self._handle = handle
        return LoadResult(
            ok=True,
            effective_device=handle.effective_device,
            effective_precision=handle.effective_precision,
            execution_provider=handle.execution_provider,
            runtime_version=handle.runtime_version,
            load_ms=load_ms,
            weights_bytes=self.model_path.stat().st_size,
            message=f"loaded on {handle.execution_provider}",
        )

    def unload(self) -> None:
        if self._handle is not None:
            self.runtime.release(self._handle)
            self._handle = None

    # --- execution -------------------------------------------------------

    def preprocess(self, request: InferenceRequest) -> PreparedInput:
        if not request.images:
            raise ConfigInvalidError("YoloV8Adapter requires at least one image")
        if len(request.images) > 1:
            raise ConfigInvalidError(
                "this export has a static batch size of 1; batched inference requires a re-export"
            )

        image = request.images[0]
        padded, meta = letterbox(image, self.input_size)
        tensor = to_model_input(padded)

        return PreparedInput(
            tensors={"images": tensor},
            context={
                "letterbox": meta,
                "confidence": request.confidence if request.confidence is not None else _DEFAULT_CONFIDENCE,
                "iou": request.iou if request.iou is not None else _DEFAULT_IOU,
                "allowed_class_ids": request.allowed_class_ids,
            },
        )

    def infer(self, prepared: PreparedInput) -> RawOutput:
        if self._handle is None:
            raise ModelLoadError("adapter is not loaded")
        tensor = next(iter(prepared.tensors.values()))
        inputs = {self._handle.input_names[0]: tensor}
        outputs = self.runtime.run(self._handle, inputs)
        return RawOutput(tensors=outputs, names=list(self._handle.output_names))

    def synchronize(self) -> None:
        """Expose the runtime's synchronization so the engine can time it correctly."""
        if self._handle is not None:
            self.runtime.synchronize(self._handle)

    def postprocess(self, raw: RawOutput, prepared: PreparedInput) -> InferenceOutput:
        ctx = prepared.context
        boxes, scores, class_ids = decode_yolov8(
            raw.tensors[0],
            ctx["confidence"],
            ctx["iou"],
            ctx["allowed_class_ids"],
        )
        if boxes.shape[0]:
            boxes = scale_boxes(boxes, ctx["letterbox"])

        provider = self._handle.execution_provider if self._handle else "unknown"
        detections = [
            Detection(
                x1=float(b[0]), y1=float(b[1]), x2=float(b[2]), y2=float(b[3]),
                confidence=float(s),
                classId=int(c),
                className=self.class_names[int(c)] if int(c) < len(self.class_names) else str(int(c)),
                inferenceBackend=f"{self.runtime.runtime_id}:{provider}",
                modelName=self.metadata.model_id,
                modelVersion=self.version,
            )
            for b, s, c in zip(boxes, scores, class_ids, strict=False)
        ]
        return InferenceOutput(detections=detections)

    # --- benchmarking support --------------------------------------------

    def synthetic_request(self, batch_size: int = 1) -> InferenceRequest:
        """A deterministic mid-grey frame.

        Flat grey rather than random noise: noise produces thousands of spurious
        low-confidence boxes, so NMS cost would dominate and the measurement would
        reflect the postprocessor rather than the model. Grey yields near-zero
        detections, isolating the runtime path.
        """
        frame = np.full((self.input_size, self.input_size, 3), 128, dtype=np.uint8)
        return InferenceRequest(
            images=[frame] * batch_size,
            confidence=_DEFAULT_CONFIDENCE,
            iou=_DEFAULT_IOU,
        )

    def evaluate(
        self,
        predictions: list[InferenceOutput],
        references: list[ReferenceOutput],
    ) -> QualityMetrics:
        """Detection quality against ground truth.

        Full COCO mAP requires the reference dataset and the pycocotools evaluation
        protocol, neither of which is wired up. Rather than emit a plausible-looking
        number from an incomplete implementation, every metric is returned
        unavailable with that as the stated reason.
        """
        reason = (
            "COCO mAP evaluation is not implemented: it requires an annotated reference "
            "dataset and the COCO evaluation protocol, neither of which is configured "
            "for this run"
        )
        unavailable = Measurement[float].unavailable(reason, unit="AP")
        return QualityMetrics(
            detection=DetectionQuality(
                map_50_95=unavailable,
                map_50=unavailable,
                map_75=unavailable,
                precision=Measurement[float].unavailable(reason),
                recall=Measurement[float].unavailable(reason),
            ),
            reference_dataset=None,
            sample_count=len(references),
        )
