"""ImageNet classification over an ONNX MobileNet-family graph.

Preprocessing follows the timm convention the weights were trained with — resize
the short side by ``1/crop_pct``, centre-crop to the input size, scale to [0,1],
then normalize by the ImageNet channel statistics. Getting any step wrong silently
degrades accuracy rather than raising, so the parameters are read from the model's
own ``config.json`` instead of being hardcoded.

Logits are converted to probabilities with a numerically-stable softmax (subtract
the max before exponentiating), because the raw graph emits unnormalized logits and
reporting those as confidences would be wrong by a factor that varies per image.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
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
from app.runtimes.base import RuntimeAdapter, SessionConfig, SessionHandle
from app.schemas.enums import DeviceKind, Modality, Phase, Precision, Task
from app.schemas.measurement import Measurement
from app.schemas.quality import ClassificationQuality, QualityMetrics

log = get_logger("adapters.mobilenet")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


class MobileNetClassifierAdapter:
    """Image classification. Runtime-agnostic; execution belongs to the runtime adapter."""

    #: Image preprocessing: resize, centre-crop and normalize.
    preprocess_phase = Phase.PREPROCESSING

    def __init__(
        self,
        model_path: str | Path,
        runtime: RuntimeAdapter,
        config_path: str | Path | None = None,
        model_id: str = "mobilenetv4-conv-small-onnx",
        display_name: str = "MobileNetV4 Conv Small (ONNX, ImageNet-1k)",
    ) -> None:
        self.model_path = Path(model_path)
        self.runtime = runtime
        self.config_path = Path(config_path) if config_path else self.model_path.parent / "config.json"
        self._handle: SessionHandle | None = None

        config = self._read_config()
        self.input_size: int = int(config.get("input_size", [3, 224, 224])[-1])
        self.crop_pct: float = float(config.get("crop_pct", 0.875))
        self.labels: list[str] = self._labels(config)

        self.metadata = ModelMetadata(
            model_id=model_id,
            display_name=display_name,
            family="mobilenetv4",
            task=Task.IMAGE_CLASSIFICATION,
            modality=Modality.IMAGE,
            source_repository="https://github.com/huggingface/pytorch-image-models",
            paper_url="https://arxiv.org/abs/2404.10518",
            model_license="Apache-2.0",
            weights_license="Apache-2.0",
            commercial_use_permitted=True,
            auto_download_permitted=True,
            parameters_millions=3.8,
            model_size_bytes=self.model_path.stat().st_size if self.model_path.exists() else None,
            revision="e2400_r224_in1k",
            supported_precisions=[Precision.FP32, Precision.INT8],
            supported_devices=[DeviceKind.CPU, DeviceKind.CUDA],
            supported_runtimes=["onnxruntime"],
            input_format=f"BGR uint8 HWC image, any size (centre-cropped to {self.input_size}x{self.input_size})",
            output_format="ranked (class_id, label, probability) tuples",
            dynamic_input_supported=True,
            streaming_supported=False,
            batch_supported=True,
            supported_quantizations=["int8-dynamic"],
            hardware_requirements=HardwareRequirement(min_ram_mb=256, min_disk_mb=20),
            known_limitations=[
                "ImageNet-1k classes only.",
                "Accuracy is sensitive to the preprocessing pipeline; this adapter follows "
                "the timm convention the weights were trained with.",
            ],
        )

    def _read_config(self) -> dict:
        if not self.config_path.exists():
            raise ModelLoadError(
                f"classifier config not found at {self.config_path}; preprocessing parameters "
                "and class labels cannot be assumed"
            )
        return json.loads(self.config_path.read_text(encoding="utf-8"))

    @staticmethod
    def _labels(config: dict) -> list[str]:
        mapping = config.get("id2label") or config.get("label2id") or {}
        if not mapping:
            raise ModelLoadError("classifier config contains no class labels")
        if isinstance(mapping, dict):
            # Keys are stringified indices; order by numeric index, not string order,
            # or class 10 would sort before class 2.
            return [mapping[k] for k in sorted(mapping, key=lambda x: int(x))]
        return list(mapping)

    # --- lifecycle -------------------------------------------------------

    def load(self, config: LoadConfig) -> LoadResult:
        if not self.model_path.exists():
            raise ModelLoadError(f"model file missing: {self.model_path}")

        t0 = time.perf_counter()
        handle = self.runtime.create_session(
            SessionConfig(
                model_path=str(self.model_path),
                device=config.device,
                device_index=config.device_index,
                precision=config.precision,
                intra_op_threads=config.thread_config.get("intra_op"),
                inter_op_threads=config.thread_config.get("inter_op"),
            )
        )
        load_ms = (time.perf_counter() - t0) * 1000.0

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
            raise ConfigInvalidError("MobileNetClassifierAdapter requires at least one image")

        resize_to = int(round(self.input_size / self.crop_pct))
        batch = np.stack([self._prepare_one(img, resize_to) for img in request.images])
        return PreparedInput(tensors={"pixel_values": batch}, context={"batch": len(request.images)})

    def _prepare_one(self, image: np.ndarray, resize_to: int) -> np.ndarray:
        h, w = image.shape[:2]
        scale = resize_to / min(h, w)
        resized = cv2.resize(
            image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_LINEAR
        )
        rh, rw = resized.shape[:2]
        top = max(0, (rh - self.input_size) // 2)
        left = max(0, (rw - self.input_size) // 2)
        cropped = resized[top : top + self.input_size, left : left + self.input_size]

        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        return normalized.transpose(2, 0, 1)

    def infer(self, prepared: PreparedInput) -> RawOutput:
        if self._handle is None:
            raise ModelLoadError("adapter is not loaded")
        inputs = {self._handle.input_names[0]: prepared.tensors["pixel_values"]}
        return RawOutput(tensors=self.runtime.run(self._handle, inputs),
                         names=list(self._handle.output_names))

    def synchronize(self) -> None:
        if self._handle is not None:
            self.runtime.synchronize(self._handle)

    def postprocess(self, raw: RawOutput, prepared: PreparedInput, top_k: int = 5) -> InferenceOutput:
        probabilities = softmax(raw.tensors[0].astype(np.float32))
        first = probabilities[0]
        ranked = np.argsort(first)[::-1][:top_k]
        return InferenceOutput(
            classifications=[
                (int(i), self.labels[int(i)] if int(i) < len(self.labels) else str(i), float(first[i]))
                for i in ranked
            ],
            extra={"batch_size": probabilities.shape[0], "num_classes": probabilities.shape[1]},
        )

    # --- benchmarking ----------------------------------------------------

    def synthetic_request(self, batch_size: int = 1) -> InferenceRequest:
        """Deterministic mid-grey input, isolating the runtime path from image content."""
        frame = np.full((self.input_size, self.input_size, 3), 128, dtype=np.uint8)
        return InferenceRequest(images=[frame] * batch_size)

    def evaluate(
        self,
        predictions: list[InferenceOutput],
        references: list[ReferenceOutput],
    ) -> QualityMetrics:
        """Top-1/top-5 accuracy against labelled references.

        Computed only when references are supplied. With none, every metric is
        unavailable with that as the reason rather than defaulting to zero — which
        would read as a model that gets everything wrong.
        """
        if not references:
            reason = "no labelled reference data was supplied for this run"
            return QualityMetrics(
                classification=ClassificationQuality(
                    top1_accuracy=Measurement[float].unavailable(reason),
                    top5_accuracy=Measurement[float].unavailable(reason),
                    f1_macro=Measurement[float].unavailable(reason),
                ),
                reference_dataset=None,
                sample_count=0,
            )

        if len(predictions) != len(references):
            raise ConfigInvalidError(
                f"{len(predictions)} predictions but {len(references)} references"
            )

        top1 = top5 = 0
        for prediction, reference in zip(predictions, references, strict=True):
            ranked = [c[0] for c in (prediction.classifications or [])]
            if reference.class_id is None or not ranked:
                continue
            if ranked[0] == reference.class_id:
                top1 += 1
            if reference.class_id in ranked[:5]:
                top5 += 1

        n = len(references)
        source = f"computed over {n} labelled examples"
        unavailable_f1 = Measurement[float].unavailable(
            "macro F1 requires per-class aggregation over a full dataset, which this "
            "per-run evaluation does not perform"
        )
        return QualityMetrics(
            classification=ClassificationQuality(
                top1_accuracy=Measurement[float].derived(top1 / n, "fraction", source),
                top5_accuracy=Measurement[float].derived(top5 / n, "fraction", source),
                f1_macro=unavailable_f1,
            ),
            reference_dataset=None,
            sample_count=n,
        )
