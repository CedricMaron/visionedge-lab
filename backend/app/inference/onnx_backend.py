"""ONNX Runtime detection backend (CPU and, when available, CUDA).

The execution provider is selected from what ORT actually reports — CUDA is used only
if ``CUDAExecutionProvider`` is genuinely available, otherwise it falls back to CPU and
records that decision. Nothing is claimed that the runtime does not offer.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from app.core.errors import ModelLoadError, RuntimeUnavailableError
from app.core.logging import get_logger
from app.core.types import Detection, HealthState
from app.inference.base import BaseDetectionBackend
from app.inference.postprocess import decode_yolov8
from app.inference.preprocess import letterbox, scale_boxes, to_model_input
from app.models.coco import COCO_CLASSES

log = get_logger("inference.onnx")


class OnnxRuntimeBackend(BaseDetectionBackend):
    backend_name = "onnxruntime"

    def __init__(
        self,
        model_path: str | Path,
        model_id: str,
        model_version: str = "v8.0",
        input_size: int = 640,
        precision: str = "fp32",
        prefer_cuda: bool = False,
        class_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.model_path = Path(model_path)
        self.model_id = model_id
        self.model_version = model_version
        self.input_size = input_size
        self.precision = precision
        self.prefer_cuda = prefer_cuda
        self.class_names = class_names or COCO_CLASSES
        self.provider: str | None = None
        self._sess = None
        self._input_name: str | None = None

    def _select_providers(self) -> list[str]:
        import onnxruntime as ort

        available = ort.get_available_providers()
        chosen: list[str] = []
        if self.prefer_cuda:
            if "CUDAExecutionProvider" in available:
                chosen.append("CUDAExecutionProvider")
            else:
                log.warning("cuda_requested_unavailable", available=available)
        chosen.append("CPUExecutionProvider")
        return chosen

    def load(self) -> None:
        import onnxruntime as ort

        if not self.model_path.exists():
            raise ModelLoadError(f"model file missing: {self.model_path}")
        self._health = HealthState.LOADING
        try:
            providers = self._select_providers()
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._sess = ort.InferenceSession(str(self.model_path), sess_options=so, providers=providers)
            self.provider = self._sess.get_providers()[0]
            self.device = "cuda" if self.provider == "CUDAExecutionProvider" else "cpu"
            self._input_name = self._sess.get_inputs()[0].name
        except Exception as exc:  # noqa: BLE001
            self._health = HealthState.ERROR
            raise ModelLoadError(f"onnx load failed: {exc}") from exc
        self._health = HealthState.READY
        log.info("onnx_loaded", model_id=self.model_id, provider=self.provider)

    def warmup(self) -> None:
        if self._sess is None:
            raise RuntimeUnavailableError("session not loaded")
        dummy = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        for _ in range(2):
            self.predict(dummy, 0.25, 0.45, None)

    def predict(
        self,
        image: np.ndarray,
        conf_threshold: float,
        iou_threshold: float,
        allowed_class_ids: set[int] | None = None,
    ) -> list[Detection]:
        if self._sess is None or self._input_name is None:
            raise RuntimeUnavailableError("session not loaded")

        padded, meta = letterbox(image, self.input_size)
        tensor = to_model_input(padded)
        out = self._sess.run(None, {self._input_name: tensor})[0]

        boxes, scores, class_ids = decode_yolov8(out, conf_threshold, iou_threshold, allowed_class_ids)
        if boxes.shape[0]:
            boxes = scale_boxes(boxes, meta)

        dets: list[Detection] = []
        for (x1, y1, x2, y2), score, cid in zip(boxes, scores, class_ids, strict=False):
            cid_i = int(cid)
            dets.append(
                Detection(
                    x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                    confidence=float(score),
                    classId=cid_i,
                    className=self.class_names[cid_i] if cid_i < len(self.class_names) else str(cid_i),
                    inferenceBackend=f"{self.backend_name}:{self.provider}",
                    modelName=self.model_id,
                    modelVersion=self.model_version,
                )
            )
        return dets

    def close(self) -> None:
        self._sess = None
        self._input_name = None
        self._health = HealthState.UNKNOWN

    def predict_timed(
        self,
        image: np.ndarray,
        conf_threshold: float,
        iou_threshold: float,
        allowed_class_ids: set[int] | None = None,
    ) -> tuple[list[Detection], dict[str, float]]:
        """Predict with a per-stage timing breakdown (ms)."""
        if self._sess is None or self._input_name is None:
            raise RuntimeUnavailableError("session not loaded")
        t0 = time.perf_counter()
        padded, meta = letterbox(image, self.input_size)
        tensor = to_model_input(padded)
        t1 = time.perf_counter()
        out = self._sess.run(None, {self._input_name: tensor})[0]
        t2 = time.perf_counter()
        boxes, scores, class_ids = decode_yolov8(out, conf_threshold, iou_threshold, allowed_class_ids)
        if boxes.shape[0]:
            boxes = scale_boxes(boxes, meta)
        dets = [
            Detection(
                x1=float(b[0]), y1=float(b[1]), x2=float(b[2]), y2=float(b[3]),
                confidence=float(s), classId=int(c),
                className=self.class_names[int(c)] if int(c) < len(self.class_names) else str(int(c)),
                inferenceBackend=f"{self.backend_name}:{self.provider}",
                modelName=self.model_id, modelVersion=self.model_version,
            )
            for b, s, c in zip(boxes, scores, class_ids, strict=False)
        ]
        t3 = time.perf_counter()
        timings = {
            "preprocess_ms": (t1 - t0) * 1000.0,
            "inference_ms": (t2 - t1) * 1000.0,
            "postprocess_ms": (t3 - t2) * 1000.0,
            "end_to_end_ms": (t3 - t0) * 1000.0,
        }
        return dets, timings
