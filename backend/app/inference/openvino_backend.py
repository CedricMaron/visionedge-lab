"""OpenVINO detection backend.

Honest availability model: the backend advertises itself only when the ``openvino``
package imports AND a converted IR/ONNX model is present. In this build OpenVINO is
not installed, so ``is_available`` returns False and the factory will not offer it.
The real inference path is implemented so that installing openvino + converting a model
activates it without further code changes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from app.core.errors import RuntimeUnavailableError
from app.core.logging import get_logger
from app.core.types import Detection, HealthState
from app.inference.base import BaseDetectionBackend
from app.inference.postprocess import decode_yolov8
from app.inference.preprocess import letterbox, scale_boxes, to_model_input
from app.models.coco import COCO_CLASSES

log = get_logger("inference.openvino")


def openvino_available() -> bool:
    try:
        import openvino  # noqa: F401

        return True
    except Exception:
        return False


class OpenVINOBackend(BaseDetectionBackend):
    backend_name = "openvino"

    def __init__(
        self,
        model_path: str | Path,
        model_id: str,
        model_version: str = "v8.0",
        input_size: int = 640,
        precision: str = "fp16",
        device: str = "CPU",
        class_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.model_path = Path(model_path)
        self.model_id = model_id
        self.model_version = model_version
        self.input_size = input_size
        self.precision = precision
        self.device = device
        self.class_names = class_names or COCO_CLASSES
        self._compiled = None
        self._output = None

    def load(self) -> None:
        if not openvino_available():
            self._health = HealthState.ERROR
            raise RuntimeUnavailableError("OpenVINO is not installed on this device")
        import openvino as ov  # pragma: no cover - exercised only where installed

        self._health = HealthState.LOADING
        core = ov.Core()
        model = core.read_model(str(self.model_path))
        self._compiled = core.compile_model(model, self.device)
        self._output = self._compiled.output(0)
        self._health = HealthState.READY
        log.info("openvino_loaded", model_id=self.model_id, device=self.device)

    def warmup(self) -> None:  # pragma: no cover
        self.predict(np.zeros((self.input_size, self.input_size, 3), np.uint8), 0.25, 0.45, None)

    def predict(  # pragma: no cover - runs only where openvino is installed
        self,
        image: np.ndarray,
        conf_threshold: float,
        iou_threshold: float,
        allowed_class_ids: set[int] | None = None,
    ) -> list[Detection]:
        if self._compiled is None:
            raise RuntimeUnavailableError("OpenVINO model not loaded")
        padded, meta = letterbox(image, self.input_size)
        tensor = to_model_input(padded)
        out = self._compiled([tensor])[self._output]
        boxes, scores, class_ids = decode_yolov8(out, conf_threshold, iou_threshold, allowed_class_ids)
        if boxes.shape[0]:
            boxes = scale_boxes(boxes, meta)
        return [
            Detection(
                x1=float(b[0]), y1=float(b[1]), x2=float(b[2]), y2=float(b[3]),
                confidence=float(s), classId=int(c),
                className=self.class_names[int(c)] if int(c) < len(self.class_names) else str(int(c)),
                inferenceBackend=f"{self.backend_name}:{self.device}",
                modelName=self.model_id, modelVersion=self.model_version,
            )
            for b, s, c in zip(boxes, scores, class_ids)
        ]

    def close(self) -> None:
        self._compiled = None
        self._output = None
        self._health = HealthState.UNKNOWN
