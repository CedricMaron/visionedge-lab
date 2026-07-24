"""PyTorch detection backend (reference).

Uses Ultralytics YOLO when it (and torch) are actually importable. If not installed,
the backend reports itself unavailable rather than pretending. This is the FP32
reference used for detection-quality / output-agreement comparison against optimized
runtimes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from app.core.errors import ModelLoadError, RuntimeUnavailableError
from app.core.logging import get_logger
from app.core.types import Detection, HealthState
from app.inference.base import BaseDetectionBackend
from app.models.coco import COCO_CLASSES

log = get_logger("inference.pytorch")


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        import ultralytics  # noqa: F401

        return True
    except Exception:
        return False


class PyTorchBackend(BaseDetectionBackend):
    backend_name = "pytorch"

    def __init__(
        self,
        weights_path: str | Path,
        model_id: str,
        model_version: str = "v8.0",
        input_size: int = 640,
        prefer_cuda: bool = False,
        class_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.weights_path = Path(weights_path)
        self.model_id = model_id
        self.model_version = model_version
        self.input_size = input_size
        self.prefer_cuda = prefer_cuda
        self.class_names = class_names or COCO_CLASSES
        self.precision = "fp32"
        self._model = None

    def load(self) -> None:
        if not torch_available():
            self._health = HealthState.ERROR
            raise RuntimeUnavailableError("PyTorch/Ultralytics not installed")
        import torch
        from ultralytics import YOLO

        if not self.weights_path.exists():
            raise ModelLoadError(f"weights missing: {self.weights_path}")
        self._health = HealthState.LOADING
        try:
            self._model = YOLO(str(self.weights_path))
            if self.prefer_cuda and torch.cuda.is_available():
                self._model.to("cuda")
                self.device = "cuda"
            else:
                self.device = "cpu"
        except Exception as exc:  # noqa: BLE001
            self._health = HealthState.ERROR
            raise ModelLoadError(f"pytorch load failed: {exc}") from exc
        self._health = HealthState.READY
        log.info("pytorch_loaded", model_id=self.model_id, device=self.device)

    def warmup(self) -> None:
        if self._model is None:
            raise RuntimeUnavailableError("model not loaded")
        self.predict(np.zeros((self.input_size, self.input_size, 3), np.uint8), 0.25, 0.45, None)

    def predict(
        self,
        image: np.ndarray,
        conf_threshold: float,
        iou_threshold: float,
        allowed_class_ids: set[int] | None = None,
    ) -> list[Detection]:
        if self._model is None:
            raise RuntimeUnavailableError("model not loaded")
        classes = sorted(allowed_class_ids) if allowed_class_ids else None
        res = self._model.predict(
            image, imgsz=self.input_size, conf=conf_threshold, iou=iou_threshold,
            classes=classes, verbose=False, device=self.device,
        )[0]
        dets: list[Detection] = []
        for b in res.boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            cid = int(b.cls[0])
            dets.append(
                Detection(
                    x1=x1, y1=y1, x2=x2, y2=y2, confidence=float(b.conf[0]),
                    classId=cid,
                    className=self.class_names[cid] if cid < len(self.class_names) else str(cid),
                    inferenceBackend=f"{self.backend_name}:{self.device}",
                    modelName=self.model_id, modelVersion=self.model_version,
                )
            )
        return dets

    def close(self) -> None:
        self._model = None
        self._health = HealthState.UNKNOWN
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
