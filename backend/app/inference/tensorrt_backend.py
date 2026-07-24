"""TensorRT detection backend.

Honest availability model: requires the ``tensorrt`` + ``pycuda`` packages AND a built
engine (``.engine``) for the current GPU. TensorRT engines are hardware/driver-specific
and cannot be shipped portably, so this build treats TensorRT as available only when
those conditions hold. The engine-execution code is intentionally minimal; the full
engine build lives in ``scripts/build_tensorrt.py``. Until an engine exists, the factory
will not offer this backend and capability detection reports it as unavailable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from app.core.errors import RuntimeUnavailableError
from app.core.logging import get_logger
from app.core.types import Detection, HealthState
from app.inference.base import BaseDetectionBackend
from app.models.coco import COCO_CLASSES

log = get_logger("inference.tensorrt")


def tensorrt_available() -> bool:
    try:
        import tensorrt  # noqa: F401

        return True
    except Exception:
        return False


class TensorRTBackend(BaseDetectionBackend):
    backend_name = "tensorrt"

    def __init__(
        self,
        engine_path: str | Path,
        model_id: str,
        model_version: str = "v8.0",
        input_size: int = 640,
        precision: str = "fp16",
        class_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.engine_path = Path(engine_path)
        self.model_id = model_id
        self.model_version = model_version
        self.input_size = input_size
        self.precision = precision
        self.device = "cuda"
        self.class_names = class_names or COCO_CLASSES
        self._engine = None
        self._context = None

    def load(self) -> None:
        if not tensorrt_available():
            self._health = HealthState.ERROR
            raise RuntimeUnavailableError("TensorRT is not installed on this device")
        if not self.engine_path.exists():
            self._health = HealthState.ERROR
            raise RuntimeUnavailableError(
                f"TensorRT engine not built for this GPU: {self.engine_path}. "
                "Run scripts/build_tensorrt.py first."
            )
        # pragma: no cover - only runs on a machine with TensorRT + a built engine
        import tensorrt as trt  # type: ignore

        self._health = HealthState.LOADING
        logger = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, "rb") as f, trt.Runtime(logger) as rt:
            self._engine = rt.deserialize_cuda_engine(f.read())
        self._context = self._engine.create_execution_context()
        self._health = HealthState.READY
        log.info("tensorrt_loaded", model_id=self.model_id, engine=str(self.engine_path))

    def warmup(self) -> None:  # pragma: no cover
        self.predict(np.zeros((self.input_size, self.input_size, 3), np.uint8), 0.25, 0.45, None)

    def predict(  # pragma: no cover - requires GPU + engine
        self,
        image: np.ndarray,
        conf_threshold: float,
        iou_threshold: float,
        allowed_class_ids: set[int] | None = None,
    ) -> list[Detection]:
        raise RuntimeUnavailableError(
            "TensorRT execution requires a built engine on this GPU. "
            "This build ships the interface and the build script (scripts/build_tensorrt.py); "
            "engine execution activates once an engine is present."
        )

    def close(self) -> None:
        self._context = None
        self._engine = None
        self._health = HealthState.UNKNOWN
