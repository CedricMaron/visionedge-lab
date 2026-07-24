"""Build a concrete DetectionBackend from an InferenceConfig + registry entry."""
from __future__ import annotations

from app.core.config import REPO_ROOT
from app.core.errors import ConfigInvalidError, ModelNotFoundError
from app.inference.base import DetectionBackend
from app.inference.config import InferenceConfig
from app.inference.onnx_backend import OnnxRuntimeBackend
from app.inference.openvino_backend import OpenVINOBackend
from app.inference.pytorch_backend import PyTorchBackend
from app.inference.tensorrt_backend import TensorRTBackend
from app.models.registry import ModelRegistry


def build_backend(cfg: InferenceConfig, registry: ModelRegistry) -> DetectionBackend:
    entry = registry.detection(cfg.model_id)
    if entry is None:
        raise ModelNotFoundError(f"unknown model_id '{cfg.model_id}'")
    if entry.deployment_status not in ("installed",):
        raise ModelNotFoundError(
            f"model '{cfg.model_id}' is not installed (status={entry.deployment_status})"
        )

    path = REPO_ROOT / entry.local_path

    if cfg.runtime in ("onnxruntime-cpu", "onnxruntime-cuda"):
        if entry.format != "onnx":
            raise ConfigInvalidError(f"model '{cfg.model_id}' is not an ONNX model")
        return OnnxRuntimeBackend(
            model_path=path, model_id=entry.model_id, model_version=entry.version,
            input_size=cfg.input_size, precision=entry.precision,
            prefer_cuda=(cfg.runtime == "onnxruntime-cuda"),
        )
    if cfg.runtime == "pytorch":
        if entry.format != "pytorch":
            raise ConfigInvalidError(f"model '{cfg.model_id}' is not a PyTorch model")
        return PyTorchBackend(
            weights_path=path, model_id=entry.model_id, model_version=entry.version,
            input_size=cfg.input_size, prefer_cuda=True,
        )
    if cfg.runtime == "openvino":
        return OpenVINOBackend(
            model_path=path, model_id=entry.model_id, model_version=entry.version,
            input_size=cfg.input_size, precision=entry.precision,
        )
    if cfg.runtime == "tensorrt":
        engine = path.with_suffix(".engine")
        return TensorRTBackend(
            engine_path=engine, model_id=entry.model_id, model_version=entry.version,
            input_size=cfg.input_size, precision=entry.precision,
        )
    raise ConfigInvalidError(f"unhandled runtime '{cfg.runtime}'")
