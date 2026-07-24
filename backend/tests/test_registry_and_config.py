"""Tests for the model registry loader/validation and inference config validation."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.capabilities.scanner import BackendCapabilities, RuntimeAvailability
from app.core.errors import ConfigInvalidError
from app.inference.config import InferenceConfig, validate_against_capabilities
from app.models.registry import load_registry, refresh_deployment_status


def test_registry_loads_and_has_nano_installed():
    reg = refresh_deployment_status(load_registry())
    nano = reg.detection("yolov8n-onnx")
    assert nano is not None
    assert nano.format == "onnx"
    assert nano.deployment_status == "installed"
    assert nano.checksum_sha256 and len(nano.checksum_sha256) == 64


def test_registry_marks_uninstalled_models():
    reg = load_registry()
    small = reg.detection("yolov8s-onnx")
    assert small is not None
    assert small.deployment_status == "not_installed"


def test_registry_has_vlm_entries():
    reg = load_registry()
    assert reg.vlm("mock-vlm") is not None
    assert reg.vlm("smolvlm-256m").license == "Apache-2.0"


def test_config_rejects_unknown_runtime():
    with pytest.raises(ValidationError):
        InferenceConfig(model_id="yolov8n-onnx", runtime="not-a-runtime")


def test_config_rejects_bad_input_size():
    with pytest.raises(ValidationError):
        InferenceConfig(model_id="yolov8n-onnx", input_size=641)  # not multiple of 32


def _caps(cuda=False, ov=False, trt=False, torch=True):
    return BackendCapabilities(
        os="Linux", os_version="x", python_version="3.12", cpu_model="cpu",
        cpu_cores_physical=6, cpu_cores_logical=12, ram_total_mb=4000, ram_available_mb=1000,
        gpus=[], nvidia_gpu_present=False,
        runtimes=RuntimeAvailability(onnxruntime=True, onnxruntime_cuda=cuda, pytorch=torch,
                                     openvino=ov, tensorrt=trt),
        supported_precisions=["fp32"],
    )


def test_validate_cuda_unavailable_raises():
    cfg = InferenceConfig(model_id="yolov8n-onnx", runtime="onnxruntime-cuda")
    with pytest.raises(ConfigInvalidError):
        validate_against_capabilities(cfg, _caps(cuda=False))


def test_validate_cpu_always_ok():
    cfg = InferenceConfig(model_id="yolov8n-onnx", runtime="onnxruntime-cpu")
    validate_against_capabilities(cfg, _caps())  # no raise


def test_validate_openvino_and_tensorrt_absent():
    for rt in ("openvino", "tensorrt"):
        with pytest.raises(ConfigInvalidError):
            validate_against_capabilities(
                InferenceConfig(model_id="yolov8n-onnx", runtime=rt), _caps()
            )
