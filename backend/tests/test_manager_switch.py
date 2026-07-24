"""Integration tests for detection model switching, rollback and memory release.

These use the real ONNX nano model (installed in models/). If the model file is missing
the tests are skipped rather than failing.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.capabilities.scanner import scan_capabilities
from app.core.config import REPO_ROOT
from app.core.types import HealthState
from app.inference.config import InferenceConfig
from app.inference.manager import DetectionManager
from app.models.registry import load_registry, refresh_deployment_status

MODEL = REPO_ROOT / "models" / "yolov8n.onnx"
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="yolov8n.onnx not installed")


@pytest.fixture
def manager():
    reg = refresh_deployment_status(load_registry())
    caps = scan_capabilities()
    mgr = DetectionManager(reg, caps)
    mgr.initialize(InferenceConfig(model_id="yolov8n-onnx", runtime="onnxruntime-cpu"))
    yield mgr
    mgr.close()


def test_initialize_ready_and_predicts(manager):
    assert manager.health() == HealthState.READY
    dets = manager.predict(np.zeros((640, 640, 3), np.uint8))
    assert isinstance(dets, list)


def test_switch_to_cuda_rolls_back(manager):
    # CUDA EP is unavailable in this environment -> must roll back, not crash
    result = manager.switch(InferenceConfig(model_id="yolov8n-onnx", runtime="onnxruntime-cuda"))
    assert result.ok is False
    assert result.rolled_back is True
    # previous working config restored and still serving
    assert manager.config.runtime == "onnxruntime-cpu"
    assert manager.health() == HealthState.READY
    assert manager.predict(np.zeros((640, 640, 3), np.uint8)) is not None


def test_switch_to_unknown_model_rolls_back(manager):
    result = manager.switch(InferenceConfig(model_id="does-not-exist", runtime="onnxruntime-cpu"))
    assert result.ok is False and result.rolled_back is True
    assert manager.config.model_id == "yolov8n-onnx"


def test_switch_records_events(manager):
    manager.switch(InferenceConfig(model_id="does-not-exist", runtime="onnxruntime-cpu"))
    kinds = [e["kind"] for e in manager.events]
    assert "switch_failed_rollback" in kinds


def test_benchmark_returns_measured_values(manager):
    result = manager.benchmark(runs=5)
    assert result.runs == 5
    assert result.fps > 0
    assert result.latency_p95_ms >= result.latency_p50_ms
