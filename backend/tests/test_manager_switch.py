"""Integration tests for detection model switching, rollback and memory release.

These use the real ONNX nano model (installed in models/). If the model file is missing
the tests are skipped rather than failing.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.capabilities.scanner import scan_capabilities
from app.core.config import REPO_ROOT
from app.core.errors import ModelLoadError
from app.core.types import HealthState
from app.inference.config import InferenceConfig
from app.inference.manager import DetectionManager, _verify_runtime_honored
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


def test_cuda_runtime_rejected_when_backend_loaded_on_cpu():
    """Hardware-independent: a CUDA runtime name must not be accepted for a CPU load.

    ONNX Runtime silently falls back to the CPU provider when the CUDA one cannot be
    created, so the manager verifies the device that actually loaded.
    """
    class _FakeBackend:
        device = "cpu"
        provider = "CPUExecutionProvider"

    cfg = InferenceConfig(model_id="yolov8n-onnx", runtime="onnxruntime-cuda")
    with pytest.raises(ModelLoadError, match="requires device 'cuda'"):
        _verify_runtime_honored(_FakeBackend(), cfg)

    _FakeBackend.device = "cuda"
    _FakeBackend.provider = "CUDAExecutionProvider"
    _verify_runtime_honored(_FakeBackend(), cfg)  # honored -> no raise

    # a CPU runtime makes no device promise beyond loading
    _verify_runtime_honored(_FakeBackend(), InferenceConfig(model_id="yolov8n-onnx"))


def test_switch_to_unknown_model_rolls_back(manager):
    result = manager.switch(InferenceConfig(model_id="does-not-exist", runtime="onnxruntime-cpu"))
    assert result.ok is False and result.rolled_back is True
    assert manager.config.model_id == "yolov8n-onnx"


def test_switch_records_events(manager):
    manager.switch(InferenceConfig(model_id="does-not-exist", runtime="onnxruntime-cpu"))
    kinds = [e["kind"] for e in manager.events]
    assert "switch_failed_rollback" in kinds


def test_benchmark_does_not_hold_the_lock_for_the_whole_run(manager):
    """A live frame must interleave with benchmark runs, not queue behind all of them.

    Measured against the benchmark's own wall time rather than a single-run sample,
    because per-inference latency on a loaded CPU varies by 3x and would make the
    threshold meaningless. Interleaving correctly, the live frame waits for roughly
    one in-flight inference (~8% of a 20-run benchmark here); holding the lock for
    the whole loop puts it at ~100%.
    """
    import threading
    import time

    blocked_ms: list[float] = []
    bench_ms: list[float] = []

    def predict_once():
        t0 = time.perf_counter()
        manager.predict(np.zeros((640, 640, 3), np.uint8))
        blocked_ms.append((time.perf_counter() - t0) * 1000.0)

    def run_bench():
        t0 = time.perf_counter()
        manager.benchmark(runs=20)
        bench_ms.append((time.perf_counter() - t0) * 1000.0)

    t = threading.Thread(target=predict_once)
    bench = threading.Thread(target=run_bench)
    bench.start()
    time.sleep(0.01)
    t.start()
    t.join()
    bench.join()

    assert blocked_ms and bench_ms
    assert blocked_ms[0] < bench_ms[0] * 0.5


def test_benchmark_records_notes(manager):
    result = manager.benchmark(runs=3, notes="measured with 7 concurrent live frames")
    assert result.notes == "measured with 7 concurrent live frames"
    assert result.runs == 3


def test_benchmark_returns_measured_values(manager):
    result = manager.benchmark(runs=5)
    assert result.runs == 5
    assert result.fps > 0
    assert result.latency_p95_ms >= result.latency_p50_ms
