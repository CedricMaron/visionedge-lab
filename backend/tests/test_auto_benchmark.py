"""The auto-benchmark job: real steps, one stored row, honest notes."""
from __future__ import annotations

import pytest

from app.benchmarking.auto import BENCHMARK_JOB_KIND, make_job_factory
from app.capabilities.scanner import scan_capabilities
from app.core.config import REPO_ROOT, get_settings
from app.core.state import AppState
from app.inference.config import InferenceConfig
from app.inference.manager import DetectionManager
from app.jobs.manager import JobManager
from app.jobs.state import JobState
from app.models.registry import load_registry, refresh_deployment_status
from app.monitoring.metrics import RollingMetrics
from app.storage.db import Database

MODEL = REPO_ROOT / "models" / "yolov8n.onnx"
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="yolov8n.onnx not installed")


@pytest.fixture
def state(tmp_path):
    registry = refresh_deployment_status(load_registry())
    caps = scan_capabilities()
    detection = DetectionManager(registry, caps)
    detection.initialize(InferenceConfig(model_id="yolov8n-onnx", runtime="onnxruntime-cpu"))
    st = AppState(
        settings=get_settings(),
        capabilities=caps,
        registry=registry,
        db=Database(tmp_path / "test.db"),
        detection=detection,
        metrics=RollingMetrics(),
    )
    st.jobs = JobManager(factory=make_job_factory(st))
    yield st
    detection.close()


def test_job_runs_one_step_per_inference_and_stores_one_row(state):
    state.jobs.submit("b1", BENCHMARK_JOB_KIND, {"runs": 4})
    state.jobs.start("b1")
    state.jobs.wait("b1", timeout=120)

    rec = state.jobs.get("b1")
    assert rec.state is JobState.COMPLETED
    assert rec.total_steps == 4
    assert rec.current_step == 4

    rows = state.db.list_benchmarks()
    assert len(rows) == 1
    assert rows[0]["model_id"] == "yolov8n-onnx"
    assert rows[0]["runs"] == 4
    assert rows[0]["fps"] > 0


def test_notes_report_concurrent_live_frames(state):
    state.metrics.record(10.0, 10.0)  # a live frame before the run
    state.jobs.submit("b2", BENCHMARK_JOB_KIND, {"runs": 2})
    state.jobs.start("b2")
    state.jobs.wait("b2", timeout=120)

    notes = state.db.list_benchmarks()[0]["notes"]
    assert "concurrent live frames" in notes
    assert "0 concurrent live frames" in notes  # none arrived DURING the run


def test_cancelled_job_stores_no_row(state):
    state.jobs.submit("b3", BENCHMARK_JOB_KIND, {"runs": 500})
    state.jobs.start("b3")
    state.jobs.cancel("b3")
    state.jobs.wait("b3", timeout=120)

    assert state.db.list_benchmarks() == []


def test_switch_submits_a_benchmark_job_and_rollback_does_not():
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app()) as c:
        ok = c.post("/api/detection/switch", json={
            "model_id": "yolov8n-onnx", "runtime": "onnxruntime-cpu",
            "execution_location": "pc_local",
        })
        assert ok.json()["ok"] is True
        jobs = c.get("/api/jobs").json()["jobs"]
        assert len([j for j in jobs if j["kind"] == "benchmark"]) == 1

        bad = c.post("/api/detection/switch", json={
            "model_id": "does-not-exist", "runtime": "onnxruntime-cpu",
            "execution_location": "pc_local",
        })
        assert bad.json()["ok"] is False
        jobs_after = c.get("/api/jobs").json()["jobs"]
        # The failed switch added nothing.
        assert len([j for j in jobs_after if j["kind"] == "benchmark"]) == 1
