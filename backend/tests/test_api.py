"""API integration tests via FastAPI TestClient (exercises startup/lifespan too)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.config import REPO_ROOT
from app.main import create_app

SAMPLE = REPO_ROOT / "benchmark-data" / "sample_bus.jpg"


@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app) as c:  # triggers lifespan startup
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_capabilities_shape(client):
    d = client.get("/api/capabilities").json()
    assert d["cpu_cores_logical"] >= 1
    assert "onnxruntime" in d["runtimes"]


def test_models_and_classes(client):
    models = client.get("/api/models").json()
    assert any(m["model_id"] == "yolov8n-onnx" for m in models["detection_models"])
    classes = client.get("/api/classes").json()
    assert len(classes["classes"]) == 80
    assert "people" in classes["groups"]


def test_metrics_endpoint(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert b"ve_frames_total" in r.content


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample image missing")
def test_infer_returns_detections(client):
    with open(SAMPLE, "rb") as f:
        r = client.post("/api/infer?confidence=0.25", files={"file": ("s.jpg", f, "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] >= 1
    assert any(d["className"] == "person" for d in body["detections"])


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample image missing")
def test_vlm_analyze_grounded(client):
    with open(SAMPLE, "rb") as f:
        r = client.post("/api/vlm/analyze-image",
                        files={"file": ("s.jpg", f, "image/jpeg")},
                        data={"ground": "true", "structured": "true"})
    assert r.status_code == 200
    body = r.json()
    assert body["response"]["model_id"] == "mock-vlm"
    assert body["disclaimer"]
    assert body["agreement"]["detected_classes"]


def test_runtime_status(client):
    d = client.get("/api/runtime-status").json()
    assert "detection" in d and "runtimes" in d


def test_infer_returns_a_real_timing_breakdown(client):
    if not SAMPLE.exists():
        pytest.skip("sample image not installed")
    with open(SAMPLE, "rb") as f:
        r = client.post("/api/infer", files={"file": ("bus.jpg", f.read(), "image/jpeg")})
    t = r.json()["timings"]
    for key in ("preprocess_ms", "inference_ms", "postprocess_ms", "end_to_end_ms"):
        assert key in t and t[key] >= 0.0
    parts = t["preprocess_ms"] + t["inference_ms"] + t["postprocess_ms"]
    assert parts <= t["end_to_end_ms"] + 1.0        # parts fit inside the whole
    assert t["inference_ms"] < t["end_to_end_ms"]   # not the same number twice


class TestDeployVerifiability:
    """/health must identify which build is running.

    A deploy that cannot tell a new process from an old one will silently leave
    stale code running and then confirm its own success — which is exactly what
    happened on the first InferenceLab deploy: the frontend was rebuilt, the
    backend scheduled task was already Running so the start request was ignored,
    and the health probe got a 200 from the old process.
    """

    def test_health_reports_the_running_version(self, client):
        body = client.get("/health").json()
        assert body["version"], "/health must report the app version"

    def test_health_reports_the_running_commit(self, client):
        body = client.get("/health").json()
        # None is acceptable (a tarball deploy has no git metadata); the key must
        # exist so a deploy can assert on it.
        assert "git_commit" in body

    def test_health_keeps_its_existing_contract(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert "detection_health" in body and "warnings" in body
