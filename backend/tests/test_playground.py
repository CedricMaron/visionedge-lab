"""Playground endpoint: one instrumented run per modality, and honest failures.

These tests assert on the *shape* of the trace rather than on any duration, because
a timing assertion would be a flake on a loaded machine. What matters is that every
phase reported is one that ran, that tensor metadata describes real arrays, and that
a step nobody timed is reported as untimed rather than as zero.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.config import REPO_ROOT, get_settings
from app.main import create_app

SAMPLE = REPO_ROOT / "benchmark-data" / "sample_bus.jpg"


@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


def _installed(relative: str) -> bool:
    return (get_settings().models_dir / relative).exists()


detection_installed = pytest.mark.skipif(
    not _installed("yolov8n.onnx") or not SAMPLE.exists(),
    reason="yolov8n.onnx or the sample image is not installed",
)
embedding_installed = pytest.mark.skipif(
    not _installed("embedding/all-MiniLM-L6-v2.onnx"),
    reason="the MiniLM embedding model is not installed",
)


@detection_installed
def test_detection_run_returns_a_full_trace(client):
    r = client.post(
        "/api/playground/infer",
        data={
            "model_id": "yolov8n-onnx",
            "runtime_id": "onnxruntime",
            "device": "cpu",
            "precision": "fp32",
            "input_size": 640,
            "confidence": 0.25,
            "iou": 0.45,
        },
        files={"file": ("bus.jpg", SAMPLE.read_bytes(), "image/jpeg")},
    )
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["task"] == "object_detection"
    assert body["execution"] == "server"
    assert body["result"]["count"] == len(body["result"]["detections"])

    stage_ids = [s["id"] for s in body["stages"]]
    assert stage_ids == ["input", "decode", "preprocess", "inference", "postprocess", "output"]

    timed = {s["id"]: s["duration_ms"] for s in body["stages"]}
    assert timed["inference"] is not None and timed["inference"] > 0
    # The framing stages are not measurements and must not pretend to be.
    assert timed["input"] is None and timed["output"] is None

    preprocess = next(s for s in body["stages"] if s["id"] == "preprocess")
    tensor = preprocess["tensors"][0]
    assert tensor["shape"] == [1, 3, 640, 640]
    assert tensor["dtype"] == "float32"
    assert tensor["layout"] == "NCHW"
    assert tensor["bytes"] == 1 * 3 * 640 * 640 * 4
    # Every declared substep says explicitly that it was not separately timed.
    assert all(step["duration_ms"] is None and step["note"] for step in preprocess["substeps"])

    raw = next(s for s in body["stages"] if s["id"] == "inference")["tensors"][0]
    assert raw["shape"][0] == 1 and raw["shape"][1] == 84


@embedding_installed
def test_text_embedding_run_reports_tokenization(client):
    r = client.post(
        "/api/playground/infer",
        data={
            "model_id": "all-minilm-l6-v2-onnx",
            "runtime_id": "onnxruntime",
            "device": "cpu",
            "precision": "fp32",
            "text": "A man is riding a bicycle next to a red car.",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["task"] == "text_embedding"
    # Text preprocessing is tokenization, and the stage is named for what it did.
    assert next(s for s in body["stages"] if s["id"] == "preprocess")["name"] == "Tokenization"
    assert [s["id"] for s in body["stages"]] == [
        "input", "preprocess", "inference", "postprocess", "output",
    ]

    embedding = body["result"]["embedding"]
    assert embedding["dimension"] == 384
    assert embedding["tokens"] and embedding["tokens"] > 0
    assert embedding["token_preview"][0] == "[CLS]"
    assert embedding["norm"] == pytest.approx(1.0, abs=1e-3)


@embedding_installed
def test_text_model_refuses_an_empty_input(client):
    r = client.post(
        "/api/playground/infer",
        data={"model_id": "all-minilm-l6-v2-onnx", "runtime_id": "onnxruntime"},
    )
    assert r.status_code == 400
    assert "text" in r.json()["detail"]


def test_unknown_model_is_rejected_with_the_available_ones(client):
    r = client.post(
        "/api/playground/infer",
        data={"model_id": "no-such-model", "runtime_id": "onnxruntime", "text": "hello"},
    )
    assert r.status_code == 400
    assert "no adapter" in r.json()["detail"]
