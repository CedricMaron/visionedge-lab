"""Tests for the VLM mock backend, structured output, grounding and manager switching."""
from __future__ import annotations

import numpy as np

from app.capabilities.scanner import scan_capabilities
from app.core.types import Detection
from app.models.registry import load_registry
from app.vlm.evaluation import detector_agreement, object_count_consistency
from app.vlm.manager import VLMManager
from app.vlm.mock_backend import MockVLMBackend
from app.vlm.prompting import detections_to_grounding
from app.vlm.structured_output import STRUCTURED_INSTRUCTION, parse_structured

IMG = np.zeros((100, 100, 3), np.uint8)


def _dets():
    return [
        Detection(x1=0, y1=0, x2=10, y2=10, confidence=0.9, classId=0, className="person",
                  inferenceBackend="t", modelName="m", modelVersion="v"),
        Detection(x1=5, y1=5, x2=20, y2=20, confidence=0.8, classId=0, className="person",
                  inferenceBackend="t", modelName="m", modelVersion="v"),
        Detection(x1=30, y1=30, x2=60, y2=60, confidence=0.7, classId=5, className="bus",
                  inferenceBackend="t", modelName="m", modelVersion="v"),
    ]


def test_mock_describe_uses_grounding():
    backend = MockVLMBackend(); backend.load()
    grounding = detections_to_grounding(_dets(), 100, 100)
    resp = backend.describe_image(IMG, "Describe.", grounding)
    assert "person" in resp.text and "bus" in resp.text
    assert resp.model_id == "mock-vlm"
    assert resp.total_latency_ms >= 0
    assert any("mock" in w.lower() for w in resp.warnings)


def test_mock_answer_counts_from_grounding():
    backend = MockVLMBackend(); backend.load()
    grounding = detections_to_grounding(_dets(), 100, 100)
    resp = backend.answer_question(IMG, "how many person are there?", grounding)
    assert "2" in resp.text


def test_mock_structured_output_valid():
    backend = MockVLMBackend(); backend.load()
    grounding = detections_to_grounding(_dets(), 100, 100)
    resp = backend.describe_image(IMG, f"Describe.\n{STRUCTURED_INSTRUCTION}", grounding)
    assert resp.structured_output is not None
    assert "person" in resp.structured_output["important_objects"]


def test_structured_parser_preserves_raw_on_failure():
    parsed, warnings, raw = parse_structured("this is not json at all")
    assert parsed is None
    assert raw == "this is not json at all"
    assert warnings


def test_structured_parser_accepts_valid_json():
    text = '{"summary": "a person", "important_objects": ["person"]}'
    parsed, warnings, _ = parse_structured(text)
    assert parsed is not None
    assert parsed.summary == "a person"


def test_object_count_consistency():
    result = object_count_consistency("There are 2 people.", _dets(), "person")
    assert result["detector_count"] == 2
    assert result["vlm_mentioned_count"] == 2
    assert result["agree"] is True


def test_detector_agreement_reports_missing():
    agree = detector_agreement("I see a bus.", _dets())
    assert "person" in agree["vlm_did_not_mention"]
    assert "bus" in agree["vlm_mentioned_detected"]


def test_vlm_manager_defaults_to_mock_and_switch_rollback():
    reg = load_registry()
    caps = scan_capabilities()
    mgr = VLMManager(reg, caps, _settings())
    mgr.initialize("mock-vlm")
    assert mgr.status()["model_id"] == "mock-vlm"
    # switching to a local model with transformers absent must roll back to mock
    result = mgr.switch("smolvlm-256m")
    if not result["ok"]:
        assert result["rolled_back"] is True
        assert mgr.status()["model_id"] == "mock-vlm"
    mgr.close()


class _settings:
    vlm_remote_url = ""
    vlm_remote_api_key = ""
    vlm_remote_timeout_s = 30.0
    vlm_remote_max_retries = 2
    allow_frame_transmission = False
