"""Comparison groups aggregate by median, never by best."""
from __future__ import annotations

from app.storage.db import Database


def _row(model_id: str, fps: float, p50: float, notes: str = "0 concurrent live frames"):
    return {
        "backend": "onnxruntime", "model_id": model_id, "input_size": 640,
        "precision": "fp32", "device": "cpu", "provider": "CPUExecutionProvider",
        "runs": 30, "fps": fps, "latency_mean_ms": p50, "latency_p50_ms": p50,
        "latency_p95_ms": p50 * 1.2, "latency_p99_ms": p50 * 1.3,
        "memory_rss_mb": 180.0, "notes": notes,
    }


def test_groups_by_config_and_uses_median_not_best(tmp_path):
    db = Database(tmp_path / "t.db")
    for fps, p50 in [(5.0, 200.0), (10.0, 100.0), (30.0, 33.0)]:
        db.insert_benchmark(_row("yolov8n-onnx", fps, p50))
    db.insert_benchmark(_row("yolov8s-onnx", 4.0, 250.0))

    groups = {g["model_id"]: g for g in db.benchmark_groups()}

    assert groups["yolov8n-onnx"]["n"] == 3
    assert groups["yolov8n-onnx"]["median_fps"] == 10.0      # not 30.0
    assert groups["yolov8n-onnx"]["median_p50_ms"] == 100.0  # not 33.0
    assert groups["yolov8s-onnx"]["n"] == 1


def test_flags_rows_measured_with_concurrent_traffic(tmp_path):
    db = Database(tmp_path / "t.db")
    db.insert_benchmark(_row("yolov8n-onnx", 10.0, 100.0))
    db.insert_benchmark(_row("yolov8n-onnx", 6.0, 160.0, notes="12 concurrent live frames during the run"))

    group = db.benchmark_groups()[0]
    assert group["any_concurrent_traffic"] is True


def test_idle_runs_are_not_flagged(tmp_path):
    db = Database(tmp_path / "t.db")
    db.insert_benchmark(_row("yolov8n-onnx", 10.0, 100.0,
                             notes="auto-benchmark after model switch; 0 concurrent live frames during the run"))

    assert db.benchmark_groups()[0]["any_concurrent_traffic"] is False


def test_empty_database_returns_no_groups(tmp_path):
    assert Database(tmp_path / "t.db").benchmark_groups() == []
