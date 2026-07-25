"""Tests for the temporal subsystem (pure-numpy, fast)."""
from __future__ import annotations

import numpy as np
import pytest

from app.temporal.anomaly import AnomalyScorer
from app.temporal.frame_buffer import FrameBuffer
from app.temporal.frame_sampler import sample_frames
from app.temporal.scene_change import SceneChangeDetector, scene_change_score


def _frame(val: int, size: int = 8) -> np.ndarray:
    return np.full((size, size, 3), val, dtype=np.uint8)


# --- frame buffer ----------------------------------------------------------
def test_frame_buffer_bounds():
    buf = FrameBuffer(capacity=3)
    for i in range(5):
        buf.add(_frame(i), ts=float(i))
    assert len(buf) == 3  # bounded
    # Oldest two dropped; timestamps 2,3,4 remain.
    tss = [f.ts for f in buf.all()]
    assert tss == [2.0, 3.0, 4.0]
    assert buf.latest().ts == 4.0


def test_frame_buffer_last_seconds():
    buf = FrameBuffer(capacity=10)
    for i in range(10):
        buf.add(_frame(i), ts=float(i))  # ts 0..9
    recent = buf.last_seconds(2.0)  # newest is 9 -> cutoff 7 -> ts 7,8,9
    assert [f.ts for f in recent] == [7.0, 8.0, 9.0]
    assert buf.last_seconds(100.0)  # everything
    assert len(buf.last_seconds(100.0)) == 10


def test_frame_buffer_sample_uniform():
    buf = FrameBuffer(capacity=10)
    for i in range(10):
        buf.add(_frame(i), ts=float(i))
    s = buf.sample(3, method="uniform")
    assert len(s) == 3
    assert buf.sample(2, method="latest")[-1].ts == 9.0


# --- scene change ----------------------------------------------------------
def test_scene_change_score_and_detector():
    a = _frame(0)
    b = _frame(255)
    assert scene_change_score(a, a, method="mad") == 0.0
    assert scene_change_score(a, b, method="mad") > 0.9
    assert scene_change_score(a, b, method="histogram") > 0.9

    det = SceneChangeDetector(method="mad", threshold=0.5)
    assert det.is_scene_change(a, b)
    assert not det.is_scene_change(a, a)


# --- frame sampler ---------------------------------------------------------
def test_sampler_uniform_and_motion():
    records = [(_frame(i * 30), float(i), None) for i in range(6)]
    uni = sample_frames(records, method="uniform", n=3)
    assert len(uni) == 3

    # Motion: big steps every frame -> keep all; identical frames -> keep only first.
    same = [(_frame(0), float(i), None) for i in range(5)]
    motion = sample_frames(same, method="motion", threshold=0.05)
    assert len(motion) == 1


def test_sampler_detection_event():
    class D:
        def __init__(self, cid):
            self.classId = cid

    recs = [
        (_frame(0), 0.0, []),
        (_frame(0), 1.0, []),
        (_frame(0), 2.0, [D(1)]),  # change: count/class differs
        (_frame(0), 3.0, [D(1)]),
    ]
    kept = sample_frames(recs, method="detection_event")
    # frame 0 (first), and frame 2 (signature change) kept.
    tss = [r[1] for r in kept]
    assert 0.0 in tss and 2.0 in tss and 3.0 not in tss


# --- anomaly ---------------------------------------------------------------
def test_anomaly_calibrate_then_score():
    rng = np.random.default_rng(0)
    normal = rng.normal(loc=1.0, scale=0.1, size=200)
    scorer = AnomalyScorer(window=200, z_threshold=3.0)
    scorer.calibrate(normal)

    # An error near the mean is not anomalous.
    near = scorer.score(1.0)
    assert abs(near["z"]) < 1.0
    assert not near["is_anomaly"]
    assert 0.0 <= near["anomaly"] <= 1.0

    # A large error is anomalous with a high 0..1 score.
    big = scorer.score(5.0)
    assert big["z"] > 3.0
    assert big["is_anomaly"]
    assert big["anomaly"] > 0.5


def test_anomaly_requires_calibration():
    scorer = AnomalyScorer()
    with pytest.raises(RuntimeError):
        scorer.score(1.0)
