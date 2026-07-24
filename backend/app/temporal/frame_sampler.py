"""Frame sampling strategies over a timestamped sequence.

Given a list of ``(frame, ts, detections)`` records (detections optional), pick a subset
of frames according to a strategy. These feed the more expensive downstream stages (VLM,
future predictor) so we only spend compute on informative frames.

Strategies:

* ``uniform`` -- evenly spaced indices.
* ``motion`` -- frames whose mean-abs-diff from the previous kept frame exceeds a
  threshold (frame-differencing).
* ``scene_change`` -- frames flagged by :mod:`app.temporal.scene_change`.
* ``detection_event`` -- frames whose detection set changes (count or class set).

All samplers are pure functions returning indices into the input list; helpers return
the selected records too.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from app.temporal.scene_change import scene_change_score

FrameRecord = tuple[np.ndarray, float, Sequence[Any] | None]


def uniform_indices(n_frames: int, n: int) -> list[int]:
    """Evenly spaced indices selecting ``n`` of ``n_frames`` frames."""
    if n_frames <= 0 or n <= 0:
        return []
    n = min(n, n_frames)
    idx = np.linspace(0, n_frames - 1, num=n).round().astype(int)
    return sorted(set(int(i) for i in idx))


def motion_indices(frames: Sequence[np.ndarray], threshold: float = 0.05) -> list[int]:
    """Indices where mean-abs-diff from the last kept frame exceeds ``threshold``.

    The first frame is always kept as a reference.
    """
    kept: list[int] = []
    ref = None
    for i, f in enumerate(frames):
        cur = np.asarray(f, dtype=np.float64)
        if cur.max() > 1.0:
            cur = cur / 255.0
        if ref is None:
            kept.append(i)
            ref = cur
            continue
        if cur.shape != ref.shape:
            kept.append(i)
            ref = cur
            continue
        if float(np.abs(cur - ref).mean()) >= threshold:
            kept.append(i)
            ref = cur
    return kept


def scene_change_indices(
    frames: Sequence[np.ndarray], threshold: float = 0.15, method: str = "mad"
) -> list[int]:
    """Indices flagged as scene changes relative to the last kept frame."""
    kept: list[int] = []
    ref_idx: int | None = None
    for i, f in enumerate(frames):
        if ref_idx is None:
            kept.append(i)
            ref_idx = i
            continue
        if scene_change_score(frames[ref_idx], f, method=method) > threshold:
            kept.append(i)
            ref_idx = i
    return kept


def _detection_signature(dets: Sequence[Any] | None) -> tuple[int, tuple]:
    """A cheap comparable signature: (count, sorted class ids)."""
    if not dets:
        return (0, tuple())
    class_ids = []
    for d in dets:
        cid = getattr(d, "classId", None)
        if cid is None and isinstance(d, dict):
            cid = d.get("classId")
        class_ids.append(cid)
    return (len(dets), tuple(sorted(str(c) for c in class_ids)))


def detection_event_indices(records: Sequence[FrameRecord]) -> list[int]:
    """Indices where the detection signature changes from the previous frame.

    The first frame is always kept. Records are ``(frame, ts, detections)``.
    """
    kept: list[int] = []
    prev_sig = None
    for i, rec in enumerate(records):
        dets = rec[2] if len(rec) > 2 else None
        sig = _detection_signature(dets)
        if prev_sig is None or sig != prev_sig:
            kept.append(i)
        prev_sig = sig
    return kept


def sample_frames(
    records: Sequence[FrameRecord],
    method: str = "uniform",
    n: int = 8,
    threshold: float = 0.05,
) -> list[FrameRecord]:
    """Dispatch to a strategy and return the selected records.

    Args:
        records: List of ``(frame, ts, detections)`` tuples.
        method: One of ``uniform``, ``motion``, ``scene_change``, ``detection_event``.
        n: Target count for ``uniform``.
        threshold: Threshold for ``motion`` / ``scene_change``.
    """
    frames = [r[0] for r in records]
    if method == "uniform":
        idx = uniform_indices(len(records), n)
    elif method == "motion":
        idx = motion_indices(frames, threshold=threshold)
    elif method == "scene_change":
        idx = scene_change_indices(frames, threshold=threshold)
    elif method == "detection_event":
        idx = detection_event_indices(records)
    else:
        raise ValueError(
            "method must be 'uniform', 'motion', 'scene_change' or 'detection_event'"
        )
    return [records[i] for i in idx]
