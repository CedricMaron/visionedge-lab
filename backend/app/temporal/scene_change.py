"""Scene-change scoring between two frames.

Two cheap detectors are provided:

* ``mad`` -- mean absolute difference of pixel intensities, normalized to [0, 1].
* ``histogram`` -- 1 minus the histogram-intersection of grayscale intensity
  histograms, also in [0, 1].

Both return a score where higher means "more changed". A configurable threshold turns
the score into a boolean scene-change decision. Pure numpy (no OpenCV required).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _to_gray(frame: np.ndarray) -> np.ndarray:
    """Convert an (H, W) or (H, W, C) uint8/float frame to float grayscale in [0,1]."""
    arr = np.asarray(frame, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr


def mad_score(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference of two frames, normalized to [0, 1]."""
    ga, gb = _to_gray(a), _to_gray(b)
    if ga.shape != gb.shape:
        raise ValueError("frames must have the same shape")
    return float(np.abs(ga - gb).mean())


def histogram_score(a: np.ndarray, b: np.ndarray, bins: int = 32) -> float:
    """1 - histogram intersection of grayscale intensity histograms, in [0, 1]."""
    ga, gb = _to_gray(a), _to_gray(b)
    ha, _ = np.histogram(ga, bins=bins, range=(0.0, 1.0), density=False)
    hb, _ = np.histogram(gb, bins=bins, range=(0.0, 1.0), density=False)
    ha = ha / max(ha.sum(), 1)
    hb = hb / max(hb.sum(), 1)
    intersection = np.minimum(ha, hb).sum()
    return float(1.0 - intersection)


def scene_change_score(a: np.ndarray, b: np.ndarray, method: str = "mad", bins: int = 32) -> float:
    """Scene-change score between frames ``a`` and ``b`` (higher = more change)."""
    if method == "mad":
        return mad_score(a, b)
    if method == "histogram":
        return histogram_score(a, b, bins=bins)
    raise ValueError("method must be 'mad' or 'histogram'")


@dataclass
class SceneChangeDetector:
    """Stateful detector applying a threshold to :func:`scene_change_score`."""

    method: str = "mad"
    threshold: float = 0.15
    bins: int = 32

    def score(self, a: np.ndarray, b: np.ndarray) -> float:
        return scene_change_score(a, b, method=self.method, bins=self.bins)

    def is_scene_change(self, a: np.ndarray, b: np.ndarray) -> bool:
        """True if the score between ``a`` and ``b`` exceeds ``threshold``."""
        return self.score(a, b) > self.threshold
