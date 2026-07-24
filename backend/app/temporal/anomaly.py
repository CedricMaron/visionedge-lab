"""Anomaly scoring from prediction errors.

A future-prediction model (temporal JEPA) yields a per-frame prediction error. This
module turns a stream of such errors into a normalized anomaly score by calibrating on a
window of "normal" errors and z-scoring new errors against that baseline.

IMPORTANT HONESTY NOTE: a high prediction error means the model was *surprised*, not that
something dangerous happened. Surprise correlates with novelty, occlusion, lighting
changes and model blind spots as much as with genuine events. Treat the score as an
attention cue for a downstream check (e.g. a VLM call), never as a safety verdict.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

import numpy as np


@dataclass
class AnomalyScorer:
    """Rolling anomaly scorer over prediction errors.

    Calibrate on a window of errors to fix ``mean``/``std``, then :meth:`score` new
    errors into a z-score and a squashed 0..1 anomaly score. Optionally keep a rolling
    window that updates the baseline as new (assumed-normal) errors arrive.
    """

    window: int = 128
    z_threshold: float = 3.0
    _errors: Deque[float] = field(default_factory=lambda: deque(maxlen=128), init=False)
    mean: float = field(default=0.0, init=False)
    std: float = field(default=1.0, init=False)
    calibrated: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._errors = deque(maxlen=self.window)

    def calibrate(self, errors) -> None:
        """Fix the baseline mean/std from a batch of assumed-normal errors."""
        arr = np.asarray(list(errors), dtype=np.float64)
        if arr.size == 0:
            raise ValueError("cannot calibrate on an empty error window")
        self.mean = float(arr.mean())
        self.std = float(arr.std())
        if self.std < 1e-9:
            self.std = 1e-9
        self._errors = deque(arr.tolist(), maxlen=self.window)
        self.calibrated = True

    def z_score(self, error: float) -> float:
        """Standardized deviation of ``error`` from the calibrated baseline."""
        if not self.calibrated:
            raise RuntimeError("AnomalyScorer must be calibrated before scoring")
        return (float(error) - self.mean) / self.std

    def score(self, error: float, update: bool = False) -> Dict[str, float]:
        """Score one error.

        Args:
            error: New prediction error.
            update: If True, add this error to the rolling window and recompute the
                baseline (online adaptation). Leave False for a fixed baseline.

        Returns:
            Dict with ``z`` (z-score), ``anomaly`` (0..1 via logistic squashing of the
            positive z-score) and ``is_anomaly`` (z > ``z_threshold``).
        """
        z = self.z_score(error)
        # Squash only positive deviations; being *more predictable* than normal is not
        # anomalous. Logistic centred on the threshold.
        anomaly = 1.0 / (1.0 + np.exp(-(z - self.z_threshold)))
        result = {
            "z": float(z),
            "anomaly": float(anomaly),
            "is_anomaly": bool(z > self.z_threshold),
        }
        if update:
            self._errors.append(float(error))
            arr = np.asarray(self._errors, dtype=np.float64)
            self.mean = float(arr.mean())
            self.std = max(float(arr.std()), 1e-9)
        return result

    def rolling_stats(self) -> Optional[Dict[str, float]]:
        """Current baseline statistics, or ``None`` if never calibrated."""
        if not self.calibrated:
            return None
        return {"mean": self.mean, "std": self.std, "count": float(len(self._errors))}
