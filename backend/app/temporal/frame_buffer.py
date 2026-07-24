"""Bounded ring buffer of recent frames with timestamps.

Keeps the most recent ``N`` frames in memory for temporal reasoning (motion, scene
change, future prediction). It does **not** persist to disk by default -- frames live in
RAM and are dropped when the buffer overflows or the process exits. This is deliberate:
retaining video frames has privacy implications, so persistence is opt-in elsewhere.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import numpy as np


@dataclass
class BufferedFrame:
    """A frame plus the wall-clock timestamp (seconds) it was captured/added."""

    frame: np.ndarray
    ts: float


class FrameBuffer:
    """Fixed-capacity ring buffer keyed by insertion, ordered oldest -> newest."""

    def __init__(self, capacity: int = 64) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = int(capacity)
        self._buf: Deque[BufferedFrame] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._buf)

    def add(self, frame: np.ndarray, ts: float) -> None:
        """Append a frame with timestamp ``ts`` (seconds). Oldest drops if full."""
        self._buf.append(BufferedFrame(frame=np.asarray(frame), ts=float(ts)))

    def clear(self) -> None:
        """Drop all buffered frames."""
        self._buf.clear()

    def all(self) -> List[BufferedFrame]:
        """All buffered frames, oldest first."""
        return list(self._buf)

    def latest(self) -> Optional[BufferedFrame]:
        """The most recent frame, or ``None`` if empty."""
        return self._buf[-1] if self._buf else None

    def last_seconds(self, sec: float) -> List[BufferedFrame]:
        """Frames whose timestamp is within ``sec`` seconds of the newest frame.

        Returns an empty list if the buffer is empty. The window is inclusive of the
        newest frame and measured relative to it (not to wall-clock now), so it works
        for replayed / offline streams too.
        """
        if not self._buf:
            return []
        newest = self._buf[-1].ts
        cutoff = newest - float(sec)
        return [f for f in self._buf if f.ts >= cutoff]

    def sample(self, n: int, method: str = "uniform") -> List[BufferedFrame]:
        """Sample ``n`` frames from the buffer.

        Args:
            n: Number of frames to return (clamped to buffer size).
            method: ``"uniform"`` -> evenly spaced across the buffer;
                ``"latest"`` -> the ``n`` most recent frames (oldest-first order);
                ``"oldest"`` -> the ``n`` oldest frames.
        """
        frames = list(self._buf)
        if not frames or n <= 0:
            return []
        n = min(n, len(frames))
        if method == "latest":
            return frames[-n:]
        if method == "oldest":
            return frames[:n]
        if method == "uniform":
            idx = np.linspace(0, len(frames) - 1, num=n).round().astype(int)
            idx = sorted(set(int(i) for i in idx))
            # Ensure exactly n where possible by padding from remaining indices.
            if len(idx) < n:
                remaining = [i for i in range(len(frames)) if i not in idx]
                idx = sorted(idx + remaining[: n - len(idx)])
            return [frames[i] for i in idx[:n]]
        raise ValueError("method must be 'uniform', 'latest' or 'oldest'")
