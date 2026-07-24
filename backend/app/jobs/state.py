"""Job state machine and serializable job records for background training jobs.

Defines the lifecycle states, the legal transitions between them, and a ``JobRecord``
dataclass that can be serialized to / from JSON for persistence.

States:
    queued    -> running | stopped
    running   -> paused | completed | failed | stopped
    paused    -> running | stopped
    completed -> (terminal)
    failed    -> (terminal)
    stopped   -> (terminal)

The transition table is the single source of truth; :func:`can_transition` and
:func:`assert_transition` enforce it.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"


TERMINAL_STATES = frozenset({JobState.STOPPED, JobState.COMPLETED, JobState.FAILED})

# Legal transitions.
_TRANSITIONS: Dict[JobState, frozenset] = {
    JobState.QUEUED: frozenset({JobState.RUNNING, JobState.STOPPED}),
    JobState.RUNNING: frozenset(
        {JobState.PAUSED, JobState.COMPLETED, JobState.FAILED, JobState.STOPPED}
    ),
    JobState.PAUSED: frozenset({JobState.RUNNING, JobState.STOPPED}),
    JobState.COMPLETED: frozenset(),
    JobState.FAILED: frozenset(),
    JobState.STOPPED: frozenset(),
}


def can_transition(src: JobState, dst: JobState) -> bool:
    """True if moving from ``src`` to ``dst`` is allowed."""
    return dst in _TRANSITIONS.get(src, frozenset())


def assert_transition(src: JobState, dst: JobState) -> None:
    """Raise ``ValueError`` if the transition is illegal."""
    if not can_transition(src, dst):
        raise ValueError(f"illegal job transition {src.value} -> {dst.value}")


@dataclass
class JobRecord:
    """Serializable description and live status of a background job."""

    job_id: str
    kind: str  # e.g. "image_jepa", "video_jepa"
    params: Dict[str, Any] = field(default_factory=dict)
    state: JobState = JobState.QUEUED
    progress: float = 0.0  # 0..1
    total_steps: int = 0
    current_step: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None
    checkpoint_path: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["state"] = self.state.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> JobRecord:
        data = dict(data)
        data["state"] = JobState(data.get("state", JobState.QUEUED.value))
        # Drop unexpected keys defensively.
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        data = {k: v for k, v in data.items() if k in known}
        return cls(**data)
