"""JobManager: in-process background training-job orchestration.

Manages the lifecycle of training jobs backed by :class:`app.jobs.worker.JobWorker`
threads. Provides ``submit`` / ``start`` / ``pause`` / ``resume`` / ``cancel``, a bounded
event stream per job (a ``deque``), progress/metric tracking, checkpoint-recovery hooks,
and best-effort persistence of job *state* to a JSON file.

HONEST PERSISTENCE CAVEAT
-------------------------
Jobs run on threads inside *this* process. The JSON file records job metadata and the
last known state/metrics/checkpoint path, but **running jobs do not survive a process
restart**. After a restart, previously-``running`` jobs are reloaded as ``failed`` (or, if
a checkpoint exists, are eligible to be resumed from it via :meth:`recover`). True
durable execution requires a separate long-lived worker process/queue, which this
in-process manager deliberately does not implement.
"""
from __future__ import annotations

import json
import os
import threading
from collections import deque
from typing import Callable, Deque, Dict, List, Optional

from app.core.logging import get_logger
from app.jobs.state import (
    JobRecord,
    JobState,
    TERMINAL_STATES,
    assert_transition,
    can_transition,
)
from app.jobs.worker import JobWorker

logger = get_logger("jobs.manager")

# A factory that, given a JobRecord, returns (step_fn, total_steps, checkpoint_fn).
JobFactory = Callable[[JobRecord], "JobPlan"]


class JobPlan:
    """What the manager needs to run a job: a step function and a step budget."""

    def __init__(
        self,
        step_fn: Callable[[int], Dict[str, float]],
        total_steps: int,
        checkpoint_fn: Optional[Callable[[int], Optional[str]]] = None,
        checkpoint_every: int = 0,
    ) -> None:
        self.step_fn = step_fn
        self.total_steps = int(total_steps)
        self.checkpoint_fn = checkpoint_fn
        self.checkpoint_every = int(checkpoint_every)


class JobManager:
    """Thread-safe registry and controller for background jobs."""

    def __init__(
        self,
        factory: JobFactory,
        persist_path: Optional[str] = None,
        event_buffer: int = 200,
    ) -> None:
        self._factory = factory
        self._persist_path = persist_path
        self._event_buffer = int(event_buffer)
        self._lock = threading.RLock()
        self._records: Dict[str, JobRecord] = {}
        self._workers: Dict[str, JobWorker] = {}
        self._events: Dict[str, Deque[str]] = {}

    # --- persistence ---------------------------------------------------------
    def _persist(self) -> None:
        if not self._persist_path:
            return
        try:
            data = {jid: rec.to_dict() for jid, rec in self._records.items()}
            tmp = f"{self._persist_path}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, default=str)
            os.replace(tmp, self._persist_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("job_persist_failed", error=str(exc))

    def load_persisted(self) -> None:
        """Reload job records from disk, marking previously-running jobs as failed.

        See the module docstring: in-process running jobs cannot survive a restart, so we
        do not pretend they are still running.
        """
        if not self._persist_path or not os.path.exists(self._persist_path):
            return
        with self._lock:
            with open(self._persist_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            for jid, raw in data.items():
                rec = JobRecord.from_dict(raw)
                if rec.state in (JobState.RUNNING, JobState.PAUSED, JobState.QUEUED):
                    rec.error = (
                        "process restarted; in-process job did not survive. "
                        "Resume from checkpoint via recover() if available."
                    )
                    rec.state = JobState.FAILED
                self._records[jid] = rec
                self._events.setdefault(jid, deque(maxlen=self._event_buffer))

    # --- callbacks from workers ---------------------------------------------
    def _emit(self, job_id: str, message: str) -> None:
        with self._lock:
            self._events.setdefault(job_id, deque(maxlen=self._event_buffer)).append(message)

    def _on_progress(self, job_id: str):
        def cb(step: int, total: int, metrics: Dict[str, float]) -> None:
            with self._lock:
                rec = self._records.get(job_id)
                if rec is None:
                    return
                rec.current_step = step
                rec.total_steps = total
                rec.progress = (step / total) if total else 1.0
                if metrics:
                    rec.metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                self._persist()

        return cb

    def _on_state(self, job_id: str):
        def cb(state: JobState, error: Optional[str]) -> None:
            import time

            with self._lock:
                rec = self._records.get(job_id)
                if rec is None:
                    return
                rec.state = state
                rec.error = error
                rec.updated_at = time.time()
                self._persist()

        return cb

    # --- public API ----------------------------------------------------------
    def submit(self, job_id: str, kind: str, params: Optional[Dict] = None) -> JobRecord:
        """Register a new queued job. Does not start it."""
        with self._lock:
            if job_id in self._records:
                raise ValueError(f"job {job_id} already exists")
            rec = JobRecord(job_id=job_id, kind=kind, params=dict(params or {}))
            self._records[job_id] = rec
            self._events[job_id] = deque(maxlen=self._event_buffer)
            self._persist()
            return rec

    def start(self, job_id: str) -> None:
        """Build the job plan and start its worker thread."""
        with self._lock:
            rec = self._require(job_id)
            assert_transition(rec.state, JobState.RUNNING)
            plan = self._factory(rec)
            rec.total_steps = plan.total_steps
            worker = JobWorker(
                job_id=job_id,
                step_fn=plan.step_fn,
                total_steps=plan.total_steps,
                on_progress=self._on_progress(job_id),
                on_state=self._on_state(job_id),
                on_event=lambda msg: self._emit(job_id, msg),
                checkpoint_fn=plan.checkpoint_fn,
                checkpoint_every=plan.checkpoint_every,
            )
            self._workers[job_id] = worker
        worker.start()

    def pause(self, job_id: str) -> None:
        with self._lock:
            rec = self._require(job_id)
            if not can_transition(rec.state, JobState.PAUSED):
                raise ValueError(f"cannot pause job in state {rec.state.value}")
            worker = self._workers.get(job_id)
        if worker:
            worker.pause()

    def resume(self, job_id: str) -> None:
        with self._lock:
            rec = self._require(job_id)
            if rec.state != JobState.PAUSED:
                raise ValueError(f"cannot resume job in state {rec.state.value}")
            worker = self._workers.get(job_id)
        if worker:
            worker.resume()

    def cancel(self, job_id: str) -> None:
        """Request a stop; the worker transitions the job to ``stopped``."""
        with self._lock:
            rec = self._require(job_id)
            worker = self._workers.get(job_id)
            if rec.state in TERMINAL_STATES:
                return
        if worker:
            worker.stop()

    def wait(self, job_id: str, timeout: Optional[float] = None) -> None:
        """Block until the job's worker thread exits (test/CLI convenience)."""
        worker = self._workers.get(job_id)
        if worker:
            worker.join(timeout)

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            return self._records.get(job_id)

    def list(self) -> List[JobRecord]:
        with self._lock:
            return list(self._records.values())

    def events(self, job_id: str) -> List[str]:
        """Snapshot of the bounded event stream for a job."""
        with self._lock:
            return list(self._events.get(job_id, deque()))

    def recover(self, job_id: str) -> Optional[str]:
        """Checkpoint-recovery hook.

        Returns the checkpoint path recorded for the job, if any, so a caller can build a
        fresh plan that resumes from it. Actual resume-from-checkpoint wiring lives in the
        job factory; this manager only surfaces the path honestly.
        """
        with self._lock:
            rec = self._records.get(job_id)
            return rec.checkpoint_path if rec else None

    def _require(self, job_id: str) -> JobRecord:
        rec = self._records.get(job_id)
        if rec is None:
            raise KeyError(f"unknown job {job_id}")
        return rec
