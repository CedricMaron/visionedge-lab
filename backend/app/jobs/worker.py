"""Job worker: runs a single training job on a background thread.

A :class:`JobWorker` executes a *step function* repeatedly for a bounded number of
steps, honouring pause/resume/stop requests via threading events. It reports progress,
metrics and lifecycle events back through callbacks supplied by the :class:`JobManager`.

The step function is any callable ``step_fn(step_index) -> dict(metrics)``. For real
training this wraps ``LightweightIJepaTrainer.train_step``; tests pass a trivial function.
Running on a plain ``threading.Thread`` keeps the FastAPI event loop unblocked (the work
is CPU-bound and cooperatively yields between steps).
"""
from __future__ import annotations

import threading
import traceback
from collections.abc import Callable

from app.core.logging import get_logger
from app.jobs.state import JobState

logger = get_logger("jobs.worker")

# Callback signatures.
StepFn = Callable[[int], dict[str, float]]
ProgressCb = Callable[[int, int, dict[str, float]], None]
StateCb = Callable[[JobState, str | None], None]
EventCb = Callable[[str], None]


class JobWorker:
    """Runs one job's steps on a background thread with pause/stop control."""

    def __init__(
        self,
        job_id: str,
        step_fn: StepFn,
        total_steps: int,
        on_progress: ProgressCb,
        on_state: StateCb,
        on_event: EventCb,
        checkpoint_fn: Callable[[int], str | None] | None = None,
        checkpoint_every: int = 0,
    ) -> None:
        self.job_id = job_id
        self.step_fn = step_fn
        self.total_steps = int(total_steps)
        self.on_progress = on_progress
        self.on_state = on_state
        self.on_event = on_event
        self.checkpoint_fn = checkpoint_fn
        self.checkpoint_every = int(checkpoint_every)

        self._thread: threading.Thread | None = None
        self._pause = threading.Event()  # set => paused
        self._stop = threading.Event()  # set => stop requested
        self._resume = threading.Event()  # used to wake from pause
        self._resume.set()

    # --- control -------------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("worker already started")
        self._thread = threading.Thread(target=self._run, name=f"job-{self.job_id}", daemon=True)
        self._thread.start()

    def pause(self) -> None:
        self._pause.set()
        self._resume.clear()

    def resume(self) -> None:
        self._pause.clear()
        self._resume.set()

    def stop(self) -> None:
        self._stop.set()
        self._resume.set()  # wake if paused so it can observe the stop

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # --- main loop -----------------------------------------------------------
    def _run(self) -> None:
        try:
            self.on_state(JobState.RUNNING, None)
            for step in range(self.total_steps):
                if self._stop.is_set():
                    self.on_event(f"stop requested at step {step}")
                    self.on_state(JobState.STOPPED, None)
                    return

                # Cooperative pause: block until resumed or stopped.
                if self._pause.is_set():
                    self.on_state(JobState.PAUSED, None)
                    self.on_event(f"paused at step {step}")
                    self._resume.wait()
                    if self._stop.is_set():
                        self.on_event(f"stop requested while paused at step {step}")
                        self.on_state(JobState.STOPPED, None)
                        return
                    self.on_state(JobState.RUNNING, None)
                    self.on_event(f"resumed at step {step}")

                metrics = self.step_fn(step)
                self.on_progress(step + 1, self.total_steps, metrics or {})

                if (
                    self.checkpoint_fn is not None
                    and self.checkpoint_every > 0
                    and (step + 1) % self.checkpoint_every == 0
                ):
                    path = self.checkpoint_fn(step + 1)
                    if path:
                        self.on_event(f"checkpoint saved at step {step + 1}: {path}")

            self.on_state(JobState.COMPLETED, None)
            self.on_event("completed")
        except Exception as exc:  # noqa: BLE001 - report, do not crash the thread
            logger.error("job_worker_failed", job_id=self.job_id, error=str(exc))
            self.on_event(f"error: {exc}\n{traceback.format_exc()}")
            self.on_state(JobState.FAILED, str(exc))
