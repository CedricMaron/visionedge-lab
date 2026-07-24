"""Tests for the background jobs subsystem."""
from __future__ import annotations

import time

from app.jobs.manager import JobManager, JobPlan
from app.jobs.state import JobRecord, JobState, can_transition


def _wait_terminal(mgr: JobManager, job_id: str, timeout: float = 3.0) -> JobRecord:
    deadline = time.time() + timeout
    while time.time() < deadline:
        rec = mgr.get(job_id)
        if rec and rec.state in (JobState.COMPLETED, JobState.FAILED, JobState.STOPPED):
            return rec
        time.sleep(0.01)
    return mgr.get(job_id)


# --- state machine ---------------------------------------------------------
def test_state_transitions():
    assert can_transition(JobState.QUEUED, JobState.RUNNING)
    assert can_transition(JobState.RUNNING, JobState.PAUSED)
    assert can_transition(JobState.PAUSED, JobState.RUNNING)
    assert can_transition(JobState.RUNNING, JobState.COMPLETED)
    # Illegal transitions.
    assert not can_transition(JobState.COMPLETED, JobState.RUNNING)
    assert not can_transition(JobState.QUEUED, JobState.PAUSED)


def test_job_record_roundtrip():
    rec = JobRecord(job_id="j1", kind="image_jepa", params={"steps": 3})
    d = rec.to_dict()
    assert d["state"] == "queued"
    back = JobRecord.from_dict(d)
    assert back.job_id == "j1" and back.state == JobState.QUEUED


# --- manager end to end ----------------------------------------------------
def test_job_manager_start_to_complete():
    results = []

    def factory(rec: JobRecord) -> JobPlan:
        steps = int(rec.params.get("steps", 3))

        def step_fn(i: int):
            results.append(i)
            return {"loss": 1.0 / (i + 1)}

        return JobPlan(step_fn=step_fn, total_steps=steps)

    mgr = JobManager(factory)
    mgr.submit("j1", "trivial", {"steps": 3})
    mgr.start("j1")
    rec = _wait_terminal(mgr, "j1")

    assert rec.state == JobState.COMPLETED
    assert rec.current_step == 3
    assert rec.progress == 1.0
    assert results == [0, 1, 2]
    assert "loss" in rec.metrics
    assert any("completed" in e for e in mgr.events("j1"))


def test_job_manager_cancel():
    def factory(rec: JobRecord) -> JobPlan:
        def step_fn(i: int):
            time.sleep(0.02)  # give cancel time to land
            return {}

        return JobPlan(step_fn=step_fn, total_steps=1000)

    mgr = JobManager(factory)
    mgr.submit("j2", "trivial")
    mgr.start("j2")
    time.sleep(0.05)
    mgr.cancel("j2")
    rec = _wait_terminal(mgr, "j2")
    assert rec.state == JobState.STOPPED


def test_job_manager_failed_job():
    def factory(rec: JobRecord) -> JobPlan:
        def step_fn(i: int):
            raise ValueError("boom")

        return JobPlan(step_fn=step_fn, total_steps=5)

    mgr = JobManager(factory)
    mgr.submit("j3", "trivial")
    mgr.start("j3")
    rec = _wait_terminal(mgr, "j3")
    assert rec.state == JobState.FAILED
    assert rec.error and "boom" in rec.error


def test_job_manager_persist_and_reload(tmp_path):
    path = str(tmp_path / "jobs.json")

    def factory(rec: JobRecord) -> JobPlan:
        return JobPlan(step_fn=lambda i: {}, total_steps=2)

    mgr = JobManager(factory, persist_path=path)
    mgr.submit("j4", "trivial", {"steps": 2})
    mgr.start("j4")
    _wait_terminal(mgr, "j4")

    # New manager reloads persisted state; completed job stays completed.
    mgr2 = JobManager(factory, persist_path=path)
    mgr2.load_persisted()
    rec = mgr2.get("j4")
    assert rec is not None
    assert rec.state == JobState.COMPLETED


def test_job_manager_reload_marks_running_as_failed(tmp_path):
    """A persisted 'running' job cannot survive a restart; reloaded as failed (honest)."""
    path = str(tmp_path / "jobs.json")
    import json

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            {"jx": JobRecord(job_id="jx", kind="t", state=JobState.RUNNING).to_dict()}, fh
        )

    def factory(rec: JobRecord) -> JobPlan:
        return JobPlan(step_fn=lambda i: {}, total_steps=1)

    mgr = JobManager(factory, persist_path=path)
    mgr.load_persisted()
    rec = mgr.get("jx")
    assert rec.state == JobState.FAILED
    assert "restart" in (rec.error or "").lower()
