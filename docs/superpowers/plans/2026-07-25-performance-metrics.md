# Performance Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show measured performance when a model is used — auto-benchmark on switch, compare models against each other, and break down single-image inference timing — without inventing numbers or hiding measurement conditions.

**Architecture:** A successful model switch submits a background benchmark job through the existing (currently unused) `app/jobs/` subsystem. The benchmark loop takes the detection manager's lock per inference rather than around the whole run, so live streaming keeps working; the number of live frames processed during the run is recorded in the stored row. A new comparison endpoint groups stored benchmark rows and aggregates by median.

**Tech Stack:** FastAPI, pydantic v2, SQLite (stdlib `sqlite3`), pytest, React 18 + TypeScript (strict), vitest, Tailwind.

## Global Constraints

- Every metric must be measured. Never hardcode, estimate, or interpolate a performance number.
- A benchmark measured while live traffic was running must say so in its stored row; it is never silently averaged in with idle runs.
- Aggregation across runs uses the **median**, never the best/minimum.
- Python: backend venv at `backend/.venv`. Run tests with `cd backend && .venv/bin/python -m pytest`.
- Lint gates that must stay green: `cd backend && .venv/bin/ruff check app tests`, and in `frontend/`: `npm run lint` (`--max-warnings 0`), `npx tsc --noEmit`, `npx vitest run`.
- Frontend types mirror backend field names EXACTLY (`frontend/src/types/index.ts` header rule).
- Do not add dependencies. Everything needed is already installed.

---

### Task 1: Per-inference locking in the benchmark loop

`DetectionManager.benchmark()` holds `self._lock` around the whole run, so a background benchmark would still freeze live inference. Split the backend's benchmark into reusable parts and re-implement the manager's loop to lock per inference.

**Files:**
- Modify: `backend/app/inference/base.py:66-105` (`BaseDetectionBackend.benchmark`)
- Modify: `backend/app/inference/manager.py:161-165` (`DetectionManager.benchmark`)
- Test: `backend/tests/test_manager_switch.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `BaseDetectionBackend.benchmark_frame() -> np.ndarray`
  - `BaseDetectionBackend.result_from_latencies(latencies: list[float], notes: str = "") -> BenchmarkResult`
  - `DetectionManager.benchmark_step(frame: np.ndarray) -> float` — one locked, timed inference, returns milliseconds
  - `DetectionManager.benchmark(runs: int = 30, notes: str = "") -> BenchmarkResult` — unchanged signature plus `notes`

- [ ] **Step 1: Write the failing test**

Add to `backend/tests/test_manager_switch.py`:

```python
def test_benchmark_does_not_hold_the_lock_for_the_whole_run(manager):
    """A concurrent predict must interleave with benchmark runs, not wait for all of them."""
    import threading
    import time

    blocked_ms = []

    def predict_once():
        t0 = time.perf_counter()
        manager.predict(np.zeros((640, 640, 3), np.uint8))
        blocked_ms.append((time.perf_counter() - t0) * 1000.0)

    single = manager.benchmark(runs=1).latency_mean_ms

    t = threading.Thread(target=predict_once)
    bench = threading.Thread(target=lambda: manager.benchmark(runs=20))
    bench.start()
    time.sleep(0.01)
    t.start()
    t.join()
    bench.join()

    # Waiting behind at most a couple of in-flight inferences is fine; waiting
    # behind all 20 is the bug this guards against.
    assert blocked_ms[0] < single * 5


def test_benchmark_records_notes(manager):
    result = manager.benchmark(runs=3, notes="measured with 7 concurrent live frames")
    assert result.notes == "measured with 7 concurrent live frames"
    assert result.runs == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_manager_switch.py -k "lock_for_the_whole_run or records_notes" -v`
Expected: FAIL — `test_benchmark_records_notes` with `TypeError: benchmark() got an unexpected keyword argument 'notes'`, and the lock test either fails the assertion or errors for the same reason.

- [ ] **Step 3: Split the backend benchmark into reusable parts**

In `backend/app/inference/base.py`, replace the body of `benchmark` with:

```python
    def benchmark_frame(self) -> np.ndarray:
        """Deterministic mid-gray frame, so results reflect the runtime path, not scene content."""
        return np.full((self.input_size, self.input_size, 3), 128, dtype=np.uint8)

    def result_from_latencies(self, latencies: list[float], notes: str = "") -> BenchmarkResult:
        """Build a BenchmarkResult from measured latencies. Values are never synthesized."""
        arr = np.array(latencies)
        return BenchmarkResult(
            backend=self.backend_name,
            model_id=self.model_id,
            input_size=self.input_size,
            precision=self.precision,
            device=self.device,
            runs=len(latencies),
            fps=float(1000.0 / arr.mean()) if arr.mean() > 0 else 0.0,
            latency_mean_ms=float(arr.mean()),
            latency_p50_ms=float(np.percentile(arr, 50)),
            latency_p95_ms=float(np.percentile(arr, 95)),
            latency_p99_ms=float(np.percentile(arr, 99)),
            memory_rss_mb=self._rss_mb(),
            provider=getattr(self, "provider", None),
            notes=notes,
        )

    def benchmark(self, runs: int = 30) -> BenchmarkResult:
        """Run ``runs`` inferences on a synthetic frame and measure latency.

        Values are always measured here, never hardcoded.
        """
        if self._health not in (HealthState.READY, HealthState.DEGRADED):
            raise RuntimeError("backend not ready for benchmarking")

        frame = self.benchmark_frame()
        for _ in range(3):  # small warm set excluded from timing
            self.predict(frame, 0.25, 0.45, None)

        lat: list[float] = []
        for _ in range(max(1, runs)):
            t0 = time.perf_counter()
            self.predict(frame, 0.25, 0.45, None)
            lat.append((time.perf_counter() - t0) * 1000.0)

        return self.result_from_latencies(lat)
```

- [ ] **Step 4: Re-implement the manager loop with per-inference locking**

In `backend/app/inference/manager.py`, replace the `benchmark` method:

```python
    def benchmark_step(self, frame) -> float:
        """One timed inference under the lock. Returns milliseconds.

        The lock is taken per inference so a long benchmark does not freeze live
        streaming — frames interleave with benchmark runs.
        """
        with self._lock:
            if self._backend is None:
                raise ModelLoadError("no backend loaded")
            t0 = time.perf_counter()
            self._backend.predict(frame, 0.25, 0.45, None)
            return (time.perf_counter() - t0) * 1000.0

    def benchmark(self, runs: int = 30, notes: str = "") -> BenchmarkResult:
        with self._lock:
            if self._backend is None:
                raise ModelLoadError("no backend loaded")
            backend = self._backend
            frame = backend.benchmark_frame()

        for _ in range(3):  # warm set, excluded from timing
            self.benchmark_step(frame)

        latencies = [self.benchmark_step(frame) for _ in range(max(1, runs))]
        return backend.result_from_latencies(latencies, notes=notes)
```

Add the imports at the top of `manager.py`:

```python
import time

from app.core.types import BenchmarkResult, Detection, HealthState
```

(`Detection` and `HealthState` are already imported from `app.core.types`; add `BenchmarkResult` to that same line and add `import time` next to `import threading`.)

- [ ] **Step 5: Run the tests**

Run: `cd backend && .venv/bin/python -m pytest tests/ -v -k "benchmark or switch"`
Expected: PASS, including the pre-existing `test_benchmark_returns_measured_values`.

- [ ] **Step 6: Run the full suite and lint**

Run: `cd backend && .venv/bin/python -m pytest -q && .venv/bin/ruff check app tests`
Expected: all pass, `All checks passed!`

- [ ] **Step 7: Commit**

```bash
git add backend/app/inference/base.py backend/app/inference/manager.py backend/tests/test_manager_switch.py
git commit -m "refactor(inference): lock per inference during benchmark runs

Holding the manager lock around the whole benchmark loop meant any background
benchmark would still freeze live streaming for its full duration. The loop now
takes the lock per inference so frames interleave, and BenchmarkResult carries a
notes field describing the measurement conditions."
```

---

### Task 2: Benchmark job plan + JobManager wired into AppState

Make `app/jobs/` a real consumer: a benchmark job whose steps are individual inferences, which writes one row on completion and records how much live traffic ran alongside it.

**Files:**
- Create: `backend/app/benchmarking/auto.py`
- Modify: `backend/app/core/state.py:24-34` (`AppState`), `backend/app/core/state.py:36-59` (`build_state`)
- Test: `backend/tests/test_auto_benchmark.py`

**Interfaces:**
- Consumes: `DetectionManager.benchmark_step`, `BaseDetectionBackend.benchmark_frame`, `BaseDetectionBackend.result_from_latencies` (Task 1).
- Produces:
  - `app.benchmarking.auto.BENCHMARK_JOB_KIND: str = "benchmark"`
  - `app.benchmarking.auto.make_job_factory(state) -> Callable[[JobRecord], JobPlan]`
  - `AppState.jobs: JobManager | None`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_auto_benchmark.py`:

```python
"""The auto-benchmark job: real steps, one stored row, honest notes."""
from __future__ import annotations

import numpy as np
import pytest

from app.benchmarking.auto import BENCHMARK_JOB_KIND, make_job_factory
from app.capabilities.scanner import scan_capabilities
from app.core.config import REPO_ROOT, get_settings
from app.core.state import AppState
from app.inference.config import InferenceConfig
from app.inference.manager import DetectionManager
from app.jobs.manager import JobManager
from app.jobs.state import JobState
from app.models.registry import load_registry, refresh_deployment_status
from app.monitoring.metrics import RollingMetrics
from app.storage.db import Database

MODEL = REPO_ROOT / "models" / "yolov8n.onnx"
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="yolov8n.onnx not installed")


@pytest.fixture
def state(tmp_path):
    registry = refresh_deployment_status(load_registry())
    caps = scan_capabilities()
    detection = DetectionManager(registry, caps)
    detection.initialize(InferenceConfig(model_id="yolov8n-onnx", runtime="onnxruntime-cpu"))
    st = AppState(
        settings=get_settings(),
        capabilities=caps,
        registry=registry,
        db=Database(tmp_path / "test.db"),
        detection=detection,
        metrics=RollingMetrics(),
    )
    st.jobs = JobManager(factory=make_job_factory(st))
    yield st
    detection.close()


def test_job_runs_one_step_per_inference_and_stores_one_row(state):
    state.jobs.submit("b1", BENCHMARK_JOB_KIND, {"runs": 4})
    state.jobs.start("b1")
    state.jobs.wait("b1", timeout=60)

    rec = state.jobs.get("b1")
    assert rec.state is JobState.COMPLETED
    assert rec.total_steps == 4
    assert rec.current_step == 4

    rows = state.db.list_benchmarks()
    assert len(rows) == 1
    assert rows[0]["model_id"] == "yolov8n-onnx"
    assert rows[0]["runs"] == 4
    assert rows[0]["fps"] > 0


def test_notes_report_concurrent_live_frames(state):
    state.metrics.record(10.0, 10.0)  # a live frame before the run
    state.jobs.submit("b2", BENCHMARK_JOB_KIND, {"runs": 2})
    state.jobs.start("b2")
    state.jobs.wait("b2", timeout=60)

    notes = state.db.list_benchmarks()[0]["notes"]
    assert "concurrent live frames" in notes
    assert "0 concurrent live frames" in notes  # none arrived DURING the run


def test_cancelled_job_stores_no_row(state):
    state.jobs.submit("b3", BENCHMARK_JOB_KIND, {"runs": 500})
    state.jobs.start("b3")
    state.jobs.cancel("b3")
    state.jobs.wait("b3", timeout=60)

    assert state.db.list_benchmarks() == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_auto_benchmark.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.benchmarking.auto'`

- [ ] **Step 3: Write the job factory**

Create `backend/app/benchmarking/auto.py`:

```python
"""Background benchmark job: one job step per measured inference.

The job exists so a model switch can leave a measured benchmark behind without
blocking the switch. Because the detection manager locks per inference, live
frames interleave with benchmark runs — which means a benchmark taken during
streaming is measuring a loaded machine. That is recorded in the stored row's
notes rather than hidden, and never silently compared against an idle run.
"""
from __future__ import annotations

from collections.abc import Callable

from app.core.logging import get_logger
from app.jobs.manager import JobPlan
from app.jobs.state import JobRecord

log = get_logger("benchmarking.auto")

BENCHMARK_JOB_KIND = "benchmark"
DEFAULT_RUNS = 30
WARMUP_RUNS = 3


def make_job_factory(state) -> Callable[[JobRecord], JobPlan]:
    """Build the JobManager factory that turns a benchmark JobRecord into a plan."""

    def factory(record: JobRecord) -> JobPlan:
        if record.kind != BENCHMARK_JOB_KIND:
            raise ValueError(f"unsupported job kind '{record.kind}'")

        runs = max(1, int(record.params.get("runs", DEFAULT_RUNS)))
        latencies: list[float] = []
        ctx: dict = {}

        def step(index: int) -> dict[str, float]:
            if index == 0:
                with state.detection._lock:  # noqa: SLF001 — snapshot backend identity once
                    backend = state.detection._backend  # noqa: SLF001
                    if backend is None:
                        raise RuntimeError("no detection backend loaded")
                    ctx["backend"] = backend
                    ctx["frame"] = backend.benchmark_frame()
                    ctx["live_frames_at_start"] = state.metrics.processed
                for _ in range(WARMUP_RUNS):
                    state.detection.benchmark_step(ctx["frame"])

            latencies.append(state.detection.benchmark_step(ctx["frame"]))

            if index == runs - 1:
                _store(state, ctx, latencies)

            return {"latency_ms": latencies[-1]}

        return JobPlan(step_fn=step, total_steps=runs)

    return factory


def _store(state, ctx: dict, latencies: list[float]) -> None:
    """Write the measured row, recording the live traffic seen during the run."""
    concurrent = max(0, state.metrics.processed - ctx["live_frames_at_start"])
    notes = f"auto-benchmark after model switch; {concurrent} concurrent live frames during the run"
    result = ctx["backend"].result_from_latencies(latencies, notes=notes)
    meta = {
        "hardware": state.capabilities.cpu_model,
        "os": f"{state.capabilities.os} {state.capabilities.os_version}",
        "runtime_versions": {"onnxruntime": state.capabilities.runtimes.onnxruntime_providers},
        "config": state.detection.config.model_dump(mode="json") if state.detection.config else None,
    }
    state.db.insert_benchmark(result.model_dump(), meta)
    log.info("auto_benchmark_stored", model_id=result.model_id, runs=result.runs,
             fps=round(result.fps, 2), concurrent_live_frames=concurrent)
```

- [ ] **Step 4: Add the JobManager to AppState**

In `backend/app/core/state.py`, add to the imports:

```python
from app.jobs.manager import JobManager
```

Add the field to `AppState` (after `metrics`):

```python
    jobs: JobManager | None = None
```

The factory needs the finished `AppState`, so the `JobManager` is attached after
the state object is constructed — not passed into the constructor. At the end of
`build_state()`, replace the existing `return AppState(...)` with:

```python
    state = AppState(
        settings=settings,
        capabilities=caps,
        registry=registry,
        db=db,
        detection=detection,
        metrics=metrics,
        vlm=vlm,
        startup_warnings=warnings,
    )
    # Attached after construction: the job factory closes over the finished state.
    state.jobs = JobManager(factory=make_job_factory(state))
    return state
```

Add both imports at the top of `state.py` with the others:

```python
from app.benchmarking.auto import make_job_factory
from app.jobs.manager import JobManager
```

If the existing `build_state()` returns the `AppState(...)` expression directly,
bind it to `state` first as shown above. Match the keyword arguments to whatever
the current constructor call passes — do not drop any.

- [ ] **Step 5: Run the tests**

Run: `cd backend && .venv/bin/python -m pytest tests/test_auto_benchmark.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Run the full suite and lint**

Run: `cd backend && .venv/bin/python -m pytest -q && .venv/bin/ruff check app tests`
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add backend/app/benchmarking/auto.py backend/app/core/state.py backend/tests/test_auto_benchmark.py
git commit -m "feat(benchmarking): background benchmark job, one step per inference

First real consumer of app/jobs/, which until now was tested but wired to
nothing. Each job step is one measured inference, so progress is real. The
stored row records how many live frames were processed during the run, because
per-inference locking means a benchmark can overlap real traffic."
```

---

### Task 3: Submit the job on a successful switch, expose job status

**Files:**
- Modify: `backend/app/api/detection.py:60-80` (the switch handler)
- Modify: `backend/app/api/meta.py` (add `GET /api/jobs`)
- Test: `backend/tests/test_auto_benchmark.py`

**Interfaces:**
- Consumes: `BENCHMARK_JOB_KIND`, `AppState.jobs` (Task 2).
- Produces: `GET /api/jobs` returning `{"jobs": [JobRecord.to_dict(), ...]}`.

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_auto_benchmark.py`:

```python
def test_switch_submits_a_benchmark_job_and_rollback_does_not():
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app()) as c:
        ok = c.post("/api/detection/switch", json={
            "model_id": "yolov8n-onnx", "runtime": "onnxruntime-cpu",
            "execution_location": "pc_local",
        })
        assert ok.json()["ok"] is True
        jobs = c.get("/api/jobs").json()["jobs"]
        assert len([j for j in jobs if j["kind"] == "benchmark"]) == 1

        bad = c.post("/api/detection/switch", json={
            "model_id": "does-not-exist", "runtime": "onnxruntime-cpu",
            "execution_location": "pc_local",
        })
        assert bad.json()["ok"] is False
        jobs_after = c.get("/api/jobs").json()["jobs"]
        # The failed switch added nothing.
        assert len([j for j in jobs_after if j["kind"] == "benchmark"]) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_auto_benchmark.py -k switch_submits -v`
Expected: FAIL — `/api/jobs` returns 404.

- [ ] **Step 3: Add the jobs endpoint**

In `backend/app/api/meta.py`, next to the existing `/api/benchmarks` handler:

```python
@router.get("/api/jobs")
async def jobs(request: Request):
    """Background job status. Empty when the jobs subsystem is not wired."""
    state = get_state(request)
    if state.jobs is None:
        return {"jobs": []}
    return {"jobs": [rec.to_dict() for rec in state.jobs.list()]}
```

- [ ] **Step 4: Submit the job on a successful switch**

In `backend/app/api/detection.py`, add near the top:

```python
import uuid

from app.benchmarking.auto import BENCHMARK_JOB_KIND
from app.jobs.state import TERMINAL_STATES
```

In the switch handler, after the `SwitchResult` is produced and only when `result.ok` is true, add:

```python
    if result.ok and state.jobs is not None:
        # A new model invalidates any benchmark still running for the old one.
        for rec in state.jobs.list():
            if rec.kind == BENCHMARK_JOB_KIND and rec.state not in TERMINAL_STATES:
                try:
                    state.jobs.cancel(rec.job_id)
                except ValueError:  # already terminal — nothing to cancel
                    pass
        job_id = f"benchmark-{uuid.uuid4().hex[:8]}"
        state.jobs.submit(job_id, BENCHMARK_JOB_KIND, {"runs": 30})
        state.jobs.start(job_id)
```

- [ ] **Step 5: Run the tests**

Run: `cd backend && .venv/bin/python -m pytest tests/test_auto_benchmark.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Run the full suite and lint**

Run: `cd backend && .venv/bin/python -m pytest -q && .venv/bin/ruff check app tests`
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add backend/app/api/detection.py backend/app/api/meta.py backend/tests/test_auto_benchmark.py
git commit -m "feat(api): auto-benchmark on successful switch, expose GET /api/jobs

A rolled-back switch submits nothing. A new switch cancels any in-flight
benchmark, since the model it was measuring is already unloaded."
```

---

### Task 4: Per-model comparison endpoint

**Files:**
- Modify: `backend/app/storage/db.py` (add `benchmark_groups`)
- Modify: `backend/app/api/meta.py` (add `GET /api/benchmarks/comparison`)
- Test: `backend/tests/test_benchmark_comparison.py`

**Interfaces:**
- Consumes: rows written by Task 2.
- Produces: `Database.benchmark_groups() -> list[dict]` and `GET /api/benchmarks/comparison` returning `{"groups": [...]}` where each group has: `model_id`, `backend`, `provider`, `device`, `input_size`, `precision`, `n`, `median_fps`, `median_p50_ms`, `latest_ts`, `latest_fps`, `any_concurrent_traffic`.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_benchmark_comparison.py`:

```python
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


def test_empty_database_returns_no_groups(tmp_path):
    assert Database(tmp_path / "t.db").benchmark_groups() == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_benchmark_comparison.py -v`
Expected: FAIL — `AttributeError: 'Database' object has no attribute 'benchmark_groups'`

- [ ] **Step 3: Implement the grouping**

Add to `backend/app/storage/db.py` (after `list_benchmarks`), and add `import re`
and `import statistics` at the top.

First the helper that reads the concurrency count back out of the notes written by
`app/benchmarking/auto.py`:

```python
_CONCURRENT_RE = re.compile(r"(\d+) concurrent live frames")


def _measured_under_load(notes: str | None) -> bool:
    """True if this row's notes report a non-zero concurrent live frame count."""
    if not notes:
        return False
    m = _CONCURRENT_RE.search(notes)
    return bool(m) and int(m.group(1)) > 0
```

Then the grouping method:

```python
    def benchmark_groups(self) -> list[dict]:
        """Group benchmark rows by configuration and aggregate by MEDIAN.

        Median rather than best, so the comparison cannot be flattered by one
        lucky run. ``n`` is returned so a single-run group is visibly weaker
        evidence than a ten-run group.
        """
        with self._conn() as c:
            rows = [dict(r) for r in c.execute("SELECT * FROM benchmarks ORDER BY ts DESC").fetchall()]

        buckets: dict[tuple, list[dict]] = {}
        for r in rows:
            key = (r["model_id"], r["backend"], r["provider"], r["device"],
                   r["input_size"], r["precision"])
            buckets.setdefault(key, []).append(r)

        groups = []
        for key, items in buckets.items():
            fps = [i["fps"] for i in items if i["fps"] is not None]
            p50 = [i["latency_p50_ms"] for i in items if i["latency_p50_ms"] is not None]
            latest = items[0]  # rows arrive newest-first
            groups.append({
                "model_id": key[0], "backend": key[1], "provider": key[2],
                "device": key[3], "input_size": key[4], "precision": key[5],
                "n": len(items),
                "median_fps": round(statistics.median(fps), 2) if fps else None,
                "median_p50_ms": round(statistics.median(p50), 2) if p50 else None,
                "latest_ts": latest["ts"],
                "latest_fps": latest["fps"],
                "any_concurrent_traffic": any(_measured_under_load(i.get("notes")) for i in items),
            })
        groups.sort(key=lambda g: (g["median_fps"] or 0), reverse=True)
        return groups
```

- [ ] **Step 4: Add the endpoint**

In `backend/app/api/meta.py`:

```python
@router.get("/api/benchmarks/comparison")
async def benchmarks_comparison(request: Request):
    """Per-configuration comparison of stored benchmark runs, aggregated by median."""
    state = get_state(request)
    return {
        "groups": state.db.benchmark_groups(),
        "note": ("Median across runs, never the best. Only comparable within this host; "
                 "rows flagged any_concurrent_traffic were measured while live inference ran."),
    }
```

- [ ] **Step 5: Run the tests**

Run: `cd backend && .venv/bin/python -m pytest tests/test_benchmark_comparison.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Run the full suite and lint**

Run: `cd backend && .venv/bin/python -m pytest -q && .venv/bin/ruff check app tests`
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add backend/app/storage/db.py backend/app/api/meta.py backend/tests/test_benchmark_comparison.py
git commit -m "feat(api): per-model benchmark comparison aggregated by median

Groups stored runs by model/backend/provider/device/size/precision and reports
the median with the run count, so a single lucky run cannot flatter a model."
```

---

### Task 5: Real timing breakdown on /api/infer

**Files:**
- Modify: `backend/app/api/detection.py:28-56` (`infer`)
- Test: `backend/tests/test_api.py`

**Interfaces:**
- Produces: `/api/infer` `timings` object with `preprocess_ms`, `inference_ms`, `postprocess_ms`, `end_to_end_ms`.

- [ ] **Step 1: Write the failing test**

Add to `backend/tests/test_api.py`:

```python
def test_infer_returns_a_real_timing_breakdown(client):
    if not SAMPLE.exists():
        pytest.skip("sample image not installed")
    with open(SAMPLE, "rb") as f:
        r = client.post("/api/infer", files={"file": ("bus.jpg", f.read(), "image/jpeg")})
    t = r.json()["timings"]
    for key in ("preprocess_ms", "inference_ms", "postprocess_ms", "end_to_end_ms"):
        assert key in t and t[key] >= 0.0
    parts = t["preprocess_ms"] + t["inference_ms"] + t["postprocess_ms"]
    assert parts <= t["end_to_end_ms"] + 1.0        # parts fit inside the whole
    assert t["inference_ms"] < t["end_to_end_ms"]   # not the same number twice
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_api.py -k timing_breakdown -v`
Expected: FAIL — `assert 'preprocess_ms' in {...}`

- [ ] **Step 3: Use predict_timed**

In `backend/app/api/detection.py`, replace the timing block of `infer`:

```python
    t0 = time.perf_counter()
    dets, timings = await run_in_threadpool(
        state.detection.predict_timed, img, confidence, iou, allowed
    )
    e2e = (time.perf_counter() - t0) * 1000.0

    backend = state.detection.config.runtime if state.detection.config else "none"
    state.metrics.record(timings["inference_ms"], e2e)
    FRAMES_TOTAL.labels(backend=backend).inc()
    INFERENCE_LATENCY.labels(backend=backend).observe(timings["inference_ms"])
    return {
        "detections": [d.model_dump() for d in dets],
        "timings": {
            "preprocess_ms": round(timings["preprocess_ms"], 2),
            "inference_ms": round(timings["inference_ms"], 2),
            "postprocess_ms": round(timings["postprocess_ms"], 2),
            "end_to_end_ms": round(e2e, 2),
        },
        "backend": backend,
        "count": len(dets),
    }
```

- [ ] **Step 4: Add the manager passthrough**

`DetectionManager` has no `predict_timed`. Add it to `backend/app/inference/manager.py` next to `predict`:

```python
    def predict_timed(self, image, conf=None, iou=None, allowed_class_ids=None):
        """Predict with a per-stage timing breakdown, applying the same config defaults."""
        with self._lock:
            if self._backend is None or not self._accepting:
                return [], {"preprocess_ms": 0.0, "inference_ms": 0.0,
                            "postprocess_ms": 0.0, "end_to_end_ms": 0.0}
            cfg = self._config
            c = conf if conf is not None else (cfg.confidence if cfg else 0.25)
            i = iou if iou is not None else (cfg.iou if cfg else 0.45)
            allowed = allowed_class_ids
            if allowed is None and cfg and cfg.allowed_class_ids is not None:
                allowed = set(cfg.allowed_class_ids)
            return self._backend.predict_timed(image, c, i, allowed)
```

- [ ] **Step 5: Run the tests**

Run: `cd backend && .venv/bin/python -m pytest tests/test_api.py -v`
Expected: PASS

- [ ] **Step 6: Run the full suite and lint**

Run: `cd backend && .venv/bin/python -m pytest -q && .venv/bin/ruff check app tests`
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add backend/app/api/detection.py backend/app/inference/manager.py backend/tests/test_api.py
git commit -m "fix(api): return a real timing breakdown from /api/infer

It previously reported inference_ms and end_to_end_ms as the same measurement.
The endpoint now uses predict_timed and returns the genuine preprocess /
inference / postprocess split."
```

---

### Task 6: Frontend — comparison table on the Benchmarks page

**Files:**
- Modify: `frontend/src/types/index.ts`
- Modify: `frontend/src/services/api.ts`
- Modify: `frontend/src/pages/BenchmarksPage.tsx`
- Test: `frontend/src/pages/BenchmarksPage.test.tsx`

**Interfaces:**
- Consumes: `GET /api/benchmarks/comparison` (Task 4).
- Produces: `BenchmarkComparisonRow`, `BenchmarkComparisonResponse`, `api.benchmarkComparison()`, and the exported `ComparisonTable` component.

- [ ] **Step 1: Write the failing test**

Create `frontend/src/pages/BenchmarksPage.test.tsx`:

```tsx
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ComparisonTable } from './BenchmarksPage';
import type { BenchmarkComparisonRow } from '@/types';

const ROWS: BenchmarkComparisonRow[] = [
  {
    model_id: 'yolov8n-onnx', backend: 'onnxruntime', provider: 'CPUExecutionProvider',
    device: 'cpu', input_size: 640, precision: 'fp32', n: 3,
    median_fps: 10, median_p50_ms: 100, latest_ts: 1, latest_fps: 12,
    any_concurrent_traffic: false,
  },
  {
    model_id: 'yolov8s-onnx', backend: 'onnxruntime', provider: 'CPUExecutionProvider',
    device: 'cpu', input_size: 640, precision: 'fp32', n: 1,
    median_fps: 4, median_p50_ms: 250, latest_ts: 2, latest_fps: 4,
    any_concurrent_traffic: true,
  },
];

describe('ComparisonTable', () => {
  it('shows the run count so single-run rows are visibly weaker evidence', () => {
    render(<ComparisonTable rows={ROWS} />);
    expect(screen.getByText('yolov8n-onnx')).toBeDefined();
    expect(screen.getByText('n=3')).toBeDefined();
    expect(screen.getByText('n=1')).toBeDefined();
  });

  it('marks groups measured while live inference was running', () => {
    render(<ComparisonTable rows={ROWS} />);
    expect(screen.getAllByTitle(/measured while live inference/i)).toHaveLength(1);
  });

  it('renders an explicit empty state rather than a zero-filled table', () => {
    render(<ComparisonTable rows={[]} />);
    expect(screen.getByText(/no benchmarks recorded yet/i)).toBeDefined();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run src/pages/BenchmarksPage.test.tsx`
Expected: FAIL — `ComparisonTable` is not exported from `./BenchmarksPage`.

- [ ] **Step 3: Add the types**

In `frontend/src/types/index.ts`, after `BenchmarksResponse`:

```ts
// One configuration's aggregated benchmark history. Median, never best — see
// backend/app/storage/db.py::benchmark_groups.
export interface BenchmarkComparisonRow {
  model_id: string;
  backend: string;
  provider: string | null;
  device: string;
  input_size: number;
  precision: string;
  n: number;
  median_fps: number | null;
  median_p50_ms: number | null;
  latest_ts: number;
  latest_fps: number | null;
  any_concurrent_traffic: boolean;
}

export interface BenchmarkComparisonResponse {
  groups: BenchmarkComparisonRow[];
  note: string;
}
```

- [ ] **Step 4: Add the API call**

In `frontend/src/services/api.ts`, add `BenchmarkComparisonResponse` to the type import and add:

```ts
  benchmarkComparison: (signal?: AbortSignal) =>
    http.get<BenchmarkComparisonResponse>('/api/benchmarks/comparison', undefined, signal),
```

- [ ] **Step 5: Add the component and render it**

In `frontend/src/pages/BenchmarksPage.tsx`, add the import of `BenchmarkComparisonRow` to the existing type import, then add the exported component:

```tsx
export function ComparisonTable({ rows }: { rows: BenchmarkComparisonRow[] }) {
  return (
    <div className="card overflow-x-auto">
      <table className="w-full min-w-[680px] text-sm">
        <thead>
          <tr className="border-b border-surface-700 text-left text-xs uppercase tracking-wide text-slate-500">
            <th className="px-3 py-2.5">Model</th>
            <th className="px-3 py-2.5">Device / provider</th>
            <th className="px-3 py-2.5">Input</th>
            <th className="px-3 py-2.5 text-right">Median FPS</th>
            <th className="px-3 py-2.5 text-right">Median p50</th>
            <th className="px-3 py-2.5 text-right">Runs</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((g) => (
            <tr key={`${g.model_id}-${g.provider}-${g.input_size}-${g.precision}`}
                className="border-b border-surface-800 last:border-0">
              <td className="px-3 py-2.5 font-mono text-xs text-slate-200">
                {g.model_id}
                {g.any_concurrent_traffic && (
                  <span
                    className="ml-2 pill bg-warn/15 text-warn"
                    title="Measured while live inference was running — the machine was loaded"
                  >
                    loaded
                  </span>
                )}
              </td>
              <td className="px-3 py-2.5 text-slate-400">{g.device} / {g.provider ?? '—'}</td>
              <td className="px-3 py-2.5 font-mono text-slate-300">{g.input_size}</td>
              <td className="px-3 py-2.5 text-right font-mono text-accent">
                {g.median_fps?.toFixed(1) ?? '—'}
              </td>
              <td className="px-3 py-2.5 text-right font-mono text-slate-300">
                {g.median_p50_ms !== null ? formatMs(g.median_p50_ms) : '—'}
              </td>
              <td className="px-3 py-2.5 text-right font-mono text-slate-400">n={g.n}</td>
            </tr>
          ))}
          {rows.length === 0 && (
            <tr>
              <td colSpan={6} className="px-3 py-6 text-center text-slate-500">
                No benchmarks recorded yet. Switch a model to record one automatically.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
```

In the page body, load and render it above the existing run table:

```tsx
  const comparison = useAsync<BenchmarkComparisonResponse>((s) => api.benchmarkComparison(s), []);
```

```tsx
      <h2 className="label mb-2 mt-6">Model comparison (median across runs)</h2>
      <ComparisonTable rows={comparison.data?.groups ?? []} />
```

- [ ] **Step 6: Run the tests, typecheck and lint**

Run: `cd frontend && npx vitest run && npx tsc --noEmit && npm run lint`
Expected: all pass, 0 eslint problems

- [ ] **Step 7: Commit**

```bash
git add frontend/src/types/index.ts frontend/src/services/api.ts frontend/src/pages/BenchmarksPage.tsx frontend/src/pages/BenchmarksPage.test.tsx
git commit -m "feat(frontend): per-model benchmark comparison table

Shows the median across runs with the run count, and marks any group whose runs
were measured while live inference was loading the machine."
```

---

### Task 7: Frontend — show the inference timing breakdown

**Files:**
- Modify: `frontend/src/types/index.ts` (`InferTimings`)
- Modify: `frontend/src/inference/types.ts` (`InferenceOutput`)
- Modify: `frontend/src/inference/serverBackend.ts:52-56`
- Test: `frontend/src/inference/serverBackend.test.ts`

**Interfaces:**
- Consumes: `/api/infer` breakdown (Task 5).
- Produces: `InferenceOutput.timings?: InferTimings`.

- [ ] **Step 1: Write the failing test**

Create `frontend/src/inference/serverBackend.test.ts`:

```ts
import { describe, it, expect, afterEach, vi } from 'vitest';
import { ServerInferenceBackend } from './serverBackend';

const BODY = {
  detections: [],
  timings: { preprocess_ms: 3, inference_ms: 140, postprocess_ms: 2, end_to_end_ms: 146 },
  backend: 'onnxruntime-cpu',
  count: 0,
};

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('ServerInferenceBackend', () => {
  it('passes the full timing breakdown through, not just inference_ms', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
      new Response(JSON.stringify(BODY), {
        status: 200, headers: { 'content-type': 'application/json' },
      }),
    ));

    const out = await new ServerInferenceBackend().infer({
      frame: new Blob(['x']), confidence: 0.25, iou: 0.45, allowedClassIds: [],
    });

    expect(out.inferenceMs).toBe(140);
    expect(out.timings?.preprocess_ms).toBe(3);
    expect(out.timings?.postprocess_ms).toBe(2);
    expect(out.timings?.end_to_end_ms).toBe(146);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run src/inference/serverBackend.test.ts`
Expected: FAIL — `out.timings` is undefined.

- [ ] **Step 3: Extend the types**

In `frontend/src/types/index.ts`, replace `InferTimings`:

```ts
export interface InferTimings {
  preprocess_ms: number;
  inference_ms: number;
  postprocess_ms: number;
  end_to_end_ms: number;
}
```

In `frontend/src/inference/types.ts`, add to `InferenceOutput`:

```ts
  timings?: InferTimings;
```

and import the type: `import type { Detection, InferTimings } from '@/types';` (keep whatever is already imported there and add `InferTimings`).

- [ ] **Step 4: Pass the timings through**

In `frontend/src/inference/serverBackend.ts`, change the return of `infer`:

```ts
    return {
      detections: res.detections,
      inferenceMs: res.timings.inference_ms,
      backend: res.backend,
      timings: res.timings,
    };
```

- [ ] **Step 5: Run the tests, typecheck and lint**

Run: `cd frontend && npx vitest run && npx tsc --noEmit && npm run lint`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add frontend/src/types/index.ts frontend/src/inference/types.ts frontend/src/inference/serverBackend.ts frontend/src/inference/serverBackend.test.ts
git commit -m "feat(frontend): carry the full inference timing breakdown

InferTimings now mirrors the backend's real preprocess/inference/postprocess
split instead of two copies of one measurement."
```

---

### Task 8: Frontend — benchmark job status, and verify the VLM panel

**Files:**
- Modify: `frontend/src/types/index.ts` (`JobRecord`)
- Modify: `frontend/src/services/api.ts` (`api.jobs`)
- Modify: `frontend/src/pages/ModelSelectorPage.tsx`
- Test: `frontend/src/pages/ModelSelectorPage.test.tsx`

**Interfaces:**
- Consumes: `GET /api/jobs` (Task 3).
- Produces: exported `BenchmarkJobStatus` component.

- [ ] **Step 1: Write the failing test**

Create `frontend/src/pages/ModelSelectorPage.test.tsx`:

```tsx
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { BenchmarkJobStatus } from './ModelSelectorPage';
import type { JobRecord } from '@/types';

function job(state: string, current = 0, total = 30): JobRecord {
  return {
    job_id: 'benchmark-abc', kind: 'benchmark', params: { runs: total },
    state, progress: total ? current / total : 0, total_steps: total,
    current_step: current, metrics: {}, error: null, checkpoint_path: null,
    created_at: 0, updated_at: 0,
  };
}

describe('BenchmarkJobStatus', () => {
  it('reports progress while a benchmark is running', () => {
    render(<BenchmarkJobStatus job={job('running', 12)} />);
    expect(screen.getByText(/benchmarking/i)).toBeDefined();
    expect(screen.getByText(/12\s*\/\s*30/)).toBeDefined();
  });

  it('renders nothing when there is no job', () => {
    const { container } = render(<BenchmarkJobStatus job={null} />);
    expect(container.textContent).toBe('');
  });

  it('surfaces a failed benchmark instead of hiding it', () => {
    const failed = { ...job('failed', 4), error: 'no detection backend loaded' };
    render(<BenchmarkJobStatus job={failed} />);
    expect(screen.getByText(/no detection backend loaded/i)).toBeDefined();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run src/pages/ModelSelectorPage.test.tsx`
Expected: FAIL — `BenchmarkJobStatus` is not exported.

- [ ] **Step 3: Add the type and API call**

In `frontend/src/types/index.ts`:

```ts
// Mirrors backend/app/jobs/state.py::JobRecord.
export interface JobRecord {
  job_id: string;
  kind: string;
  params: Record<string, unknown>;
  state: string;
  progress: number;
  total_steps: number;
  current_step: number;
  metrics: Record<string, number>;
  error: string | null;
  checkpoint_path: string | null;
  created_at: number;
  updated_at: number;
}

export interface JobsResponse {
  jobs: JobRecord[];
}
```

In `frontend/src/services/api.ts` (add `JobsResponse` to the type import):

```ts
  jobs: (signal?: AbortSignal) => http.get<JobsResponse>('/api/jobs', undefined, signal),
```

- [ ] **Step 4: Add the component and poll for it**

In `frontend/src/pages/ModelSelectorPage.tsx`, add the exported component:

```tsx
export function BenchmarkJobStatus({ job }: { job: JobRecord | null }) {
  if (!job) return null;
  if (job.state === 'failed') {
    return (
      <p className="mt-2 text-sm text-bad">
        Benchmark failed: {job.error ?? 'unknown error'}
      </p>
    );
  }
  if (job.state === 'running' || job.state === 'queued') {
    return (
      <p className="mt-2 text-sm text-slate-400">
        Benchmarking this model in the background… {job.current_step} / {job.total_steps} runs
      </p>
    );
  }
  if (job.state === 'completed') {
    return <p className="mt-2 text-sm text-good">Benchmark recorded — see the Benchmarks page.</p>;
  }
  return null;
}
```

Add the polling hook to the same file, above the page component:

```tsx
const TERMINAL_JOB_STATES = ['completed', 'failed', 'stopped'];

function useBenchmarkJob(switchStatus: string): JobRecord | null {
  const [job, setJob] = useState<JobRecord | null>(null);

  useEffect(() => {
    if (switchStatus !== 'success') return;
    let cancelled = false;

    const tick = async () => {
      try {
        const { jobs } = await api.jobs();
        const latest = jobs
          .filter((j) => j.kind === 'benchmark')
          .sort((a, b) => b.created_at - a.created_at)[0] ?? null;
        if (cancelled) return;
        setJob(latest);
        if (latest && TERMINAL_JOB_STATES.includes(latest.state)) {
          clearInterval(timer);
        }
      } catch {
        // Job status is advisory — a failed poll must not break the page.
      }
    };

    const timer = setInterval(tick, 1000);
    void tick();
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [switchStatus]);

  return job;
}
```

In the page component, call it and render the status next to the existing switch
message:

```tsx
  const benchmarkJob = useBenchmarkJob(status);
```

```tsx
        <BenchmarkJobStatus job={benchmarkJob} />
```

Add `JobRecord` to the type import at the top of the file, and confirm `useEffect`
and `useState` are already imported from `react` (they are).

- [ ] **Step 5: Run the tests, typecheck and lint**

Run: `cd frontend && npx vitest run && npx tsc --noEmit && npm run lint`
Expected: all pass

- [ ] **Step 6: Verify the VLM metrics render (build nothing)**

Start the backend (`make backend`) and the frontend (`make frontend`), open the Multimodal Assistant, attach an image and ask a question. Confirm the panel shows time-to-first-token, generation, total latency, prompt/generated tokens and memory.

If it renders: this satisfies the "VLM performance metrics" scope item — no code required, it was only ever hidden by the crash fixed in `1b517d7`. If it does not render, stop and report what is missing rather than building a parallel implementation.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/types/index.ts frontend/src/services/api.ts frontend/src/pages/ModelSelectorPage.tsx frontend/src/pages/ModelSelectorPage.test.tsx
git commit -m "feat(frontend): show background benchmark job progress after a switch

Surfaces queued/running progress and failures from GET /api/jobs, so an
auto-benchmark is visible rather than silently happening."
```

---

## Final verification

- [ ] `cd backend && .venv/bin/python -m pytest -q` — all pass
- [ ] `cd backend && .venv/bin/ruff check app tests` — clean
- [ ] `cd frontend && npx vitest run` — all pass
- [ ] `cd frontend && npx tsc --noEmit` — clean
- [ ] `cd frontend && npm run lint` — 0 problems
- [ ] `cd frontend && npm run build` — succeeds
- [ ] Manual: switch a model in the UI, confirm the benchmark job appears, completes, and its row shows up in the comparison table with the correct `n`.
