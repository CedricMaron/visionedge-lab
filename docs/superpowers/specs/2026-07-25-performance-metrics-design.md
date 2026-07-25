# Performance metrics when using models — design

**Date:** 2026-07-25
**Status:** approved, not yet implemented

## Goal

Make measured performance visible at the moment a model is used, and comparable
across models, without inventing numbers or hiding the conditions a number was
measured under.

## What already exists

Established by inspection before designing, so the work doesn't rebuild it:

- `RollingMetrics` (`app/monitoring/metrics.py`) — FPS, mean/p50/p95/p99 inference
  latency, end-to-end latency, processed/dropped/skipped counters over a 300-sample
  window. Surfaced at `/api/detection/status` and `/api/runtime-status`.
- `PerformancePage` — live charts and stat cards over those metrics.
- `benchmarks` SQLite table (`app/storage/db.py`) with model, backend, provider,
  precision, device, runs, fps, latency percentiles, RSS, hardware, OS, versions,
  checksum. Written by `POST /api/detection/benchmark`, read by `BenchmarksPage`.
- `BaseDetectionBackend.benchmark()` — measured, never hardcoded.
- `VLMResponsePanel` — already renders time-to-first-token, generation latency,
  total latency, prompt/generated tokens and memory.
- `app/jobs/` — `JobManager` + `JobWorker` with states, progress, persistence,
  pause/resume/cancel. Fully tested (7 tests) and **wired to nothing**.

## Scope

Four items. One is verification only.

### 1. Auto-benchmark on model switch

A successful switch submits a background benchmark job; the result lands in the
`benchmarks` table without the user asking.

### 2. Per-model comparison view

A table comparing stored benchmark rows across models, so "which model is faster
on this machine" is answerable from the UI.

### 3. Timing breakdown on single-image inference

`/api/infer` returns a real preprocess / inference / postprocess split.

### 4. VLM metrics — verify only

Already implemented. It was invisible only because `VLMResponsePanel` crashed on
every answer (fixed in commit `1b517d7`). Confirm it renders; build nothing.

## Architecture decisions

### Benchmark concurrency: per-inference locking

`DetectionManager.benchmark()` currently holds `self._lock` around the entire run
loop, so moving it to a background thread would still stall live inference for the
duration (~4 s for 30 runs on the reference machine).

**Decision:** acquire the lock per inference instead of around the loop, so live
frames interleave with benchmark frames.

Rejected alternatives:

- *Keep the whole-loop lock* — "background" would be a lie; streaming freezes after
  every switch.
- *Second backend instance for benchmarking* — no contention, but doubles model
  memory on a 6 GB card where the project already documents that detector + VLM +
  JEPA do not co-fit.

**Honesty requirement:** interleaving means a benchmark run concurrent with live
traffic is measuring a loaded machine. The job records the number of live frames
processed during the run and stores it in the report's `notes` field, and the
comparison view labels any row measured with concurrent traffic. A contaminated
number is reported as contaminated, never silently averaged in.

### Comparison source: controlled benchmark runs

The comparison view reads the `benchmarks` table (fixed run count, identical
synthetic mid-gray frame) rather than live rolling metrics. Live metrics reflect
real scenes but are confounded by session length, scene content and sample count,
so cross-model differences would not be attributable to the model.

### Aggregation: latest + median, never best

Per `(model_id, runtime, provider, input_size, precision)` group, the API returns
the latest run and the **median** FPS and p50 across that group's runs, plus the
run count `n`. Median rather than minimum-latency ("best") so the table cannot
cherry-pick a lucky run. `n` is displayed so a single-run row is visibly weaker
evidence than a ten-run row.

## Components

### Backend

**`app/core/state.py`** — `AppState` gains a `JobManager` instance. First real
consumer of `app/jobs/`.

**`app/benchmarking/auto.py`** (new) — builds the benchmark job:
- one job step = one timed inference, so `JobWorker` progress is real, not faked;
- on completion, writes a row through the existing `Database.insert_benchmark`;
- counts live frames processed during the run (delta of
  `RollingMetrics.processed`) and writes it into `notes`.

**`app/inference/manager.py`** — `benchmark()` takes the lock per inference.

**`app/api/detection.py`**:
- `POST /api/detection/switch` — on `ok=True` only, cancel any in-flight benchmark
  job (its model is gone) and submit a new one. A failed or rolled-back switch
  submits nothing.
- `POST /api/infer` — use `predict_timed` and return
  `{preprocess_ms, inference_ms, postprocess_ms, end_to_end_ms}`. Today it returns
  `inference_ms` and `end_to_end_ms` set to the same measurement.

**`app/api/meta.py`**:
- `GET /api/jobs` — job status from `JobManager.list()`.
- `GET /api/benchmarks/comparison` — grouped rows as described above.

### Frontend

**`src/types/index.ts`** — `InferTimings` gains `preprocess_ms` and
`postprocess_ms`; new `BenchmarkComparisonRow` and `JobRecord` types.

**`BenchmarksPage`** — a comparison table above the existing run list: one row per
model+runtime group, showing median FPS, median p50, `n`, device/provider, and a
marker when the run had concurrent live traffic.

**`ModelSelectorPage`** — after applying a switch, show the benchmark job's state
(queued / running with step progress / done / failed) polled from `/api/jobs`.

**Single-image inference** — display the four-part timing breakdown wherever
`/api/infer` results are shown.

## Data flow

```
switch ok ──> cancel stale job ──> submit benchmark job ──> JobWorker thread
                                                              │ per step:
                                                              │   lock, 1 inference, unlock
                                                              ▼
                                              insert_benchmark(row + notes)
                                                              │
       BenchmarksPage <── GET /api/benchmarks/comparison ─────┘
```

## Error handling

- Benchmark job failure never affects the switch — the switch already returned.
  The job moves to `failed` with its error, visible via `/api/jobs`.
- A switch during a running benchmark cancels it; a cancelled job writes no row.
- If no backend is loaded when the job runs, it fails cleanly rather than raising
  into the worker thread.
- `/api/benchmarks/comparison` on an empty table returns an empty list, and the UI
  renders an explicit empty state rather than a zero-filled table.

## Testing

Backend:
- switch succeeds → job submitted; switch rolls back → no job submitted;
- job completion writes exactly one `benchmarks` row with the expected model/runtime;
- second switch cancels the first job and writes no row for it;
- comparison endpoint groups correctly and returns the median, not the best, for a
  known set of rows;
- `/api/infer` timing parts sum to `end_to_end_ms` within tolerance;
- `benchmark()` no longer blocks a concurrent `predict()` for the whole run.

Frontend:
- comparison table renders rows, `n`, and the concurrent-traffic marker;
- empty state when no benchmarks exist;
- timing breakdown displays all four parts.

## Out of scope

- Live rolling metrics partitioned per model (rejected above as confounded).
- Any change to `VLMResponsePanel` beyond confirming it renders.
- Cross-machine benchmark comparison — results remain valid only within one host.
