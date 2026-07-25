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
                backend, frame = state.detection.benchmark_target()
                ctx["backend"] = backend
                ctx["frame"] = frame
                ctx["live_frames_at_start"] = state.metrics.processed
                for _ in range(WARMUP_RUNS):
                    state.detection.benchmark_step(frame)

            latencies.append(state.detection.benchmark_step(ctx["frame"]))

            if index == runs - 1:
                _store(state, ctx, latencies)

            return {"latency_ms": latencies[-1]}

        return JobPlan(step_fn=step, total_steps=runs)

    return factory


def _store(state, ctx: dict, latencies: list[float]) -> None:
    """Write the measured row, recording the live traffic seen during the run."""
    concurrent = max(0, state.metrics.processed - ctx["live_frames_at_start"])
    notes = (
        f"auto-benchmark after model switch; {concurrent} concurrent live frames during the run"
    )
    result = ctx["backend"].result_from_latencies(latencies, notes=notes)
    meta = {
        "hardware": state.capabilities.cpu_model,
        "os": f"{state.capabilities.os} {state.capabilities.os_version}",
        "runtime_versions": {"onnxruntime": state.capabilities.runtimes.onnxruntime_providers},
        "config": state.detection.config.model_dump(mode="json") if state.detection.config else None,
    }
    state.db.insert_benchmark(result.model_dump(), meta)
    log.info(
        "auto_benchmark_stored",
        model_id=result.model_id,
        runs=result.runs,
        fps=round(result.fps, 2),
        concurrent_live_frames=concurrent,
    )
