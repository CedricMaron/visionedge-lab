"""InferenceLab benchmarking API.

Everything here reads through the same engine, registry and probes as the CLI. No
endpoint computes a metric of its own — routes select, filter and serialize, so the
web UI and the command line cannot disagree about what a number means.

Long benchmarks run through the existing background job manager rather than blocking
a request, and a run is cancellable while it executes.
"""
from __future__ import annotations

import threading
import uuid

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.adapters.base import LoadConfig
from app.benchmark.engine import BenchmarkEngine, EngineOptions
from app.benchmark.export import iterations_to_csv, summary_to_csv, to_json, to_markdown
from app.benchmark.scenarios import load_all
from app.core.config import get_settings
from app.core.errors import InferenceLabError
from app.core.logging import get_logger
from app.core.state import get_state
from app.runtimes.registry import capability_matrix, probe_all
from app.schemas.enums import BenchmarkMode, DeviceKind, ExecutionLocation, Precision
from app.schemas.environment import RuntimeReference
from app.storage.runs import RunStore

log = get_logger("api.lab")
router = APIRouter(prefix="/api/lab", tags=["inferencelab"])

#: Cancellation flags for in-flight runs, keyed by run token.
_CANCEL_FLAGS: dict[str, threading.Event] = {}


def _store() -> RunStore:
    return RunStore(get_settings().db_path)


# --- discovery ---------------------------------------------------------------


@router.get("/runtimes")
async def list_runtimes():
    """Every known runtime with its probe result. Never claims the unverified."""
    return {
        "runtimes": [
            {
                "runtime_id": c.runtime_id,
                "available": c.available,
                "unavailable_reason": c.unavailable_reason,
                "version": c.version,
                "execution_providers": c.execution_providers,
                "devices": [d.value for d in c.devices],
                "precisions_by_device": {
                    d.value: [p.value for p in ps] for d, ps in c.precisions_by_device.items()
                },
                "supports_device_synchronization": c.supports_device_synchronization,
                "supports_profiling": c.supports_profiling,
                "notes": c.notes,
            }
            for c in probe_all()
        ]
    }


@router.get("/capability-matrix")
async def get_capability_matrix():
    """Runtime x device x precision, with a reason on every unsupported cell."""
    return {"cells": capability_matrix()}


@router.get("/models")
async def list_models(request: Request, include_test_adapters: bool = False):
    """Adapter-architecture models with licences and disk-derived status.

    Test adapters are excluded unless explicitly requested, so a fabricating
    adapter cannot appear in a normal listing.
    """
    registry = get_state(request).registry
    entries = registry.models if include_test_adapters else registry.production_models()
    return {"models": [m.model_dump() for m in entries]}


@router.get("/scenarios")
async def list_scenarios():
    return {
        "scenarios": [
            {**spec.model_dump(mode="json"), "has_sufficient_samples": spec.has_sufficient_samples}
            for spec in load_all().values()
        ]
    }


@router.get("/system")
async def system_info(request: Request):
    """Low-level environment overview for the System page."""
    from app.instrumentation.environment import collect_hardware, collect_software

    hardware = await run_in_threadpool(collect_hardware)
    software = await run_in_threadpool(collect_software)
    return {
        "hardware": hardware.model_dump(),
        "software": software.model_dump(),
        "runtimes": (await list_runtimes())["runtimes"],
    }


# --- running -----------------------------------------------------------------


class RunRequest(BaseModel):
    scenario_id: str
    model_id: str
    runtime_id: str = "onnxruntime"
    device: DeviceKind = DeviceKind.CPU
    precision: Precision = Precision.FP32
    mode: BenchmarkMode | None = None
    measured_iterations: int | None = Field(default=None, ge=1, le=1000)
    warmup_iterations: int | None = Field(default=None, ge=0, le=100)
    seed: int | None = None
    label: str | None = Field(default=None, max_length=120)
    enable_sampler: bool = True


@router.post("/runs")
async def create_run(request: Request, body: RunRequest):
    """Execute a benchmark synchronously and persist it.

    Bounded by the scenario's own timeout. The iteration ceiling above keeps a
    single HTTP request from pinning the box indefinitely; longer sweeps belong on
    the CLI, which the response's reproduction command provides.
    """
    from app.cli import _MODELS, _build_adapter  # shared resolution, no duplicate logic

    scenarios = load_all()
    if body.scenario_id not in scenarios:
        raise HTTPException(404, f"unknown scenario '{body.scenario_id}'")
    if body.model_id not in _MODELS:
        raise HTTPException(404, f"unknown model '{body.model_id}'")

    scenario = scenarios[body.scenario_id]
    updates: dict = {}
    if body.measured_iterations is not None:
        updates["measured_iterations"] = body.measured_iterations
    if body.warmup_iterations is not None:
        updates["warmup_iterations"] = body.warmup_iterations
    if body.mode is not None:
        updates["mode"] = body.mode
    if body.seed is not None:
        updates["random_seed"] = body.seed
    if updates:
        scenario = scenario.model_copy(update=updates)

    reproduce = (
        f"inference-lab benchmark run --scenario {scenario.id} --model {body.model_id} "
        f"--runtime {body.runtime_id} --device {body.device.value} "
        f"--precision {body.precision.value} --iterations {scenario.measured_iterations}"
    )

    token = uuid.uuid4().hex[:12]
    cancel = threading.Event()
    _CANCEL_FLAGS[token] = cancel

    try:
        adapter = _build_adapter(body.model_id, body.runtime_id, scenario.input_size)
    except InferenceLabError as exc:
        _CANCEL_FLAGS.pop(token, None)
        raise HTTPException(400, exc.user_message) from exc

    engine = BenchmarkEngine(
        EngineOptions(
            enable_sampler=body.enable_sampler,
            reproduction_command=reproduce,
            label=body.label,
        )
    )
    try:
        run = await run_in_threadpool(
            engine.run,
            adapter,
            scenario,
            LoadConfig(
                runtime_id=body.runtime_id, device=body.device, precision=body.precision,
                input_size=scenario.input_size,
            ),
            RuntimeReference(
                runtime_id=body.runtime_id, device=body.device, precision=body.precision
            ),
            ExecutionLocation.IN_PROCESS,
            None,
            cancel,
        )
    finally:
        engine.close()
        adapter.unload()
        _CANCEL_FLAGS.pop(token, None)

    await run_in_threadpool(_store().save, run)
    return {"run_token": token, "run": run.model_dump(mode="json")}


@router.post("/runs/{token}/cancel")
async def cancel_run(token: str):
    flag = _CANCEL_FLAGS.get(token)
    if flag is None:
        raise HTTPException(404, "no in-flight run with that token")
    flag.set()
    return {"cancelled": True, "run_token": token}


# --- results -----------------------------------------------------------------


@router.get("/runs")
async def list_runs(
    limit: int = Query(50, ge=1, le=500),
    task: str | None = None,
    model_id: str | None = None,
    fingerprint: str | None = None,
):
    return {"runs": await run_in_threadpool(
        _store().list, limit, task, model_id, fingerprint
    )}


@router.get("/runs/{run_id}")
async def get_run(run_id: str):
    run = await run_in_threadpool(_store().get, run_id)
    if run is None:
        raise HTTPException(404, f"no run with id '{run_id}'")
    return run.model_dump(mode="json")


@router.get("/runs/{run_id}/iterations")
async def get_run_iterations(run_id: str):
    """Raw per-iteration samples — the evidence behind every aggregate."""
    store = _store()
    if await run_in_threadpool(store.get, run_id) is None:
        raise HTTPException(404, f"no run with id '{run_id}'")
    samples = await run_in_threadpool(store.iterations, run_id)
    return {"iterations": [s.model_dump(mode="json") for s in samples]}


@router.get("/runs/{run_id}/utilization")
async def get_run_utilization(run_id: str):
    samples = await run_in_threadpool(_store().utilization, run_id)
    return {"samples": [s.model_dump(mode="json") for s in samples]}


@router.get("/runs/{run_id}/export")
async def export_run(run_id: str, format: str = Query("json", pattern="^(json|csv|markdown)$")):
    from fastapi.responses import PlainTextResponse

    run = await run_in_threadpool(_store().get, run_id)
    if run is None:
        raise HTTPException(404, f"no run with id '{run_id}'")
    if format == "json":
        return PlainTextResponse(to_json(run), media_type="application/json")
    if format == "csv":
        return PlainTextResponse(iterations_to_csv(run), media_type="text/csv")
    return PlainTextResponse(to_markdown(run), media_type="text/markdown")


class CompareRequest(BaseModel):
    run_ids: list[str] = Field(min_length=2, max_length=8)


@router.post("/compare")
async def compare_runs(body: CompareRequest):
    """Compare runs, refusing rather than silently normalizing incompatible ones.

    The refusal carries the specific differences so the UI can explain itself
    instead of greying out a button with no reason.
    """
    store = _store()
    runs = []
    for run_id in body.run_ids:
        run = await run_in_threadpool(store.get, run_id)
        if run is None:
            raise HTTPException(404, f"no run with id '{run_id}'")
        runs.append(run)

    baseline = runs[0]
    comparisons = []
    for other in runs[1:]:
        comparable, reasons = baseline.is_comparable_to(other)
        comparisons.append({
            "run_id": other.identity.run_id,
            "comparable": comparable,
            "blocking_differences": reasons,
        })

    all_comparable = all(c["comparable"] for c in comparisons)
    return {
        "baseline_run_id": baseline.identity.run_id,
        "all_comparable": all_comparable,
        "comparisons": comparisons,
        "warning": None if all_comparable else (
            "Some runs measured materially different things. Differences are listed per run; "
            "placing them on a shared axis would be misleading."
        ),
        "runs": [r.model_dump(mode="json") for r in runs],
    }


@router.get("/export/summary")
async def export_summary(limit: int = Query(100, ge=1, le=1000)):
    from fastapi.responses import PlainTextResponse

    store = _store()
    rows = await run_in_threadpool(store.list, limit)
    runs = [r for r in (await run_in_threadpool(store.get, row["run_id"]) for row in rows) if r]
    return PlainTextResponse(summary_to_csv(runs), media_type="text/csv")


# --- overview ----------------------------------------------------------------


@router.get("/overview")
async def overview(request: Request):
    """Landing-page data: recent runs, per-task leaders, runtime availability.

    Leaders are computed per task, never pooled across tasks — a combined ranking
    over detection and embedding latency would be meaningless.
    """
    store = _store()
    rows = await run_in_threadpool(store.list, 200)

    by_task: dict[str, list[dict]] = {}
    for row in rows:
        by_task.setdefault(row["task"], []).append(row)

    leaders = {}
    for task, task_rows in by_task.items():
        timed = [r for r in task_rows if r["latency_p50_ms"] is not None]
        memoried = [r for r in task_rows if r["peak_rss_mb"] is not None]
        leaders[task] = {
            "fastest": min(timed, key=lambda r: r["latency_p50_ms"]) if timed else None,
            "most_memory_efficient": (
                min(memoried, key=lambda r: r["peak_rss_mb"]) if memoried else None
            ),
            "run_count": len(task_rows),
        }

    probes = probe_all()
    return {
        "recent_runs": rows[:10],
        "leaders_by_task": leaders,
        "total_runs": len(rows),
        "recent_failures": [
            r for r in rows if r["status"] in ("failed", "partial", "timed_out")
        ][:5],
        "runtimes_available": [c.runtime_id for c in probes if c.available],
        "runtimes_unavailable": len([c for c in probes if not c.available]),
    }
