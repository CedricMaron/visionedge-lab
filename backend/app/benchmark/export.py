"""Export benchmark runs to JSON, CSV and Markdown.

Every format carries the same disclosures the UI shows: sample counts alongside
means, unavailable metrics with their reasons, and integrity warnings. An export
that dropped the caveats would be a way to launder an uncertain number into a
confident-looking one.
"""
from __future__ import annotations

import csv
import io
import json

from app.schemas.measurement import Measurement
from app.schemas.run import BenchmarkRun
from app.schemas.timing import DurationStats


def to_json(run: BenchmarkRun, indent: int = 2) -> str:
    """The complete document, losing nothing."""
    return json.dumps(run.model_dump(mode="json"), indent=indent)


def iterations_to_csv(run: BenchmarkRun) -> str:
    """Raw per-iteration samples — the evidence behind every aggregate."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow([
        "run_id", "index", "group", "counted_in_statistics", "succeeded",
        "total_ms", "preprocessing_ms", "model_execution_ms", "postprocessing_ms",
        "device_synchronized", "error_type", "error_message",
    ])
    for it in run.iterations:
        spans = {s.phase.value: s for s in it.spans if s.parent is None}
        model_span = spans.get("model_execution")
        writer.writerow([
            run.identity.run_id,
            it.index,
            it.group.value,
            it.counts_toward_statistics,
            it.succeeded,
            _round(it.total_ms),
            _round(getattr(spans.get("preprocessing"), "duration_ms", None)),
            _round(getattr(model_span, "duration_ms", None)),
            _round(getattr(spans.get("postprocessing"), "duration_ms", None)),
            getattr(model_span, "device_synchronized", ""),
            it.error_type or "",
            it.error_message or "",
        ])
    return buffer.getvalue()


def summary_to_csv(runs: list[BenchmarkRun]) -> str:
    """One row per run, for comparing many runs in a spreadsheet."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow([
        "run_id", "created_at", "status", "task", "scenario", "model", "runtime",
        "device", "precision", "mode", "fingerprint", "batch_size",
        "n_measured", "n_failed", "p50_ms", "p90_ms", "p95_ms", "p99_ms",
        "mean_ms", "stddev_ms", "cv", "requests_per_s", "peak_rss_mb",
        "energy_j", "warnings",
    ])
    for run in runs:
        t = run.timings.total
        writer.writerow([
            run.identity.run_id,
            run.identity.created_at,
            run.status.value,
            run.task.value,
            run.scenario.id,
            run.model.model_id,
            run.runtime.runtime_id,
            run.runtime.device.value,
            run.runtime.precision.value,
            run.mode.value,
            run.fingerprint.digest,
            run.scenario.batch_size,
            t.n,
            run.failed_iterations,
            _round(t.p50_ms), _round(t.p90_ms), _round(t.p95_ms), _round(t.p99_ms),
            _round(t.mean_ms), _round(t.stddev_ms),
            _round(t.coefficient_of_variation, 4),
            _measurement_csv(run.throughput.requests_per_second),
            _measurement_csv(run.memory.peak_process_rss_mb),
            _measurement_csv(run.energy.total_energy_j),
            " | ".join(run.warnings),
        ])
    return buffer.getvalue()


def to_markdown(run: BenchmarkRun) -> str:
    """A human-readable report, warnings first."""
    lines: list[str] = []
    a = lines.append

    a(f"# Benchmark run `{run.identity.run_id}`")
    a("")
    a(f"**{run.model.display_name or run.model.model_id}** · {run.task.value} · "
      f"{run.runtime.runtime_id} on {run.runtime.device.value} · {run.runtime.precision.value}")
    a("")
    a(f"Status: **{run.status.value}** · scenario `{run.scenario.id}` · "
      f"mode `{run.mode.value}` · fingerprint `{run.fingerprint.digest}`")
    a("")

    if run.warnings:
        a("## Warnings")
        a("")
        for w in run.warnings:
            a(f"- {w}")
        a("")

    if run.errors.failure_count:
        a(f"## Failures ({run.errors.failure_count})")
        a("")
        a("Statistics below exclude these iterations.")
        a("")
        for f in run.errors.failures[:20]:
            a(f"- iteration {f.index}: `{f.error_type}` — {f.error_message}")
        a("")

    a("## Latency decomposition")
    a("")
    a("| Phase | p50 ms | p95 ms | mean ms | n |")
    a("|---|---:|---:|---:|---:|")
    for phase, stats in sorted(
        run.timings.phases.items(), key=lambda kv: -(kv[1].mean_ms or 0)
    ):
        a(f"| {phase.value} | {_fmt(stats.p50_ms)} | {_fmt(stats.p95_ms)} | "
          f"{_fmt(stats.mean_ms)} | {stats.n} |")
    if run.timings.residual_ms is not None:
        a(f"| _residual overhead_ | | | {_fmt(run.timings.residual_ms)} | |")
    t = run.timings.total
    a(f"| **end-to-end** | **{_fmt(t.p50_ms)}** | **{_fmt(t.p95_ms)}** | "
      f"**{_fmt(t.mean_ms)}** | **{t.n}** |")
    a("")
    a(_distribution_line(t))
    a("")

    a("## Cold start vs. steady state")
    a("")
    c = run.cold_warm
    a(f"- model load: {_fmt(c.model_load_ms)} ms")
    a(f"- first inference: {_fmt(c.first_inference_ms)} ms")
    a(f"- cold start total: {_fmt(c.cold_start_total_ms)} ms")
    a(f"- warm p50: {_fmt(c.warm_inference.p50_ms)} ms over {c.warm_inference.n} iterations")
    a("")

    a("## Throughput")
    a("")
    a("| Metric | Value | Kind |")
    a("|---|---|---|")
    for name, m in _iter_measurements(run.throughput):
        a(f"| {name} | {_measurement_md(m)} | {m.kind.value if m.available else '—'} |")
    a("")

    a("## Memory")
    a("")
    for name, m in _iter_measurements(run.memory, skip={"snapshots"}):
        a(f"- **{name}**: {_measurement_md(m)}")
    a("")

    a("## Energy")
    a("")
    for name, m in _iter_measurements(run.energy):
        a(f"- **{name}**: {_measurement_md(m)}")
    a("")

    a("## Environment")
    a("")
    hw, sw = run.hardware, run.software
    a(f"- CPU: {hw.cpu_model} ({hw.cpu_cores_logical} logical cores)")
    if hw.cpu_instruction_sets:
        a(f"- ISA: {', '.join(hw.cpu_instruction_sets)}")
    a(f"- RAM: {hw.ram_total_mb} MB")
    a(f"- GPU: {', '.join(g.name for g in hw.gpus) if hw.gpus else 'none detected'}")
    a(f"- CUDA: {hw.cuda_version or 'n/a'} · cuDNN: {hw.cudnn_version or 'n/a'}")
    a(f"- OS: {sw.os} {sw.os_version}")
    a(f"- Python: {sw.python_version}")
    for pkg, ver in sorted(sw.package_versions.items()):
        a(f"  - {pkg} {ver}")
    a("")

    a("## Reproducibility")
    a("")
    r = run.reproducibility
    a(f"- git commit: `{r.git_commit or 'unknown'}`" + (" **(dirty tree)**" if r.git_dirty else ""))
    a(f"- seed: {r.random_seed} · deterministic: {r.deterministic_mode}")
    if r.reproduction_command:
        a("")
        a("```bash")
        a(r.reproduction_command)
        a("```")
    a("")
    a(f"_Raw samples retained: {len(run.iterations)} iterations "
      f"({run.warmup_iterations_run} warm-up, {run.successful_iterations} measured, "
      f"{run.failed_iterations} failed)._")

    return "\n".join(lines)


# --- helpers ---------------------------------------------------------------


def _round(value: float | None, digits: int = 3) -> str:
    return "" if value is None else f"{round(value, digits)}"


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def _distribution_line(stats: DurationStats) -> str:
    if stats.n == 0:
        return "_No measured iterations._"
    parts = [f"n = {stats.n}", f"min {_fmt(stats.min_ms)} ms", f"max {_fmt(stats.max_ms)} ms"]
    if stats.stddev_ms is not None:
        parts.append(f"stddev {_fmt(stats.stddev_ms)} ms")
    if stats.coefficient_of_variation is not None:
        parts.append(f"CV {stats.coefficient_of_variation:.3f}")
    return "_" + " · ".join(parts) + "_"


def _measurement_md(m: Measurement) -> str:
    if not m.available:
        return f"_unavailable — {m.unavailable_reason}_"
    text = f"{m.value:.4g} {m.unit}".strip() if isinstance(m.value, float) else f"{m.value} {m.unit}".strip()
    if m.note:
        text += f" <br><sub>{m.note}</sub>"
    return text


def _measurement_csv(m: Measurement) -> str:
    return "" if not m.available else str(m.value)


def _iter_measurements(model, skip: set[str] | None = None):
    """Yield (field_name, Measurement) pairs from a metrics model."""
    skip = skip or set()
    for name in type(model).model_fields:
        if name in skip:
            continue
        value = getattr(model, name)
        if isinstance(value, Measurement):
            yield name.replace("_", " "), value
