"""``inference-lab`` command line interface.

Calls exactly the same :class:`~app.benchmark.engine.BenchmarkEngine` as the HTTP
API (§24), so a benchmark cannot mean one thing in the web UI and another on the
command line. Every completed run prints the command that reproduces it.

    inference-lab runtimes
    inference-lab scenarios
    inference-lab system
    inference-lab benchmark run --scenario single-image-detection --model yolov8n-onnx
    inference-lab results list
    inference-lab results show <run_id> --format markdown
"""
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from app.adapters.base import LoadConfig
from app.adapters.classification.mobilenet import MobileNetClassifierAdapter
from app.adapters.detection.yolov8 import YoloV8Adapter
from app.adapters.embedding.minilm import MiniLmEmbeddingAdapter
from app.benchmark.engine import BenchmarkEngine, EngineOptions
from app.benchmark.export import iterations_to_csv, summary_to_csv, to_json, to_markdown
from app.benchmark.scenarios import get_scenario, load_all
from app.core.config import REPO_ROOT, get_settings
from app.core.errors import InferenceLabError
from app.runtimes.registry import capability_matrix, get_adapter, probe_all
from app.schemas.enums import BenchmarkMode, DeviceKind, ExecutionLocation, Precision
from app.schemas.environment import RuntimeReference
from app.storage.runs import RunStore

PROG = "inference-lab"


# --- adapter construction ---------------------------------------------------

#: model_id -> (relative path under models/, adapter factory, install hint).
#: Explicit rather than dynamic: resolving an adapter class by name from user input
#: would be an arbitrary-import primitive (§25).
_MODELS: dict[str, tuple[str, str, str]] = {
    "yolov8n-onnx": ("yolov8n.onnx", "yolov8", "run `make model`"),
    "yolov8s-onnx": ("yolov8s.onnx", "yolov8", "run `scripts/export_onnx.py --model small`"),
    "yolov8m-onnx": ("yolov8m.onnx", "yolov8", "run `scripts/export_onnx.py --model medium`"),
    "mobilenetv4-conv-small-onnx": (
        "classification/mobilenetv4_conv_small.onnx", "mobilenet",
        "run `python scripts/download_models.py --model mobilenetv4-conv-small-onnx`",
    ),
    "all-minilm-l6-v2-onnx": (
        "embedding/all-MiniLM-L6-v2.onnx", "minilm",
        "run `python scripts/download_models.py --model all-minilm-l6-v2-onnx`",
    ),
}


def _build_adapter(model_id: str, runtime_id: str, input_size: int | None):
    """Map a model id to its adapter."""
    entry = _MODELS.get(model_id)
    if entry is None:
        available = ", ".join(sorted(_MODELS))
        raise InferenceLabError(
            f"unknown model '{model_id}'",
            user_message=f"'{model_id}' has no adapter. Available: {available}",
        )

    relative, kind, hint = entry
    path = get_settings().models_dir / relative
    if not path.exists():
        raise InferenceLabError(
            f"model file not found: {path}",
            user_message=f"'{model_id}' is not installed — {hint}.",
        )

    runtime = get_adapter(runtime_id)
    if kind == "yolov8":
        return YoloV8Adapter(path, runtime, model_id=model_id, input_size=input_size or 640)
    if kind == "mobilenet":
        return MobileNetClassifierAdapter(path, runtime, model_id=model_id)
    if kind == "minilm":
        return MiniLmEmbeddingAdapter(path, runtime, model_id=model_id)
    raise InferenceLabError(f"no adapter factory for kind '{kind}'")


def _reproduction_command(args: argparse.Namespace) -> str:
    parts = [
        PROG, "benchmark", "run",
        "--scenario", args.scenario,
        "--model", args.model,
        "--runtime", args.runtime,
        "--device", args.device,
        "--precision", args.precision,
    ]
    if args.iterations is not None:
        parts += ["--iterations", str(args.iterations)]
    if args.warmup is not None:
        parts += ["--warmup", str(args.warmup)]
    if args.mode:
        parts += ["--mode", args.mode]
    if args.seed is not None:
        parts += ["--seed", str(args.seed)]
    return " ".join(shlex.quote(p) for p in parts)


# --- commands ---------------------------------------------------------------

def cmd_runtimes(args: argparse.Namespace) -> int:
    print(f"{'runtime':24} {'status':12} {'version':10} detail")
    print("-" * 100)
    for cap in probe_all():
        status = "available" if cap.available else "unavailable"
        detail = cap.unavailable_reason or (
            f"devices: {', '.join(d.value for d in cap.devices)}"
        )
        print(f"{cap.runtime_id:24} {status:12} {cap.version or '-':10} {detail}")
    return 0


def cmd_matrix(args: argparse.Namespace) -> int:
    print(f"{'runtime':24} {'device':6} {'precision':10} {'ok':4} reason")
    print("-" * 110)
    for row in capability_matrix():
        if args.supported_only and not row["supported"]:
            continue
        mark = "yes" if row["supported"] else "no"
        print(f"{row['runtime_id']:24} {row['device']:6} {row['precision']:10} {mark:4} "
              f"{row['reason'] or ''}")
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    models_dir = get_settings().models_dir
    print(f"{'model_id':32} {'status':14} {'task':22} path")
    print("-" * 108)
    for model_id, (relative, kind, hint) in sorted(_MODELS.items()):
        path = models_dir / relative
        installed = path.exists()
        task = {"yolov8": "object_detection", "mobilenet": "image_classification",
                "minilm": "text_embedding"}.get(kind, "unknown")
        status = "installed" if installed else "not installed"
        print(f"{model_id:32} {status:14} {task:22} {relative}")
        if not installed:
            print(f"{'':32} -> {hint}")
    return 0


def cmd_scenarios(args: argparse.Namespace) -> int:
    scenarios = load_all()
    if not scenarios:
        print("No scenarios found in benchmarks/scenarios/", file=sys.stderr)
        return 1
    for spec in scenarios.values():
        print(f"{spec.id:34} {spec.task.value:22} "
              f"warmup={spec.warmup_iterations} measured={spec.measured_iterations} "
              f"batch={spec.batch_size} mode={spec.mode.value}")
        if spec.description:
            print(f"{'':34} {' '.join(spec.description.split())[:100]}")
    return 0


def cmd_system(args: argparse.Namespace) -> int:
    from app.instrumentation.environment import collect_hardware, collect_software

    hw, sw = collect_hardware(), collect_software()
    print(f"OS        : {sw.os} {sw.os_version} (kernel {sw.kernel_version})")
    print(f"Python    : {sw.python_version}")
    print(f"CPU       : {hw.cpu_model}")
    print(f"Cores     : {hw.cpu_cores_physical} physical / {hw.cpu_cores_logical} logical")
    print(f"ISA       : {', '.join(hw.cpu_instruction_sets) or 'undetected'}")
    print(f"RAM       : {hw.ram_total_mb} MB")
    print(f"GPUs      : {len(hw.gpus)}")
    for g in hw.gpus:
        print(f"  [{g.index}] {g.name} · {g.memory_total_mb} MB · driver {g.driver_version} "
              f"· compute {g.compute_capability}")
    print(f"CUDA      : {hw.cuda_version or 'n/a'}   cuDNN: {hw.cudnn_version or 'n/a'}")
    print(f"NVML      : {'available' if hw.nvml_available else 'unavailable'}")
    print("Packages  :")
    for pkg, ver in sorted(sw.package_versions.items()):
        print(f"  {pkg:28} {ver}")
    return 0


def cmd_benchmark_run(args: argparse.Namespace) -> int:
    scenario = get_scenario(args.scenario)
    if args.iterations is not None:
        scenario = scenario.model_copy(update={"measured_iterations": args.iterations})
    if args.warmup is not None:
        scenario = scenario.model_copy(update={"warmup_iterations": args.warmup})
    if args.mode:
        scenario = scenario.model_copy(update={"mode": BenchmarkMode(args.mode)})
    if args.seed is not None:
        scenario = scenario.model_copy(update={"random_seed": args.seed})

    device = DeviceKind(args.device)
    precision = Precision(args.precision)
    adapter = _build_adapter(args.model, args.runtime, scenario.input_size)

    engine = BenchmarkEngine(
        EngineOptions(
            enable_sampler=not args.no_sampler,
            reproduction_command=_reproduction_command(args),
            label=args.label,
        )
    )
    try:
        run = engine.run(
            adapter,
            scenario,
            LoadConfig(
                runtime_id=args.runtime, device=device, precision=precision,
                input_size=scenario.input_size,
            ),
            RuntimeReference(
                runtime_id=args.runtime, device=device, precision=precision,
            ),
            execution_location=ExecutionLocation.IN_PROCESS,
        )
    finally:
        engine.close()
        adapter.unload()

    if not args.no_save:
        RunStore(get_settings().db_path).save(run)

    if args.format == "json":
        print(to_json(run))
    elif args.format == "markdown":
        print(to_markdown(run))
    else:
        _print_summary(run)

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = {"json": to_json, "markdown": to_markdown, "csv": iterations_to_csv}[
            args.output_format
        ](run)
        path.write_text(content, encoding="utf-8")
        print(f"\nwrote {args.output_format} to {path}", file=sys.stderr)

    return 0 if run.status.value in ("completed", "partial") else 1


def _print_summary(run) -> None:
    t = run.timings.total
    print(f"run {run.identity.run_id}  status={run.status.value}  "
          f"fingerprint={run.fingerprint.digest}")
    print(f"{run.model.model_id} · {run.runtime.runtime_id}/{run.runtime.device.value}"
          f"/{run.runtime.precision.value} · scenario {run.scenario.id} · mode {run.mode.value}")
    print()
    print(f"{'phase':22} {'p50 ms':>9} {'p95 ms':>9} {'mean ms':>9} {'n':>5}")
    print("-" * 60)
    for phase, stats in sorted(run.timings.phases.items(), key=lambda kv: -(kv[1].mean_ms or 0)):
        print(f"{phase.value:22} {_n(stats.p50_ms):>9} {_n(stats.p95_ms):>9} "
              f"{_n(stats.mean_ms):>9} {stats.n:>5}")
    if run.timings.residual_ms is not None:
        print(f"{'residual overhead':22} {'':>9} {'':>9} {_n(run.timings.residual_ms):>9}")
    print("-" * 60)
    print(f"{'end-to-end':22} {_n(t.p50_ms):>9} {_n(t.p95_ms):>9} {_n(t.mean_ms):>9} {t.n:>5}")
    if t.stddev_ms is not None:
        print(f"\nspread: min {_n(t.min_ms)} / max {_n(t.max_ms)} / stddev {_n(t.stddev_ms)} ms"
              f" · CV {t.coefficient_of_variation:.3f}")
    print(f"cold start: {_n(run.cold_warm.cold_start_total_ms)} ms "
          f"(load {_n(run.cold_warm.model_load_ms)} + first inference "
          f"{_n(run.cold_warm.first_inference_ms)})")

    rps = run.throughput.requests_per_second
    if rps.available:
        print(f"throughput: {rps.value:.2f} req/s")
    energy = run.energy.total_energy_j
    print(f"energy: {f'{energy.value:.2f} J' if energy.available else 'unavailable'}")
    if not energy.available:
        print(f"  reason: {energy.unavailable_reason}")

    if run.errors.failure_count:
        print(f"\nFAILURES: {run.errors.failure_count} (excluded from statistics)")
        for f in run.errors.failures[:5]:
            print(f"  iteration {f.index}: {f.error_type}: {f.error_message}")
    if run.warnings:
        print("\nwarnings:")
        for w in run.warnings:
            print(f"  - {w}")
    if run.reproducibility.reproduction_command:
        print(f"\nreproduce:\n  {run.reproducibility.reproduction_command}")


def _n(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def cmd_results_list(args: argparse.Namespace) -> int:
    rows = RunStore(get_settings().db_path).list(limit=args.limit, task=args.task,
                                                 model_id=args.model)
    if not rows:
        print("no runs stored yet")
        return 0
    print(f"{'run_id':18} {'status':10} {'task':20} {'model':18} {'p50 ms':>8} {'n':>4} fingerprint")
    print("-" * 105)
    for r in rows:
        p50 = "—" if r["latency_p50_ms"] is None else f"{r['latency_p50_ms']:.2f}"
        print(f"{r['run_id']:18} {r['status']:10} {r['task']:20} {r['model_id']:18} "
              f"{p50:>8} {r['measured_iterations']:>4} {r['fingerprint']}")
    return 0


def cmd_results_show(args: argparse.Namespace) -> int:
    run = RunStore(get_settings().db_path).get(args.run_id)
    if run is None:
        print(f"no run with id '{args.run_id}'", file=sys.stderr)
        return 1
    if args.format == "json":
        print(to_json(run))
    elif args.format == "csv":
        print(iterations_to_csv(run))
    elif args.format == "summary":
        _print_summary(run)
    else:
        print(to_markdown(run))
    return 0


def cmd_results_export(args: argparse.Namespace) -> int:
    store = RunStore(get_settings().db_path)
    runs = [r for r in (store.get(row["run_id"]) for row in store.list(limit=args.limit)) if r]
    if not runs:
        print("no runs to export", file=sys.stderr)
        return 1
    Path(args.output).write_text(summary_to_csv(runs), encoding="utf-8")
    print(f"exported {len(runs)} run(s) to {args.output}")
    return 0


# --- parser -----------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG, description="InferenceLab — multimodal AI inference, profiling and benchmarking."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("runtimes", help="probe every runtime and report what is usable").set_defaults(
        func=cmd_runtimes
    )
    matrix = sub.add_parser("matrix", help="runtime x device x precision capability matrix")
    matrix.add_argument("--supported-only", action="store_true")
    matrix.set_defaults(func=cmd_matrix)

    sub.add_parser("models", help="list models with an adapter").set_defaults(func=cmd_models)
    sub.add_parser("scenarios", help="list benchmark scenarios").set_defaults(func=cmd_scenarios)
    sub.add_parser("system", help="show hardware and software environment").set_defaults(
        func=cmd_system
    )

    bench = sub.add_parser("benchmark", help="run benchmarks")
    bench_sub = bench.add_subparsers(dest="benchmark_command", required=True)
    run = bench_sub.add_parser("run", help="execute one scenario")
    run.add_argument("--scenario", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--runtime", default="onnxruntime")
    run.add_argument("--device", default="cpu", choices=[d.value for d in DeviceKind])
    run.add_argument("--precision", default="fp32", choices=[p.value for p in Precision])
    run.add_argument("--iterations", type=int, help="override measured_iterations")
    run.add_argument("--warmup", type=int, help="override warmup_iterations")
    run.add_argument("--mode", choices=[m.value for m in BenchmarkMode])
    run.add_argument("--seed", type=int)
    run.add_argument("--label")
    run.add_argument("--format", default="summary", choices=["summary", "json", "markdown"])
    run.add_argument("--output", help="also write the report to this path")
    run.add_argument("--output-format", default="json", choices=["json", "markdown", "csv"])
    run.add_argument("--no-save", action="store_true", help="do not persist the run")
    run.add_argument("--no-sampler", action="store_true", help="disable hardware sampling")
    run.set_defaults(func=cmd_benchmark_run)

    results = sub.add_parser("results", help="inspect stored runs")
    results_sub = results.add_subparsers(dest="results_command", required=True)
    listing = results_sub.add_parser("list")
    listing.add_argument("--limit", type=int, default=25)
    listing.add_argument("--task")
    listing.add_argument("--model")
    listing.set_defaults(func=cmd_results_list)

    show = results_sub.add_parser("show")
    show.add_argument("run_id")
    show.add_argument("--format", default="markdown",
                      choices=["markdown", "json", "csv", "summary"])
    show.set_defaults(func=cmd_results_show)

    export = results_sub.add_parser("export", help="export a summary CSV of many runs")
    export.add_argument("--output", default=str(REPO_ROOT / "benchmarks" / "results" / "summary.csv"))
    export.add_argument("--limit", type=int, default=100)
    export.set_defaults(func=cmd_results_export)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except InferenceLabError as exc:
        print(f"error: {exc.user_message}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
