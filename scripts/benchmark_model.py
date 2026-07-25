#!/usr/bin/env python3
"""Benchmark a detection model's real inference latency + throughput. MEASURED, never faked.

For .onnx models this loads the actual ``OnnxRuntimeBackend`` and times the full
inference path on the sample image (or a zero image), reporting mean / p50 / p95 / p99
latency, FPS, and process RSS. A JSON report is written to benchmark-data/ capturing the
hardware (from the real capability scanner), OS, runtime versions, model checksum, config
and the measured metrics.

No numbers are invented: every metric here comes from ``time.perf_counter()`` around real
runs on this machine. Results are only comparable within the same machine/run.

Example:
    python scripts/benchmark_model.py --model models/yolov8n.onnx --runs 50 --warmup 5
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import _common as C

C.bootstrap_path()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True, type=Path, help="model to benchmark (.onnx)")
    p.add_argument("--runs", type=int, default=50, help="timed runs (default 50)")
    p.add_argument("--warmup", type=int, default=5, help="warmup runs, untimed (default 5)")
    p.add_argument("--size", type=int, default=640, help="input size (default 640)")
    p.add_argument("--image", type=Path, default=C.BENCHMARK_DIR / "sample_bus.jpg",
                   help="image to run (falls back to a zero image if missing)")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.model.exists():
        C.die(f"model not found: {args.model}")
    if args.runs <= 0:
        C.die("--runs must be >= 1")
    if args.model.suffix.lower() != ".onnx":
        C.die(f"only .onnx models are benchmarked by this script (got {args.model.suffix}); "
              "OpenVINO/TensorRT timing must run on their target runtimes.")

    import numpy as np

    # Real image or a deterministic zero image (still a real, measured run).
    image = None
    image_source = "zero_image"
    if args.image.exists():
        import cv2

        image = cv2.imread(str(args.image))
        if image is not None:
            image_source = str(args.image)
    if image is None:
        image = np.zeros((args.size, args.size, 3), dtype=np.uint8)

    from app.capabilities.scanner import scan_capabilities
    from app.inference.onnx_backend import OnnxRuntimeBackend

    C.info(f"loading {args.model} in onnxruntime")
    backend = OnnxRuntimeBackend(model_path=args.model, model_id=args.model.stem, input_size=args.size)
    backend.load()
    C.info(f"provider: {backend.provider}")

    C.info(f"warmup: {args.warmup} runs")
    for _ in range(args.warmup):
        backend.predict(image, args.conf, args.iou, None)

    C.info(f"timing: {args.runs} runs on {image_source}")
    latencies_ms: list[float] = []
    infer_ms: list[float] = []
    rss_before = C.rss_mb()
    for _ in range(args.runs):
        t0 = time.perf_counter()
        _dets, timings = backend.predict_timed(image, args.conf, args.iou, None)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)
        infer_ms.append(timings["inference_ms"])
    rss_after = C.rss_mb()
    backend.close()

    mean = statistics.mean(latencies_ms)
    metrics = {
        "end_to_end_ms": {
            "mean": round(mean, 3),
            "p50": round(C.percentile(latencies_ms, 50), 3),
            "p95": round(C.percentile(latencies_ms, 95), 3),
            "p99": round(C.percentile(latencies_ms, 99), 3),
            "min": round(min(latencies_ms), 3),
            "max": round(max(latencies_ms), 3),
            "stdev": round(statistics.pstdev(latencies_ms), 3),
        },
        "inference_only_ms": {
            "mean": round(statistics.mean(infer_ms), 3),
            "p50": round(C.percentile(infer_ms, 50), 3),
            "p95": round(C.percentile(infer_ms, 95), 3),
        },
        "fps_end_to_end": round(1000.0 / mean, 2) if mean > 0 else None,
        "rss_mb_before": round(rss_before, 1),
        "rss_mb_after": round(rss_after, 1),
    }

    caps = scan_capabilities()
    report = {
        "kind": "benchmark",
        "timestamp": C.now_iso(),
        "model": {
            "path": str(args.model),
            "sha256": C.sha256_file(args.model),
            "size_bytes": C.file_size(args.model),
        },
        "config": {
            "runs": args.runs, "warmup": args.warmup, "size": args.size,
            "conf": args.conf, "iou": args.iou, "image_source": image_source,
        },
        "runtime": {
            "backend": "onnxruntime",
            "provider": backend.provider,
            "versions": C.runtime_versions(),
        },
        "hardware": {
            "cpu_model": caps.cpu_model,
            "cpu_cores_physical": caps.cpu_cores_physical,
            "cpu_cores_logical": caps.cpu_cores_logical,
            "ram_total_mb": caps.ram_total_mb,
            "ram_available_mb": caps.ram_available_mb,
            "gpus": [g.model_dump() for g in caps.gpus],
            "nvidia_gpu_present": caps.nvidia_gpu_present,
        },
        "os": {"system": caps.os, "version": caps.os_version, "python": caps.python_version},
        "metrics": metrics,
        "note": "MEASURED on this machine; only comparable within the same host/run.",
    }

    # --- print table ---
    e = metrics["end_to_end_ms"]
    print("\n=== BENCHMARK (measured) ===")
    print(f"model:     {args.model.name}  ({C.human_size(report['model']['size_bytes'])})")
    print(f"provider:  {backend.provider}")
    print(f"runs:      {args.runs} (warmup {args.warmup})  image: {image_source}")
    print(f"{'metric':<18}{'end-to-end (ms)':>18}")
    for key in ("mean", "p50", "p95", "p99", "min", "max", "stdev"):
        print(f"{key:<18}{e[key]:>18.3f}")
    print(f"{'fps':<18}{metrics['fps_end_to_end']:>18}")
    print(f"{'inference mean':<18}{metrics['inference_only_ms']['mean']:>18.3f}")
    print(f"{'rss MB (after)':<18}{metrics['rss_mb_after']:>18.1f}")
    print("============================\n")

    C.BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    out = C.BENCHMARK_DIR / f"bench_{args.model.stem}_{ts}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    C.ok(f"report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
