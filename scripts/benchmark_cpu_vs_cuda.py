#!/usr/bin/env python3
"""Benchmark the SAME ONNX model on the CPU vs the CUDA execution provider. MEASURED.

Honest by construction:
  * it reports the provider ONNX Runtime *actually* selected for each run — asking for
    CUDA does not mean CUDA ran, so the fallback to CPU is printed as such;
  * a speedup is only computed when the CUDA run genuinely used ``CUDAExecutionProvider``;
  * detection outputs from both runs are compared, so a "faster" provider that changes
    the results cannot pass unnoticed;
  * every latency comes from ``time.perf_counter()`` around real inference on this
    machine. Results are only comparable within the same host/run.

Example:
    python scripts/benchmark_cpu_vs_cuda.py --model models/yolov8n.onnx --runs 50
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import _common as C

C.bootstrap_path()


@dataclass
class RunResult:
    """One measured pass of the model under a requested provider preference."""

    requested: str
    provider: str
    latencies_ms: list[float]
    infer_ms: list[float]
    rss_mb_after: float
    detections: list[Any] = field(default_factory=list)

    @property
    def used_cuda(self) -> bool:
        return self.provider == "CUDAExecutionProvider"

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.latencies_ms)

    @property
    def p50_ms(self) -> float:
        return C.percentile(self.latencies_ms, 50)

    def metrics(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "provider_actually_used": self.provider,
            "end_to_end_ms": {
                "mean": round(self.mean_ms, 3),
                "p50": round(self.p50_ms, 3),
                "p95": round(C.percentile(self.latencies_ms, 95), 3),
                "p99": round(C.percentile(self.latencies_ms, 99), 3),
                "min": round(min(self.latencies_ms), 3),
                "max": round(max(self.latencies_ms), 3),
                "stdev": round(statistics.pstdev(self.latencies_ms), 3),
            },
            "inference_only_ms": {
                "mean": round(statistics.mean(self.infer_ms), 3),
                "p50": round(C.percentile(self.infer_ms, 50), 3),
                "p95": round(C.percentile(self.infer_ms, 95), 3),
            },
            "fps_end_to_end": round(1000.0 / self.mean_ms, 2) if self.mean_ms > 0 else None,
            "rss_mb_after": round(self.rss_mb_after, 1),
            "detections": len(self.detections),
        }


def _measure(model: Path, image, args, prefer_cuda: bool) -> RunResult:
    from app.inference.onnx_backend import OnnxRuntimeBackend

    requested = "CUDAExecutionProvider" if prefer_cuda else "CPUExecutionProvider"
    backend = OnnxRuntimeBackend(
        model_path=model, model_id=model.stem, input_size=args.size, prefer_cuda=prefer_cuda
    )
    backend.load()
    C.info(f"requested {requested} -> ORT selected {backend.provider}")

    for _ in range(args.warmup):
        backend.predict(image, args.conf, args.iou, None)

    latencies_ms: list[float] = []
    infer_ms: list[float] = []
    detections: list[Any] = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        dets, timings = backend.predict_timed(image, args.conf, args.iou, None)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)
        infer_ms.append(timings["inference_ms"])
        detections = dets

    result = RunResult(
        requested=requested,
        provider=backend.provider or "unknown",
        latencies_ms=latencies_ms,
        infer_ms=infer_ms,
        rss_mb_after=C.rss_mb(),
        detections=detections,
    )
    backend.close()
    return result


def compare_detections(cpu_dets: list[Any], cuda_dets: list[Any]) -> dict[str, Any]:
    """Compare two detection lists: counts, class agreement, and worst numeric drift.

    Detections are matched by sorted order (confidence desc, then class id) so the
    comparison does not depend on the backend's emission order.
    """
    def _key(d):
        return (-d.confidence, d.classId, d.x1, d.y1)

    a = sorted(cpu_dets, key=_key)
    b = sorted(cuda_dets, key=_key)
    counts_a = sorted((d.classId, d.className) for d in a)
    counts_b = sorted((d.classId, d.className) for d in b)

    report: dict[str, Any] = {
        "cpu_detections": len(a),
        "cuda_detections": len(b),
        "same_count": len(a) == len(b),
        "same_classes": counts_a == counts_b,
    }
    if len(a) == len(b) and a:
        report["max_box_delta_px"] = round(
            max(
                max(abs(x.x1 - y.x1), abs(x.y1 - y.y1), abs(x.x2 - y.x2), abs(x.y2 - y.y2))
                for x, y in zip(a, b, strict=True)
            ),
            4,
        )
        report["max_confidence_delta"] = round(
            max(abs(x.confidence - y.confidence) for x, y in zip(a, b, strict=True)), 6
        )
    report["agree"] = bool(report["same_count"] and report["same_classes"])
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", type=Path, default=C.MODELS_DIR / "yolov8n.onnx",
                   help="model to benchmark (.onnx)")
    p.add_argument("--runs", type=int, default=50, help="timed runs per provider (default 50)")
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
        C.die(f"only .onnx models can be compared across ORT providers (got {args.model.suffix})")

    import numpy as np
    import onnxruntime as ort

    available = ort.get_available_providers()
    print(f"onnxruntime {ort.__version__}  available providers: {available}")
    if "CUDAExecutionProvider" not in available:
        C.warn("this onnxruntime build does not offer CUDAExecutionProvider at all; "
               "both runs below will be CPU. Install onnxruntime-gpu to compare.")

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

    C.info(f"model: {args.model}  image: {image_source}  runs: {args.runs} (warmup {args.warmup})")
    cpu = _measure(args.model, image, args, prefer_cuda=False)
    cuda = _measure(args.model, image, args, prefer_cuda=True)

    agreement = compare_detections(cpu.detections, cuda.detections)
    cuda_really_ran = cuda.used_cuda and not cpu.used_cuda

    # --- print table ---
    print("\n=== CPU vs CUDA (measured on this machine) ===")
    header = f"{'requested':<10}{'provider used':<26}{'FPS':>8}{'mean ms':>10}{'p50':>9}{'p95':>9}{'p99':>9}"
    print(header)
    for label, r in (("CPU", cpu), ("CUDA", cuda)):
        m = r.metrics()["end_to_end_ms"]
        print(f"{label:<10}{r.provider:<26}{1000.0 / r.mean_ms:8.1f}"
              f"{m['mean']:10.2f}{m['p50']:9.2f}{m['p95']:9.2f}{m['p99']:9.2f}")

    print()
    if cuda_really_ran:
        speedup = cpu.p50_ms / cuda.p50_ms if cuda.p50_ms > 0 else float("nan")
        print(f"CUDA genuinely ran on the GPU. Measured p50 speedup vs CPU: {speedup:.2f}x")
    else:
        speedup = None
        print(f"NO GPU COMPARISON: the CUDA request resolved to {cuda.provider!r}.")
        print("Both rows above ran on the CPU, so no speedup is claimed — any difference "
              "between them is run-to-run variance on the same provider. This usually means "
              "onnxruntime-gpu is not installed, or CUDA/cuDNN is missing or version-mismatched.")

    print(f"\ndetection agreement: {'OK' if agreement['agree'] else 'MISMATCH'}  {agreement}")
    print("==============================================\n")

    caps = scan_capabilities()
    report = {
        "kind": "benchmark_cpu_vs_cuda",
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
            "available_providers": available,
            "versions": C.runtime_versions(),
        },
        "hardware": {
            "cpu_model": caps.cpu_model,
            "cpu_cores_physical": caps.cpu_cores_physical,
            "cpu_cores_logical": caps.cpu_cores_logical,
            "ram_total_mb": caps.ram_total_mb,
            "gpus": [g.model_dump() for g in caps.gpus],
            "nvidia_gpu_present": caps.nvidia_gpu_present,
        },
        "os": {"system": caps.os, "version": caps.os_version, "python": caps.python_version},
        "runs": {"cpu": cpu.metrics(), "cuda": cuda.metrics()},
        "comparison": {
            "cuda_actually_used": cuda_really_ran,
            "p50_speedup_vs_cpu": round(speedup, 3) if speedup is not None else None,
            "detection_agreement": agreement,
        },
        "note": ("MEASURED on this machine; only comparable within the same host/run. "
                 "A speedup is reported only when CUDAExecutionProvider was actually selected."),
    }

    C.BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    out = C.BENCHMARK_DIR / f"cpu_vs_cuda_{args.model.stem}_{ts}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    C.ok(f"report -> {out}")

    if not agreement["agree"] and cuda_really_ran:
        C.warn("providers disagree on detections — investigate before trusting the speedup")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
