#!/usr/bin/env python3
"""Quantize an ONNX detector to FP16 or INT8, then verify + report output agreement.

FP16:
    Uses ``onnxconverter_common.float16.convert_float_to_float16`` when available,
    otherwise the onnxruntime transformers ``float16`` tool, otherwise a clear install
    error. FP16 gives a real speed/size win mainly on GPU / FP16-capable runtimes; on a
    pure-CPU ORT build it mostly just halves file size (compute may upcast).

INT8:
    Static quantization (``onnxruntime.quantization.quantize_static``) using a
    CalibrationDataReader that letterboxes images from --calibration-dir with the SAME
    preprocessing as inference (app.inference.preprocess). Or dynamic (--dynamic), which
    needs no calibration data.

    HONEST NOTE: INT8 quality depends heavily on representative calibration data. Poor or
    too-few calibration images degrade detection quality. INT8 execution also depends on
    the target runtime/hardware supporting int8 kernels; this script produces a valid
    quantized graph and verifies it loads + runs, but does not claim a speedup on a
    runtime that lacks int8 support.

Examples:
    python scripts/quantize_onnx.py --input models/yolov8n.onnx --precision fp16 --output /tmp/fp16.onnx
    python scripts/quantize_onnx.py --input models/yolov8n.onnx --precision int8 --dynamic --output /tmp/int8d.onnx
    python scripts/quantize_onnx.py --input models/yolov8n.onnx --precision int8 --calibration-dir calibration
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _common as C

C.bootstrap_path()


def _quantize_fp16(input_path: Path, output: Path) -> str:
    import onnx

    model = onnx.load(str(input_path))
    # Preferred: onnxconverter-common
    try:
        from onnxconverter_common import float16

        fp16_model = float16.convert_float_to_float16(model, keep_io_types=True)
        onnx.save(fp16_model, str(output))
        return "onnxconverter_common.float16"
    except Exception:
        pass
    # Fallback: onnxruntime transformers float16 tool
    try:
        from onnxruntime.transformers.float16 import convert_float_to_float16

        fp16_model = convert_float_to_float16(model, keep_io_types=True)
        onnx.save(fp16_model, str(output))
        return "onnxruntime.transformers.float16"
    except Exception:
        pass
    C.die(
        "no FP16 conversion tool available. Install one into the venv:\n"
        "  pip install onnxconverter-common\n"
        "(or ensure onnxruntime ships the transformers float16 tool)"
    )
    return ""  # unreachable


def _build_calibration_reader(calib_dir: Path, input_name: str, size: int, limit: int):
    """A CalibrationDataReader yielding letterboxed tensors from real images."""
    import cv2
    import numpy as np
    from app.inference.preprocess import letterbox, to_model_input
    from onnxruntime.quantization import CalibrationDataReader

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(p for p in calib_dir.rglob("*") if p.suffix.lower() in exts)
    if not images:
        C.die(f"no calibration images found under {calib_dir} "
              "(supported: jpg/png/bmp/webp). Run scripts/calibrate.py to prepare a set, "
              "or use --dynamic for calibration-free INT8.")
    images = images[:limit]
    C.info(f"calibration set: {len(images)} images from {calib_dir}")

    class _Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self._it = iter(images)

        def get_next(self):
            for path in self._it:
                img = cv2.imread(str(path))
                if img is None:
                    continue
                padded, _ = letterbox(img, size)
                tensor = to_model_input(padded).astype(np.float32)
                return {input_name: tensor}
            return None

    return _Reader(), len(images)


def _quantize_int8(input_path: Path, output: Path, dynamic: bool,
                   calib_dir: Path | None, size: int, samples: int) -> dict:
    from onnxruntime.quantization import QuantType, quantize_dynamic, quantize_static

    meta: dict = {"mode": "dynamic" if dynamic else "static"}
    if dynamic:
        quantize_dynamic(
            model_input=str(input_path), model_output=str(output),
            weight_type=QuantType.QInt8,
        )
        meta["calibration_images"] = 0
        return meta

    if calib_dir is None:
        C.die("static INT8 needs --calibration-dir with representative images "
              "(or pass --dynamic for calibration-free INT8).")
    if not calib_dir.exists():
        C.die(f"calibration dir not found: {calib_dir}")

    import onnxruntime as ort

    sess = ort.InferenceSession(str(input_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    reader, n = _build_calibration_reader(calib_dir, input_name, size, samples)
    quantize_static(
        model_input=str(input_path), model_output=str(output),
        calibration_data_reader=reader,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
    )
    meta["calibration_images"] = n
    meta["calibration_dir"] = str(calib_dir)
    return meta


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", required=True, type=Path, help="input .onnx (fp32) model")
    p.add_argument("--precision", required=True, choices=["fp16", "int8"])
    p.add_argument("--dynamic", action="store_true",
                   help="INT8 only: dynamic quantization (no calibration data needed)")
    p.add_argument("--calibration-dir", type=Path, default=None,
                   help="INT8 static: directory of representative images")
    p.add_argument("--samples", type=int, default=100,
                   help="max calibration images to use (default 100)")
    p.add_argument("--size", type=int, default=640, help="model input size")
    p.add_argument("--output", type=Path, default=None,
                   help="output path (default <input>.<precision>.onnx)")
    p.add_argument("--reference", type=Path, default=C.MODELS_DIR / "yolov8n.pt",
                   help="reference .pt for output-agreement (optional)")
    p.add_argument("--image", type=Path, default=C.BENCHMARK_DIR / "sample_bus.jpg",
                   help="image for output-agreement check")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.input.exists():
        C.die(f"input model not found: {args.input}")
    try:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
    except Exception:
        C.die("onnx/onnxruntime not installed; pip install -r backend/requirements/base.txt")

    output = (args.output or args.input.with_suffix(f".{args.precision}.onnx")).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    C.info(f"quantizing {args.input} -> {args.precision}")
    quant_meta: dict = {}
    if args.precision == "fp16":
        tool = _quantize_fp16(args.input, output)
        quant_meta["tool"] = tool
        C.ok(f"FP16 via {tool}")
    else:
        quant_meta = _quantize_int8(
            args.input, output, args.dynamic, args.calibration_dir, args.size, args.samples
        )
        C.ok(f"INT8 ({quant_meta['mode']}) done")

    C.ok(f"wrote {output} ({C.human_size(C.file_size(output))})")

    # --- verification: loads + runs + finite ---
    try:
        run = C.ort_run_zero(output)
        io = C.onnx_io_info(output)
    except Exception as exc:  # noqa: BLE001
        C.die(f"quantized model failed to load/run in onnxruntime: {exc}")
    if not run["all_finite"]:
        C.die("verification failed: quantized model produced non-finite output")
    C.ok(f"verified: output_shapes={run['output_shapes']} finite=True")

    # --- output agreement vs reference (best-effort) ---
    agreement = None
    try:
        from validate_onnx import run_agreement

        if args.image.exists():
            agreement = run_agreement(
                output, args.reference, args.image, conf=0.25, iou=0.45, size=args.size
            )
            from validate_onnx import _print_report
            _print_report(agreement)
        else:
            C.warn(f"agreement image not found ({args.image}); skipping agreement check")
    except Exception as exc:  # noqa: BLE001
        C.warn(f"agreement check skipped: {exc}")

    metadata = {
        "kind": "onnx_quantize",
        "created_utc": C.now_iso(),
        "precision": args.precision,
        "quantization": quant_meta,
        "source": str(args.input),
        "source_sha256": C.sha256_file(args.input),
        "source_size_bytes": C.file_size(args.input),
        "file": output.name,
        "size_bytes": C.file_size(output),
        "sha256": C.sha256_file(output),
        "opset": io["opset"],
        "inputs": io["inputs"],
        "outputs": io["outputs"],
        "verification": {
            "loaded_in_onnxruntime": True,
            "zero_input_all_finite": run["all_finite"],
            "output_shapes": run["output_shapes"],
        },
        "output_agreement": agreement,
        "notes": (
            "INT8 quality depends on representative calibration data; FP16/INT8 speedups "
            "depend on runtime/hardware support. This build verifies the graph loads and "
            "runs with finite output and reports agreement vs the FP32 reference."
        ),
        "tooling": C.runtime_versions(),
    }
    sidecar = C.write_sidecar(output, metadata)
    C.ok(f"metadata sidecar -> {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
