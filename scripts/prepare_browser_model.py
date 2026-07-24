#!/usr/bin/env python3
"""Prepare an ONNX detector for ONNX Runtime Web (browser inference, Phase 3).

This validates that a model is a reasonable candidate for in-browser inference with ONNX
Runtime Web (onnxruntime-web / WASM+WebGPU), optionally converts it to FP16 to shrink the
download, copies it into frontend/public/models/, and writes a manifest json the frontend
can read (name, size, sha256, input/output, opset).

Browser inference is NOT wired up in this build -- the detection pipeline runs server-side.
This script is the Phase-3 seam: it produces the asset + manifest so a future frontend can
load it. It changes nothing in the running system.

Example:
    python scripts/prepare_browser_model.py --input models/yolov8n.onnx
    python scripts/prepare_browser_model.py --input models/yolov8n.onnx --fp16
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import _common as C

C.bootstrap_path()

# ONNX Runtime Web supports opsets broadly; very new opsets may lack WASM kernels.
# 12-19 is a safe, widely supported window for detection models.
MIN_SAFE_OPSET = 10
MAX_SAFE_OPSET = 21


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", required=True, type=Path, help="input .onnx model")
    p.add_argument("--output", type=Path, default=C.FRONTEND_MODELS_DIR,
                   help=f"output directory (default {C.FRONTEND_MODELS_DIR})")
    p.add_argument("--fp16", action="store_true",
                   help="convert to FP16 to shrink the browser download")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.input.exists():
        C.die(f"input model not found: {args.input}")

    try:
        io = C.onnx_io_info(args.input)
    except Exception as exc:  # noqa: BLE001
        C.die(f"not a readable ONNX model: {exc}")

    opset = io["opset"]
    C.info(f"opset={opset} inputs={io['inputs']} outputs={io['outputs']}")
    if opset is None:
        C.warn("could not determine opset")
    elif opset < MIN_SAFE_OPSET or opset > MAX_SAFE_OPSET:
        C.warn(f"opset {opset} is outside the {MIN_SAFE_OPSET}-{MAX_SAFE_OPSET} window "
               "commonly supported by ONNX Runtime Web; some WASM kernels may be missing. "
               "Consider re-exporting with --opset 12.")
    else:
        C.ok(f"opset {opset} is within the ONNX Runtime Web compatibility window")

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    src = args.input
    precision = "fp32"
    tmp_fp16: Path | None = None
    if args.fp16:
        C.info("converting to FP16 for a smaller browser download ...")
        tmp_fp16 = out_dir / (args.input.stem + ".fp16.onnx")
        try:
            import onnx

            model = onnx.load(str(args.input))
            try:
                from onnxconverter_common import float16

                model = float16.convert_float_to_float16(model, keep_io_types=True)
            except Exception:
                from onnxruntime.transformers.float16 import convert_float_to_float16

                model = convert_float_to_float16(model, keep_io_types=True)
            onnx.save(model, str(tmp_fp16))
        except Exception as exc:  # noqa: BLE001
            C.die(f"FP16 conversion failed (install onnxconverter-common): {exc}")
        src = tmp_fp16
        precision = "fp16"

    dest = out_dir / (args.input.stem + (".fp16.onnx" if precision == "fp16" else ".onnx"))
    if src.resolve() != dest.resolve():
        shutil.copyfile(src, dest)
    if tmp_fp16 and tmp_fp16.exists() and tmp_fp16.resolve() != dest.resolve():
        tmp_fp16.unlink()
    C.ok(f"copied model -> {dest} ({C.human_size(C.file_size(dest))})")

    # --- verification: the copied asset still loads/runs ---
    try:
        run = C.ort_run_zero(dest)
    except Exception as exc:  # noqa: BLE001
        C.die(f"prepared browser model failed to load/run: {exc}")
    if not run["all_finite"]:
        C.die("verification failed: prepared model produced non-finite output")
    C.ok(f"verified: output_shapes={run['output_shapes']} finite=True")

    sha = C.sha256_file(dest)
    manifest = {
        "kind": "browser_model_manifest",
        "created_utc": C.now_iso(),
        "target_runtime": "onnxruntime-web",
        "status": "asset prepared; browser inference is Phase 3 (not wired up in this build)",
        "name": dest.name,
        "precision": precision,
        "size_bytes": C.file_size(dest),
        "sha256": sha,
        "opset": opset,
        "inputs": io["inputs"],
        "outputs": io["outputs"],
        "source": str(args.input),
        "source_sha256": C.sha256_file(args.input),
        "notes": (
            "Load with onnxruntime-web (WASM or WebGPU EP). Preprocess must match the "
            "server: letterbox to a square, RGB, NCHW float32, /255. Postprocess is the "
            "YOLOv8 decode + NMS (see backend/app/inference/postprocess.py)."
        ),
    }
    manifest_path = out_dir / (dest.stem + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    C.ok(f"manifest -> {manifest_path}")
    C.info("browser inference is a future phase; this only stages the asset + manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
