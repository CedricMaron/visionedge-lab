#!/usr/bin/env python3
"""Simplify / optimize an ONNX graph with onnxslim, then verify it still runs.

onnxslim folds constants, fuses ops and prunes dead nodes. The optimized model must
load in onnxruntime and keep the same output shape as the input model.

Example:
    python scripts/simplify_onnx.py --input models/yolov8n.onnx --output /tmp/slim.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _common as C

C.bootstrap_path()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", required=True, type=Path, help="input .onnx model")
    p.add_argument("--output", type=Path, default=None,
                   help="output path (default <input>.slim.onnx)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.input.exists():
        C.die(f"input model not found: {args.input}")

    try:
        import onnxslim
    except Exception:
        C.die("onnxslim not installed; pip install -r backend/requirements/base.txt")

    output = (args.output or args.input.with_suffix(".slim.onnx")).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    # Record the pre-simplification output shapes for comparison.
    try:
        before = C.ort_run_zero(args.input)
    except Exception as exc:  # noqa: BLE001
        C.die(f"input model does not load/run in onnxruntime: {exc}")

    in_size = C.file_size(args.input)
    C.info(f"simplifying {args.input} ({C.human_size(in_size)})")

    try:
        # onnxslim.slim accepts a path and (optionally) an output path; returns a model.
        model = onnxslim.slim(str(args.input))
        import onnx
        onnx.save(model, str(output))
    except Exception as exc:  # noqa: BLE001
        C.die(f"onnxslim failed: {exc}")

    out_size = C.file_size(output)
    C.ok(f"wrote {output} ({C.human_size(out_size)})")

    # --- verification ---
    try:
        after = C.ort_run_zero(output)
        io = C.onnx_io_info(output)
    except Exception as exc:  # noqa: BLE001
        C.die(f"simplified model failed to load/run: {exc}")

    if not after["all_finite"]:
        C.die("verification failed: simplified model produced non-finite output")
    if before["output_shapes"] != after["output_shapes"]:
        C.die(f"verification failed: output shape changed "
              f"{before['output_shapes']} -> {after['output_shapes']}")
    C.ok(f"verified: output_shapes unchanged {after['output_shapes']} finite=True")

    delta = in_size - out_size
    pct = (delta / in_size * 100.0) if in_size else 0.0
    C.info(f"size: {C.human_size(in_size)} -> {C.human_size(out_size)} "
           f"({'-' if delta >= 0 else '+'}{abs(pct):.1f}%)")

    metadata = {
        "kind": "onnx_simplify",
        "created_utc": C.now_iso(),
        "tool": "onnxslim",
        "source": str(args.input),
        "source_sha256": C.sha256_file(args.input),
        "source_size_bytes": in_size,
        "file": output.name,
        "size_bytes": out_size,
        "sha256": C.sha256_file(output),
        "opset": io["opset"],
        "inputs": io["inputs"],
        "outputs": io["outputs"],
        "verification": {
            "output_shapes_unchanged": True,
            "output_shapes": after["output_shapes"],
            "zero_input_all_finite": after["all_finite"],
        },
        "tooling": C.runtime_versions(),
    }
    sidecar = C.write_sidecar(output, metadata)
    C.ok(f"metadata sidecar -> {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
