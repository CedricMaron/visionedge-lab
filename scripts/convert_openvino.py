#!/usr/bin/env python3
"""Convert an ONNX detector to the OpenVINO IR format.

OpenVINO is an OPT-IN accelerator (best on Intel CPU / iGPU). It is NOT installed in the
default CPU stack, so this script exits with a clean, actionable message rather than a
traceback when ``openvino`` is missing. When it IS installed, the model is converted with
``openvino.convert_model`` and saved as IR (.xml + .bin), then verified by re-reading it.

Example:
    python scripts/convert_openvino.py --input models/yolov8n.onnx --precision fp16
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
    p.add_argument("--precision", choices=["fp16", "fp32"], default="fp16",
                   help="IR weight precision (default fp16)")
    p.add_argument("--output", type=Path, default=None,
                   help="output IR .xml path (default <input>.openvino.xml)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.input.exists():
        C.die(f"input model not found: {args.input}")

    try:
        import openvino as ov
    except Exception:
        C.die("OpenVINO not installed; pip install -r backend/requirements/openvino.txt")
        return 1  # unreachable, keeps type checkers happy

    output = (args.output or args.input.with_suffix(".openvino.xml")).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    C.info(f"converting {args.input} -> OpenVINO IR (precision={args.precision})")
    try:
        ov_model = ov.convert_model(str(args.input))
        compress = args.precision == "fp16"
        ov.save_model(ov_model, str(output), compress_to_fp16=compress)
    except Exception as exc:  # noqa: BLE001
        C.die(f"OpenVINO conversion failed: {exc}")

    bin_path = output.with_suffix(".bin")
    C.ok(f"wrote {output} + {bin_path.name}")

    # --- verification: re-read the IR and run a zero input ---
    try:
        import numpy as np

        core = ov.Core()
        model = core.read_model(str(output))
        compiled = core.compile_model(model, "CPU")
        inp = compiled.input(0)
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.partial_shape.get_max_shape()] \
            if hasattr(inp, "partial_shape") else [1, 3, 640, 640]
        out = compiled([np.zeros(shape, dtype=np.float32)])
        finite = all(bool(np.isfinite(v).all()) for v in out.values())
        shapes = [list(v.shape) for v in out.values()]
    except Exception as exc:  # noqa: BLE001
        C.die(f"produced IR failed to load/run: {exc}")
    if not finite:
        C.die("verification failed: IR produced non-finite output")
    C.ok(f"verified: output_shapes={shapes} finite=True")

    metadata = {
        "kind": "openvino_ir",
        "created_utc": C.now_iso(),
        "precision": args.precision,
        "source": str(args.input),
        "source_sha256": C.sha256_file(args.input),
        "file_xml": output.name,
        "file_bin": bin_path.name,
        "xml_size_bytes": C.file_size(output),
        "bin_size_bytes": C.file_size(bin_path) if bin_path.exists() else None,
        "verification": {"loaded": True, "output_shapes": shapes, "all_finite": finite},
        "tooling": {**C.runtime_versions(), "openvino": getattr(ov, "__version__", "unknown")},
    }
    sidecar = C.write_sidecar(output, metadata)
    C.ok(f"metadata sidecar -> {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
