#!/usr/bin/env python3
"""Export an Ultralytics YOLOv8 detector to ONNX, with a metadata sidecar + verification.

Uses ``ultralytics.YOLO(...).export(format="onnx")``. Nano reproduces the reference
``models/yolov8n.onnx``. Small/medium download the corresponding ultralytics weights
(yolov8s.pt ~22MB, yolov8m.pt ~50MB) on first use -- the download size is printed first.

Example:
    python scripts/export_onnx.py --model nano --size 640
    python scripts/export_onnx.py --model small --size 640 --output models/yolov8s.onnx
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import _common as C

C.bootstrap_path()

# model key -> (ultralytics weights name, approximate .pt download size)
MODEL_WEIGHTS = {
    "nano": ("yolov8n.pt", "~6.5 MB"),
    "small": ("yolov8s.pt", "~22 MB"),
    "medium": ("yolov8m.pt", "~50 MB"),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True, choices=sorted(MODEL_WEIGHTS),
                   help="which YOLOv8 size to export")
    p.add_argument("--size", type=int, default=640, help="square input size (default 640)")
    p.add_argument("--opset", type=int, default=12, help="ONNX opset version (default 12)")
    p.add_argument("--output", type=Path, default=None,
                   help="output .onnx path (default models/<weights>.onnx)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.size <= 0 or args.size % 32 != 0:
        C.die(f"--size must be a positive multiple of 32, got {args.size}")

    try:
        from ultralytics import YOLO
    except Exception:
        C.die("ultralytics not installed; pip install -r backend/requirements/base.txt "
              "(ultralytics is required to export)")

    weights_name, dl_size = MODEL_WEIGHTS[args.model]
    weights_path = C.MODELS_DIR / weights_name
    default_out = weights_name.replace(".pt", ".onnx")
    output = (args.output or (C.MODELS_DIR / default_out)).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if not weights_path.exists():
        C.info(f"weights {weights_name} not present locally -- ultralytics will download {dl_size}")
    else:
        C.info(f"using local weights {weights_path} ({C.human_size(C.file_size(weights_path))})")

    C.info(f"exporting {args.model} (weights={weights_name}) to ONNX "
           f"opset={args.opset} imgsz={args.size}")

    # Ultralytics writes the .onnx next to the .pt weights. Point it at models/.
    try:
        model = YOLO(str(weights_path))
        exported = model.export(format="onnx", imgsz=args.size, opset=args.opset, dynamic=False)
    except Exception as exc:  # noqa: BLE001
        C.die(f"ultralytics export failed: {exc}")

    exported_path = Path(exported)
    if exported_path.resolve() != output:
        shutil.move(str(exported_path), str(output))
    C.ok(f"wrote {output} ({C.human_size(C.file_size(output))})")

    # --- verification: introspect + run a zero input ---
    try:
        io = C.onnx_io_info(output)
        run = C.ort_run_zero(output)
    except Exception as exc:  # noqa: BLE001
        C.die(f"produced model failed to load/run in onnxruntime: {exc}")

    if not run["all_finite"]:
        C.die("verification failed: model produced non-finite output on a zero input")
    # YOLOv8 detection head output is rank-3: [1, 4+nc, anchors]
    if 3 not in run["output_ranks"]:
        C.warn(f"unexpected output rank(s) {run['output_ranks']} (expected a rank-3 tensor)")
    C.ok(f"verified: provider={run['provider']} output_shapes={run['output_shapes']} finite=True")

    sha = C.sha256_file(output)
    metadata = {
        "kind": "onnx_export",
        "created_utc": C.now_iso(),
        "model": args.model,
        "source_weights": weights_name,
        "source_weights_sha256": C.sha256_file(weights_path) if weights_path.exists() else None,
        "opset": io["opset"],
        "ir_version": io["ir_version"],
        "imgsz": args.size,
        "inputs": io["inputs"],
        "outputs": io["outputs"],
        "file": output.name,
        "size_bytes": C.file_size(output),
        "sha256": sha,
        "verification": {
            "loaded_in_onnxruntime": True,
            "zero_input_all_finite": run["all_finite"],
            "output_shapes": run["output_shapes"],
        },
        "tooling": C.runtime_versions(),
    }
    sidecar = C.write_sidecar(output, metadata)
    C.ok(f"metadata sidecar -> {sidecar}")
    C.info(f"sha256={sha}")

    # If reproducing nano, note whether it matches the committed reference checksum.
    ref = C.MODELS_DIR / "yolov8n.onnx"
    if args.model == "nano" and ref.exists() and output != ref:
        ref_sha = C.sha256_file(ref)
        if ref_sha == sha:
            C.ok("byte-identical to committed models/yolov8n.onnx")
        else:
            C.info("differs byte-wise from models/yolov8n.onnx (expected: export is "
                   "not always bit-reproducible across ultralytics/torch versions). "
                   "Run scripts/validate_onnx.py to confirm output agreement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
