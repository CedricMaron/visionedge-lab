#!/usr/bin/env python3
"""Build / inspect an INT8 calibration set from a directory of images.

Validates that images decode, applies the SAME letterbox preprocessing used at inference
(app.inference.preprocess), reports the per-channel mean/std and shape stats of the
preprocessed tensors, and writes a calibration manifest that scripts/quantize_onnx.py can
reference for INT8 static quantization.

WHY THIS MATTERS: INT8 static quantization estimates activation ranges from these images.
If the calibration set is not REPRESENTATIVE of real deployment scenes (lighting, scale,
classes, clutter), INT8 detection quality degrades. Prefer a few hundred varied real frames
over many near-duplicates. This tool only inspects/prepares data; it never fabricates stats.

Example:
    python scripts/calibrate.py --images calibration --samples 100
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _common as C

C.bootstrap_path()

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--images", required=True, type=Path, help="directory of calibration images")
    p.add_argument("--samples", type=int, default=100, help="max images to use (default 100)")
    p.add_argument("--size", type=int, default=640, help="letterbox size (default 640)")
    p.add_argument("--output", type=Path, default=None,
                   help="manifest path (default <images>/calibration_manifest.json)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.images.exists() or not args.images.is_dir():
        C.die(f"images directory not found: {args.images}")

    import cv2
    import numpy as np
    from app.inference.preprocess import letterbox, to_model_input

    all_imgs = sorted(p for p in args.images.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not all_imgs:
        C.die(f"no images found under {args.images} (supported: {sorted(IMG_EXTS)}). "
              "Populate this directory with representative deployment frames.")

    chosen = all_imgs[:args.samples]
    C.info(f"found {len(all_imgs)} images; using {len(chosen)} (size={args.size})")

    ok_files: list[str] = []
    bad_files: list[str] = []
    # Running per-channel (RGB) mean/std over the normalized [0,1] NCHW tensors.
    ch_sum = np.zeros(3, dtype=np.float64)
    ch_sqsum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for path in chosen:
        img = cv2.imread(str(path))
        if img is None:
            bad_files.append(str(path))
            continue
        padded, _ = letterbox(img, args.size)
        tensor = to_model_input(padded)  # (1,3,H,W) float32 in [0,1], RGB
        t = tensor[0]
        ch_sum += t.reshape(3, -1).sum(axis=1)
        ch_sqsum += (t.reshape(3, -1) ** 2).sum(axis=1)
        pixel_count += t.shape[1] * t.shape[2]
        ok_files.append(str(path))

    if not ok_files:
        C.die("no images decoded successfully; check the files.")
    if bad_files:
        C.warn(f"{len(bad_files)} file(s) failed to decode and were skipped")

    mean = ch_sum / pixel_count
    var = ch_sqsum / pixel_count - mean ** 2
    std = np.sqrt(np.clip(var, 0.0, None))

    print("\n=== CALIBRATION SET INSPECTION ===")
    print(f"images used:     {len(ok_files)} (skipped {len(bad_files)})")
    print(f"tensor shape:    [1, 3, {args.size}, {args.size}] float32 in [0,1], RGB, letterboxed")
    print(f"per-channel mean (R,G,B): {[round(float(x), 4) for x in mean]}")
    print(f"per-channel std  (R,G,B): {[round(float(x), 4) for x in std]}")
    if len(ok_files) < 50:
        print("NOTE: < 50 images. INT8 calibration is more reliable with a few hundred "
              "varied, representative frames.")
    print("==================================\n")

    manifest = {
        "kind": "calibration_manifest",
        "created_utc": C.now_iso(),
        "images_dir": str(args.images),
        "size": args.size,
        "preprocess": "letterbox + RGB + NCHW float32 /255 (app.inference.preprocess)",
        "num_images_used": len(ok_files),
        "num_images_skipped": len(bad_files),
        "per_channel_mean_rgb": [float(x) for x in mean],
        "per_channel_std_rgb": [float(x) for x in std],
        "files": ok_files,
        "skipped_files": bad_files,
        "guidance": (
            "Use representative deployment frames. Non-representative or near-duplicate "
            "images degrade INT8 quality. Feed this dir to "
            "scripts/quantize_onnx.py --precision int8 --calibration-dir <dir>."
        ),
    }
    out = (args.output or (args.images / "calibration_manifest.json")).resolve()
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    C.ok(f"manifest -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
