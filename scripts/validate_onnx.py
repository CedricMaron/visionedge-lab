#!/usr/bin/env python3
"""Validate an ONNX detector by running it and comparing against a reference.

Runs the ONNX model through the project's real ``OnnxRuntimeBackend`` and, when the
PyTorch reference (``.pt`` via Ultralytics) is available, computes an OUTPUT-AGREEMENT
report: detection count, per-class multiset overlap, mean confidence delta, and mean IoU
of greedily matched boxes.

This is deliberately labelled AGREEMENT, not mAP: there is no labelled validation set
here, so it measures how closely the optimized runtime reproduces the FP32 reference on
sample images -- not absolute accuracy.

Example:
    python scripts/validate_onnx.py --input models/yolov8n.onnx
    python scripts/validate_onnx.py --input /tmp/fp16.onnx --reference models/yolov8n.pt
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import _common as C

C.bootstrap_path()

DEFAULT_IMAGE = C.BENCHMARK_DIR / "sample_bus.jpg"
DEFAULT_REFERENCE = C.MODELS_DIR / "yolov8n.pt"


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _match(dets_a: list, dets_b: list, iou_thr: float = 0.5):
    """Greedy per-class IoU matching. Returns (matches, unmatched_a, unmatched_b).

    matches is a list of (det_a, det_b, iou, conf_delta).
    """
    used_b: set[int] = set()
    matches = []
    for da in dets_a:
        best_j, best_iou = -1, iou_thr
        for j, db in enumerate(dets_b):
            if j in used_b or db.classId != da.classId:
                continue
            v = _iou((da.x1, da.y1, da.x2, da.y2), (db.x1, db.y1, db.x2, db.y2))
            if v >= best_iou:
                best_iou, best_j = v, j
        if best_j >= 0:
            db = dets_b[best_j]
            used_b.add(best_j)
            matches.append((da, db, best_iou, abs(da.confidence - db.confidence)))
    unmatched_a = len(dets_a) - len(matches)
    unmatched_b = len(dets_b) - len(used_b)
    return matches, unmatched_a, unmatched_b


def _load_onnx_backend(onnx_path: Path, size: int):
    from app.inference.onnx_backend import OnnxRuntimeBackend

    be = OnnxRuntimeBackend(model_path=onnx_path, model_id=onnx_path.stem, input_size=size)
    be.load()
    return be


def run_agreement(onnx_path: Path, reference: Path | None, image_path: Path,
                  conf: float, iou: float, size: int) -> dict:
    """Run ONNX (and reference if available) on one image; return an agreement dict.

    Importable by other scripts (quantize_onnx uses it).
    """
    import cv2

    img = cv2.imread(str(image_path))
    if img is None:
        C.die(f"could not decode image: {image_path}")

    onnx_be = _load_onnx_backend(onnx_path, size)
    onnx_dets = onnx_be.predict(img, conf, iou, None)
    onnx_be.close()

    result: dict = {
        "image": str(image_path),
        "onnx_model": str(onnx_path),
        "conf_threshold": conf,
        "iou_threshold": iou,
        "onnx_detection_count": len(onnx_dets),
        "onnx_class_multiset": dict(Counter(d.className for d in onnx_dets)),
        "onnx_mean_confidence": (
            round(sum(d.confidence for d in onnx_dets) / len(onnx_dets), 4)
            if onnx_dets else None
        ),
        "reference_available": False,
    }

    if reference is None or not reference.exists():
        result["reference_note"] = (
            "no PyTorch reference available; reported ONNX detections only "
            "(this is not an accuracy score)"
        )
        return result

    from app.inference.pytorch_backend import PyTorchBackend, torch_available

    if not torch_available():
        result["reference_note"] = (
            "reference .pt present but torch/ultralytics not importable; "
            "install backend/requirements/base.txt to enable comparison"
        )
        return result

    ref_be = PyTorchBackend(weights_path=reference, model_id=reference.stem, input_size=size)
    ref_be.load()
    ref_dets = ref_be.predict(img, conf, iou, None)
    ref_be.close()

    matches, unmatched_onnx, unmatched_ref = _match(onnx_dets, ref_dets, iou_thr=0.5)
    mean_iou = round(sum(m[2] for m in matches) / len(matches), 4) if matches else None
    mean_conf_delta = round(sum(m[3] for m in matches) / len(matches), 4) if matches else None

    onnx_ms = Counter(d.className for d in onnx_dets)
    ref_ms = Counter(d.className for d in ref_dets)
    multiset_agree = onnx_ms == ref_ms

    result.update({
        "reference_available": True,
        "reference_model": str(reference),
        "reference_detection_count": len(ref_dets),
        "reference_class_multiset": dict(ref_ms),
        "reference_mean_confidence": (
            round(sum(d.confidence for d in ref_dets) / len(ref_dets), 4)
            if ref_dets else None
        ),
        "agreement": {
            "detection_count_delta": len(onnx_dets) - len(ref_dets),
            "class_multiset_identical": multiset_agree,
            "matched_boxes": len(matches),
            "unmatched_onnx": unmatched_onnx,
            "unmatched_reference": unmatched_ref,
            "mean_iou_matched": mean_iou,
            "mean_confidence_delta_matched": mean_conf_delta,
        },
    })
    return result


def _print_report(r: dict) -> None:
    print("\n=== OUTPUT-AGREEMENT REPORT (not mAP; no labelled set) ===")
    print(f"image:        {r['image']}")
    print(f"onnx model:   {r['onnx_model']}")
    print(f"conf/iou:     {r['conf_threshold']} / {r['iou_threshold']}")
    print(f"onnx dets:    {r['onnx_detection_count']}  classes={r['onnx_class_multiset']}")
    print(f"onnx mean conf: {r['onnx_mean_confidence']}")
    if not r["reference_available"]:
        print(f"reference:    UNAVAILABLE -- {r.get('reference_note', '')}")
        print("=========================================================\n")
        return
    a = r["agreement"]
    print(f"ref  dets:    {r['reference_detection_count']}  classes={r['reference_class_multiset']}")
    print(f"ref mean conf:  {r['reference_mean_confidence']}")
    print("--- agreement ---")
    print(f"detection count delta:     {a['detection_count_delta']}")
    print(f"class multiset identical:  {a['class_multiset_identical']}")
    print(f"matched boxes:             {a['matched_boxes']} "
          f"(unmatched onnx={a['unmatched_onnx']}, ref={a['unmatched_reference']})")
    print(f"mean IoU (matched):        {a['mean_iou_matched']}")
    print(f"mean conf delta (matched): {a['mean_confidence_delta_matched']}")
    print("=========================================================\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", required=True, type=Path, help="ONNX model to validate")
    p.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE,
                   help=f"reference .pt weights (default {DEFAULT_REFERENCE})")
    p.add_argument("--image", type=Path, default=DEFAULT_IMAGE,
                   help=f"sample image (default {DEFAULT_IMAGE})")
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--size", type=int, default=640, help="model input size")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.input.exists():
        C.die(f"input model not found: {args.input}")
    if not args.image.exists():
        C.die(f"image not found: {args.image}")
    r = run_agreement(args.input, args.reference, args.image, args.conf, args.iou, args.size)
    _print_report(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
