"""From-scratch YOLOv8 detection-head decoding and NMS (pure NumPy).

This deliberately avoids the ultralytics runtime: the raw ONNX output tensor is
decoded here so the inference path depends only on NumPy. The YOLOv8 detection head
emits, per image, a tensor of shape ``[1, 4 + num_classes, num_anchors]`` (commonly
``[1, 84, 8400]`` for COCO at 640). The 4 box channels are ``(cx, cy, w, h)`` in model
input pixels; class channels are already activated (0..1).
"""
from __future__ import annotations

import numpy as np


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert center-form (cx,cy,w,h) to corner-form (x1,y1,x2,y2)."""
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return out


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy non-maximum suppression. Returns kept indices (into the inputs).

    ``boxes`` is (N,4) in xyxy. Standard IoU-based greedy suppression.
    """
    if boxes.shape[0] == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = rest[iou <= iou_threshold]
    return keep


def _normalize_output(output: np.ndarray) -> np.ndarray:
    """Return a (num_anchors, 4+num_classes) array regardless of transpose layout."""
    arr = output
    if arr.ndim == 3:
        arr = arr[0]  # drop batch -> (C, A) or (A, C)
    if arr.ndim != 2:
        raise ValueError(f"Unexpected YOLO output rank: {output.shape}")
    # Detection head has 4 + num_classes channels (>= 5, typically 84). The anchor
    # dimension (e.g. 8400) is far larger, so the smaller axis is the channel axis.
    if arr.shape[0] < arr.shape[1]:
        arr = arr.T  # (C, A) -> (A, C)
    return arr


def decode_yolov8(
    output: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
    allowed_class_ids: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a raw YOLOv8 output tensor into boxes/scores/class ids (model space).

    Class filtering is applied *before* NMS when ``allowed_class_ids`` is provided,
    i.e. as early as the runtime reasonably allows.

    Returns ``(boxes_xyxy, scores, class_ids)`` in model-input pixel coordinates.
    """
    arr = _normalize_output(output)
    boxes_xywh = arr[:, :4]
    class_scores = arr[:, 4:]

    class_ids = class_scores.argmax(axis=1)
    scores = class_scores.max(axis=1)

    mask = scores >= conf_threshold
    if allowed_class_ids is not None:
        allowed = np.fromiter(allowed_class_ids, dtype=np.int64) if allowed_class_ids else np.empty(0, np.int64)
        mask &= np.isin(class_ids, allowed)

    boxes_xywh = boxes_xywh[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]
    if boxes_xywh.shape[0] == 0:
        return np.empty((0, 4), np.float32), np.empty((0,), np.float32), np.empty((0,), np.int64)

    boxes = xywh_to_xyxy(boxes_xywh)

    # Class-aware NMS: suppress within each class independently.
    keep_all: list[int] = []
    for cid in np.unique(class_ids):
        idx = np.where(class_ids == cid)[0]
        kept = nms(boxes[idx], scores[idx], iou_threshold)
        keep_all.extend(idx[kept].tolist())

    keep_all_arr = np.array(sorted(keep_all), dtype=np.int64)
    return boxes[keep_all_arr], scores[keep_all_arr], class_ids[keep_all_arr]
