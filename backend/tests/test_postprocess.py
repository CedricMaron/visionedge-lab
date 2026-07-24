"""Tests for the from-scratch YOLOv8 decoder and NMS."""
from __future__ import annotations

import numpy as np

from app.inference.postprocess import decode_yolov8, nms, xywh_to_xyxy


def test_xywh_to_xyxy_basic():
    boxes = np.array([[10.0, 20.0, 4.0, 6.0]])
    out = xywh_to_xyxy(boxes)
    assert out.tolist() == [[8.0, 17.0, 12.0, 23.0]]


def test_nms_suppresses_overlapping():
    boxes = np.array([
        [0, 0, 10, 10],
        [1, 1, 11, 11],   # heavily overlaps box 0
        [100, 100, 110, 110],  # separate
    ], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    keep = nms(boxes, scores, iou_threshold=0.5)
    assert 0 in keep and 2 in keep
    assert 1 not in keep  # suppressed by box 0


def test_nms_keeps_all_when_disjoint():
    boxes = np.array([[0, 0, 5, 5], [10, 10, 15, 15]], dtype=np.float32)
    scores = np.array([0.5, 0.6], dtype=np.float32)
    assert sorted(nms(boxes, scores, 0.5)) == [0, 1]


def _synthetic_output(num_classes=80, num_anchors=100):
    """Build a [1, 4+C, A] YOLOv8-style output with one strong detection."""
    arr = np.zeros((1, 4 + num_classes, num_anchors), dtype=np.float32)
    # anchor 0: box centered (50,50) size 20x20, class 5 high score
    arr[0, 0, 0] = 50; arr[0, 1, 0] = 50; arr[0, 2, 0] = 20; arr[0, 3, 0] = 20
    arr[0, 4 + 5, 0] = 0.9
    # anchor 1: near-duplicate of anchor 0 (should be NMS-suppressed), lower score
    arr[0, 0, 1] = 51; arr[0, 1, 1] = 51; arr[0, 2, 1] = 20; arr[0, 3, 1] = 20
    arr[0, 4 + 5, 1] = 0.7
    # anchor 2: different class 10 elsewhere
    arr[0, 0, 2] = 200; arr[0, 1, 2] = 200; arr[0, 2, 2] = 10; arr[0, 3, 2] = 10
    arr[0, 4 + 10, 2] = 0.8
    return arr


def test_decode_transposed_layout_and_nms():
    out = _synthetic_output()
    boxes, scores, class_ids = decode_yolov8(out, conf_threshold=0.25, iou_threshold=0.5)
    # duplicate anchor 1 suppressed -> two survivors (class 5 and class 10)
    assert sorted(class_ids.tolist()) == [5, 10]
    assert scores.max() >= 0.8


def test_decode_handles_both_axis_orders():
    out = _synthetic_output()
    boxes_a, _, _ = decode_yolov8(out, 0.25, 0.5)
    # transpose to [1, A, 4+C] and confirm identical decode
    out_t = np.transpose(out, (0, 2, 1))
    boxes_b, _, _ = decode_yolov8(out_t, 0.25, 0.5)
    assert boxes_a.shape == boxes_b.shape


def test_class_filter_applied_before_nms():
    out = _synthetic_output()
    boxes, scores, class_ids = decode_yolov8(out, 0.25, 0.5, allowed_class_ids={10})
    assert set(class_ids.tolist()) == {10}


def test_confidence_threshold_filters():
    out = _synthetic_output()
    boxes, scores, class_ids = decode_yolov8(out, conf_threshold=0.85, iou_threshold=0.5)
    # only the 0.9 (class5) and 0.8(class10)->filtered; 0.9 stays, 0.8 below 0.85
    assert set(class_ids.tolist()) == {5}


def test_empty_when_all_below_threshold():
    out = _synthetic_output()
    boxes, scores, class_ids = decode_yolov8(out, conf_threshold=0.99, iou_threshold=0.5)
    assert boxes.shape[0] == 0
