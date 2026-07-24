"""Image preprocessing for YOLO-family models: letterbox resize + normalization.

Kept dependency-light (NumPy + OpenCV) so it runs without PyTorch.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxMeta:
    """Geometry needed to map model-space boxes back to the original image."""

    scale: float          # resize ratio applied to the original image
    pad_x: float          # left padding added (pixels, model space)
    pad_y: float          # top padding added (pixels, model space)
    orig_w: int
    orig_h: int


def letterbox(image: np.ndarray, size: int = 640, color: int = 114) -> tuple[np.ndarray, LetterboxMeta]:
    """Resize preserving aspect ratio and pad to a square ``size``x``size``.

    Returns the padded image (HWC, uint8, BGR) and the geometry metadata.
    """
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((size, size, 3), color, dtype=np.uint8)
    pad_x = (size - nw) / 2.0
    pad_y = (size - nh) / 2.0
    top, left = int(round(pad_y - 0.1)), int(round(pad_x - 0.1))
    canvas[top : top + nh, left : left + nw] = resized
    return canvas, LetterboxMeta(scale=scale, pad_x=left, pad_y=top, orig_w=w, orig_h=h)


def to_model_input(padded_bgr: np.ndarray) -> np.ndarray:
    """Convert a letterboxed BGR uint8 image to a normalized NCHW float32 tensor."""
    rgb = cv2.cvtColor(padded_bgr, cv2.COLOR_BGR2RGB)
    chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return chw[None, ...]  # add batch dim


def scale_boxes(boxes_xyxy: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    """Map boxes from letterboxed model space back to original pixel coordinates."""
    out = boxes_xyxy.copy().astype(np.float32)
    out[:, [0, 2]] -= meta.pad_x
    out[:, [1, 3]] -= meta.pad_y
    out /= meta.scale
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, meta.orig_w)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, meta.orig_h)
    return out
