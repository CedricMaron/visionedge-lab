"""Image decode helpers shared by REST and WebSocket handlers."""
from __future__ import annotations

import base64

import cv2
import numpy as np

from app.core.errors import InferenceError


def decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode JPEG/PNG bytes to a BGR uint8 image. Raises InferenceError on failure."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise InferenceError("frame decode failed", user_message="Could not decode the image frame.")
    return img


def decode_data_url(data_url: str) -> np.ndarray:
    """Decode a base64 data URL (``data:image/jpeg;base64,...``) or bare base64."""
    payload = data_url.split(",", 1)[1] if data_url.startswith("data:") else data_url
    return decode_image_bytes(base64.b64decode(payload))
