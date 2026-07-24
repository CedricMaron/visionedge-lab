"""Visual encoders producing embeddings for the representation subsystem.

Defines the :class:`VisualEncoder` protocol and a dependency-light baseline encoder that
turns images into deterministic embeddings via a fixed random projection of a pooled
pixel signature. The baseline is meant for tests and smoke-checks, not accuracy.

Pretrained encoders (I-JEPA / timm ViT) are provided as **opt-in hooks** that raise a
clear "install/enable" message when their dependency is missing. Nothing is downloaded at
import time, and importing this module never requires torch or timm.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class VisualEncoder(Protocol):
    """Interface for anything that maps images to embedding vectors.

    Implementations expose an ``embed_dim`` and an ``encode`` method taking a batch of
    images ``(B, H, W, C)`` (or ``(B, C, H, W)``) and returning ``(B, embed_dim)``.
    """

    embed_dim: int

    def encode(self, images: np.ndarray) -> np.ndarray: ...


@dataclass
class BaselineEncoderConfig:
    embed_dim: int = 16
    pool_grid: int = 4  # images are average-pooled to pool_grid x pool_grid x C first
    seed: int = 0


class RandomProjectionEncoder:
    """Deterministic baseline encoder: pooled pixels -> fixed random projection.

    Steps: convert to float, average-pool each image to a ``pool_grid x pool_grid`` grid
    per channel, flatten, then project through a fixed (seeded) Gaussian matrix and L2-
    normalize. Identical inputs always yield identical embeddings, which the tests use.

    This captures coarse colour/spatial structure only -- it is a stand-in encoder, not a
    learned representation.
    """

    def __init__(self, config: BaselineEncoderConfig | None = None) -> None:
        self.config = config or BaselineEncoderConfig()
        self.embed_dim = self.config.embed_dim
        self._proj = None  # lazily built once the input feature size is known
        self._in_features = None

    def _ensure_projection(self, in_features: int) -> None:
        if self._proj is None or self._in_features != in_features:
            rng = np.random.default_rng(self.config.seed)
            self._proj = rng.normal(size=(in_features, self.embed_dim)) / np.sqrt(in_features)
            self._in_features = in_features

    @staticmethod
    def _to_bhwc(images: np.ndarray) -> np.ndarray:
        arr = np.asarray(images, dtype=np.float64)
        if arr.ndim == 3:  # single image (H, W, C) or (C, H, W)
            arr = arr[None]
        if arr.ndim != 4:
            raise ValueError("images must be (B,H,W,C) or (B,C,H,W)")
        # Heuristic: channels-first if axis 1 is small (<=4) and axis 3 is not.
        if arr.shape[1] <= 4 and arr.shape[3] > 4:
            arr = np.transpose(arr, (0, 2, 3, 1))
        return arr

    def _pool(self, img: np.ndarray) -> np.ndarray:
        """Average-pool one (H, W, C) image to (pool_grid, pool_grid, C)."""
        g = self.config.pool_grid
        h, w, c = img.shape
        hs = np.array_split(np.arange(h), g)
        ws = np.array_split(np.arange(w), g)
        out = np.zeros((g, g, c))
        for i, ri in enumerate(hs):
            for j, cj in enumerate(ws):
                out[i, j] = img[np.ix_(ri, cj)].mean(axis=(0, 1))
        return out

    def encode(self, images: np.ndarray) -> np.ndarray:
        """Encode a batch of images to ``(B, embed_dim)`` L2-normalized embeddings."""
        batch = self._to_bhwc(images)
        if batch.max() > 1.0:
            batch = batch / 255.0
        feats = np.stack([self._pool(img).reshape(-1) for img in batch], axis=0)
        self._ensure_projection(feats.shape[1])
        emb = feats @ self._proj
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / np.clip(norms, 1e-12, None)


def load_pretrained_ijepa_encoder(*args, **kwargs):
    """Opt-in hook for a pretrained I-JEPA encoder.

    Not implemented in this scaffold: obtaining real I-JEPA / V-JEPA weights requires a
    separate, explicit download of Meta's released checkpoints under their license. We do
    not bundle or fetch them. Raises with guidance rather than pretending.
    """
    raise NotImplementedError(
        "Pretrained I-JEPA/V-JEPA weights are not bundled. Download the official "
        "checkpoints from Meta under their license and wire them in here explicitly; "
        "this scaffold ships only the from-scratch lightweight reimplementation."
    )


def load_timm_encoder(model_name: str = "vit_tiny_patch16_224", pretrained: bool = True):
    """Opt-in hook for a timm ViT backbone as a :class:`VisualEncoder`.

    Raises:
        RuntimeError: with an install message when ``timm`` (or torch) is unavailable.
    """
    try:
        import timm  # type: ignore
        import torch  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "load_timm_encoder requires the optional 'timm' package (and torch). "
            "Install with `pip install timm` to use a pretrained ViT backbone. "
            "Note: pretrained=True will download weights on first use."
        ) from exc

    model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
    model.eval()
    return model
