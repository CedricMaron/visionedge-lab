"""Tiny ViT-style context encoder for the lightweight I-JEPA reimplementation.

This is a *faithful lightweight reimplementation of the published I-JEPA architecture,
at an educational / CPU scale* -- it is emphatically NOT Meta's V-JEPA or V-JEPA2 and
carries none of their pretrained weights or performance.

The context encoder is a small Vision Transformer: a patchifying convolution, learned
positional embeddings, and a couple of transformer blocks. It can optionally encode
only a subset of patches (the I-JEPA "context"), selected by flat patch indices.

``torch`` is imported at module load time but guarded: if the import fails the module
still imports and the classes raise a clear error only when instantiated.
"""
from __future__ import annotations

from dataclasses import dataclass

try:  # heavy import guarded so the module always imports
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
    _Base = nn.Module
except Exception:  # pragma: no cover - exercised only when torch is absent
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False
    _Base = object  # type: ignore[misc,assignment]


@dataclass
class ViTConfig:
    """Configuration for the tiny ViT encoder / predictor.

    Defaults are intentionally minuscule so a full train step runs in well under a
    second on CPU.
    """

    image_size: int = 32
    patch_size: int = 8
    in_channels: int = 3
    embed_dim: int = 16
    depth: int = 2
    num_heads: int = 2
    mlp_ratio: float = 2.0
    predictor_dim: int = 16
    predictor_depth: int = 1

    @property
    def grid_size(self) -> int:
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.grid_size * self.grid_size


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise RuntimeError(
            "torch is required for the JEPA neural modules but is not importable. "
            "Install a CPU build of torch to use ContextEncoder/TargetEncoder/Predictor."
        )


class PatchEmbed(_Base):
    """Conv-based patch embedding: (B, C, H, W) -> (B, num_patches, embed_dim)."""

    def __init__(self, config: ViTConfig) -> None:
        _require_torch()
        super().__init__()
        self.proj = nn.Conv2d(
            config.in_channels,
            config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def forward(self, x):  # noqa: D401
        x = self.proj(x)  # (B, D, gh, gw)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class ContextEncoder(_Base):
    """Tiny ViT patch encoder.

    ``forward(x, keep_indices=None)`` returns per-patch tokens ``(B, N, D)``. When
    ``keep_indices`` (1-D LongTensor of flat patch indices) is provided, only those
    patches are embedded and passed through the transformer, returning
    ``(B, len(keep_indices), D)`` -- this is the I-JEPA "context only" path. The same
    ``keep_indices`` is applied to every item in the batch (a documented scaffold
    simplification; per-sample masks would batch via padding).
    """

    def __init__(self, config: ViTConfig) -> None:
        _require_torch()
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbed(config)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.num_patches, config.embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=int(config.embed_dim * config.mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            layer, num_layers=config.depth, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(config.embed_dim)

    def forward(self, x, keep_indices=None):  # noqa: D401
        tokens = self.patch_embed(x) + self.pos_embed  # (B, N, D)
        if keep_indices is not None:
            idx = keep_indices.to(tokens.device).long()
            tokens = tokens.index_select(1, idx)  # (B, K, D)
        tokens = self.blocks(tokens)
        tokens = self.norm(tokens)
        return tokens
