"""I-JEPA predictor: context tokens -> predicted target embeddings.

Given the context encoder's output tokens (at their patch positions) and the flat
indices of the target patches, the predictor produces a prediction of the target
encoder's embedding at each target position. It follows the published I-JEPA design: a
learned mask token plus positional embeddings mark where predictions are needed, a small
transformer mixes context and mask tokens, and the target slots are read back out.

Faithful lightweight reimplementation, educational scale. Not V-JEPA/V-JEPA2. ``torch``
is guarded so the module always imports.
"""
from __future__ import annotations

from app.jepa.context_encoder import (
    _TORCH_AVAILABLE,
    ViTConfig,
    _Base,
    _require_torch,
)

if _TORCH_AVAILABLE:
    import torch
    import torch.nn as nn


class Predictor(_Base):
    """Small transformer predictor mapping context tokens -> target embeddings."""

    def __init__(self, config: ViTConfig) -> None:
        _require_torch()
        super().__init__()
        self.config = config
        pdim = config.predictor_dim
        self.in_proj = nn.Linear(config.embed_dim, pdim)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.num_patches, pdim))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pdim))
        nn.init.normal_(self.mask_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=pdim,
            nhead=config.num_heads,
            dim_feedforward=int(pdim * config.mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            layer, num_layers=config.predictor_depth, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(pdim)
        self.out_proj = nn.Linear(pdim, config.embed_dim)

    def forward(self, context_tokens, context_indices, target_indices):  # noqa: D401
        """Predict target embeddings.

        Args:
            context_tokens: ``(B, Kctx, embed_dim)`` context encoder output.
            context_indices: 1-D LongTensor of the flat patch indices for
                ``context_tokens`` (same order, shared across the batch).
            target_indices: 1-D LongTensor of flat target patch indices to predict.

        Returns:
            ``(B, Ktgt, embed_dim)`` predicted target embeddings.
        """
        b = context_tokens.shape[0]
        ctx_idx = context_indices.to(context_tokens.device).long()
        tgt_idx = target_indices.to(context_tokens.device).long()

        ctx = self.in_proj(context_tokens) + self.pos_embed.index_select(1, ctx_idx)

        mask_tokens = self.mask_token.expand(b, tgt_idx.shape[0], -1)
        mask_tokens = mask_tokens + self.pos_embed.index_select(1, tgt_idx)

        x = torch.cat([ctx, mask_tokens], dim=1)
        x = self.blocks(x)
        x = self.norm(x)
        pred = x[:, ctx.shape[1] :, :]  # target slots
        return self.out_proj(pred)


__all__ = ["Predictor", "ViTConfig", "_TORCH_AVAILABLE"]
