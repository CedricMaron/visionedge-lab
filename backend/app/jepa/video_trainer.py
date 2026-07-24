"""Compact temporal JEPA trainer: predict a future frame's embedding from the past.

Architecture (educational scale, faithful in spirit to video JEPA / predictive coding,
but NOT Meta's V-JEPA/V-JEPA2):

    past frames --(shared frame encoder)--> per-frame embeddings
                --(temporal aggregator: GRU or MLP)--> predicted future embedding
    future frame --(EMA target frame encoder, no grad)--> target embedding
    loss = smooth-L1 between predicted and target embedding

Two aggregators are provided: a ``GRU`` recurrent baseline and a ``mlp`` baseline that
flattens the past-embedding stack. ``torch`` is imported lazily inside methods; the
module always imports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


from app.core.logging import get_logger
from app.jepa.collapse_monitor import collapse_metrics, collapse_warning
from app.jepa.context_encoder import ViTConfig

logger = get_logger("jepa.video_trainer")


@dataclass
class VideoTrainConfig:
    """Hyper-parameters for the temporal trainer."""

    vit: ViTConfig = field(default_factory=ViTConfig)
    context_frames: int = 3
    aggregator: str = "gru"  # "gru" or "mlp"
    hidden_dim: int = 32
    learning_rate: float = 1e-3
    ema_momentum: float = 0.996
    seed: int = 0
    device: str = "cpu"


class LightweightVideoJepaTrainer:
    """Temporal predictor trainer over tiny synthetic clips."""

    def __init__(self, config: Optional[VideoTrainConfig] = None) -> None:
        import torch
        import torch.nn as nn

        from app.jepa.context_encoder import ContextEncoder
        from app.jepa.target_encoder import TargetEncoder

        self.config = config or VideoTrainConfig()
        if self.config.aggregator not in ("gru", "mlp"):
            raise ValueError("aggregator must be 'gru' or 'mlp'")
        torch.manual_seed(self.config.seed)

        self.device = torch.device(self.config.device)
        embed_dim = self.config.vit.embed_dim

        self.frame_encoder = ContextEncoder(self.config.vit).to(self.device)
        self.target_encoder = TargetEncoder(self.config.vit).to(self.device)
        self.target_encoder.copy_from(self.frame_encoder)

        if self.config.aggregator == "gru":
            self.aggregator = nn.GRU(
                input_size=embed_dim,
                hidden_size=self.config.hidden_dim,
                batch_first=True,
            ).to(self.device)
            self.head = nn.Linear(self.config.hidden_dim, embed_dim).to(self.device)
        else:  # mlp baseline over the flattened past-embedding stack
            self.aggregator = nn.Sequential(
                nn.Linear(embed_dim * self.config.context_frames, self.config.hidden_dim),
                nn.GELU(),
                nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            ).to(self.device)
            self.head = nn.Linear(self.config.hidden_dim, embed_dim).to(self.device)

        params = (
            list(self.frame_encoder.parameters())
            + list(self.aggregator.parameters())
            + list(self.head.parameters())
        )
        self.optimizer = torch.optim.AdamW(params, lr=self.config.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self._step = 0

    def _pool(self, tokens):
        """Mean-pool patch tokens ``(B, N, D)`` -> frame embedding ``(B, D)``."""
        return tokens.mean(dim=1)

    def synthetic_clip(self, batch_size: int = 2):
        """A random clip batch ``(B, T+1, C, H, W)`` (past frames + 1 future frame)."""
        import torch

        g = torch.Generator().manual_seed(self.config.seed + self._step)
        vit = self.config.vit
        t = self.config.context_frames + 1
        return torch.rand(
            batch_size, t, vit.in_channels, vit.image_size, vit.image_size, generator=g
        ).to(self.device)

    def train_step(self, clip) -> Dict[str, float]:
        """One temporal-prediction step.

        Args:
            clip: Tensor ``(B, T+1, C, H, W)``. The first ``context_frames`` frames are
                the past; the last frame is the prediction target.

        Returns:
            Metrics dict: ``loss``, ``embed_std``, ``avg_pairwise_cosine``,
            ``effective_rank``, ``aggregator``, ``collapse_warning``, ``step``.
        """
        import torch

        clip = clip.to(self.device)
        t_ctx = self.config.context_frames
        b = clip.shape[0]
        vit = self.config.vit

        past = clip[:, :t_ctx]  # (B, T, C, H, W)
        future = clip[:, t_ctx]  # (B, C, H, W)

        # Encode past frames (with grad).
        past_flat = past.reshape(b * t_ctx, vit.in_channels, vit.image_size, vit.image_size)
        past_emb = self._pool(self.frame_encoder(past_flat)).reshape(b, t_ctx, vit.embed_dim)

        if self.config.aggregator == "gru":
            out, _ = self.aggregator(past_emb)  # (B, T, H)
            summary = out[:, -1, :]  # last step
        else:
            summary = self.aggregator(past_emb.reshape(b, -1))  # (B, H)
        predicted = self.head(summary)  # (B, D)

        # Target: EMA encoder on the future frame (no grad).
        target_emb = self._pool(self.target_encoder.encode(future))  # (B, D)

        loss = self.loss_fn(predicted, target_emb.detach())

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        self.target_encoder.ema_update(self.frame_encoder, self.config.ema_momentum)
        self._step += 1

        with torch.no_grad():
            flat = target_emb.detach().cpu().numpy()
        metrics = collapse_metrics(flat)

        return {
            "loss": float(loss.item()),
            "embed_std": metrics["mean_std"],
            "avg_pairwise_cosine": metrics["avg_pairwise_cosine"],
            "effective_rank": metrics["effective_rank"],
            "aggregator": self.config.aggregator,
            "collapse_warning": collapse_warning(metrics),
            "step": self._step,
        }
