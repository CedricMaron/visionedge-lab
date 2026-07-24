"""LightweightIJepaTrainer: one self-supervised training step on images.

Wires together the tiny context encoder, the EMA target encoder, the predictor, the
block-mask generator and a smooth-L1 representation loss into a single ``train_step``.
It runs on CPU with minuscule tensors so it is fully testable.

This is a *faithful lightweight reimplementation of the published I-JEPA architecture at
an educational scale*. It is NOT Meta's V-JEPA/V-JEPA2 and ships no pretrained weights.

Real datasets: CIFAR-10 / STL-10 loaders would attach at :meth:`synthetic_batch` (via
``torchvision.datasets``). They are intentionally opt-in and not imported here, so the
module has no torchvision dependency and never downloads anything at import time.
``torch`` itself is imported lazily inside methods.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.core.logging import get_logger
from app.jepa.collapse_monitor import collapse_metrics, collapse_warning
from app.jepa.context_encoder import ViTConfig
from app.jepa.masking import BlockMaskConfig, generate_block_masks

logger = get_logger("jepa.image_trainer")


@dataclass
class ImageTrainConfig:
    """Hyper-parameters for the image trainer (all tiny by default)."""

    vit: ViTConfig = field(default_factory=ViTConfig)
    mask: BlockMaskConfig = field(default_factory=BlockMaskConfig)
    learning_rate: float = 1e-3
    ema_momentum: float = 0.996
    seed: int = 0
    device: str = "cpu"


class LightweightIJepaTrainer:
    """Trainer holding the online encoder, EMA target encoder and predictor."""

    def __init__(self, config: ImageTrainConfig | None = None) -> None:
        import torch  # lazy

        from app.jepa.context_encoder import ContextEncoder
        from app.jepa.predictor import Predictor
        from app.jepa.target_encoder import TargetEncoder

        self.config = config or ImageTrainConfig()
        torch.manual_seed(self.config.seed)
        self._rng = np.random.default_rng(self.config.seed)

        self.device = torch.device(self.config.device)
        self.context_encoder = ContextEncoder(self.config.vit).to(self.device)
        self.predictor = Predictor(self.config.vit).to(self.device)
        self.target_encoder = TargetEncoder(self.config.vit).to(self.device)
        self.target_encoder.copy_from(self.context_encoder)

        params = list(self.context_encoder.parameters()) + list(self.predictor.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=self.config.learning_rate)
        self.loss_fn = torch.nn.SmoothL1Loss()
        self._step = 0

    def synthetic_batch(self, batch_size: int = 2):
        """A random image batch ``(B, C, H, W)`` on the trainer's device.

        This is the seam where a real ``torchvision`` CIFAR-10 / STL-10 DataLoader would
        be plugged in (opt-in). Kept synthetic so tests never touch the network.
        """
        import torch

        g = torch.Generator().manual_seed(self.config.seed + self._step)
        vit = self.config.vit
        return torch.rand(
            batch_size, vit.in_channels, vit.image_size, vit.image_size, generator=g
        ).to(self.device)

    def train_step(self, batch) -> dict[str, float]:
        """Run one I-JEPA training step on an image batch.

        Args:
            batch: Float tensor ``(B, C, H, W)`` in roughly [0, 1].

        Returns:
            A metrics dict: ``loss``, ``embed_std`` (mean per-dim std of target
            embeddings), ``avg_pairwise_cosine``, ``effective_rank``, ``num_context``,
            ``num_target``, and ``collapse_warning`` (str or ``None``).
        """
        import torch

        batch = batch.to(self.device)
        vit = self.config.vit
        grid = vit.grid_size

        masks = generate_block_masks(grid, grid, self.config.mask, self._rng)
        ctx_idx = torch.as_tensor(masks.context_indices(), dtype=torch.long, device=self.device)
        tgt_idx = torch.as_tensor(masks.target_indices(), dtype=torch.long, device=self.device)

        # Targets from the EMA encoder (no grad).
        target_tokens_all = self.target_encoder.encode(batch)  # (B, N, D)
        target_tokens = target_tokens_all.index_select(1, tgt_idx)  # (B, Ktgt, D)

        # Context path (with grad).
        context_tokens = self.context_encoder(batch, keep_indices=ctx_idx)  # (B, Kctx, D)
        predicted = self.predictor(context_tokens, ctx_idx, tgt_idx)  # (B, Ktgt, D)

        loss = self.loss_fn(predicted, target_tokens.detach())

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        # EMA step for the target encoder.
        self.target_encoder.ema_update(self.context_encoder, self.config.ema_momentum)
        self._step += 1

        # Diagnostics from the (detached) target embeddings.
        with torch.no_grad():
            flat = target_tokens_all.reshape(-1, vit.embed_dim).cpu().numpy()
        metrics = collapse_metrics(flat)
        warning = collapse_warning(metrics)

        return {
            "loss": float(loss.item()),
            "embed_std": metrics["mean_std"],
            "avg_pairwise_cosine": metrics["avg_pairwise_cosine"],
            "effective_rank": metrics["effective_rank"],
            "num_context": int(ctx_idx.numel()),
            "num_target": int(tgt_idx.numel()),
            "collapse_warning": warning,
            "step": self._step,
        }
