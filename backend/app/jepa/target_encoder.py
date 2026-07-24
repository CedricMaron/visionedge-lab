"""EMA-updated target encoder for the lightweight I-JEPA reimplementation.

The target encoder shares the exact architecture of the :class:`ContextEncoder` but is
**never trained by gradients**. Its parameters are updated as an exponential moving
average of the online context encoder (see :mod:`app.jepa.ema`). Its forward pass runs
under ``torch.no_grad`` and its parameters have ``requires_grad = False``.

Educational scale; a faithful lightweight reimplementation of the published I-JEPA
architecture, not Meta's V-JEPA/V-JEPA2.
"""
from __future__ import annotations

from app.jepa.context_encoder import (
    _TORCH_AVAILABLE,
    ContextEncoder,
    ViTConfig,
    _Base,
    _require_torch,
)
from app.jepa.ema import EMA


class TargetEncoder(_Base):
    """A frozen, EMA-tracked twin of :class:`ContextEncoder`.

    Use :meth:`copy_from` once at init to mirror an online encoder, then
    :meth:`ema_update` each step. :meth:`encode` computes target tokens with no grad.
    """

    def __init__(self, config: ViTConfig) -> None:
        _require_torch()
        super().__init__()
        self.encoder = ContextEncoder(config)
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def copy_from(self, online: ContextEncoder) -> None:
        """Hard-copy weights from an online encoder (used at initialization)."""
        import torch

        with torch.no_grad():
            for t, o in zip(self.encoder.parameters(), online.parameters(), strict=False):
                t.copy_(o)
                t.requires_grad_(False)
            for t, o in zip(self.encoder.buffers(), online.buffers(), strict=False):
                t.copy_(o)

    def ema_update(self, online: ContextEncoder, momentum: float) -> None:
        """Advance the target toward ``online`` by one EMA step."""
        EMA(momentum).update_modules(self.encoder, online)

    def encode(self, x, keep_indices=None):
        """No-grad forward returning target tokens ``(B, N or K, D)``."""
        import torch

        with torch.no_grad():
            return self.encoder(x, keep_indices=keep_indices)

    # Keep an nn.Module-style forward for convenience; still no grad.
    def forward(self, x, keep_indices=None):  # noqa: D401
        return self.encode(x, keep_indices=keep_indices)


__all__ = ["TargetEncoder", "ViTConfig", "_TORCH_AVAILABLE"]
