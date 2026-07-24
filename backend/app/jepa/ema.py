"""Exponential Moving Average (EMA) updates for the JEPA target encoder.

In I-JEPA the target encoder is not trained by gradient descent. Instead its weights
track the online (context) encoder as an exponential moving average::

    target = momentum * target + (1 - momentum) * online

The target encoder must therefore **never receive gradients**. For torch tensors the
update is performed inside ``torch.no_grad()`` and mutates the target tensors in place;
for numpy arrays a new list of arrays is returned (numpy has no autograd to guard).

The functions detect the input type at runtime so the same helper works for both a
pure-numpy prototype and the torch modules.
"""
from __future__ import annotations

from typing import List, Sequence


def _is_torch_tensor(x: object) -> bool:
    """True if ``x`` looks like a torch.Tensor, without importing torch eagerly."""
    return type(x).__module__.startswith("torch") and type(x).__name__ == "Tensor"


def ema_update(
    target_params: Sequence,
    online_params: Sequence,
    momentum: float,
) -> List:
    """Move ``target_params`` a fraction ``(1 - momentum)`` toward ``online_params``.

    Args:
        target_params: List of numpy arrays or torch tensors (the EMA / target side).
        online_params: Matching list of numpy arrays or torch tensors (trained side).
        momentum: EMA decay in [0, 1). Higher = slower target movement.

    Returns:
        The list of updated target params. For torch tensors these are the same
        objects (updated in place under ``no_grad``); for numpy arrays they are new
        arrays.

    Raises:
        ValueError: If the two sequences differ in length or momentum is out of range.
    """
    if not 0.0 <= momentum < 1.0:
        raise ValueError("momentum must be in [0, 1)")
    if len(target_params) != len(online_params):
        raise ValueError("target_params and online_params must have the same length")

    if target_params and _is_torch_tensor(target_params[0]):
        import torch

        updated: List = []
        with torch.no_grad():  # target must not accumulate gradients
            for t, o in zip(target_params, online_params):
                t.mul_(momentum).add_(o.detach(), alpha=1.0 - momentum)
                updated.append(t)
        return updated

    # numpy path
    import numpy as np

    updated_np: List = []
    for t, o in zip(target_params, online_params):
        updated_np.append(momentum * np.asarray(t) + (1.0 - momentum) * np.asarray(o))
    return updated_np


class EMA:
    """Small stateful EMA helper.

    Wraps a momentum value and applies :func:`ema_update` to (target, online) param
    lists. For torch ``nn.Module`` pairs use :meth:`update_modules`, which copies the
    online module's parameters and buffers into the target under ``no_grad`` and never
    lets the target require grad.
    """

    def __init__(self, momentum: float = 0.996) -> None:
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        self.momentum = float(momentum)

    def update(self, target_params: Sequence, online_params: Sequence) -> List:
        """Apply one EMA step to raw param lists."""
        return ema_update(target_params, online_params, self.momentum)

    def update_modules(self, target_module, online_module) -> None:
        """EMA-update a torch target ``nn.Module`` from an online ``nn.Module``.

        Both parameters and buffers are tracked. The target's parameters keep
        ``requires_grad = False``.
        """
        import torch

        with torch.no_grad():
            for t, o in zip(target_module.parameters(), online_module.parameters()):
                t.mul_(self.momentum).add_(o.detach(), alpha=1.0 - self.momentum)
                t.requires_grad_(False)
            for t, o in zip(target_module.buffers(), online_module.buffers()):
                t.copy_(o)
