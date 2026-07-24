"""Checkpoint save/load for JEPA training runs.

Wraps ``torch.save`` / ``torch.load`` (guarded) to persist a dict containing model
state dicts, the training config, metrics and metadata (epoch, timestamp). ``torch`` is
imported lazily so the module always imports even without torch present.

Checkpoints are ordinary pickled dicts; only load ones you trust (``torch.load`` uses
Python pickle).
"""
from __future__ import annotations

import time
from dataclasses import asdict, is_dataclass
from typing import Any


def _config_to_dict(config: Any) -> Any:
    """Best-effort serialization of a (possibly nested) dataclass config."""
    if config is None:
        return None
    if is_dataclass(config) and not isinstance(config, type):
        return asdict(config)
    if isinstance(config, dict):
        return config
    return repr(config)


def build_checkpoint(
    model_states: dict[str, Any],
    epoch: int,
    config: Any = None,
    metrics: dict[str, float] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a checkpoint dict (does not touch disk).

    Args:
        model_states: Mapping name -> ``state_dict`` (e.g. ``{"context_encoder": sd}``).
        epoch: Epoch/step number.
        config: Training config (dataclass, dict or anything ``repr``-able).
        metrics: Latest metrics.
        extra: Any additional user payload.
    """
    return {
        "format_version": 1,
        "created_at": time.time(),
        "epoch": int(epoch),
        "config": _config_to_dict(config),
        "metrics": dict(metrics or {}),
        "model_states": model_states,
        "extra": dict(extra or {}),
    }


def save_checkpoint(path: str, checkpoint: dict[str, Any]) -> str:
    """Persist a checkpoint dict with ``torch.save``. Returns the path written."""
    import torch  # guarded lazy import

    torch.save(checkpoint, path)
    return path


def load_checkpoint(path: str, map_location: str = "cpu") -> dict[str, Any]:
    """Load a checkpoint dict with ``torch.load`` (defaults to CPU mapping)."""
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Older torch without the weights_only kwarg.
        return torch.load(path, map_location=map_location)


def save_training_checkpoint(
    path: str,
    model_states: dict[str, Any],
    epoch: int,
    config: Any = None,
    metrics: dict[str, float] | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Convenience: build + save in one call."""
    ckpt = build_checkpoint(model_states, epoch, config=config, metrics=metrics, extra=extra)
    return save_checkpoint(path, ckpt)
