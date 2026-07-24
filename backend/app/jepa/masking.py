"""Block masking for I-JEPA (faithful lightweight reimplementation, educational scale).

This implements the multi-block masking strategy described in the published I-JEPA
paper (Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding
Predictive Architecture"). It is a small, dependency-free (pure numpy) reimplementation
intended for teaching and CPU-scale experiments. It is *not* Meta's V-JEPA / V-JEPA2
and makes no claim of parity with those systems.

The generator produces, over a ``(grid_h, grid_w)`` patch grid:

* a single boolean *context* mask (which patches the context encoder may see), and
* a list of boolean *target* masks (rectangular blocks the predictor must predict).

Targets are removed from the context so context and targets never overlap, matching
the I-JEPA recipe. Given the same :class:`numpy.random.Generator` the output is fully
deterministic, which the tests rely on for reproducibility.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class BlockMaskConfig:
    """Configuration for :func:`generate_block_masks`.

    Attributes:
        num_context_blocks: How many large context blocks to sample and union.
        num_target_blocks: How many target blocks to sample.
        target_scale: (min, max) fraction of the grid area for each target block.
        context_scale: (min, max) fraction of the grid area for each context block.
        aspect_ratio: (min, max) aspect ratio (width/height) for target blocks.
        max_attempts: Retries when a sampled block would be degenerate.
    """

    num_context_blocks: int = 1
    num_target_blocks: int = 4
    target_scale: Tuple[float, float] = (0.15, 0.2)
    context_scale: Tuple[float, float] = (0.85, 1.0)
    aspect_ratio: Tuple[float, float] = (0.75, 1.5)
    max_attempts: int = 20


@dataclass
class MaskResult:
    """Result of a masking call.

    All masks are flat boolean arrays of length ``grid_h * grid_w`` where ``True``
    marks a selected patch. Patch index ``i`` corresponds to grid cell
    ``(i // grid_w, i % grid_w)`` (row-major).
    """

    grid_h: int
    grid_w: int
    context_mask: np.ndarray
    target_masks: List[np.ndarray] = field(default_factory=list)

    @property
    def num_patches(self) -> int:
        return self.grid_h * self.grid_w

    def context_indices(self) -> np.ndarray:
        """Flat indices of patches visible to the context encoder."""
        return np.flatnonzero(self.context_mask)

    def target_indices(self) -> np.ndarray:
        """Sorted unique flat indices covered by any target block."""
        if not self.target_masks:
            return np.array([], dtype=np.int64)
        union = np.zeros(self.num_patches, dtype=bool)
        for m in self.target_masks:
            union |= m
        return np.flatnonzero(union)


def _sample_block(
    grid_h: int,
    grid_w: int,
    scale: Tuple[float, float],
    aspect_ratio: Tuple[float, float],
    rng: np.random.Generator,
    max_attempts: int,
) -> np.ndarray:
    """Sample one rectangular block as a flat boolean mask.

    Returns a mask with at least one patch set. Falls back to a single centre
    patch if no valid rectangle is found within ``max_attempts``.
    """
    total = grid_h * grid_w
    for _ in range(max_attempts):
        s = float(rng.uniform(scale[0], scale[1]))
        area = max(1.0, s * total)
        ar = float(rng.uniform(aspect_ratio[0], aspect_ratio[1]))
        # area = h * w ; ar = w / h  ->  h = sqrt(area / ar)
        h = int(round(np.sqrt(area / ar)))
        w = int(round(np.sqrt(area * ar)))
        h = int(np.clip(h, 1, grid_h))
        w = int(np.clip(w, 1, grid_w))
        top = int(rng.integers(0, grid_h - h + 1))
        left = int(rng.integers(0, grid_w - w + 1))
        mask = np.zeros((grid_h, grid_w), dtype=bool)
        mask[top : top + h, left : left + w] = True
        if mask.any():
            return mask.reshape(-1)
    # Degenerate fallback: centre patch.
    mask = np.zeros((grid_h, grid_w), dtype=bool)
    mask[grid_h // 2, grid_w // 2] = True
    return mask.reshape(-1)


def generate_block_masks(
    grid_h: int,
    grid_w: int,
    config: BlockMaskConfig,
    rng: np.random.Generator,
) -> MaskResult:
    """Generate I-JEPA context + target masks over a ``(grid_h, grid_w)`` grid.

    Args:
        grid_h: Number of patch rows.
        grid_w: Number of patch columns.
        config: Sampling configuration.
        rng: A seeded ``numpy`` generator; identical generators produce identical
            masks (used by tests for determinism).

    Returns:
        A :class:`MaskResult`. The context mask has all target patches removed, so
        context and targets are disjoint. It is guaranteed to leave at least one
        context patch and at least one target patch (a scaffold-level guarantee so
        downstream training never sees an empty side).
    """
    if grid_h < 1 or grid_w < 1:
        raise ValueError("grid dimensions must be >= 1")

    total = grid_h * grid_w

    # 1) Sample target blocks.
    target_masks: List[np.ndarray] = []
    for _ in range(max(1, config.num_target_blocks)):
        target_masks.append(
            _sample_block(
                grid_h, grid_w, config.target_scale, config.aspect_ratio, rng, config.max_attempts
            )
        )

    target_union = np.zeros(total, dtype=bool)
    for m in target_masks:
        target_union |= m

    # 2) Sample context block(s) with (near) square aspect and subtract targets.
    context = np.zeros(total, dtype=bool)
    for _ in range(max(1, config.num_context_blocks)):
        context |= _sample_block(
            grid_h, grid_w, config.context_scale, (1.0, 1.0), rng, config.max_attempts
        )
    context &= ~target_union

    # 3) Guarantee non-empty sides (scaffold safety net, documented above).
    if not context.any():
        free = np.flatnonzero(~target_union)
        if free.size == 0:
            # Everything is a target; give the first patch back to context.
            first = int(np.flatnonzero(target_union)[0])
            target_union[first] = False
            target_masks = [m & target_union for m in target_masks]
            context[first] = True
        else:
            context[int(free[0])] = True

    if not target_union.any():
        # Put a single target on a patch that is not context, if possible.
        free = np.flatnonzero(~context)
        idx = int(free[0]) if free.size else 0
        m = np.zeros(total, dtype=bool)
        m[idx] = True
        target_masks = [m]

    return MaskResult(grid_h=grid_h, grid_w=grid_w, context_mask=context, target_masks=target_masks)
