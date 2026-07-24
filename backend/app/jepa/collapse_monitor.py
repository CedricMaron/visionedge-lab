"""Representation-collapse diagnostics for joint-embedding SSL.

Joint-embedding methods such as I-JEPA can *collapse*: the encoder maps every input to
(almost) the same vector, driving the loss to zero while learning nothing useful. These
pure-numpy helpers compute cheap diagnostics from an ``(N, D)`` embedding matrix so a
training loop (or a test) can detect the failure early.

None of these metrics is a guarantee of good representations -- they only detect the
*absence* of variation, which is a necessary (not sufficient) health signal.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def per_dimension_std(embeddings: np.ndarray) -> np.ndarray:
    """Standard deviation of each embedding dimension. Shape ``(D,)``."""
    emb = np.asarray(embeddings, dtype=np.float64)
    return emb.std(axis=0)


def mean_std(embeddings: np.ndarray) -> float:
    """Mean over dimensions of the per-dimension std. Near 0 suggests collapse."""
    return float(per_dimension_std(embeddings).mean())


def embedding_variance(embeddings: np.ndarray) -> float:
    """Total variance = mean squared distance of rows from the centroid."""
    emb = np.asarray(embeddings, dtype=np.float64)
    centered = emb - emb.mean(axis=0, keepdims=True)
    return float((centered ** 2).sum(axis=1).mean())


def effective_rank(embeddings: np.ndarray, eps: float = 1e-12) -> float:
    """Effective rank via the entropy of the normalized singular-value spectrum.

    Defined as ``exp(H(p))`` where ``p_i = s_i / sum(s)`` are the normalized singular
    values of the centered matrix and ``H`` is Shannon entropy (nats). A collapsed
    representation has effective rank ~1; a full-rank one approaches ``min(N, D)``.
    """
    emb = np.asarray(embeddings, dtype=np.float64)
    centered = emb - emb.mean(axis=0, keepdims=True)
    if centered.size == 0:
        return 0.0
    s = np.linalg.svd(centered, compute_uv=False)
    total = s.sum()
    if total <= eps:
        return 1.0
    p = s / total
    p = p[p > eps]
    entropy = -(p * np.log(p)).sum()
    return float(np.exp(entropy))


def average_pairwise_cosine(embeddings: np.ndarray, eps: float = 1e-12) -> float:
    """Mean cosine similarity over all distinct row pairs.

    Values close to 1.0 mean rows point the same way (a collapse signature). For a
    single row (or none) returns 1.0 by convention.
    """
    emb = np.asarray(embeddings, dtype=np.float64)
    n = emb.shape[0]
    if n < 2:
        return 1.0
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    normalized = emb / np.clip(norms, eps, None)
    sims = normalized @ normalized.T
    # Average of the strict upper triangle.
    iu = np.triu_indices(n, k=1)
    return float(sims[iu].mean())


def collapse_metrics(embeddings: np.ndarray) -> Dict[str, float]:
    """Bundle all scalar diagnostics into a dict."""
    return {
        "mean_std": mean_std(embeddings),
        "embedding_variance": embedding_variance(embeddings),
        "effective_rank": effective_rank(embeddings),
        "avg_pairwise_cosine": average_pairwise_cosine(embeddings),
        "min_dim_std": float(per_dimension_std(embeddings).min()) if np.asarray(embeddings).size else 0.0,
    }


def collapse_warning(metrics: Dict[str, float], std_threshold: float = 1e-3) -> Optional[str]:
    """Return a human-readable warning if metrics indicate collapse, else ``None``.

    Args:
        metrics: Output of :func:`collapse_metrics` (or a compatible dict).
        std_threshold: If ``mean_std`` falls below this, collapse is flagged.
    """
    ms = metrics.get("mean_std", None)
    if ms is not None and ms < std_threshold:
        return (
            f"Possible representation collapse: mean per-dim std {ms:.2e} < "
            f"threshold {std_threshold:.2e}. Effective rank "
            f"{metrics.get('effective_rank', float('nan')):.2f}, avg pairwise cosine "
            f"{metrics.get('avg_pairwise_cosine', float('nan')):.3f}."
        )
    cos = metrics.get("avg_pairwise_cosine", None)
    if cos is not None and cos > 0.999:
        return (
            f"Possible representation collapse: avg pairwise cosine {cos:.4f} ~ 1.0 "
            "(embeddings nearly identical in direction)."
        )
    return None
