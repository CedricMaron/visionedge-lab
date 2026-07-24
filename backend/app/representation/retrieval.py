"""Retrieval helpers over embeddings and the :class:`EmbeddingStore`.

Cosine nearest-neighbour search and recall@k evaluation. These wrap the store or operate
directly on numpy matrices, keeping the math in one place.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from app.representation.embedding_store import EmbeddingStore


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, eps, None)


def cosine_nearest(
    query: np.ndarray, matrix: np.ndarray, k: int = 5
) -> list[tuple[int, float]]:
    """Return ``(index, similarity)`` for the top-``k`` rows of ``matrix``."""
    if matrix.shape[0] == 0:
        return []
    q = _l2_normalize(np.asarray(query, dtype=np.float64).reshape(-1))
    m = _l2_normalize(np.asarray(matrix, dtype=np.float64))
    sims = m @ q
    k = min(k, matrix.shape[0])
    order = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in order]


def nearest_in_store(
    store: EmbeddingStore, query: np.ndarray, k: int = 5
) -> list[tuple[str, float, dict[str, Any]]]:
    """Convenience wrapper: nearest records in an :class:`EmbeddingStore`."""
    return store.nearest(query, k=k)


def recall_at_k(
    query_matrix: np.ndarray,
    query_labels: Sequence,
    gallery_matrix: np.ndarray,
    gallery_labels: Sequence,
    k: int = 5,
) -> float:
    """Recall@k using cosine similarity.

    Fraction of queries whose top-``k`` gallery neighbours contain at least one item
    with the same label. Provide a gallery distinct from the queries to avoid trivial
    self-matches.
    """
    q = _l2_normalize(np.asarray(query_matrix, dtype=np.float64))
    g = _l2_normalize(np.asarray(gallery_matrix, dtype=np.float64))
    ql = np.asarray(list(query_labels))
    gl = np.asarray(list(gallery_labels))
    if g.shape[0] == 0 or q.shape[0] == 0:
        return 0.0
    sims = q @ g.T
    k = min(k, g.shape[0])
    topk = np.argsort(-sims, axis=1)[:, :k]
    hits = [(gl[topk[i]] == ql[i]).any() for i in range(q.shape[0])]
    return float(np.mean(hits))
