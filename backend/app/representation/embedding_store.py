"""In-memory embedding store with optional npz persistence.

A minimal vector store: add ``(id, vector, meta)`` records, query cosine nearest
neighbours, enumerate all records, clear, and optionally save/load to a ``.npz`` file.
Backed by numpy; no external vector DB. Intended for modest collections (thousands of
vectors) held in RAM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class StoredEmbedding:
    """One stored vector with its id and free-form metadata."""

    id: str
    vector: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)


class EmbeddingStore:
    """Cosine-similarity in-memory embedding store."""

    def __init__(self, dim: Optional[int] = None) -> None:
        self.dim = dim
        self._ids: List[str] = []
        self._vectors: List[np.ndarray] = []
        self._meta: List[Dict[str, Any]] = []
        self._index: Dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, id: str, vec, meta: Optional[Dict[str, Any]] = None) -> None:
        """Add or overwrite the vector for ``id``.

        The first vector added fixes the store's dimensionality if it was not set.
        """
        v = np.asarray(vec, dtype=np.float64).reshape(-1)
        if self.dim is None:
            self.dim = v.shape[0]
        elif v.shape[0] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {v.shape[0]}")

        if id in self._index:
            i = self._index[id]
            self._vectors[i] = v
            self._meta[i] = dict(meta or {})
        else:
            self._index[id] = len(self._ids)
            self._ids.append(id)
            self._vectors.append(v)
            self._meta.append(dict(meta or {}))

    def get(self, id: str) -> Optional[StoredEmbedding]:
        """Return the record for ``id`` or ``None``."""
        i = self._index.get(id)
        if i is None:
            return None
        return StoredEmbedding(self._ids[i], self._vectors[i], self._meta[i])

    def all(self) -> List[StoredEmbedding]:
        """All stored records in insertion order."""
        return [
            StoredEmbedding(self._ids[i], self._vectors[i], self._meta[i])
            for i in range(len(self._ids))
        ]

    def matrix(self) -> np.ndarray:
        """Stacked ``(N, D)`` matrix of all vectors (empty ``(0, dim)`` if none)."""
        if not self._vectors:
            return np.zeros((0, self.dim or 0), dtype=np.float64)
        return np.vstack(self._vectors)

    def clear(self) -> None:
        """Remove all records (keeps the fixed ``dim``)."""
        self._ids.clear()
        self._vectors.clear()
        self._meta.clear()
        self._index.clear()

    def nearest(self, vec, k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Return up to ``k`` nearest records by cosine similarity.

        Returns a list of ``(id, similarity, meta)`` sorted by descending similarity.
        """
        if not self._vectors:
            return []
        q = np.asarray(vec, dtype=np.float64).reshape(-1)
        mat = self.matrix()
        qn = q / max(np.linalg.norm(q), 1e-12)
        mn = mat / np.clip(np.linalg.norm(mat, axis=1, keepdims=True), 1e-12, None)
        sims = mn @ qn
        k = min(k, len(self._ids))
        order = np.argsort(-sims)[:k]
        return [(self._ids[i], float(sims[i]), self._meta[i]) for i in order]

    # --- persistence ---------------------------------------------------------
    def save(self, path: str) -> str:
        """Persist to a ``.npz`` file (ids, vectors, metadata as JSON). Opt-in."""
        import json

        ids = np.array(self._ids, dtype=object)
        vectors = self.matrix()
        meta = np.array([json.dumps(m) for m in self._meta], dtype=object)
        np.savez(path, ids=ids, vectors=vectors, meta=meta, dim=np.array([self.dim or 0]))
        return path

    @classmethod
    def load(cls, path: str) -> EmbeddingStore:
        """Load a store previously written by :meth:`save`."""
        import json

        data = np.load(path, allow_pickle=True)
        dim = int(data["dim"][0]) or None
        store = cls(dim=dim)
        ids = list(data["ids"])
        vectors = data["vectors"]
        meta = list(data["meta"])
        for i, sid in enumerate(ids):
            store.add(str(sid), vectors[i], json.loads(meta[i]))
        return store
