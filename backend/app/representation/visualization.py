"""Dimensionality reduction for embedding visualization.

PCA to 2-D / 3-D is implemented from scratch via numpy's SVD so it works with **no
sklearn dependency**. Non-linear methods (t-SNE, UMAP) are offered as optional hooks that
raise a clear "install X" message when the library is unavailable -- we never silently
fall back or fabricate a layout.
"""
from __future__ import annotations


import numpy as np


def pca(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Project ``(N, D)`` embeddings to ``(N, n_components)`` via PCA (SVD).

    The data is mean-centered, then the top ``n_components`` right-singular vectors are
    used as the projection basis. Deterministic up to sign; a sign convention is applied
    (the largest-magnitude entry of each component is made positive) so repeated calls on
    the same data give identical output.

    Args:
        embeddings: Input matrix ``(N, D)``.
        n_components: Target dimensionality (e.g. 2 or 3), clamped to ``min(N, D)``.
    """
    x = np.asarray(embeddings, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("embeddings must be a 2-D (N, D) array")
    n, d = x.shape
    k = min(n_components, n, d)
    if k < 1:
        raise ValueError("need at least one sample and one feature")

    centered = x - x.mean(axis=0, keepdims=True)
    # SVD: centered = U S Vt ; principal directions are rows of Vt.
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:k]  # (k, D)

    # Deterministic sign: make the max-abs entry of each component positive.
    for i in range(k):
        j = int(np.argmax(np.abs(components[i])))
        if components[i, j] < 0:
            components[i] = -components[i]

    projected = centered @ components.T  # (N, k)
    # Pad with zeros if the caller asked for more components than available.
    if k < n_components:
        pad = np.zeros((n, n_components - k))
        projected = np.hstack([projected, pad])
    return projected


def pca_2d(embeddings: np.ndarray) -> np.ndarray:
    """Convenience: PCA to 2 dimensions."""
    return pca(embeddings, n_components=2)


def pca_3d(embeddings: np.ndarray) -> np.ndarray:
    """Convenience: PCA to 3 dimensions."""
    return pca(embeddings, n_components=3)


def explained_variance_ratio(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Fraction of total variance captured by each of the top ``n_components``."""
    x = np.asarray(embeddings, dtype=np.float64)
    centered = x - x.mean(axis=0, keepdims=True)
    s = np.linalg.svd(centered, compute_uv=False)
    var = s ** 2
    total = var.sum()
    if total <= 0:
        return np.zeros(min(n_components, len(var)))
    return (var / total)[:n_components]


def tsne(embeddings: np.ndarray, n_components: int = 2, **kwargs) -> np.ndarray:
    """t-SNE embedding via scikit-learn, if installed.

    Raises:
        RuntimeError: with an actionable install message when scikit-learn is absent.
    """
    try:
        from sklearn.manifold import TSNE  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional dep
        raise RuntimeError(
            "t-SNE requires scikit-learn, which is not installed. "
            "Install it with `pip install scikit-learn` to use tsne()."
        ) from exc
    return TSNE(n_components=n_components, **kwargs).fit_transform(
        np.asarray(embeddings, dtype=np.float64)
    )


def umap(embeddings: np.ndarray, n_components: int = 2, **kwargs) -> np.ndarray:
    """UMAP embedding via umap-learn, if installed.

    Raises:
        RuntimeError: with an actionable install message when umap-learn is absent.
    """
    try:
        import umap as umap_lib  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional dep
        raise RuntimeError(
            "UMAP requires umap-learn, which is not installed. "
            "Install it with `pip install umap-learn` to use umap()."
        ) from exc
    return umap_lib.UMAP(n_components=n_components, **kwargs).fit_transform(
        np.asarray(embeddings, dtype=np.float64)
    )
