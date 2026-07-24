"""Tests for the representation subsystem (pure-numpy, fast)."""
from __future__ import annotations

import numpy as np
import pytest

from app.representation.embedding_store import EmbeddingStore
from app.representation.encoders import RandomProjectionEncoder, load_pretrained_ijepa_encoder
from app.representation.retrieval import cosine_nearest, recall_at_k
from app.representation.visualization import pca, pca_2d


# --- embedding store -------------------------------------------------------
def test_embedding_store_nearest_correct():
    store = EmbeddingStore()
    store.add("a", [1.0, 0.0, 0.0], {"label": "a"})
    store.add("b", [0.0, 1.0, 0.0], {"label": "b"})
    store.add("c", [0.9, 0.1, 0.0], {"label": "c"})

    res = store.nearest([1.0, 0.0, 0.0], k=2)
    ids = [r[0] for r in res]
    assert ids[0] == "a"  # exact match first
    assert "c" in ids  # closest neighbour second
    assert res[0][1] == pytest.approx(1.0, abs=1e-6)
    assert len(store) == 3


def test_embedding_store_overwrite_and_clear():
    store = EmbeddingStore()
    store.add("x", [1.0, 0.0])
    store.add("x", [0.0, 1.0])  # overwrite
    assert len(store) == 1
    assert store.get("x").vector.tolist() == [0.0, 1.0]
    store.clear()
    assert len(store) == 0


def test_embedding_store_npz_roundtrip(tmp_path):
    store = EmbeddingStore()
    store.add("a", [1.0, 2.0], {"k": 1})
    store.add("b", [3.0, 4.0], {"k": 2})
    path = str(tmp_path / "store.npz")
    store.save(path)
    loaded = EmbeddingStore.load(path)
    assert len(loaded) == 2
    assert loaded.get("b").vector.tolist() == [3.0, 4.0]
    assert loaded.get("a").meta["k"] == 1


# --- retrieval -------------------------------------------------------------
def test_cosine_nearest_and_recall():
    mat = np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]])
    res = cosine_nearest([1.0, 0.0], mat, k=2)
    assert res[0][0] == 0  # exact match index

    gallery = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    glabels = [0, 0, 1, 1]
    queries = np.array([[1.0, 0.05], [0.05, 1.0]])
    qlabels = [0, 1]
    r = recall_at_k(queries, qlabels, gallery, glabels, k=1)
    assert r == 1.0


# --- visualization (PCA) ---------------------------------------------------
def test_pca_reduces_dims_and_deterministic():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 8))
    p1 = pca_2d(x)
    p2 = pca_2d(x)
    assert p1.shape == (30, 2)
    assert np.allclose(p1, p2)  # deterministic (sign-fixed)
    # 3-D reduction shape.
    assert pca(x, 3).shape == (30, 3)


def test_pca_recovers_main_axis():
    # Data varies mostly along axis 0; PC1 should capture it.
    rng = np.random.default_rng(1)
    x = np.zeros((50, 3))
    x[:, 0] = rng.normal(scale=10.0, size=50)
    x[:, 1] = rng.normal(scale=0.1, size=50)
    proj = pca_2d(x)
    # First projected coordinate should have far more variance than the second.
    assert proj[:, 0].var() > 10 * proj[:, 1].var()


# --- encoders --------------------------------------------------------------
def test_random_projection_encoder_deterministic():
    enc = RandomProjectionEncoder()
    rng = np.random.default_rng(0)
    imgs = rng.integers(0, 256, size=(2, 16, 16, 3)).astype(np.uint8)
    e1 = enc.encode(imgs)
    e2 = RandomProjectionEncoder().encode(imgs)  # fresh instance, same seed
    assert e1.shape == (2, 16)
    assert np.allclose(e1, e2)  # deterministic across instances
    # L2-normalized.
    assert np.allclose(np.linalg.norm(e1, axis=1), 1.0, atol=1e-6)


def test_pretrained_hook_raises_clearly():
    with pytest.raises(NotImplementedError) as exc:
        load_pretrained_ijepa_encoder()
    assert "not bundled" in str(exc.value).lower()
