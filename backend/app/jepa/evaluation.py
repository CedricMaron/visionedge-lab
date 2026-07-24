"""Downstream evaluation of frozen representations.

Cheap, mostly-numpy evaluators used to judge whether a (frozen) encoder produces useful
embeddings:

* :func:`linear_probe` -- train a small multinomial logistic-regression head on top of
  frozen features and report test accuracy.
* :func:`nearest_neighbor_accuracy` -- 1-NN (or k-NN) classification accuracy.
* :func:`retrieval_recall_at_k` -- fraction of queries whose top-k gallery neighbours
  contain at least one same-label item.

These are diagnostics, not leaderboard numbers; nothing here is hardcoded.
"""
from __future__ import annotations

import numpy as np


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, eps, None)


def linear_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    num_classes: int | None = None,
    epochs: int = 200,
    lr: float = 0.1,
    weight_decay: float = 1e-4,
    seed: int = 0,
) -> dict[str, float]:
    """Fit a numpy softmax-regression probe on frozen features.

    Standardizes features using train statistics, runs full-batch gradient descent on
    the cross-entropy loss, and returns train/test accuracy. Deterministic given
    ``seed``.
    """
    rng = np.random.default_rng(seed)
    xtr = np.asarray(train_features, dtype=np.float64)
    xte = np.asarray(test_features, dtype=np.float64)
    ytr = np.asarray(train_labels).astype(int)
    yte = np.asarray(test_labels).astype(int)

    if num_classes is None:
        num_classes = int(max(ytr.max(), yte.max())) + 1

    mean = xtr.mean(axis=0, keepdims=True)
    std = xtr.std(axis=0, keepdims=True) + 1e-8
    xtr = (xtr - mean) / std
    xte = (xte - mean) / std

    n, d = xtr.shape
    w = rng.normal(scale=0.01, size=(d, num_classes))
    b = np.zeros(num_classes)
    y_onehot = np.eye(num_classes)[ytr]

    for _ in range(epochs):
        logits = xtr @ w + b
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        grad_logits = (probs - y_onehot) / n
        grad_w = xtr.T @ grad_logits + weight_decay * w
        grad_b = grad_logits.sum(axis=0)
        w -= lr * grad_w
        b -= lr * grad_b

    def _acc(x, y):
        pred = (x @ w + b).argmax(axis=1)
        return float((pred == y).mean())

    return {"train_accuracy": _acc(xtr, ytr), "test_accuracy": _acc(xte, yte)}


def nearest_neighbor_accuracy(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    k: int = 1,
    metric: str = "cosine",
) -> float:
    """k-NN classification accuracy of test items against the train set."""
    xtr = np.asarray(train_features, dtype=np.float64)
    xte = np.asarray(test_features, dtype=np.float64)
    ytr = np.asarray(train_labels).astype(int)
    yte = np.asarray(test_labels).astype(int)

    if metric == "cosine":
        a = _l2_normalize(xte)
        b = _l2_normalize(xtr)
        sim = a @ b.T  # higher = closer
        order = np.argsort(-sim, axis=1)
    elif metric == "euclidean":
        d2 = (
            (xte ** 2).sum(1, keepdims=True)
            - 2 * xte @ xtr.T
            + (xtr ** 2).sum(1, keepdims=True).T
        )
        order = np.argsort(d2, axis=1)
    else:
        raise ValueError("metric must be 'cosine' or 'euclidean'")

    k = min(k, xtr.shape[0])
    topk_labels = ytr[order[:, :k]]
    # Majority vote per row.
    preds = np.array(
        [np.bincount(row, minlength=int(ytr.max()) + 1).argmax() for row in topk_labels]
    )
    return float((preds == yte).mean())


def retrieval_recall_at_k(
    query_features: np.ndarray,
    query_labels: np.ndarray,
    gallery_features: np.ndarray,
    gallery_labels: np.ndarray,
    k: int = 5,
) -> float:
    """Recall@k: fraction of queries with a same-label item in the top-k gallery hits.

    Uses cosine similarity. If a query vector also appears in the gallery it is not
    excluded; supply a distinct gallery to avoid trivial self-matches.
    """
    q = _l2_normalize(np.asarray(query_features, dtype=np.float64))
    g = _l2_normalize(np.asarray(gallery_features, dtype=np.float64))
    ql = np.asarray(query_labels).astype(int)
    gl = np.asarray(gallery_labels).astype(int)

    sim = q @ g.T
    k = min(k, g.shape[0])
    topk = np.argsort(-sim, axis=1)[:, :k]
    hits = [(gl[topk[i]] == ql[i]).any() for i in range(q.shape[0])]
    return float(np.mean(hits))
