"""Tests for the lightweight I-JEPA reimplementation subsystem (CPU-only, tiny)."""
from __future__ import annotations

import numpy as np
import pytest

from app.jepa.collapse_monitor import collapse_metrics, collapse_warning, effective_rank
from app.jepa.context_encoder import ViTConfig
from app.jepa.ema import EMA, ema_update
from app.jepa.masking import BlockMaskConfig, generate_block_masks


# --- masking ---------------------------------------------------------------
def test_mask_shapes_and_disjoint():
    cfg = BlockMaskConfig(num_context_blocks=1, num_target_blocks=4)
    rng = np.random.default_rng(0)
    res = generate_block_masks(4, 4, cfg, rng)

    assert res.context_mask.shape == (16,)
    assert res.context_mask.dtype == bool
    assert len(res.target_masks) == 4
    # Context and targets never overlap.
    ctx = set(res.context_indices().tolist())
    tgt = set(res.target_indices().tolist())
    assert ctx.isdisjoint(tgt)
    # Both sides non-empty (scaffold guarantee).
    assert len(ctx) > 0 and len(tgt) > 0


def test_mask_determinism():
    cfg = BlockMaskConfig()
    a = generate_block_masks(6, 6, cfg, np.random.default_rng(42))
    b = generate_block_masks(6, 6, cfg, np.random.default_rng(42))
    assert np.array_equal(a.context_mask, b.context_mask)
    assert np.array_equal(a.target_indices(), b.target_indices())
    # Different seed generally differs.
    c = generate_block_masks(6, 6, cfg, np.random.default_rng(7))
    assert not np.array_equal(a.context_mask, c.context_mask)


# --- EMA -------------------------------------------------------------------
def test_ema_moves_target_toward_online_numpy():
    target = [np.zeros(4)]
    online = [np.ones(4)]
    updated = ema_update(target, online, momentum=0.9)
    # New target = 0.9*0 + 0.1*1 = 0.1 ; strictly between old target and online.
    assert np.allclose(updated[0], 0.1)


def test_ema_torch_moves_target_and_no_grad():
    torch = pytest.importorskip("torch")
    target = [torch.zeros(4, requires_grad=False)]
    online = [torch.ones(4, requires_grad=True)]
    ema_update(target, online, momentum=0.8)
    assert torch.allclose(target[0], torch.full((4,), 0.2))
    # Target must not track gradients.
    assert target[0].requires_grad is False
    assert target[0].grad_fn is None


def test_ema_class_updates_modules_no_grad():
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    online = nn.Linear(4, 4)
    target = nn.Linear(4, 4)
    for p in target.parameters():
        p.requires_grad_(False)
    # Make them different.
    with torch.no_grad():
        for p in online.parameters():
            p.add_(1.0)
    before = target.weight.detach().clone()
    EMA(0.5).update_modules(target, online)
    assert not torch.allclose(before, target.weight)  # moved
    assert target.weight.requires_grad is False


# --- collapse monitor ------------------------------------------------------
def test_collapse_flags_constant_matrix():
    const = np.ones((8, 4))  # every row identical -> collapsed
    metrics = collapse_metrics(const)
    assert metrics["mean_std"] < 1e-6
    warning = collapse_warning(metrics)
    assert warning is not None and "collapse" in warning.lower()


def test_collapse_ok_for_varied_matrix():
    rng = np.random.default_rng(0)
    varied = rng.normal(size=(32, 8))
    metrics = collapse_metrics(varied)
    assert metrics["mean_std"] > 0.1
    assert collapse_warning(metrics) is None
    # Effective rank of random data should be well above 1.
    assert effective_rank(varied) > 1.5


# --- trainer smoke ---------------------------------------------------------
def test_image_trainer_single_step_runs():
    pytest.importorskip("torch")
    from app.jepa.image_trainer import ImageTrainConfig, LightweightIJepaTrainer

    cfg = ImageTrainConfig(vit=ViTConfig(image_size=32, patch_size=8, embed_dim=16))
    trainer = LightweightIJepaTrainer(cfg)
    batch = trainer.synthetic_batch(batch_size=2)
    out = trainer.train_step(batch)
    assert "loss" in out and np.isfinite(out["loss"])
    assert out["num_context"] > 0 and out["num_target"] > 0


def test_video_trainer_step_gru_and_mlp():
    pytest.importorskip("torch")
    from app.jepa.video_trainer import LightweightVideoJepaTrainer, VideoTrainConfig

    for agg in ("gru", "mlp"):
        cfg = VideoTrainConfig(
            vit=ViTConfig(image_size=32, patch_size=8, embed_dim=16),
            context_frames=3,
            aggregator=agg,
        )
        trainer = LightweightVideoJepaTrainer(cfg)
        clip = trainer.synthetic_clip(batch_size=2)
        out = trainer.train_step(clip)
        assert np.isfinite(out["loss"])
        assert out["aggregator"] == agg


# --- evaluation & checkpoints ---------------------------------------------
def test_linear_probe_and_knn_separable():
    from app.jepa.evaluation import linear_probe, nearest_neighbor_accuracy

    rng = np.random.default_rng(0)
    a = rng.normal(loc=+3, size=(20, 4))
    b = rng.normal(loc=-3, size=(20, 4))
    x = np.vstack([a, b])
    y = np.array([0] * 20 + [1] * 20)
    res = linear_probe(x, y, x, y, num_classes=2, epochs=100)
    assert res["test_accuracy"] > 0.9
    assert nearest_neighbor_accuracy(x, y, x, y, k=1) > 0.9


def test_checkpoint_roundtrip(tmp_path):
    pytest.importorskip("torch")
    from app.jepa.checkpoints import load_checkpoint, save_training_checkpoint

    path = str(tmp_path / "ckpt.pt")
    save_training_checkpoint(
        path, {"enc": {"w": [1, 2, 3]}}, epoch=5, config={"lr": 0.1}, metrics={"loss": 0.3}
    )
    ckpt = load_checkpoint(path)
    assert ckpt["epoch"] == 5
    assert ckpt["metrics"]["loss"] == 0.3
    assert ckpt["model_states"]["enc"]["w"] == [1, 2, 3]
