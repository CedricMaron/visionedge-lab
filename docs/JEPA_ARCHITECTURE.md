# JEPA Architecture

This document explains the Joint-Embedding Predictive Architecture (JEPA) slice of
VisionEdge Lab: the core principle, the context/target encoders and why the target uses an
EMA, why prediction happens in *representation* space, how I-JEPA / V-JEPA / V-JEPA 2
differ, the implementation-maturity levels, collapse monitoring, and linear probing.

> **Scope honesty.** This repo contains a **faithful, lightweight reimplementation of the
> published I-JEPA architecture at educational / CPU / ViT-tiny scale** (embedding
> dimension 16, 32×32 inputs, depth-2 transformer). It is **not** Meta's I-JEPA/V-JEPA
> system, is not trained at their scale, and makes no claim to their representation
> quality. The value here is a correct, inspectable implementation you can read and run on
> a laptop — not state-of-the-art numbers.

## 1. The JEPA principle

Most self-supervised vision methods are either **generative** (predict masked *pixels*,
e.g. MAE) or **contrastive** (pull augmented views together, push negatives apart). JEPA
takes a third path: predict the **representation** of one part of the input from another
part, in an abstract latent space, and never reconstruct pixels.

The intuition (LeCun, *A Path Towards Autonomous Machine Intelligence*, 2022,
<https://openreview.net/forum?id=BZ5a1r-kVsf>): pixel prediction wastes capacity modelling
unpredictable detail (exact textures, noise), while a good world model should predict the
*abstract* consequences and ignore what is inherently unpredictable. Predicting in
representation space lets the model discard nuisance detail by construction.

## 2. Context encoder, target encoder, predictor

The implementation (`backend/app/jepa/`) has three modules, all sharing one tiny-ViT
config `ViTConfig` (`image_size=32, patch_size=8, embed_dim=16, depth=2, num_heads=2,
predictor_dim=16, predictor_depth=1`, giving a 4×4 = 16-patch grid):

- **`ContextEncoder`** (`context_encoder.py`) — the *online* encoder. Patchify via a
  `Conv2d`, add a learned positional embedding, run a small `TransformerEncoder`,
  LayerNorm. Its `forward(x, keep_indices=None)` can encode **only a subset of patches**
  (the "context" patches) — the key I-JEPA move.
- **`TargetEncoder`** (`target_encoder.py`) — wraps a `ContextEncoder` whose parameters are
  frozen (`requires_grad_(False)`) and encodes under `torch.no_grad()`. It produces the
  *targets* the predictor must match.
- **`Predictor`** (`predictor.py`) — takes the encoded context tokens plus the *positions*
  of the masked target patches, inserts a learned `mask_token` at those positions, runs its
  own small transformer, and outputs a predicted embedding **per target patch**, projected
  back to `embed_dim`. Signature: `forward(context_tokens, context_indices,
  target_indices) -> (B, K_target, embed_dim)`.

Flow (spatial / I-JEPA case, `image_trainer.py::train_step`):

```
image ──► block-mask ──► context patches ──► ContextEncoder ─┐
                          target patches ──► TargetEncoder(EMA, no-grad) ─► target embeddings
                                                             │                       │
                    Predictor(context_tokens, ctx_idx, tgt_idx) ─► predicted embeddings
                                                             │                       │
                                       SmoothL1Loss(predicted, target.detach()) ◄────┘
```

Only the context encoder and predictor receive gradients. The target encoder is updated
separately (next section).

## 3. Why an EMA target, and why it must be stop-gradient

If the target encoder were trained by the same loss, the network could **collapse**: map
every input to the same constant vector, making the prediction loss trivially zero while
the representations become useless. JEPA prevents this by making the target a **slowly
moving average** of the online encoder rather than a directly optimized copy.

`ema.py` implements `target = momentum * target + (1 - momentum) * online` (default
`momentum = 0.996`), applied in `torch.no_grad()`, with buffers hard-copied and target
params kept non-trainable (`TargetEncoder.ema_update`). Two properties matter:

1. **Stop-gradient**: no gradient flows into the target, so the model cannot cheat by
   degenerating the targets.
2. **Slow update**: the target lags the online encoder, giving a stable, self-distilled
   objective. This asymmetry (predict a slowly-updated teacher, with stop-gradient) is the
   same anti-collapse mechanism used by BYOL and is central to I-JEPA.

## 4. Why predict in representation space, not pixels

Confirmed in the code: the predictor's output dimension equals `embed_dim` and the loss
(`SmoothL1Loss`) compares **predicted embeddings** to the **EMA target encoder's
embeddings** (`image_trainer.py`, `video_trainer.py`). There is no decoder and no pixel
reconstruction anywhere. Consequences:

- The model is free to be *invariant* to unpredictable pixel detail — it is scored on
  matching abstract features, not RGB values.
- The representation is what downstream tasks consume (linear probe, kNN, retrieval,
  anomaly scoring), so optimizing it directly is the point.

## 5. I-JEPA vs V-JEPA vs V-JEPA 2

| | Predicts | Domain | In this repo |
| --- | --- | --- | --- |
| **I-JEPA** | masked image *patches'* embeddings from visible patches | single images (spatial) | `LightweightIJepaTrainer` — reimplemented at tiny scale |
| **V-JEPA** | masked spatiotemporal region embeddings | video (space + time) | `LightweightVideoJepaTrainer` — temporal *future-frame* variant, tiny scale |
| **V-JEPA 2** | + action-conditioned prediction for planning | video + robot control | not reimplemented (conceptual reference only) |

- **I-JEPA** (Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding
  Predictive Architecture*, CVPR 2023, <https://arxiv.org/abs/2301.08243>): masks several
  target *blocks* in an image and predicts their representations from a single context
  block. Our `image_trainer.py` follows this exactly — multi-block masking
  (`masking.py`, `BlockMaskConfig`: 1 context block, 4 target blocks, disjoint) and
  block-embedding prediction.
- **V-JEPA** (Bardes et al., *Revisiting Feature Prediction for Learning Visual
  Representations from Video*, 2024, <https://arxiv.org/abs/2404.08471>): extends feature
  prediction to video with spatiotemporal masking. Our `video_trainer.py` implements a
  **temporal predictive-coding variant**: a shared per-frame encoder pools patch tokens to
  a frame embedding, a GRU (or MLP) aggregator over the past `context_frames` (default 3)
  predicts the **next frame's** EMA embedding (`SmoothL1Loss`). It is explicitly *not*
  Meta's V-JEPA masking scheme — it is the "predict the future in representation space"
  idea at tiny scale, which is exactly what the world-model experiment needs.
- **V-JEPA 2** (Meta, *V-JEPA 2: Self-Supervised Video Models Enable Understanding,
  Prediction and Planning*, 2025, <https://arxiv.org/abs/2506.09985>): adds
  action-conditioned prediction usable for robot planning. Referenced as the direction of
  travel; **not** implemented here.

Key implemented difference: **image = spatial block masking within one frame; video =
temporal future-embedding prediction across frames** (no masking, GRU/MLP aggregator).

## 6. Implementation-maturity levels

The code does not encode a formal "4 levels" enum; the following describes the **honest
maturity gradient** actually present, from fully implemented to opt-in/interface-only. Read
it as tiers of how much is really built, not as four separate trained systems.

1. **Level 1 — Spatial I-JEPA from scratch (implemented, runs on CPU).**
   `LightweightIJepaTrainer` + tiny ViT + multi-block masking + EMA target + representation
   prediction. Trainable on synthetic or CIFAR/STL-scale data.
2. **Level 2 — Temporal / video JEPA from scratch (implemented, runs on CPU).**
   `LightweightVideoJepaTrainer` predicts future-frame embeddings; this is what feeds the
   world-model / anomaly experiment.
3. **Level 3 — Pretrained I-JEPA/V-JEPA encoders (opt-in, not wired).**
   `representation/encoders.py` exposes a `load_pretrained_ijepa_encoder` seam that
   currently raises `NotImplementedError`, and a `load_timm_encoder` path (opt-in, needs
   `timm`). Intent: *wrap*, never reimplement, Meta's released weights.
4. **Level 4 — Action-conditioned world model / planning (conceptual).** The V-JEPA 2
   direction. **Interface-defined at most; implementation planned (later phase).**

## 7. Collapse monitoring

Because collapse is the primary failure mode, `collapse_monitor.py` computes diagnostics
from an `(N, D)` embedding matrix on every training step and exposes them via
`collapse_metrics(embeddings)`:

- `mean_std` / `min_dim_std` — per-dimension standard deviation; near 0 ⇒ collapse.
- `embedding_variance` — mean squared distance of rows from their centroid.
- `effective_rank` — `exp(H(p))` where `p` is the normalized singular-value spectrum
  (SVD entropy); ~1 ⇒ collapsed to a line, → `min(N, D)` ⇒ full-rank spread.
- `avg_pairwise_cosine` — mean cosine similarity over row pairs; ~1 ⇒ collapse.

`collapse_warning(metrics)` flags a problem when `mean_std < 1e-3` **or**
`avg_pairwise_cosine > 0.999`. These are surfaced in every `train_step` return
(`embed_std`, `avg_pairwise_cosine`, `effective_rank`, `collapse_warning`) so a collapsing
run is visible immediately rather than after a silent, useless training job.

## 8. Linear probing and downstream evaluation

A JEPA encoder is trained without labels, so its quality is judged by how *linearly usable*
its frozen features are. `jepa/evaluation.py` provides:

- `linear_probe(train_features, train_labels, test_features, test_labels, …)` — a NumPy
  logistic-regression probe on **frozen** features; returns train/test accuracy.
- `nearest_neighbor_accuracy(…, k=1, metric="cosine")` — kNN accuracy (no training).
- `retrieval_recall_at_k(query, gallery, …, k=5)` — cosine-similarity Recall@k.

The convention is standard: freeze the encoder, train only a linear head (or use kNN), and
report accuracy — a high linear-probe score means the self-supervised objective produced
*semantically organized* features. See `MULTIMODAL_EVALUATION.md` for how these fit the
broader evaluation story (including anomaly AUROC, which lives in the temporal slice).

## References

- LeCun, *A Path Towards Autonomous Machine Intelligence*, 2022.
  <https://openreview.net/forum?id=BZ5a1r-kVsf>
- Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive
  Architecture* (I-JEPA), CVPR 2023. <https://arxiv.org/abs/2301.08243>
- Bardes et al., *Revisiting Feature Prediction for Learning Visual Representations from
  Video* (V-JEPA), 2024. <https://arxiv.org/abs/2404.08471>
- Assran et al. / Meta AI, *V-JEPA 2: Self-Supervised Video Models Enable Understanding,
  Prediction and Planning*, 2025. <https://arxiv.org/abs/2506.09985>
- Grill et al., *Bootstrap Your Own Latent* (BYOL, the EMA-target anti-collapse
  mechanism), 2020. <https://arxiv.org/abs/2006.07733>
