# World-Model Experiment: Anomaly from Prediction Error

This document describes the temporal experiment in VisionEdge Lab: using a JEPA-style
future-embedding predictor as a small **world model**, and turning its *prediction error*
into an attention signal ("something unexpected is happening here"). It also states,
plainly, what that signal is **not**.

> **The one caveat that matters most.** High prediction error means the model was
> **surprised**, not that anything **dangerous** or **bad** happened. Surprise and danger
> are different things. This pipeline produces an *attention cue* for a downstream check
> (e.g. asking the VLM to describe the frame), never a safety verdict.

## 1. The idea

A world model predicts what happens next. If you can predict the near future of a scene in
representation space, then a **large gap between prediction and reality** is exactly the
definition of *novelty* — the current moment is not explained by the recent past.

We already have a future-embedding predictor: the `LightweightVideoJepaTrainer`
(`backend/app/jepa/video_trainer.py`, see `JEPA_ARCHITECTURE.md` §5). It encodes the last
`context_frames` frames, aggregates them (GRU/MLP), and predicts the **next frame's**
embedding as produced by the EMA target encoder. The training loss for that prediction —
`SmoothL1` distance between predicted and actual next-frame embedding — is, at inference
time, a per-frame **prediction-error / surprise** score.

```
frames ... f(t-3) f(t-2) f(t-1) ─► aggregator ─► predicted embedding ê(t)
                                   f(t) ─► EMA encoder ─► actual embedding e(t)
                         surprise(t) = SmoothL1( ê(t), e(t) )   # small = expected, large = novel
```

## 2. From surprise to an anomaly score (calibration)

Raw prediction error is not interpretable on its own: its scale depends on the scene, the
encoder, and lighting. `backend/app/temporal/anomaly.py` (`AnomalyScorer`) converts it into
a calibrated, bounded signal:

1. **Calibrate on "normal".** `calibrate(errors)` takes a window of prediction errors from
   an assumed-normal period and fixes a baseline `mean` and `std` (std floored to avoid
   division by zero). This is what makes the score scene-specific rather than a magic global
   threshold.
2. **Z-score.** `z_score(error) = (error - mean) / std` measures how many standard
   deviations the current surprise is above the calibrated baseline.
3. **Score.** `score(error, update=False)` returns `{"z", "anomaly", "is_anomaly"}` where
   `anomaly = 1 / (1 + exp(-(z - z_threshold)))` (default `z_threshold = 3.0`). Note this
   squashes only the **positive** deviation: being *more* predictable than usual is not an
   anomaly.
4. **Optional online adaptation.** With `update=True` the new error joins a rolling window
   (`window = 128`) and the baseline is recomputed, so slow scene drift (dusk falling, a
   camera slowly warming) does not read as a permanent anomaly.

`rolling_stats()` exposes the current `{mean, std, count}` for inspection.

## 3. Selecting frames worth predicting

Running the predictor and (especially) a VLM on every frame is wasteful. The temporal slice
picks informative frames first:

- `temporal/frame_buffer.py` — a bounded, **RAM-only** ring buffer (`capacity = 64`,
  privacy by construction: nothing is written to disk).
- `temporal/scene_change.py` — `mad_score` / `histogram_score` and a stateful
  `SceneChangeDetector` (mean-absolute-difference or histogram, thresholded).
- `temporal/frame_sampler.py` — `sample_frames(records, method, …)` dispatching to uniform,
  motion, scene-change, or detection-event sampling.

So the pipeline can react to motion / scene change / detector events, run the future
predictor on those frames, and only escalate to the VLM when the calibrated anomaly crosses
threshold.

## 4. End-to-end pipeline

```
camera ─► frame buffer (RAM) ─► frame sampler (motion / scene-change / event)
                                        │
                                 future-embedding predictor (video JEPA)
                                        │  prediction error (surprise)
                                 AnomalyScorer (calibrated z-score → 0..1)
                                        │  is_anomaly?
                       ┌────────────────┴───────────────┐
                    below threshold                 above threshold
                    (log, adapt baseline)     escalate: VLM describes the frame,
                                              event router records an event
```

The anomaly score is an **attention router**, deciding *where to spend the expensive
semantic model*, not a classifier of good/bad.

## 5. Honest limitations

- **Surprise ≠ danger.** A person doing something perfectly safe but novel (a new pose, a
  new object entering frame) scores high; a slow, genuinely dangerous change that the model
  has "gotten used to" scores low. The module's own docstring says this explicitly.
- **Calibration is only as good as the "normal" window.** If the calibration period
  contained anomalies, or is unrepresentative, thresholds are wrong. Garbage-in.
- **Tiny model.** The predictor is a ViT-tiny / GRU at educational scale (see
  `JEPA_ARCHITECTURE.md`); its notion of "expected" is limited. This is a demonstration of
  the *mechanism*, not a production anomaly detector.
- **No labelled anomalies here.** There is no labelled anomaly benchmark in this repo, so
  we report the *mechanism* and calibrated scores, not a validated AUROC on real incidents.
  Anomaly-AUROC-style evaluation is described in `MULTIMODAL_EVALUATION.md` as the correct
  way to measure this once labelled data exists.
- **The right use is human-in-the-loop.** Treat a high score as "a human (or a VLM) should
  look at this frame", never as an automated alarm with consequences.

## References

- LeCun, *A Path Towards Autonomous Machine Intelligence* (world models, prediction in
  representation space), 2022. <https://openreview.net/forum?id=BZ5a1r-kVsf>
- Bardes et al., *Revisiting Feature Prediction for Learning Visual Representations from
  Video* (V-JEPA), 2024. <https://arxiv.org/abs/2404.08471>
- Assran et al. / Meta AI, *V-JEPA 2* (world models for prediction and planning), 2025.
  <https://arxiv.org/abs/2506.09985>
