# Multimodal Evaluation

How VisionEdge Lab evaluates its three visual-intelligence layers — the vision-language
model, the JEPA representation, and the system as a whole. The theme throughout: **pick
metrics that match what the model actually does, and never overclaim.**

## 1. Why exact-match is the wrong VLM metric

If you ask a VLM "describe this scene" and it answers *"A city bus with several people
waiting beside it"*, an exact-string match against a reference caption *"Bus and pedestrians
on a street"* scores **0** — even though the answer is correct. Open-ended generation has
many valid phrasings. Exact-match (and even BLEU-style n-gram overlap) punishes correct
paraphrase and rewards copying. So we evaluate VLM output along several **complementary,
honest axes** instead of one brittle number.

### 1a. Performance / cost metrics (measured, on `VLMResponse`)

These ride on every response (`backend/app/core/types.py`) and are real measurements:

| Metric | Field | Meaning |
| --- | --- | --- |
| Total latency | `total_latency_ms` | end-to-end wall clock |
| Generation latency | `generation_latency_ms` | model generation time |
| Time to first token (TTFT) | `time_to_first_token_ms` | responsiveness (None when not streamed) |
| Prompt / generated tokens | `prompt_tokens`, `generated_tokens` | cost + verbosity |
| Tokens/sec | derived: `generated_tokens / generation_latency_ms` | throughput |
| Memory | `memory_usage_mb` | resident footprint |

TTFT is honestly `None` for backends that do not stream (the local SmolVLM path), rather
than a faked number.

### 1b. Content / grounding metrics (lexical, non-fabricating)

`backend/app/vlm/evaluation.py` — pure functions over already-produced text, never calling
a model:

- **`keyword_coverage(answer, expected_concepts)`** → fraction of expected concepts present
  (case-insensitive substring). A soft recall proxy that tolerates paraphrase; empty
  expectation returns `1.0`.
- **`object_count_consistency(answer, detections, class_name)`** → `{detector_count,
  vlm_mentioned_count, agree}`; parses the first integer in the answer and compares it to
  the detector's count for that class.
- **`detector_agreement(answer, detections)`** → which detector classes the VLM mentioned
  vs. omitted, with an explicit note that *neither model is assumed correct*. This is the
  one the API attaches (with a disclaimer) when grounding is on.

### 1c. Structured validity

For structured requests, validity is whether `parse_structured` returned a valid
`SceneUnderstanding` (see `VLM_ARCHITECTURE.md` §6). The signal is binary per response
(parsed / not parsed) plus the warnings list; a validity **rate** over a batch is the
useful aggregate. The parser never fabricates a valid object to inflate this rate.

### 1d. Human rating

For genuinely open descriptions, the honest gold standard is **human judgement** (accuracy,
grounding, hallucination, usefulness on a small Likert scale). This is **interface-level
only** in this build — there is no human-rating function in code; it is noted here as the
intended qualitative complement, not something the system computes automatically.

## 2. JEPA / representation evaluation

A JEPA encoder is trained without labels, so it is judged by the *usefulness and health* of
its frozen features, not a training accuracy.

### 2a. Training-health metrics (from `collapse_monitor.py`)

Reported every step (see `JEPA_ARCHITECTURE.md` §7): representation loss (`SmoothL1` in
embedding space), `embed_std` / `min_dim_std`, `embedding_variance`, `effective_rank`
(SVD-entropy), `avg_pairwise_cosine`, and a boolean `collapse_warning`. These catch the
dominant failure mode (collapse) early.

### 2b. Downstream-quality metrics (from `jepa/evaluation.py`)

With the encoder **frozen**:

- **Linear probe accuracy** — `linear_probe(...)`; train only a linear head. High score ⇒
  features are linearly separable ⇒ the SSL objective organized them semantically.
- **kNN accuracy** — `nearest_neighbor_accuracy(..., metric="cosine")`; no training at all,
  a clean test of local structure.
- **Retrieval Recall@k** — `retrieval_recall_at_k(...)`; do same-class items retrieve each
  other by cosine similarity?

### 2c. Anomaly evaluation (temporal slice)

For the world-model experiment (`WORLD_MODEL_EXPERIMENT.md`), the appropriate metric is
**AUROC of the calibrated anomaly score against labelled anomalies** — it is
threshold-free and handles class imbalance. **Honest status:** there is no labelled anomaly
set in this repo, so we report the calibrated z-scores and the *mechanism*; AUROC is
documented as the correct evaluation *once labelled incidents exist*, not a number we
currently claim.

## 3. System evaluation

Beyond per-model metrics, the platform as a whole is judged on:

- **End-to-end frame latency** and **sustained FPS** under the real backpressure policy
  (bounded frame queue, frame-drop under load) — measured, not modelled.
- **Detector inference latency** (`scripts/benchmark_model.py`: mean/p50/p95/p99, FPS, RSS)
  — see `MODEL_OPTIMIZATION.md`.
- **Output agreement across runtimes** — `scripts/validate_onnx.py` compares an optimized
  ONNX/FP16/INT8 model to the FP32 PyTorch reference by detection count, class multiset,
  matched-box mean IoU and confidence delta. Labelled **agreement**, never mAP, because
  there is no labelled validation set.
- **Resource footprint** — RAM/VRAM headroom from the capability scanner, so a
  configuration that would OOM is caught before deployment.

## 4. The honesty rules for every metric here

1. **Measure, don't estimate.** Latency/throughput/memory come from real runs on the
   reporting machine and are only comparable within that machine.
2. **Name the metric for what it is.** Agreement is agreement; a linear probe is a probe;
   surprise is surprise. We do not relabel a proxy as ground-truth accuracy.
3. **Report uncertainty and gaps.** Where there is no labelled data (mAP, anomaly AUROC,
   human ratings), we say so and describe what would be needed — rather than inventing a
   score. See `RESEARCH_LIMITATIONS.md`.

## References

- Liu et al., *Visual Instruction Tuning* (LLaVA; open-ended VLM evaluation challenges),
  2023. <https://arxiv.org/abs/2304.08485>
- Assran et al., *I-JEPA* (linear-probe / frozen-feature evaluation of SSL encoders), 2023.
  <https://arxiv.org/abs/2301.08243>
- Fawcett, *An introduction to ROC analysis* (AUROC for imbalanced anomaly detection),
  2006. <https://doi.org/10.1016/j.patrec.2005.10.010>
