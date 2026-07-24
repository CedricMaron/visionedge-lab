# Research Limitations (Brutally Honest)

This document exists so that nothing in VisionEdge Lab is oversold. It states exactly what
is **pretrained**, what is **reimplemented**, what is **simplified**, and what is **not
built yet** — plus the hardware and data constraints, and what scaling up would actually
require. If a claim elsewhere in the repo seems impressive, check it against this page.

## 1. What is pretrained (not ours)

- **YOLOv8 detector weights** (`yolov8n/s/m.pt`) are Ultralytics' pretrained COCO models
  (AGPL-3.0). We export and run them; we did **not** train them. Our contribution around
  them is the from-scratch NumPy post-processing (decode + NMS), the ONNX runtime path, and
  the optimization tooling.
- **SmolVLM / Qwen2.5-VL** (opt-in VLMs) are third-party pretrained models loaded via
  `transformers`. We wrote the *integration* (backend, grounding, structured output), not
  the models.

## 2. What is reimplemented from scratch (faithful, small scale)

- **YOLOv8 decoding + NMS** — pure NumPy (`backend/app/inference/postprocess.py`),
  independent of the Ultralytics runtime.
- **I-JEPA** (`backend/app/jepa/`) — a faithful reimplementation of the *published
  architecture*: tiny ViT context/target encoders, EMA target, multi-block masking,
  representation-space prediction, collapse monitoring, linear/kNN/retrieval probes. It is
  correct in structure but **tiny** (embed dim 16, 32×32 inputs; see below). It is **not**
  Meta's I-JEPA and reproduces none of its scale or benchmark numbers.
- **Temporal / video JEPA** — a from-scratch future-embedding predictor (GRU/MLP
  aggregator) used for the world-model/anomaly experiment. It is inspired by V-JEPA's
  "predict features of the future", not a reimplementation of V-JEPA's masking system.
- **The mock VLM** — deterministic, grounded on detector context; a stand-in that is
  honest about being a stand-in, not a model.

## 3. What is simplified

- **JEPA is at educational scale.** `ViTConfig` defaults: `image_size=32, patch_size=8,
  embed_dim=16, depth=2, num_heads=2`. That is orders of magnitude smaller than a research
  JEPA (ViT-L/H, 224²+ inputs, hundreds of embedding dims, huge datasets). Expect the
  representations to *demonstrate the mechanism*, not to be competitive features.
- **Masking/index sharing.** The context/target patch indices are shared across a batch in
  the scaffold (documented in code) rather than sampled per-example — a simplification for
  clarity and CPU speed.
- **Video VLM is single-frame.** `analyze_video` analyzes the middle frame and warns; there
  is no cross-frame temporal reasoning inside the VLM (that lives, separately, in the
  temporal slice).
- **Structured output is best-effort.** `parse_structured` tolerates and reports malformed
  JSON rather than guaranteeing schema-valid output from every model.

## 4. What is interface-only / planned

- **OpenVINO** and **TensorRT** backends and their scripts are implemented but **inactive**
  here (packages not installed / no usable GPU); they exit with clean install messages.
- **Browser (ONNX Runtime Web) inference** — the asset-prep script and manifest exist, but
  in-browser inference itself is **Phase 3** (not wired up).
- **Pretrained I-JEPA/V-JEPA encoder loading** — a seam exists
  (`representation/encoders.py`) but `load_pretrained_ijepa_encoder` currently raises
  `NotImplementedError`; the `timm` path is opt-in.
- **Action-conditioned world model / planning (V-JEPA 2 direction)** — conceptual only.
- **Human-rating evaluation and labelled mAP / anomaly-AUROC** — described in
  `MULTIMODAL_EVALUATION.md` as the correct methods, but **not computed** (no labelled data
  in-repo).

## 5. Hardware and data constraints

- **CPU-only reference box.** The default stack is CPU (`torch … +cpu`,
  `onnxruntime` CPU/Azure providers; `torch.cuda.is_available()` is `False`). GPU paths
  (CUDA ORT, TensorRT, GPU VLMs) are real in code but **unexercised** on this machine.
- **Single developer GPU at most.** Where a GPU is present, it is one consumer card — not a
  cluster. Qwen2.5-VL-3B at FP16 (~7 GB) already crowds a 6–8 GB card alongside the
  detector; that is why it is marked server/quantize-only in the registry.
- **RAM limits.** Small local VLMs (SmolVLM-256M) can run on CPU but are slow and
  memory-hungry; `<4 GB` free RAM risks OOM. The capability scanner surfaces available RAM
  so a doomed configuration is visible beforehand.
- **No labelled evaluation set.** There is a single sample image (`sample_bus.jpg`) and no
  COCO val split in-repo. Therefore: **no mAP**, no labelled anomaly AUROC, no human-rating
  corpus. Cross-runtime quality is reported as *output agreement* against the FP32
  reference, which measures fidelity of an optimization, **not** absolute accuracy.
- **Benchmarks are host-specific.** All latency/FPS/RSS numbers come from real runs on the
  reporting machine and are only comparable within that host and run.

## 6. What scaling up would actually require

- **JEPA that competes:** ViT-L/H encoders, 224²+ inputs, large curated image/video
  datasets (ImageNet-scale for images; large video corpora for V-JEPA), multi-GPU training
  for days, and per-example masking. That is a training-infrastructure project, not a
  config change.
- **Trustworthy anomaly detection:** a labelled incident dataset, calibration per
  deployment, AUROC/PR evaluation, and human review — plus acceptance that "surprise ≠
  danger" (see `WORLD_MODEL_EXPERIMENT.md`).
- **Production VLM:** a served GPU (or a hosted API) for Qwen-class models, streaming for
  real TTFT, and a human-rating loop for open-ended quality.
- **Real accuracy numbers:** a labelled validation set (COCO val for detection) to report
  mAP, so "agreement" can be replaced by measured accuracy.
- **GPU-accelerated runtimes:** installing and validating CUDA ORT / OpenVINO / TensorRT on
  the actual target hardware, then rebuilding TensorRT engines per target.

## 7. The one-line summary

VisionEdge Lab is an **honest, runnable, educational-scale** demonstration of a modern
vision stack — pretrained detector + real optimization tooling + faithful-but-tiny
reimplementations of JEPA/world-model ideas + an opt-in VLM slice. It is **not** a
production system and does not claim state-of-the-art accuracy, benchmarks, or users.
