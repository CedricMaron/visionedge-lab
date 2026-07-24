# Interview Guide — VisionEdge Lab

Honest, strong answers. Nothing here claims users, production traffic, or benchmark numbers that weren't measured. Where the project is a reimplementation or a mock, say so — that candour is itself a signal.

## The one-paragraph pitch
VisionEdge Lab is a hardware-aware multimodal vision platform. It runs real-time object detection, layers vision-language scene understanding on top, and adds a JEPA-style predictive module that forecasts *future representations* (not pixels) to score anomalies. The interesting engineering isn't any single model — it's the deployment layer: a common backend interface across runtimes (ONNX Runtime, PyTorch, OpenVINO, TensorRT), honest capability detection, runtime model-switching with rollback, measured benchmarking, and explicit local-vs-server trade-offs.

---

## Detection & deployment

**How was the model deployed?**
YOLOv8n is exported from PyTorch to ONNX and served by ONNX Runtime behind a `DetectionBackend` interface (`load/warmup/predict/benchmark/close`). The detection *post-processing* (anchor decode + NMS) is a from-scratch NumPy reimplementation, so the serving path depends only on numpy + onnxruntime + opencv — no ultralytics at runtime. I validated the decoder against Ultralytics' own output on a real image: 5/6 detections matched with near-identical boxes; the one difference was a class right at the confidence threshold — which the "output agreement" tool is designed to surface.

**Why several runtimes?**
Because the fastest configuration is device-dependent. An RTX GPU wants TensorRT FP16; a CPU-only box wants ONNX Runtime or OpenVINO; a phone wants a tiny quantized model in WASM/WebGPU. A single interface with per-runtime adapters lets the app pick what the hardware actually supports. Crucially, each adapter *probes* availability — it never claims CUDA/OpenVINO/TensorRT exist unless the runtime imports and (for TensorRT) an engine is built for that GPU.

**How did you compare PyTorch, ONNX, OpenVINO and TensorRT?**
Two axes. **Speed:** `benchmark_model.py` measures P50/P95/P99 latency, FPS, and RSS on the actual device — never hardcoded. **Quality:** `validate_onnx.py` runs "output agreement" against the FP32 PyTorch reference — detection count, class-multiset agreement, confidence delta, and box IoU. I'm careful to label that as *agreement*, not mAP; formal mAP needs a labeled val set (which the tool supports optionally).

**How did you handle unsupported hardware / failures?**
Capability detection drives a recommendation, but the real safety net is the switch state machine: stop accepting frames → drain → unload → free → load → warmup → health-check → resume, with **rollback to the last known-good config** on any failure. I demonstrate this: switching to the ONNX CUDA provider on a box without it doesn't crash — it rolls back to CPU and logs the fallback event. GPU→CPU and server→local fallbacks use the same pattern.

**How did you monitor latency?**
Per-stage timings (preprocess/inference/postprocess/e2e), a rolling window computing P50/P95/P99 + FPS + dropped frames, Prometheus `/metrics` for scraping, and structured JSON logs carrying session/model/runtime/frame-id/timings. The WebSocket transport uses a bounded queue and drops stale frames under backpressure, counting the drops.

---

## Vision-language

**How does a VLM differ from a detector, and how do they complement each other?**
A detector answers *what/where* with boxes; a VLM answers *what's happening* in language and can do VQA. They complement: I optionally inject detector results into the VLM prompt as **grounding** ("a detector reported these objects; treat as a hint, trust the image over it"), then report agreement/disagreement — e.g., detector sees 4 people, does the VLM's answer mention them? Neither model is assumed correct.

**Which VLM did you use?**
The default backend is a **deterministic mock** — honestly labelled, used for tests/CI and for demoing the grounding/structured-output plumbing without a multi-GB download. Real backends are opt-in: **SmolVLM-256M/500M** (Apache-2.0) locally via Transformers, and any **OpenAI-compatible** endpoint remotely (env-configured key, TLS, retries, and a privacy gate that refuses to transmit frames unless explicitly enabled). This keeps the architecture model-agnostic — nothing is hardwired to one family.

**How did you validate quantized / optimized VLM output?**
The optimization-report contract requires measured before/after latency and memory plus embedding cosine-similarity to the reference and a VLM-eval delta — and the rule that an "optimization" that doesn't actually reduce latency or memory must explain why. For open-ended answers I use keyword/concept coverage and detector agreement, **not** exact-match alone.

---

## JEPA

**What is a Joint-Embedding Predictive Architecture, and why predict embeddings instead of pixels?**
JEPA predicts the *representation* of a masked/future region from visible context, in latent space. Predicting pixels wastes capacity modelling unpredictable high-frequency detail (exact textures, noise); predicting embeddings focuses the model on structure that's actually predictable. I-JEPA does this for image blocks; V-JEPA/V-JEPA 2 extend it to video/time.

**Context vs. target encoder, and why EMA?**
The context encoder (gradient-trained) encodes visible patches; a predictor maps those to predicted target embeddings; the target encoder produces the targets and is updated as an **exponential moving average** of the context encoder — it receives no gradients. The EMA target is what prevents the trivial "everything maps to a constant" collapse: the target is a slowly-moving, non-trivial teacher.

**How did you avoid representation collapse, and how do you know?**
Safeguards: EMA target, predictor asymmetry, normalization, mask diversity, gradient clipping, LR warmup. And I *monitor* it: per-dimension std, embedding variance, effective (SVD) rank, and average pairwise cosine similarity, with a warning when embeddings go near-constant. I don't claim collapse is impossible — I claim it's monitored.

**Is this a full world model / V-JEPA?**
No, and I never say it is. This is a **faithful lightweight reimplementation** of the published I-JEPA architecture at educational scale (ViT-tiny, CIFAR/STL-scale), plus a compact temporal predictor. It reproduces the *mechanisms* — masked representation prediction, EMA targets, collapse monitoring, linear probing — not foundation-model performance.

**How is anomaly detection derived from prediction error?**
Predict the future embedding from recent frames; when the real future frame arrives, encode it and measure prediction error. Calibrate on a "normal" window (mean/std of error), normalize new errors to an anomaly score, and threshold. Honest caveat surfaced in the UI: **high prediction error ≠ danger** — it can just mean camera motion or an unusual-but-benign event.

---

## System design

**Why invoke the VLM selectively?** VLMs are expensive. An invocation policy runs the VLM only on triggers — anomaly over threshold, a detector event, a user question, a substantial scene change, a timer. This trades a bounded quality loss for large savings in GPU time, tokens, and (for remote) cost, all measurable.

**How did you manage GPU memory across models?** Detector + VLM + JEPA don't co-fit in 6 GB. A resource manager tracks loaded models and estimated memory, prioritizes real-time detection, can offload/quantize the VLM, and **pauses training rather than silently killing it**, logging every decision.

**Why can the fastest config differ across devices?** Different accelerators, memory ceilings, and runtime support. TensorRT FP16 wins on an RTX GPU; on a CPU box OpenVINO/ONNX win; on a phone a 320px INT8 model in WebGPU wins. The execution planner recommends per-stage placement (e.g., detector local, big VLM remote & event-triggered) *with reasons*, and shows measured trade-offs rather than declaring a universal optimum.

**What failed during development?** Prebuilt COCO YOLOv8 ONNX mirrors were gated/404, so I exported from source — which also made the PyTorch reference backend real. The dev box has only ~1.7 GB free RAM, which is exactly why real VLMs and pretrained JEPA encoders are opt-in and the resource-manager story is central rather than decorative.

**What would production require?** A labeled val set for real mAP/accuracy tracking; a persistent training worker (jobs here don't survive process restart, and I say so); real TensorRT engines built per target GPU; authn/z + rate limiting hardening; and native mobile runtimes (TFLite/NNAPI/Core ML) for true phone-local inference.

---

## Suggested CV bullets (only claims the code supports)
- Built a hardware-aware multimodal computer-vision platform combining real-time object detection, vision-language inference and JEPA-inspired future-representation prediction across PC, browser and server runtimes.
- Implemented a from-scratch YOLOv8 ONNX decoder + NMS, a multi-runtime backend interface (ONNX Runtime/PyTorch/OpenVINO/TensorRT), capability detection, and runtime model-switching with last-known-good rollback.
- Implemented masked representation prediction with EMA target encoders, collapse monitoring, linear-probe evaluation and prediction-error anomaly scoring.
- Designed measured benchmarking, quantization/conversion tooling, and memory-aware execution planning for local GPU, CPU and remote inference.
