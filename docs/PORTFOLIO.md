# Portfolio Assets — VisionEdge Lab

Only claims supported by the actual implementation. No invented users, traffic, or benchmark numbers.

## GitHub repository description (short)
> Hardware-aware multimodal vision platform: real-time object detection + vision-language understanding + JEPA-style future-representation prediction, compared across CPU/GPU/browser/server runtimes. FastAPI + ONNX Runtime backend, React/TS frontend, honest capability detection, runtime model-switching with rollback, and measured benchmarking.

## Portfolio project description (long)
VisionEdge Lab is a multimodal computer-vision platform built to demonstrate deployment engineering, not just model usage. It runs YOLOv8 object detection (ONNX Runtime, with a from-scratch NumPy decoder), layers a pluggable vision-language backend for scene description and grounded VQA, and adds a faithful lightweight I-JEPA reimplementation that predicts future *representations* to score anomalies. A common backend interface spans ONNX Runtime, PyTorch, OpenVINO and TensorRT; a capability scanner detects real hardware and runtime availability; a switch state machine changes models at runtime with last-known-good rollback; and an execution planner recommends where each pipeline stage should run given the device, network and privacy constraints. Everything is honest about what is a working integration, an opt-in model, or a simplified experiment — including a resource manager motivated by the real constraint that a detector, a VLM and a JEPA model don't co-fit in 6 GB of VRAM.

## LinkedIn post
> I built **VisionEdge Lab** — a hardware-aware multimodal vision platform that compares object detection, vision-language understanding, and JEPA-style predictive representation learning across edge, browser, and server runtimes.
>
> The interesting part wasn't any single model — it was the deployment layer: a common interface across ONNX Runtime / PyTorch / OpenVINO / TensorRT, honest capability detection, runtime model-switching with rollback, measured benchmarking (never hardcoded), and an execution planner that reasons about latency vs. privacy vs. memory. I also reimplemented I-JEPA (context/target encoders, EMA targets, collapse monitoring) at educational scale to show *why* predicting embeddings beats predicting pixels.
>
> Built with FastAPI, ONNX Runtime, React/TypeScript, and a lot of attention to being honest about what's a real integration vs. an opt-in model vs. a simplified experiment. #ComputerVision #MLOps #EdgeAI

## CV bullet points
- Built a hardware-aware multimodal computer-vision platform combining real-time object detection, vision-language inference and JEPA-inspired future-representation prediction across PC, browser and server runtimes (FastAPI, ONNX Runtime, React/TypeScript).
- Implemented a from-scratch YOLOv8 ONNX decoder + NMS (validated to 0.95 mean box-IoU against the PyTorch reference) behind a multi-runtime backend interface with capability detection and runtime model-switching with last-known-good rollback.
- Reimplemented masked representation prediction (I-JEPA) with EMA target encoders, collapse monitoring, linear-probe evaluation and prediction-error anomaly scoring at educational scale.
- Designed measured benchmarking, ONNX/FP16/INT8 quantization tooling with output-agreement validation, and memory-aware execution planning for local GPU, CPU and remote inference.
- Wrote 97 automated tests (87 backend, 10 frontend) covering NMS correctness, runtime switching/rollback, structured VLM output, invocation policy and collapse detection.

## Technical interview talking points
- **From-scratch decoder:** why I reimplemented YOLOv8 post-processing in NumPy (runtime-dependency reduction) and how I validated it (output-agreement vs. reference, 5/6 detections, 0.95 IoU, threshold-borderline diff surfaced).
- **Multi-runtime honesty:** every runtime is *probed*, never assumed; CUDA/OpenVINO/TensorRT report unavailable unless truly present; switching to an unavailable provider rolls back and logs a fallback event.
- **JEPA fundamentals:** context vs. target encoder, why EMA prevents collapse, why representation-space prediction beats pixel prediction, and honest scope ("faithful lightweight reimplementation, not V-JEPA").
- **Systems trade-offs:** selective VLM invocation, GPU-memory coordination, and per-stage execution placement with measured trade-offs.

## Three-minute demo script
1. Open **Live Inference**, start the camera, show real-time boxes + live FPS/latency.
2. Open **Class Selector**, switch to "People only", show filtering.
3. Open **Model Selector**, switch runtime; show the switch + rollback messaging.
4. Open **Multimodal Assistant**, capture a frame, toggle detector-grounding, show the grounded description + detector-agreement.
5. Open **Optimization Advisor**, show the per-stage recommendation with reasons for *this* device.

## Ten-minute technical demo script
Add to the above: **Device Capabilities** (real RTX 2060/ORT-provider detection) → **Benchmarks** (run a live measured benchmark) → `scripts/validate_onnx.py` (output-agreement) and `scripts/quantize_onnx.py --precision fp16` (size drop + IoU) in a terminal → **Architecture** + **Research Notes** pages → close on `docs/RESEARCH_LIMITATIONS.md`, explaining honestly what's real vs. planned.
