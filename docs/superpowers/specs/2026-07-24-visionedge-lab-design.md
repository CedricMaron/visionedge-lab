# VisionEdge Lab — Reconciled Design (Detection + VLM + JEPA)

**Date:** 2026-07-24
**Status:** Approved for build (Foundation A + VLM slice B + scaffold C)

## 1. Positioning

> A hardware-aware multimodal vision platform for comparing detection, vision-language
> understanding and predictive representation learning across edge, browser and server
> environments.

Three complementary levels of visual intelligence:
1. **Detection** — what/where objects are (YOLOv8n ONNX, NumPy NMS).
2. **Vision-language** — what is happening / QA over frames (VLM adapters).
3. **JEPA** — how the scene changes / future-representation prediction (from-scratch I-JEPA + temporal predictor).

## 2. Ground truth at design time

- Repo started empty on 2026-07-24. Nothing pre-existing; this doc is the first reference.
- Hardware: RTX 2060 (6 GB VRAM), WSL2 with **3.7 GB system RAM (~1.7 GB free)**, 12 cores,
  Python 3.12, Node 24. Internet available.
- Consequence: detector + VLM + JEPA cannot co-reside in memory. This *motivates* the resource
  manager and execution planner rather than undermining them.

## 3. What is honest about each component

| Component | Nature |
|---|---|
| YOLOv8n detection | Pretrained weights, exported to ONNX; **NMS/post-processing is a from-scratch NumPy reimplementation** (no ultralytics at runtime). |
| ONNX Runtime CPU/CUDA backends | Real inference integrations. |
| PyTorch / OpenVINO / TensorRT backends | Interfaces + honest availability probes; report `available: false` unless the runtime is actually installed. Never faked. |
| VLM mock backend | Deterministic, real code; the CI/test default. Explicitly labelled mock. |
| VLM SmolVLM local | Real integration via `transformers`, **opt-in download** (Apache-2.0). May OOM on 3.7 GB RAM — documented. |
| VLM remote backend | Real OpenAI-compatible client behind abstraction; opt-in, env-configured, TLS, privacy warning. |
| I-JEPA (Levels 2–3) | **Faithful lightweight reimplementation** of the published architecture (context/target encoder, predictor, mask generator, EMA, representation loss, collapse monitor). ViT-tiny scale — educational, not foundation-scale. |
| Pretrained I-JEPA / V-JEPA encoders | Opt-in downloads; wrapped, not reimplemented. Never claimed equivalent to Meta's V-JEPA/V-JEPA 2. |

## 4. Model selections (license-checked)

- **Detector:** YOLOv8n → ONNX. Nano installed; small/medium in registry as `not_installed` (per-model install).
- **VLM default:** `mock` (tests). **Local opt-in:** SmolVLM-256M / 500M-Instruct (Apache-2.0, image + multi-image, some video). **Server opt-in:** Qwen2.5-VL-3B-Instruct (Apache-2.0) / any OpenAI-compatible endpoint.
- **JEPA:** from-scratch ViT-tiny I-JEPA trained on CIFAR-10 / STL-10 / user folder. Pretrained I-JEPA/V-JEPA as opt-in encoders.

## 5. Shared contracts (single source of truth)

Backend Pydantic models mirror frontend TS types exactly.

- `Detection`: x1,y1,x2,y2,confidence,classId,className,inferenceBackend,modelName,modelVersion.
- `DetectionBackend` Protocol: load/warmup/predict/benchmark/close.
- `VisionLanguageBackend` Protocol: load/warmup/describe_image/answer_question/analyze_video/unload.
- `VLMResponse`: text, structured_output, model_id, runtime, execution_location, prompt/generated tokens, TTFT, latencies, memory, warnings.

## 6. Architecture (build order A → B → C)

**A. Detection foundation** (fully runnable + tested):
FastAPI app; `DetectionBackend` interface + `OnnxRuntimeBackend` (CPU + optional CUDA EP) + guarded `PyTorchBackend` + honest `OpenVINO`/`TensorRT` stubs; capability scanner (psutil + nvidia-smi + ORT providers); model registry (`models/registry.json`); backend-switch state machine (stop→drain→unload→free→load→warmup→health→resume, rollback on failure); WS frame transport (bounded queue, frame drop) + REST `/api/infer`; monitoring endpoints; SQLite store schema; structured JSON logging; Prometheus `/metrics`. React+TS+Vite+Tailwind shell with Live Inference / Device Capabilities / Model Selector / Class Selector / Settings functional.

**B. VLM vertical slice** (fully runnable + tested):
`vlm/` package (base, registry, local_backend, remote_backend, prompting, structured_output, evaluation); mock default; VLM API (`/api/vlm/*`); Multimodal Assistant page; structured output (Pydantic-validated, raw preserved on failure); latency/token/memory metrics; detector-grounding toggle wiring.

**C. Scaffold the rest** (real interfaces, honest not-implemented states, core testable pieces done):
`representation/` (encoders, embedding_store, retrieval, visualization), `jepa/` (context/target encoder, predictor, masking, ema, image/video trainer, collapse_monitor, evaluation, checkpoints), `temporal/` (frame_buffer, frame_sampler, scene_change, anomaly), `orchestration/` (invocation_policy, execution_planner, resource_manager), `jobs/` (worker, manager, state). Frontend pages/services stubbed as real components with explicit "phase N" status, not fake outputs. Genuinely-testable primitives implemented now: masking, EMA update, collapse metrics, frame buffer, invocation policy, resource manager, anomaly normalization.

## 7. Testing

Real, run this session. Backend pytest: NMS correctness, registry validation, capability scan, backend switch + rollback, config validation, VLM mock + structured-output validation + failed-load fallback, invocation policy, EMA/frozen-target, mask generation, collapse detection, anomaly normalization, resource coordination. Frontend vitest: class-selection store, model-switch store. Models are **mocked** in automated tests — no multi-GB downloads in CI.

## 8. Non-negotiable honesty rules (from both specs)

No fake outputs, no hardcoded benchmarks, no silent large downloads, confirm before >5 GB, never claim an unavailable runtime, never call server-assisted "fully local", never equate the lightweight I-JEPA with Meta V-JEPA, label VLM answers as model interpretation not verified truth, don't store camera frames by default, release model memory on switch.

## 9. Explicitly deferred (documented, not faked)

Phone browser inference (ORT-Web/WebGPU), phone-local VLM, real TensorRT/OpenVINO engines, pretrained V-JEPA integration, full training runs, PWA polish, UMAP/t-SNE (PCA ships), remote GPU server deployment. Each surfaces in the UI as an honest "not yet implemented in this build" state with a rationale.
