# InferenceLab

**Multimodal AI inference, profiling and benchmarking.**

InferenceLab is a hardware-aware multimodal benchmarking platform that decomposes AI
inference from input decoding to device execution and network transport. It compares
vision, language, speech, image and video models across local, browser, edge and remote
runtimes using reproducible latency, throughput, memory, utilization, energy and quality
measurements.

> Formerly *VisionEdge Lab*. See [`docs/MIGRATION.md`](docs/MIGRATION.md) for what changed
> and how existing deployments carry forward.

InferenceLab demonstrates three complementary levels of visual intelligence and — just as importantly — the *deployment engineering* around them: model conversion, quantization, runtime selection, capability detection, benchmarking, graceful fallback, and local-vs-server trade-offs.

1. **Object detection** — *what* objects are present and *where* (YOLOv8, ONNX Runtime, from-scratch NumPy decoding).
2. **Vision-language understanding** — *what is happening*; scene description and visual question answering (pluggable VLM backends).
3. **JEPA-style predictive understanding** — *how the scene changes*; future-representation prediction and anomaly-from-prediction-error (faithful lightweight I-JEPA reimplementation).

> ### Honesty first
> This is a portfolio/research project built on a single laptop-class GPU (RTX 2060, 6 GB) in a RAM-constrained WSL2 environment. It never fabricates capabilities, benchmark numbers, or model quality. Where a component is a mock, an opt-in integration, or a simplified reimplementation, it says so — see [What is real vs. planned](#what-is-real-vs-planned) and [`docs/RESEARCH_LIMITATIONS.md`](docs/RESEARCH_LIMITATIONS.md). The lightweight JEPA implementation is **not** Meta's V-JEPA/V-JEPA 2 and is never claimed to be.

---

## What is real vs. planned

Measured on the reference box (i7-9750H, 12 threads, RTX 2060 6 GB, ~1 GB free RAM,
WSL2). Every "working" row below is covered by tests in this repository.

### Benchmarking platform

| Capability | Status |
|---|---|
| Phase-decomposed latency on a monotonic clock, with device synchronization | ✅ **Working & tested** |
| Residual overhead reported explicitly, never folded into a real phase | ✅ **Working & tested** |
| Raw per-iteration retention; percentiles recomputable from stored samples | ✅ **Working & tested** |
| Warm-up / cold-start / steady-state separation | ✅ **Working & tested** |
| Failed-iteration tracking, cancellation, timeouts | ✅ **Working & tested** |
| Integrity warnings (thin sample, background load, throttling, dirty tree) | ✅ **Working & tested** |
| Environment fingerprint + comparison guard that refuses incompatible runs | ✅ **Working & tested** |
| CPU/RAM probes (psutil), GPU probes (NVML: utilization, clocks, temp, power, VRAM) | ✅ **Working** — every field a real probe |
| Energy by integrating measured GPU power | ✅ **Working** — `derived`, GPU-only, refuses below 3 samples |
| Sampler self-cost measurement; perturbation measured at <4% (interleaved) | ✅ **Working & tested** |
| Versioned result schema, SQLite migrations, JSON/CSV/Markdown export | ✅ **Working & tested** |
| CLI sharing one engine with the API (same fingerprint for the same config) | ✅ **Working & tested** |
| Remote inference: correlation ids, server timing envelope, transport residual | ✅ **Working & tested** — verified over real HTTP |
| CPU package energy (RAPL) | ⛔ **Not readable** under WSL2 — reported unavailable with that reason |
| Framework GPU allocated/reserved memory | ⛔ **Not exposed by ONNX Runtime** — reported unavailable, NVML device totals shown separately |

### Modalities

| Task | Model | Status |
|---|---|---|
| Object detection | YOLOv8n (ONNX, AGPL-3.0) | ✅ **Working & tested** — from-scratch NumPy decode, equivalence-tested |
| Image classification | MobileNetV4 Conv Small (ONNX, Apache-2.0) | ✅ **Working & tested** — timm preprocessing read from the model config |
| Text embedding | all-MiniLM-L6-v2 (ONNX, Apache-2.0) | ✅ **Working & tested** — masked mean pooling, semantic ordering asserted |
| Vision-language | mock backend; SmolVLM opt-in | 🟡 **Mock working & labelled**; local VLM guarded/opt-in |
| Speech-to-text, TTS, image/video generation, LLM | — | ⛔ **Contracts and scenarios only** — no runtime fits ~1 GB free RAM here; every entry says so |

### Runtimes

| Runtime | Status |
|---|---|
| ONNX Runtime (CPU) | ✅ **Working** |
| ONNX Runtime (CUDA) | 🟡 **Probed and refused** — provider is listed but session creation fails (`libcublasLt.so.12` missing), so the load is rejected rather than silently falling back to CPU |
| PyTorch, TensorRT, OpenVINO, CoreML, TFLite, MLX, llama.cpp, vLLM, TGI, browser, remote-streaming | 🟡 **Declared and probed** — each reports whether the dependency is missing or the adapter is unimplemented |

### Vision slice (from VisionEdge Lab, preserved)

| Capability | Status |
|---|---|
| Live WebSocket inference with bounded queue and frame drop | ✅ **Working** |
| Model switching with rollback, auto-benchmark on switch | ✅ **Working & tested** |
| Caddy + compose deployment, per-IP rate limiting, TLS | ✅ **Working** — live |
| JEPA primitives, temporal buffer, anomaly scorer, orchestration | 🟡 **Reimplemented primitives + tests**; full training runs planned |

---

## Architecture

```
                           ┌──────────────────────┐
                           │ Camera / Image/Video │
                           └──────────┬───────────┘
                                      │
                             ┌────────▼────────┐
                             │ Frame Scheduler │
                             └───────┬─────────┘
               ┌─────────────────────┼─────────────────────┐
      ┌────────▼────────┐   ┌────────▼────────┐   ┌────────▼────────┐
      │ Object Detector │   │ Visual Encoder  │   │ Frame Buffer     │
      └────────┬────────┘   └────────┬────────┘   └────────┬────────┘
      ┌────────▼────────┐   ┌────────▼────────┐   ┌────────▼────────┐
      │ Tracker / Zones │   │ Embedding Store │   │ JEPA Predictor  │
      └────────┬────────┘   └────────┬────────┘   └────────┬────────┘
               └──────────────┬──────┴──────────────┬──────┘
                     ┌────────▼────────┐   ┌────────▼────────┐
                     │ Event Router    │   │ Anomaly Scorer  │
                     └────────┬────────┘   └────────┬────────┘
                              └──────────┬──────────┘
                                ┌────────▼────────┐
                                │ VLM Invocation  │  (selective, event-triggered)
                                │ Policy          │
                                └────────┬────────┘
                                ┌────────▼────────┐
                                │ Local / Server  │
                                │ VLM Backend     │
                                └────────┬────────┘
                                ┌────────▼────────┐
                                │ UI / Logs / DB  │
                                └─────────────────┘
```

Full detail: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Supported execution modes

| Mode | Camera | Inference | Status |
|---|---|---|---|
| **PC local** | PC/browser | On the PC (ONNX Runtime CPU today; CUDA/OpenVINO/TensorRT opt-in) | ✅ Working |
| **Local server** | Phone/PC | On a PC on your LAN via WebSocket | ✅ Working (same WS transport) |
| **Remote server** | Any | Configurable remote endpoint (VLM: OpenAI-compatible, opt-in, TLS, auth) | 🟡 VLM path implemented; detection remote is same transport |
| **Phone local (browser)** | Phone | In the mobile browser (ORT-Web/WebGPU) | ⛔ Planned (interfaces stubbed) |

### Trade-offs at a glance

| | Latency | Bandwidth | Privacy | Battery | Model quality | Hardware need |
|---|---|---|---|---|---|---|
| PC local | Low | None | High (frames stay local) | PC-powered | Bounded by local GPU/CPU | GPU helps |
| Phone local | Low-med | None | Highest | Drains phone | Small models only | Modern phone |
| Local server | Low (LAN) | LAN only | High (stays on LAN) | Light on phone | Larger models | A PC on LAN |
| Remote server | Network-bound | Uploads frames | **Lowest** (frames leave device) | Light on client | Largest models | None on client |

---

## Installation

Prereqs: **Python 3.11+**, **Node 18+**. A CUDA GPU is optional.

```bash
git clone <this-repo> visionedge-lab && cd visionedge-lab
cp .env.example .env

# --- backend (CPU-first) ---
make venv
make install                 # base.txt: fastapi, onnxruntime, opencv, numpy, ...
make model                   # export YOLOv8n -> models/yolov8n.onnx (+ checksum)

# --- frontend ---
make frontend-install
```

### CPU-only setup
The default. `make install` + `make model` is all you need — detection runs on ONNX Runtime CPU and the VLM defaults to the deterministic mock.

### NVIDIA / CUDA setup
```bash
# PyTorch CUDA (pick your CUDA version's index-url) + onnxruntime-gpu:
backend/.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
backend/.venv/bin/pip install -r backend/requirements/cuda.txt   # onnxruntime-gpu
```
Then the ONNX **CUDA** execution provider and PyTorch-CUDA backend light up automatically (verified via `/api/capabilities`).

### OpenVINO setup
```bash
backend/.venv/bin/pip install -r backend/requirements/openvino.txt
backend/.venv/bin/python scripts/convert_openvino.py --input models/yolov8n.onnx --precision fp16
```

### TensorRT setup
TensorRT engines are GPU/driver-specific and must be built on the target machine:
```bash
backend/.venv/bin/pip install -r backend/requirements/tensorrt.txt
backend/.venv/bin/python scripts/build_tensorrt.py --input models/yolov8n.onnx --precision fp16
```

### Local VLM (SmolVLM) setup
```bash
backend/.venv/bin/pip install -r backend/requirements/vlm.txt   # transformers, pillow
# then switch the VLM to smolvlm-256m in the Multimodal Assistant page (opt-in download).
```
> On <4 GB RAM this may be slow or OOM — the backend reports that honestly and rolls back to the mock.

---

## Running

```bash
make backend      # FastAPI on http://localhost:8000  (docs at /docs)
make frontend     # Vite dev server (defaults to http://localhost:5173)
```

Open the frontend, go to **Live Inference**, allow camera access, and press Start. Or try detection headlessly:

```bash
curl -F "file=@benchmark-data/sample_bus.jpg" "http://localhost:8000/api/infer?confidence=0.25"
```

### Phone on your LAN
Set `VITE_API_BASE` in `.env` to your PC's LAN IP over **https** (browsers require a secure context for camera access on non-localhost). See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for an HTTPS dev-cert recipe. IPs are never hardcoded — they come from env only.

---

## API

| Endpoint | Purpose |
|---|---|
| `GET /health`, `/ready`, `/metrics` | Liveness, readiness, Prometheus metrics |
| `GET /api/capabilities` | Real hardware + runtime detection |
| `GET /api/models`, `/api/model-registry`, `/api/classes` | Registry + COCO classes/groups |
| `POST /api/infer` | Single-image detection |
| `POST /api/detection/switch` | Runtime model/runtime switch (with rollback) |
| `POST /api/detection/benchmark` | Measured benchmark |
| `WS /api/ws/detect` | Real-time detection (bounded queue, frame dropping) |
| `GET /api/runtime-status`, `/api/sessions`, `/api/benchmarks` | Status/history |
| `GET /api/benchmarks/comparison` | Per-model comparison, median across runs with run count |
| `GET /api/jobs` | Background job status (auto-benchmark progress) |
| `GET /api/vlm/models`, `POST /api/vlm/analyze-image`, `/api/vlm/ask`, `/api/vlm/analyze-frames` | VLM (mock default) |

---

## Model conversion, quantization, calibration, benchmarking

All under `scripts/` (each has `--help`). Examples:
```bash
python scripts/export_onnx.py     --model nano --size 640
python scripts/validate_onnx.py   --input models/yolov8n.onnx      # output-agreement vs PyTorch reference
python scripts/quantize_onnx.py   --input models/yolov8n.onnx --precision int8 --calibration-dir calibration/
python scripts/benchmark_model.py --model models/yolov8n.onnx --runs 50   # MEASURED, never hardcoded
```
Details: [`docs/MODEL_OPTIMIZATION.md`](docs/MODEL_OPTIMIZATION.md).

---

## Testing

```bash
make test            # backend pytest (detection, registry, switching/rollback, VLM, API, JEPA/temporal/orchestration)
make frontend-test   # frontend vitest (class selection, model switching)
```

---

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — system + deployment architecture
- [`docs/VLM_ARCHITECTURE.md`](docs/VLM_ARCHITECTURE.md) — vision-language design
- [`docs/JEPA_ARCHITECTURE.md`](docs/JEPA_ARCHITECTURE.md) — JEPA principles & the reimplementation
- [`docs/WORLD_MODEL_EXPERIMENT.md`](docs/WORLD_MODEL_EXPERIMENT.md) — future-prediction & anomaly
- [`docs/MODEL_OPTIMIZATION.md`](docs/MODEL_OPTIMIZATION.md) — conversion/quantization
- [`docs/MULTIMODAL_EVALUATION.md`](docs/MULTIMODAL_EVALUATION.md) — evaluation methodology
- [`docs/RESEARCH_LIMITATIONS.md`](docs/RESEARCH_LIMITATIONS.md) — honest limitations
- [`docs/INTERVIEW_GUIDE.md`](docs/INTERVIEW_GUIDE.md) — interview talking points

## Deployment

The app is served from a single origin behind a Caddy reverse proxy: Caddy serves
the built SPA and proxies `/api` (including the `/api/ws/detect` WebSocket) and
`/health` to the backend. Same origin means no CORS, `wss://` derived for free, and
one certificate. Camera capture requires HTTPS, which Caddy's automatic TLS provides.

```bash
# 1. Point an A/AAAA record at the host, and open ports 80 and 443
#    (Caddy needs 80 for the ACME challenge).

# 2. Fetch the detection weights (12.8 MB, checksum-verified against the registry).
#    A fresh clone has no .onnx: model binaries are gitignored.
python scripts/download_models.py --install yolov8n-onnx

# 3. Start it.
SITE_ADDRESS=visionedge.c-maron.space docker compose -f docker-compose.prod.yml up -d
```

**On a Windows host without WSL2**, the compose file cannot run — those are Linux
containers, and nested virtualization is disabled on many Windows VPS plans. See
[`docs/DEPLOY_WINDOWS.md`](docs/DEPLOY_WINDOWS.md) for the native path: same
single-origin design with Caddy and the backend as ordinary Windows processes.

Set `IL_RATE_LIMIT_PER_MIN` for anything internet-facing — every `/api/infer` and
`/api/vlm/*` request is a real model forward pass, so an unlimited public endpoint
is a cost and abuse vector. The limit is enforced in the backend
(`app/api/ratelimit.py`) rather than at the proxy, so it holds however you deploy.
The SQLite database must stay on a mounted volume (`IL_DB_PATH`), or benchmark
history and sessions are discarded on every restart.

### The hosted demo runs on different hardware

Every number this project reports is measured on the machine it runs on. This
README's framing — a laptop-class RTX 2060 with 6 GB in WSL2 — describes the
**development** machine. On a hosted deployment, the Device Capabilities page and
every benchmark will honestly report **that server's** hardware instead, which is
typically a CPU-only VPS with no GPU at all. The two sets of numbers are not
comparable, and neither is wrong: they are measurements of different machines.

Note also that the YOLOv8 weights are AGPL-3.0 (see [License](#license)); AGPL
covers network use, so a public deployment serving them should keep its
corresponding source available — which this public repository does.

## Privacy & security

- Camera frames are **not stored by default**. Remote frame transmission is off unless you set `IL_ALLOW_FRAME_TRANSMISSION=true`.
- Remote endpoints use env-based API keys and TLS. No public server is exposed by default.
- CORS is `*` for local dev only — set explicit origins for any deployment.

## Known limitations

See [`docs/RESEARCH_LIMITATIONS.md`](docs/RESEARCH_LIMITATIONS.md). Highlights: single-GPU/low-RAM dev environment; JEPA is educational-scale (not foundation-scale); "output agreement" is not formal mAP; phone-local browser inference and native mobile are planned, not built.

## License

Code in this repository: MIT. YOLOv8 weights: AGPL-3.0 (Ultralytics). SmolVLM/Qwen2.5-VL: Apache-2.0. Review model licenses before any commercial use.
