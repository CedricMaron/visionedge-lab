# VisionEdge Lab → InferenceLab: migration analysis

**Date:** 2026-07-26
**Baseline commit:** `626fb9f`
**Baseline test state:** 108 backend tests passing (`pytest -q`, 24.4 s)

This document records what the repository actually contains today, what survives the
migration, what is replaced, and in what order the work lands. It is written from a
full inspection of the tree, not from assumption.

---

## 1. Current architecture, as measured

### Stack

| Layer | Technology | Notes |
|---|---|---|
| Backend | FastAPI + Uvicorn, Python 3.12 | `backend/app`, factory pattern in `main.py::create_app` |
| Validation | Pydantic v2 + pydantic-settings | `core/types.py` is the declared contract source of truth |
| Frontend | React 18 + TypeScript 5.6 + Vite 5 | `frontend/src`, path alias `@/` |
| State | Zustand (3 stores, persisted) | `settingsStore`, `classStore`, `modelSwitchStore` |
| Styling | Tailwind 3.4, `darkMode: 'class'` | custom `surface-*` dark scale, 165 hard-coded usages |
| Charts | Recharts 2.12 | already a dependency, unused in most pages |
| Persistence | SQLite via stdlib `sqlite3` | `storage/db.py`, 2 tables, no migration system |
| Metrics | prometheus-client | `monitoring/metrics.py`, `/metrics` endpoint |
| Logging | structlog, JSON by default | `core/logging.py` |
| Tests | pytest (108, backend) + vitest (frontend) | `asyncio_mode = "auto"` |
| Lint | ruff (E,F,I,B,UP), line-length 110 | backend only; eslint for frontend |
| Container | Two Dockerfiles + compose (dev, prod) | CPU-first, GPU opt-in and commented out |
| Deploy | Caddy on a Windows VPS, PowerShell scripts | `deploy/`, three sites, live at visionedge.c-maron.space |
| CI | **none** | no `.github/workflows` — a gap, not a constraint |
| Auth | **none** | rate limiting per IP only (`api/ratelimit.py`) |

### Backend module inventory

```
app/
  api/          meta, detection, imaging, vlm, advisor, ratelimit    (6 routers)
  capabilities/ scanner.py — real probes: CPU, RAM, NVML-free GPU via nvidia-smi, ORT/torch/OV/TRT
  core/         config (env, VE_ prefix), errors, logging, state (AppState), types (contracts)
  inference/    base, config, factory, manager, onnx/openvino/pytorch/tensorrt backends,
                preprocess (letterbox), postprocess (from-scratch YOLOv8 decode + NMS)
  models/       registry.py (pydantic-validated registry.json), coco.py
  benchmarking/ auto.py — background benchmark on model switch
  jobs/         manager, state, worker — generic background job runner with progress
  monitoring/   metrics.py — RollingMetrics + prometheus collectors
  storage/      db.py — benchmarks + sessions tables
  vlm/          base, manager, mock/local/remote backends, prompting, structured_output, evaluation
  jepa/         12 modules — masking, EMA, encoders, trainers, collapse monitor
  temporal/     frame buffer, sampler, scene change, anomaly
  representation/ embedding store, encoders, retrieval, visualization
  orchestration/  execution planner, invocation policy, resource manager
```

### Measured environment (this development box)

| Property | Value | Consequence for the migration |
|---|---|---|
| CPU | 12 logical cores | fine |
| RAM | **3 GB total, ~1 GB available** | **binding constraint** — rules out torch-GPU, transformers, diffusers, vLLM locally |
| GPU | RTX 2060, 6 GB, driver via WSL2 | usable |
| NVML | `nvidia-ml-py` installed **and working** | power draw, temperature, clocks, utilization, VRAM are all genuinely readable |
| ONNX Runtime | 1.20.1 **gpu build** | providers: TensorRT, CUDA, CPU |
| torch | 2.13.0 **+cpu** | CPU-only; GPU work must go through ORT |
| Network | HuggingFace + PyPI reachable | model downloads are possible |

Verified directly:

```
providers: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
NVML name: NVIDIA GeForce RTX 2060
NVML power W: 6.025      NVML util: 0      NVML temp: 58      NVML mem: 356/6144 MB
```

This is the single most important finding for the brief: **energy and hardware
utilization metrics can be genuinely measured on this machine**, so §10 and §11 of the
brief do not have to degrade to "unavailable". The production VPS has no GPU, so both
paths — probe present and probe absent — will be exercised in practice.

---

## 2. What is reused

The existing codebase is closer to the target than a rewrite would suggest. These
assets are kept and built upon rather than replaced:

| Asset | Why it survives |
|---|---|
| `core/types.py` contract discipline | Backend Pydantic models mirrored exactly by `frontend/src/types`. This is the pattern the new versioned schemas extend, not replace. |
| `capabilities/scanner.py` | Already the "never claim what you cannot probe" implementation the brief asks for in §5 and §23. Gains NVML, instruction sets, and per-runtime versions. |
| `inference/manager.py` | The load → warmup → verify → rollback state machine, and `_verify_runtime_honored` which refuses to report `onnxruntime-cuda` when ORT silently fell back to CPU. This is exactly §31's "do not silently fall back to another runtime". Generalized from detection to any task. |
| `inference/postprocess.py` | From-scratch NumPy YOLOv8 decode + NMS. Moves into the detection adapter unchanged. |
| `models/registry.py` | Pydantic-validated registry with checksum verification and disk-derived deployment status. Extended with adapter/task/modality fields. |
| `jobs/` | Generic job manager with progress, cancellation and terminal states. Becomes the benchmark run executor. |
| `storage/db.py` | Kept, but gains a migration mechanism and raw-iteration storage. |
| `monitoring/metrics.py` | Prometheus collectors and rolling window stay. |
| `api/ratelimit.py`, error mapping, structlog setup | Unchanged. |
| Deployment stack (Caddy, compose, PowerShell) | Unchanged except for renamed hostnames/paths where safe. |
| Frontend shell: routing, `useAsync`, `http.ts`, stores | Structure is sound; presentation is retheming. |

**Estimated reuse: roughly 70% of backend modules, 60% of frontend.**

## 3. What is replaced or generalized

| Current | Problem against the brief | Action |
|---|---|---|
| `DetectionBackend` protocol | Task-specific: `predict(image, conf, iou, classes) -> list[Detection]`. Cannot express tokenization, streaming, batching, or an audio input. | Generalize to `ModelAdapter` with `load/preprocess/infer/postprocess/evaluate/unload`. `DetectionBackend` becomes one adapter family. |
| Runtime choice welded into backend classes | `OnnxRuntimeBackend`, `PyTorchBackend`, … each re-implement session setup **and** decode. §5 requires runtime independent of model. | Split: `RuntimeAdapter` owns session/provider/threads/precision; `ModelAdapter` owns pre/post. |
| `BenchmarkResult` (flat, 13 fields, averages only) | §7/§19 require raw per-iteration samples, P90/P99, stddev, CV, sample counts, failed iterations. Storing only aggregates is explicitly forbidden. | New versioned `BenchmarkRun` schema; raw iterations persisted in their own table. |
| Timing: 3 stages (`preprocess/inference/postprocess`) | §6 requires up to 16 phases including transfers, queueing, serialization, and device synchronization. | Hierarchical span timeline on a monotonic clock. |
| `benchmark()` measures CPU dispatch only | On CUDA this measures dispatch, not execution — §6 forbids this. | Explicit device synchronization before stopping the clock; CUDA events where the runtime exposes them. |
| GPU probing via `nvidia-smi` subprocess | ~8 ms per call, coarse, no power/clocks. | NVML in-process, with the `nvidia-smi` path retained as fallback. |
| `execution_location` hardcoded to `PC_LOCAL` in the WS handler | Wrong data written to every session row. | Client declares its device; server records what actually connected. |
| Dark-only Tailwind palette | §2 asks for minimal/technical/professional. | Semantic CSS-variable tokens, light default. |
| `VE_` env prefix, `visionedge.db`, package names | Rebrand. | `IL_` prefix with `VE_` accepted as a deprecated alias for one release; DB file renamed with automatic adoption of the old file if present. |

## 4. What is explicitly *not* attempted

Honesty about scope matters as much as honesty about metrics. On a box with ~1 GB free
RAM and a CPU-only torch build:

- **vLLM, TGI, diffusers, TensorRT engine building, MLX, Core ML, TFLite** — interfaces
  are defined per §5 so the architecture is complete, but each reports
  `available: false` with the concrete reason from its capability probe. None will be
  claimed as working.
- **Image generation, video generation, TTS, STT with real weights** — adapter contracts
  and scenario definitions land; the models do not fit this machine. Their registry
  entries carry `not_installed_reason` and install instructions.
- The mock adapter (§26) is the only adapter that fabricates anything, it is labelled a
  test adapter, and it is filtered out of production model listings.

The three modalities that will genuinely work end to end, chosen because they fit in
RAM and run on the ONNX Runtime that is already installed and GPU-capable:

1. **Object detection** — YOLOv8n ONNX (existing, already validated)
2. **Image classification** — a small ONNX classifier
3. **Text embedding** — a small ONNX sentence embedding model

This satisfies acceptance criterion §30.5 ("at least three modalities work end to end")
with real measurements rather than three shallow integrations.

---

## 5. Migration order

Each phase ends with tests, lint, typecheck and a commit. Existing behaviour is verified
after every phase — the 108-test baseline must never regress.

| Phase | Content | Risk |
|---|---|---|
| **1. Foundation** | Rename to InferenceLab. Versioned schemas (`BenchmarkRun`, `PhaseTiming`, `IterationSample`, `EnvironmentFingerprint`). `ModelAdapter` + `RuntimeAdapter` contracts. Task/Modality enums. No behaviour change. | Low — additive |
| **2. Instrumentation** | Span timeline on monotonic clocks, percentile aggregation, psutil + NVML samplers, memory allocated/reserved/peak, energy integration, overhead measurement. Standalone and unit-tested before anything depends on it. | Low — new package |
| **3. Vision on adapters** | Detection re-expressed as a `ModelAdapter` over an ORT `RuntimeAdapter`. Old `DetectionBackend` kept as a thin shim so `api/detection.py` and its tests keep working. Mock adapter added. | **Medium** — the one place existing behaviour can regress |
| **4. Modalities 2 and 3** | Classification and embedding adapters + registry entries + download script. | Low |
| **5. Benchmark engine + CLI** | Scenarios, warmup/measured/cooldown, cold vs warm, failure tracking, raw persistence with DB migration, exports, `inference-lab` CLI sharing the engine with the API. | Medium — DB migration |
| **6. Frontend** | Rebrand, semantic tokens/light theme, new navigation, latency decomposition, hardware time series, system page with capability matrix, comparison guards. | Medium — wide but mechanical |
| **7. Remote timing** | Correlation IDs, server-reported phase timings, explicit *residual overhead* labelling per §18. | Low |
| **8. Docs + CI** | Methodology and metrics docs finalized against the shipped code; add the missing CI workflow. | Low |

Phases 1 and 2 are pure additions and carry no regression risk. Phase 3 is the hinge:
it is where the existing, working, tested detection path is re-pointed at the new
contracts, and it is guarded by the existing test suite plus new adapter tests.

## 6. Compatibility commitments

- Existing API routes keep working. `/api/infer`, `/api/detection/*`, `/api/ws/detect`,
  `/api/capabilities`, `/api/models`, `/api/benchmarks` retain their paths and response
  shapes; new fields are additive.
- The WebSocket protocol accepts both the current bare-JPEG frames and the new
  correlation-tagged frames, so an old client cannot break.
- `VE_`-prefixed environment variables continue to be read, with a deprecation warning,
  alongside the new `IL_` prefix.
- The existing SQLite file is adopted rather than discarded; migrations are additive and
  versioned via `PRAGMA user_version`.
- The three deployed sites and their Caddy configuration keep working throughout.

## 7. Known risks

| Risk | Mitigation |
|---|---|
| 1 GB free RAM makes some model downloads impossible | Model set chosen to fit; download script checks available RAM and disk first and refuses with a clear message rather than OOM-ing |
| ORT CUDA provider is listed but may fail to create a session | Already handled: `_verify_runtime_honored` raises instead of silently reporting CUDA while running on CPU. Behaviour is kept and tested. |
| NVML present on dev box, absent on the CPU-only VPS | Every probe is optional and returns an explicit "unavailable + reason", which the UI renders as such. Both paths are tested. |
| Instrumentation distorts what it measures | Overhead is itself measured and reported per benchmark mode (Standard/Detailed/Profiler), and the modes are marked non-comparable. |
| Wide frontend retheme touching 19 files | Old palette classes are removed from the Tailwind config so a missed usage fails the build rather than rendering a dark patch. |
