# Architecture

> **InferenceLab** — multimodal AI inference, profiling and benchmarking.
>
> This document describes the platform architecture after the InferenceLab migration.
> For the original VisionEdge Lab design and what carried forward, see
> [MIGRATION.md](MIGRATION.md).

## Layered design

The platform separates concerns that are usually tangled, and the separation is what
makes the (model × runtime × device × precision) matrix real rather than asserted:

```
                       ┌──────────────────────────────┐
                       │  Web UI          CLI         │   one engine, two front doors
                       └──────────────┬───────────────┘
                       ┌──────────────▼───────────────┐
                       │      Benchmark engine        │   lifecycle, integrity rules
                       └──┬────────────┬───────────┬──┘
          ┌───────────────▼──┐  ┌──────▼───────┐  ┌▼──────────────────┐
          │  Model adapters  │  │ Instrumenta- │  │  Runtime adapters │
          │  (task-specific) │  │ tion         │  │  (execution)      │
          └──────────────────┘  └──────────────┘  └───────────────────┘
                       ┌──────────────────────────────┐
                       │  Schemas · Storage · Probes  │
                       └──────────────────────────────┘
```

| Layer | Owns | Never touches |
|---|---|---|
| **Schemas** (`app/schemas/`) | Versioned contracts, `Measurement` provenance | I/O |
| **Model adapters** (`app/adapters/`) | Pre/post-processing, quality evaluation, metadata | Sessions, providers, clocks |
| **Runtime adapters** (`app/runtimes/`) | Sessions, devices, threads, synchronization | Tensor meaning |
| **Instrumentation** (`app/instrumentation/`) | Timing, probes, sampling, energy, environment | Execution |
| **Benchmark engine** (`app/benchmark/`) | Lifecycle, statistics, integrity warnings | Model or runtime specifics |
| **Storage** (`app/storage/`) | Versioned SQLite, raw iteration retention | Metric computation |
| **API / CLI** | Selection, filtering, serialization | Computing any metric |

The last row matters: **no route computes a metric.** Both front doors call the same
engine, so the web UI and the command line cannot disagree about what a number means.

## Core invariants

1. **No metric without provenance.** Every value is a `Measurement` carrying its unit,
   kind (measured / derived / estimated), instrumentation source, and — when absent —
   the reason. Pydantic validators reject a valueless measurement with no reason and an
   estimate with no documented methodology.
2. **No claim without a probe.** A runtime is offered only if its capability probe
   succeeded on this machine.
3. **No silent fallback.** A session that landed on a different device than requested
   is refused, not adopted.
4. **No averages without evidence.** Raw per-iteration samples are persisted; any
   percentile can be recomputed from stored data.
5. **No hidden exclusions.** Warm-up and failed iterations are retained, marked, and
   excluded from statistics — with the exclusion stated in the result.

## Documentation map

| Document | Contents |
|---|---|
| [MIGRATION.md](MIGRATION.md) | VisionEdge Lab → InferenceLab: inventory, reuse, order |
| [BENCHMARK_METHODOLOGY.md](BENCHMARK_METHODOLOGY.md) | Clocks, synchronization, sampling, energy, integrity |
| [METRICS.md](METRICS.md) | Every metric: unit, source, formula, limits |
| [MODEL_ADAPTERS.md](MODEL_ADAPTERS.md) | Adapter contract and correctness traps |
| [RUNTIMES.md](RUNTIMES.md) | Runtime contract and capability states |
| [DEVELOPMENT.md](DEVELOPMENT.md) | Setup, commands, conventions |

---

## Original VisionEdge Lab architecture

Retained below for the vision slice, which continues to work unchanged.

VisionEdge Lab combines three layers of visual intelligence over a single camera stream:
**detection** (where are the objects), **vision-language** (what is happening), and
**JEPA / world-model** (how the scene changes and what comes next). This document shows the
end-to-end data flow, the resource constraint that shapes it, the deployment modes, and the
honest status of each part.

## 1. End-to-end data flow

```
 ┌──────────┐
 │  Camera  │  webcam / phone / RTSP  (frames decoded to BGR uint8)
 └────┬─────┘
      ▼
 ┌──────────────────┐   bounded queue (max_frame_queue), drop-under-load
 │ Frame Scheduler  │   backpressure: newest frames win, no unbounded buffering
 └────┬─────────────┘
      ▼
 ┌───────────────────────────────────────────────────────────────┐
 │  Per-frame fan-out (only ONE heavy model resident at a time)   │
 │                                                                 │
 │   ┌───────────┐     ┌───────────┐     ┌────────────────────┐   │
 │   │ Detector  │     │  Encoder  │     │  Frame Buffer      │   │
 │   │ (YOLOv8→  │     │ (JEPA ViT │     │  (RAM ring, 64;    │   │
 │   │  ONNX/ORT)│     │  features)│     │   nothing on disk) │   │
 │   └────┬──────┘     └────┬──────┘     └─────────┬──────────┘   │
 └────────┼─────────────────┼──────────────────────┼─────────────┘
          ▼                 ▼                       ▼
   ┌────────────┐    ┌──────────────┐      ┌──────────────────┐
   │  Tracker   │    │  Embedding   │      │  JEPA future     │
   │ (box IDs)  │    │  store /     │      │  predictor       │
   │            │    │  retrieval   │      │  (video trainer) │
   └─────┬──────┘    └──────┬───────┘      └────────┬─────────┘
         │                  │                       │ prediction error
         └──────────┬───────┴───────────────────────┘
                    ▼
          ┌───────────────────────────┐
          │  Event Router / Anomaly   │  calibrated surprise (z-score → 0..1);
          │  (temporal/anomaly.py)    │  scene-change + detector events
          └────────────┬──────────────┘
                       ▼  escalate only when it is worth the cost
          ┌───────────────────────────┐
          │  VLM Invocation Policy     │  when/whether to call the expensive model
          │  (orchestration/…)         │
          └───────┬───────────┬────────┘
                  ▼           ▼
        ┌──────────────┐  ┌──────────────────┐
        │  Local VLM   │  │  Server VLM      │  (opt-in; privacy-gated)
        │ (mock /      │  │ (remote OpenAI-  │
        │  SmolVLM)    │  │  compatible)     │
        └──────┬───────┘  └────────┬─────────┘
               └───────┬───────────┘
                       ▼
        ┌───────────────────────────────────────────┐
        │  UI  ·  Logs / Metrics  ·  DB (SQLite)     │  results, events, telemetry
        └───────────────────────────────────────────┘
```

Where each block lives in code:

| Block | Module(s) |
| --- | --- |
| Frame scheduler / backpressure | `app/transport`, `app/jobs` (`worker.py`, `manager.py`) |
| Detector | `app/inference` (`onnx_backend.py`, `factory.py`, `manager.py`) |
| JEPA encoder / embeddings | `app/jepa`, `app/representation` (`embedding_store.py`, `retrieval.py`) |
| Frame buffer / sampling / scene change | `app/temporal` (`frame_buffer.py`, `frame_sampler.py`, `scene_change.py`) |
| Anomaly / event routing | `app/temporal/anomaly.py`, `app/orchestration` |
| Invocation policy / resource coordination | `app/orchestration` (`invocation_policy.py`, `resource_manager.py`, `execution_planner.py`) |
| VLM backends | `app/vlm` (`mock_backend.py`, `local_backend.py`, `remote_backend.py`, `manager.py`) |
| API | `app/api` (`detection.py`, `vlm.py`, `imaging.py`, `advisor.py`, `meta.py`) |
| Storage / metrics | `app/storage/db.py`, `app/monitoring/metrics.py` |

## 2. The resource constraint that shapes everything

On the reference hardware the detector, a VLM, and a JEPA model **cannot all reside in
memory at once**. Rather than pretend otherwise, the architecture makes this explicit: a
**resource manager / invocation policy** (`app/orchestration`) coordinates which heavy
model is loaded, and model switching **releases memory** (backends implement `close()` /
`unload()`; the VLM manager unloads the previous model on a successful switch and rolls back
on failure). This is why the pipeline is a *fan-out with a policy gate* rather than three
always-on models.

## 3. Deployment modes and their honest status

| Mode | What runs where | Status in this build |
| --- | --- | --- |
| **PC-local** | Detector (ONNX/ORT CPU) + mock VLM + JEPA all on one machine, no network | **Implemented and runs.** The default. |
| **Phone-local** | Browser inference (ONNX Runtime Web / WebGPU) + phone-local VLM | **Planned (Phase 3).** `prepare_browser_model.py` stages the ONNX asset + manifest; in-browser inference is not wired up. |
| **Local-server** | Detector/JEPA on a workstation, browser/phone as a thin client over LAN | **Partially implemented.** Server + API run; no host/IP is hardcoded (see `.env.example`). Server-assisted mode is never called "fully local". |
| **Remote-server** | Heavy VLM (Qwen2.5-VL class) on a GPU server via OpenAI-compatible API | **Interface implemented, opt-in.** `RemoteVLMBackend` works when an endpoint is configured; **off by default and privacy-gated** (`allow_frame_transmission` / `VE_ALLOW_FRAME_TRANSMISSION`). No frame leaves the machine unless explicitly enabled. |

## 4. Design invariants (enforced, not aspirational)

- **No fabricated outputs or benchmarks.** Every metric is measured; the mock VLM is
  labelled a mock everywhere it appears.
- **Never claim an unavailable runtime.** ORT selects providers from what it truly reports;
  OpenVINO/TensorRT are offered only when importable *and* a model/engine exists.
- **Frames are not stored by default.** The frame buffer is RAM-only; remote transmission is
  an explicit opt-in.
- **Server-assisted is never "fully local."** The execution location travels with every VLM
  response (`execution_location`).
- **The lightweight I-JEPA is never equated with Meta's V-JEPA.** See
  `JEPA_ARCHITECTURE.md` and `RESEARCH_LIMITATIONS.md`.
- **Model switches release memory.** Enforced by `close()`/`unload()` and the manager's
  last-known-good rollback.

## 5. Where to read more

- Detection optimization + runtimes: `MODEL_OPTIMIZATION.md`
- Vision-language slice: `VLM_ARCHITECTURE.md`
- JEPA / representation learning: `JEPA_ARCHITECTURE.md`
- World-model / anomaly experiment: `WORLD_MODEL_EXPERIMENT.md`
- How everything is evaluated: `MULTIMODAL_EVALUATION.md`
- The unvarnished caveats: `RESEARCH_LIMITATIONS.md`
