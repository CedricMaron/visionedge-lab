# Model Optimization

How VisionEdge Lab makes detection models smaller and faster without lying about the
result. This covers numeric precisions, the export/runtime pipeline, quantization and
calibration, the report schema, and the non-negotiable rule that **every optimization must
be validated** before it is trusted. The tooling lives in `scripts/` (see
`scripts/README.md`); this document is the conceptual companion.

## 1. Numeric precisions

| Precision | Bits | What it buys | Honest caveats |
| --- | --- | --- | --- |
| **FP32** | 32 | reference accuracy | largest, slowest — the baseline everything is compared to |
| **FP16** | 16 | ~½ size; faster on FP16-capable HW | speed win needs GPU / OpenVINO / TensorRT; on pure-CPU ORT it mostly just halves the file |
| **BF16** | 16 | FP32-like range, half size | needs BF16-capable hardware (recent CPUs/GPUs); **not exercised in this CPU build** |
| **INT8** | 8 | ~¼ size; big speedup on INT8 kernels | needs calibration; quality drops with poor calibration data; speedup needs a runtime with int8 kernels |

The capability scanner (`backend/app/capabilities/scanner.py`) reports which precisions the
current machine can actually accelerate (`supported_precisions`), so the UI/tooling never
offers a precision the hardware cannot use.

## 2. The export + runtime pipeline

```
YOLOv8 .pt (Ultralytics)
      │  scripts/export_onnx.py   (opset 12, imgsz 640, static shapes)
      ▼
   .onnx (FP32)  ──► scripts/simplify_onnx.py (onnxslim: fold/fuse/prune)
      │
      ├─► scripts/quantize_onnx.py --precision fp16   → .fp16.onnx
      ├─► scripts/quantize_onnx.py --precision int8   → .int8.onnx  (static/dynamic)
      ├─► scripts/convert_openvino.py                 → OpenVINO IR (.xml/.bin)
      └─► scripts/build_tensorrt.py                   → .engine (GPU-specific)
```

- **ONNX** is the portable interchange format. Export uses **static shapes** and opset 12
  by default (widely supported, including ONNX Runtime Web).
- **ONNX Runtime (ORT)** is the default execution engine; it is what
  `backend/app/inference/onnx_backend.py` and `scripts/benchmark_model.py` actually run.
  ORT selects execution providers from what it genuinely reports (CPU here; CUDA only if
  truly available) — see `RESEARCH_LIMITATIONS.md`.
- **OpenVINO** targets Intel CPU/iGPU. Interface + conversion script are implemented;
  **openvino is not installed in this build**, so `convert_openvino.py` exits with a clean
  install message.
- **TensorRT** targets NVIDIA GPUs. Engines are **hardware/driver/version-specific and must
  be built on the target machine** — they are never shipped. Interface + build script are
  implemented; **tensorrt is not installed / no usable GPU here**, so `build_tensorrt.py`
  exits cleanly.

## 3. Quantization and calibration

- **FP16** (`quantize_onnx.py --precision fp16`): uses `onnxconverter_common.float16` when
  installed, else the `onnxruntime.transformers.float16` tool, else a clear install error.
  Halves file size; measured to reproduce the FP32 reference closely on the sample image
  (high matched-box IoU) — but as above, the *latency* win depends on the runtime.
- **INT8 dynamic** (`--precision int8 --dynamic`): weights quantized, activations quantized
  at runtime. No calibration data needed. Smallest file; good default when you lack
  representative frames.
- **INT8 static** (`--precision int8 --calibration-dir DIR`): activation ranges are
  estimated from real images via a `CalibrationDataReader` that applies the **exact
  inference-time letterbox preprocessing** (`app.inference.preprocess`). Usually higher
  quality than dynamic — *if* the calibration set is representative.

### Calibration data matters — a lot

INT8 static quantization is only as good as its calibration images. Non-representative or
near-duplicate frames give wrong activation ranges and degrade detection quality.
`scripts/calibrate.py` builds/inspects a calibration set (decodes each image, letterboxes
it, reports per-channel mean/std, warns when there are too few images) and writes a
manifest. Prefer a few hundred varied, real deployment frames over many similar ones.

## 4. The optimization-report schema

Every optimization script writes a sidecar `<model>.json`; `benchmark_model.py` writes a
timestamped report into `benchmark-data/`. The shared shape:

```jsonc
{
  "kind": "onnx_export | onnx_simplify | onnx_quantize | openvino_ir | tensorrt_engine | benchmark",
  "created_utc": "ISO-8601",
  "source": "path to the input model",
  "source_sha256": "…",
  "file": "produced file name",
  "size_bytes": 6543210,
  "sha256": "…",
  "opset": 12,
  "inputs":  [{"name": "images",  "shape": [1, 3, 640, 640]}],
  "outputs": [{"name": "output0", "shape": [1, 84, 8400]}],
  "verification": {                       // proof the artifact actually works
    "loaded_in_onnxruntime": true,
    "zero_input_all_finite": true,
    "output_shapes": [[1, 84, 8400]]
  },
  "output_agreement": { /* validate_onnx result vs FP32 reference, when available */ },
  "tooling": { "python": "3.12.x", "onnx": "…", "onnxruntime": "…", "torch": "…" }
}
```

The benchmark report additionally carries `hardware` (from the capability scanner — CPU
model, cores, RAM, GPUs), `os`, `runtime` (provider + versions), `config` (runs/warmup/
size), and `metrics` (`end_to_end_ms` mean/p50/p95/p99, `fps_end_to_end`, `rss_mb`). These
numbers are **measured on the reporting machine** and are only comparable within that host.

## 5. The validation rule (non-negotiable)

An "optimized" model is worthless — or worse, silently wrong — if it is not checked. Every
script therefore verifies its output, and the workflow adds semantic validation:

1. **It loads.** The produced model is re-opened in its runtime (ORT / OpenVINO / TensorRT
   deserialize). A model that will not load is a hard failure, not a warning.
2. **It is finite.** A zero (or real) input is run and every output element is checked with
   `np.isfinite`. NaN/Inf ⇒ hard failure.
3. **Shape is preserved.** The output rank/shape must match the FP32 model (e.g.
   `[1, 84, 8400]` for YOLOv8-640). `simplify_onnx.py` fails if the shape changes.
4. **It still agrees with the reference.** `validate_onnx.py` (and the agreement step baked
   into `quantize_onnx.py`) compares detections to the FP32 `.pt` reference: detection
   count, class multiset, **mean IoU of matched boxes**, and confidence delta. A good FP16
   model shows near-identical detections and high matched-box IoU. This is reported as
   **output agreement, never mAP**, because there is no labelled validation set — a point
   worth repeating so no one mistakes agreement for accuracy.
5. **It is actually faster / smaller.** File size is reported by every script; latency is
   measured by `benchmark_model.py`. An "optimization" that does not improve size or latency
   on the target runtime is not an optimization there, and the numbers make that visible
   rather than assumed.

Cosine similarity of raw output tensors to the reference is a natural additional check for
FP16/INT8 (closeness in logit space before decode); the current tooling uses the stronger,
task-level detection-agreement check (matched-box IoU + confidence delta) as the primary
signal, with tensor-level finiteness/shape as the gate.

## 6. Implemented vs. interface-only in this build

| Technique | Status here |
| --- | --- |
| ONNX export (nano/small/medium), simplify | **Implemented, runs on CPU** |
| FP16 conversion | **Implemented** (via ORT transformers float16; `onnxconverter-common` preferred if installed) |
| INT8 dynamic + INT8 static (with calibration) | **Implemented, runs on CPU** |
| ORT CPU benchmark + output-agreement validation | **Implemented, runs** |
| OpenVINO IR conversion | Script implemented; **openvino not installed** → clean error |
| TensorRT engine build | Script implemented; **tensorrt/GPU not available** → clean error |
| BF16 | Not exercised (needs BF16 hardware) |
| Browser (ONNX Runtime Web) asset prep | Implemented; **browser inference is Phase 3** |

## References

- Ultralytics YOLOv8 export docs. <https://docs.ultralytics.com/modes/export/>
- ONNX Runtime — quantization. <https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html>
- Jacob et al., *Quantization and Training of Neural Networks for Efficient
  Integer-Arithmetic-Only Inference*, 2018. <https://arxiv.org/abs/1712.05877>
- Intel OpenVINO documentation. <https://docs.openvino.ai/>
- NVIDIA TensorRT documentation. <https://docs.nvidia.com/deeplearning/tensorrt/>
