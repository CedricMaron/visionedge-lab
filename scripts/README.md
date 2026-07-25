# Model-optimization scripts

Command-line tools for exporting, optimizing, validating and benchmarking the detection
models used by VisionEdge Lab. They reuse the real inference code in `backend/app`
(`OnnxRuntimeBackend`, the letterbox preprocess, the from-scratch YOLOv8 decode, the
capability scanner, the model registry) so results reflect the actual runtime, not a
parallel implementation.

## Ground rules these scripts follow

- **No fabricated numbers.** `benchmark_model.py` measures with `time.perf_counter()` on
  this machine. Nothing prints a metric it did not measure.
- **Honest availability.** A script never claims a runtime is present when it is not. If
  OpenVINO / TensorRT / an FP16 tool is missing, it prints a clear, actionable install
  message and exits non-zero (no traceback).
- **Every produced artifact is verified** (reloaded in its runtime, checked for the
  expected output shape and finite values) and gets a sidecar `<model>.json` with sha256,
  size, shapes, opset and the tooling versions used.
- **Optimization != accuracy.** Quality is reported as *output agreement* against the FP32
  reference, never as mAP — there is no labelled validation set here.

## Running

From the repo root, with the backend on the path:

```bash
cd /path/to/demo_vision
YOLO_CONFIG_DIR=/tmp/Ultralytics PYTHONPATH=backend backend/.venv/bin/python scripts/<name>.py --help
```

Every script supports `--help`.

## Commands

| Script | Purpose | Example |
| --- | --- | --- |
| `export_onnx.py` | Export YOLOv8 nano/small/medium to ONNX via Ultralytics; sidecar + verify | `python scripts/export_onnx.py --model nano --size 640` |
| `simplify_onnx.py` | Simplify/fuse an ONNX graph with onnxslim; verify same output shape | `python scripts/simplify_onnx.py --input models/yolov8n.onnx --output /tmp/s.onnx` |
| `validate_onnx.py` | Run ONNX via the real backend; output-AGREEMENT report vs the `.pt` reference | `python scripts/validate_onnx.py --input models/yolov8n.onnx` |
| `quantize_onnx.py` | FP16 or INT8 (static/dynamic) quantization; verify + agreement | `python scripts/quantize_onnx.py --input models/yolov8n.onnx --precision fp16 --output /tmp/fp16.onnx` |
| `convert_openvino.py` | Convert ONNX to OpenVINO IR (opt-in; clean error if not installed) | `python scripts/convert_openvino.py --input models/yolov8n.onnx --precision fp16` |
| `build_tensorrt.py` | Build a TensorRT engine (GPU-only, opt-in; clean error if not installed) | `python scripts/build_tensorrt.py --input models/yolov8n.onnx --precision fp16` |
| `benchmark_model.py` | MEASURE latency (mean/p50/p95/p99), FPS, RSS; write a JSON report | `python scripts/benchmark_model.py --model models/yolov8n.onnx --runs 50` |
| `benchmark_cpu_vs_cuda.py` | Same model on the CPU vs CUDA provider; reports the provider ORT *actually* used, compares detections, claims a speedup only if CUDA really ran | `python scripts/benchmark_cpu_vs_cuda.py --model models/yolov8n.onnx --runs 50` |
| `download_models.py` | List registry models / install a single one (with a >5GB guard) | `python scripts/download_models.py --list` |
| `checksum.py` | sha256 + size; optionally update the registry entry | `python scripts/checksum.py --input models/yolov8n.onnx --update-registry` |
| `prepare_browser_model.py` | Stage an ONNX for ONNX Runtime Web (Phase 3) with a manifest | `python scripts/prepare_browser_model.py --input models/yolov8n.onnx --fp16` |
| `calibrate.py` | Build/inspect an INT8 calibration set (decode, letterbox, mean/std, manifest) | `python scripts/calibrate.py --images calibration --samples 100` |

`_common.py` is a shared helper module (path bootstrap, checksums, sidecar writing, ONNX
introspection, consistent error reporting) imported by the others; it is not run directly.

## Typical workflows

**Export → validate → benchmark**

```bash
python scripts/export_onnx.py --model small --size 640
python scripts/validate_onnx.py --input models/yolov8s.onnx
python scripts/benchmark_model.py --model models/yolov8s.onnx --runs 50
```

**Quantize with agreement + benchmark**

```bash
python scripts/calibrate.py --images calibration --samples 100
python scripts/quantize_onnx.py --input models/yolov8n.onnx --precision int8 --calibration-dir calibration
python scripts/benchmark_model.py --model models/yolov8n.int8.onnx --runs 50
```

## What actually runs in this build

| Path | Status here |
| --- | --- |
| ONNX export / simplify / FP16 / INT8 (dynamic + static) | Implemented and runs on CPU |
| ONNX Runtime CPU benchmark + validation | Implemented and runs |
| ONNX Runtime CUDA provider | Script implemented; on this box the EP **fails to load** (missing CUDA libs) → the comparison says so and claims no speedup |
| OpenVINO IR conversion | Interface + script implemented; **openvino not installed** → clean error |
| TensorRT engine build | Interface + script implemented; **tensorrt not installed / no GPU** → clean error |
| Browser (ONNX Runtime Web) asset prep | Implemented; **browser inference itself is Phase 3** |

FP16/INT8 produce valid, verified graphs on CPU, but the *speed* benefit of low precision
is realized mainly on runtimes/hardware with native FP16/INT8 kernels (GPU, OpenVINO,
TensorRT). On a pure-CPU ORT build the main win is a smaller file. This is stated in each
script's output rather than assumed.
