#!/usr/bin/env python3
"""Build a TensorRT engine from an ONNX detector (GPU-only, opt-in).

IMPORTANT: TensorRT engines are HARDWARE- AND DRIVER-SPECIFIC. An engine built on one
GPU/driver/TensorRT version will not (reliably) load on another. Engines therefore must be
built on the TARGET machine and are never shipped in this repo. This build does not have
tensorrt installed, so the script exits with a clean, actionable message rather than a
traceback. On a machine with tensorrt + pycuda, it builds and serializes an engine, then
verifies the engine deserializes.

Example:
    python scripts/build_tensorrt.py --input models/yolov8n.onnx --precision fp16
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _common as C

C.bootstrap_path()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", required=True, type=Path, help="input .onnx model")
    p.add_argument("--precision", required=True, choices=["fp16", "int8"])
    p.add_argument("--calibration-dir", type=Path, default=None,
                   help="INT8 only: directory of representative images for calibration")
    p.add_argument("--output", type=Path, default=None,
                   help="output .engine path (default <input>.<precision>.engine)")
    p.add_argument("--workspace-mb", type=int, default=1024,
                   help="builder workspace pool size in MB (default 1024)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.input.exists():
        C.die(f"input model not found: {args.input}")

    try:
        import tensorrt as trt  # type: ignore
    except Exception:
        C.die(
            "TensorRT not installed; pip install -r backend/requirements/tensorrt.txt\n"
            "Note: TensorRT engines are GPU/driver/version-specific and must be built on "
            "the TARGET machine (NVIDIA GPU + matching CUDA/driver). They are never shipped "
            "portably in this repo."
        )
        return 1  # unreachable

    if args.precision == "int8" and args.calibration_dir is None:
        C.die("INT8 TensorRT requires --calibration-dir with representative images.")

    output = (args.output or args.input.with_suffix(f".{args.precision}.engine")).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    C.info(f"building TensorRT engine from {args.input} (precision={args.precision})")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(args.input, "rb") as f:
        if not parser.parse(f.read()):
            errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            C.die(f"failed to parse ONNX for TensorRT: {errs}")

    config = builder.create_builder_config()
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mb << 20)
    except Exception:
        config.max_workspace_size = args.workspace_mb << 20  # older TRT API

    if args.precision == "fp16":
        if not builder.platform_has_fast_fp16:
            C.warn("this GPU reports no fast FP16; engine may fall back to FP32")
        config.set_flag(trt.BuilderFlag.FP16)
    elif args.precision == "int8":
        if not builder.platform_has_fast_int8:
            C.warn("this GPU reports no fast INT8; engine may fall back")
        config.set_flag(trt.BuilderFlag.INT8)
        C.warn("INT8 TensorRT calibration requires an IInt8Calibrator over the calibration "
               "images; quality depends on representative data. Provide a calibrator wired to "
               "--calibration-dir for production use.")

    try:
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            C.die("TensorRT engine build returned None (build failed)")
        with open(output, "wb") as f:
            f.write(serialized)
    except Exception as exc:  # noqa: BLE001
        C.die(f"TensorRT build failed: {exc}")

    C.ok(f"wrote {output} ({C.human_size(C.file_size(output))})")

    # --- verification: deserialize the engine ---
    try:
        runtime = trt.Runtime(logger)
        with open(output, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            C.die("verification failed: engine did not deserialize")
        num_io = engine.num_io_tensors if hasattr(engine, "num_io_tensors") else engine.num_bindings
    except Exception as exc:  # noqa: BLE001
        C.die(f"engine verification failed: {exc}")
    C.ok(f"verified: engine deserialized ({num_io} IO tensors)")

    metadata = {
        "kind": "tensorrt_engine",
        "created_utc": C.now_iso(),
        "precision": args.precision,
        "source": str(args.input),
        "source_sha256": C.sha256_file(args.input),
        "file": output.name,
        "size_bytes": C.file_size(output),
        "sha256": C.sha256_file(output),
        "tensorrt_version": trt.__version__,
        "gpu_specific": True,
        "portable": False,
        "verification": {"deserialized": True, "io_tensors": int(num_io)},
        "notes": "Engine is specific to this GPU/driver/TensorRT version; rebuild on each target.",
        "tooling": C.runtime_versions(),
    }
    sidecar = C.write_sidecar(output, metadata)
    C.ok(f"metadata sidecar -> {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
