"""ONNX Runtime adapter — the platform's primary working runtime.

Covers the CPU and CUDA execution providers. TensorRT is probed separately even
though it is also reached through ORT, because "TensorRT EP is listed" and
"TensorRT can build an engine for this model on this machine" are different claims
and only the second one matters.

A note on what ``session.run()`` measures: the Python API is blocking and returns
host-resident NumPy arrays. On the CUDA provider that means the returned time
already includes kernel execution *and* the device-to-host copy — there is no
outstanding asynchronous work to synchronize afterwards. This is genuinely
synchronized timing, but it is not purely kernel time, and the span carries that
note so nobody reads it as a pure GPU-execution figure.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from app.core.logging import get_logger
from app.runtimes.base import (
    BaseRuntimeAdapter,
    RuntimeCapability,
    SessionConfig,
    SessionHandle,
)
from app.schemas.enums import DeviceKind, Precision

log = get_logger("runtimes.onnxruntime")

_PROVIDER_FOR_DEVICE = {
    DeviceKind.CUDA: "CUDAExecutionProvider",
    DeviceKind.CPU: "CPUExecutionProvider",
}

_DEVICE_FOR_PROVIDER = {
    "CUDAExecutionProvider": DeviceKind.CUDA,
    "TensorrtExecutionProvider": DeviceKind.CUDA,
    "CPUExecutionProvider": DeviceKind.CPU,
    "OpenVINOExecutionProvider": DeviceKind.INTEL_GPU,
}

_OPT_LEVELS = {
    "disabled": "ORT_DISABLE_ALL",
    "basic": "ORT_ENABLE_BASIC",
    "extended": "ORT_ENABLE_EXTENDED",
    "all": "ORT_ENABLE_ALL",
}


class OnnxRuntimeAdapter(BaseRuntimeAdapter):
    """Executes ONNX graphs. Knows nothing about what the tensors mean."""

    runtime_id = "onnxruntime"

    def probe(self) -> RuntimeCapability:
        ort, version, error = self._import_version("onnxruntime")
        if ort is None:
            return RuntimeCapability(
                runtime_id=self.runtime_id,
                available=False,
                unavailable_reason=f"onnxruntime is not importable ({error})",
            )

        try:
            providers = list(ort.get_available_providers())
        except Exception as exc:  # noqa: BLE001
            return RuntimeCapability(
                runtime_id=self.runtime_id,
                available=False,
                unavailable_reason=f"onnxruntime imported but get_available_providers() failed: {exc}",
            )

        devices = [DeviceKind.CPU]
        if "CUDAExecutionProvider" in providers:
            devices.append(DeviceKind.CUDA)

        # fp16 is deliberately absent from the CPU list. ORT will run an fp16 graph on
        # CPU, but by inserting cast nodes around fp32 kernels — it is slower than fp32
        # and offers no memory benefit, so advertising it would send users toward a
        # configuration that is strictly worse.
        precisions_by_device = {
            DeviceKind.CPU: [Precision.FP32, Precision.INT8],
            DeviceKind.CUDA: [Precision.FP32, Precision.FP16, Precision.INT8],
        }
        precisions_by_device = {d: p for d, p in precisions_by_device.items() if d in devices}

        notes = [
            "session.run() is blocking and returns host-resident arrays, so measured "
            "model-execution time includes the device-to-host copy on GPU providers.",
        ]
        if "CUDAExecutionProvider" in providers:
            notes.append(
                "CUDAExecutionProvider is listed as available, which does not guarantee a "
                "session can be created on it — that is verified per load, not per probe."
            )

        return RuntimeCapability(
            runtime_id=self.runtime_id,
            available=True,
            version=version,
            execution_providers=providers,
            devices=devices,
            precisions_by_device=precisions_by_device,
            supports_device_synchronization=True,
            supports_profiling=True,
            notes=notes,
        )

    def create_session(self, config: SessionConfig) -> SessionHandle:
        import onnxruntime as ort

        path = Path(config.model_path)
        if not path.exists():
            raise FileNotFoundError(f"ONNX model not found: {path}")

        available = list(ort.get_available_providers())
        wanted = _PROVIDER_FOR_DEVICE.get(config.device)
        providers: list[str] = []
        if wanted and wanted != "CPUExecutionProvider":
            if wanted in available:
                providers.append(wanted)
            else:
                # Not an error yet — the honored check below turns it into one. Logging
                # it here means the reason is in the record even if the caller only sees
                # the mismatch message.
                log.warning(
                    "provider_unavailable", requested=wanted, available=available,
                )
        providers.append("CPUExecutionProvider")

        so = ort.SessionOptions()
        level = _OPT_LEVELS.get((config.graph_optimization_level or "all").lower())
        if level:
            so.graph_optimization_level = getattr(ort.GraphOptimizationLevel, level)
        if config.intra_op_threads is not None:
            so.intra_op_num_threads = config.intra_op_threads
        if config.inter_op_threads is not None:
            so.inter_op_num_threads = config.inter_op_threads
        if config.enable_profiling:
            so.enable_profiling = True

        t0 = time.perf_counter()
        session = ort.InferenceSession(str(path), sess_options=so, providers=providers)
        load_ms = (time.perf_counter() - t0) * 1000.0

        actual_provider = session.get_providers()[0]
        effective_device = _DEVICE_FOR_PROVIDER.get(actual_provider, DeviceKind.CPU)

        return SessionHandle(
            session=session,
            runtime_id=self.runtime_id,
            runtime_version=getattr(ort, "__version__", None),
            execution_provider=actual_provider,
            effective_device=effective_device,
            effective_precision=config.precision,
            requested_device=config.device,
            input_names=[i.name for i in session.get_inputs()],
            output_names=[o.name for o in session.get_outputs()],
            load_ms=load_ms,
            thread_config={
                "intra_op": so.intra_op_num_threads,
                "inter_op": so.inter_op_num_threads,
            },
        )

    def run(self, handle: SessionHandle, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        if handle.session is None:
            raise RuntimeError("session has been released")
        return handle.session.run(None, inputs)

    def synchronize(self, handle: SessionHandle) -> None:
        """No-op with a real justification.

        ``run()`` above has already blocked until outputs were copied to host memory,
        so there is no outstanding device work at this point. Spans around it are
        therefore correctly marked as synchronized.
        """
        return None

    def end_profiling(self, handle: SessionHandle) -> str | None:
        """Flush the ORT profile and return its path, when profiling was enabled."""
        if handle.session is None:
            return None
        try:
            return handle.session.end_profiling()
        except Exception as exc:  # noqa: BLE001
            log.warning("end_profiling_failed", error=str(exc))
            return None
