"""Runtime registry: every runtime the platform knows of, and what it can prove.

The brief (§5) asks for interfaces covering a broad set of runtimes while only
implementing those the environment supports. The honest way to express that is a
registry where every entry is probed and reports one of three states:

* **available** — an adapter exists and its probe succeeded here.
* **declared, not installed** — an adapter exists, but the dependency is absent.
* **declared, no adapter** — the platform knows the runtime exists but has not
  implemented execution for it.

The third state is the one usually hidden behind a greyed-out checkbox with no
explanation. It is spelled out instead, because a recruiter reading the capability
matrix should be able to tell the difference between "your machine lacks this" and
"this was never built".
"""
from __future__ import annotations

from dataclasses import dataclass

from app.runtimes.base import RuntimeAdapter, RuntimeCapability
from app.runtimes.onnxruntime_adapter import OnnxRuntimeAdapter
from app.schemas.enums import DeviceKind, Precision


@dataclass(frozen=True, slots=True)
class DeclaredRuntime:
    """A runtime the platform models but does not (yet) execute."""

    runtime_id: str
    display_name: str
    probe_module: str | None
    reason_when_missing: str
    adapter_status: str  # "implemented" | "not_implemented"
    notes: str = ""
    #: When set, the dependency counts as present only if ONNX Runtime actually lists
    #: this execution provider. Without it, anything probing the `onnxruntime` module
    #: would claim to be installed just because ORT is.
    required_ort_provider: str | None = None


# Runtimes with a working adapter in this repository.
_IMPLEMENTED: dict[str, type[RuntimeAdapter]] = {
    "onnxruntime": OnnxRuntimeAdapter,
}

# Runtimes modelled by the interfaces but not executed by this build. Each says why.
_DECLARED: tuple[DeclaredRuntime, ...] = (
    DeclaredRuntime(
        "pytorch", "PyTorch", "torch",
        "torch is not installed",
        "not_implemented",
        "torch is installed as a CPU-only build on the reference box; a PyTorch execution "
        "adapter is defined by the interface but not implemented, so it is not offered.",
    ),
    DeclaredRuntime(
        "pytorch-compile", "PyTorch (torch.compile)", "torch",
        "torch is not installed",
        "not_implemented",
        "Requires a compilation phase that the timeline models as graph_compilation.",
    ),
    DeclaredRuntime(
        "torchscript", "TorchScript", "torch",
        "torch is not installed", "not_implemented",
    ),
    DeclaredRuntime(
        "onnxruntime-directml", "ONNX Runtime (DirectML)", "onnxruntime",
        "DmlExecutionProvider is not among this ONNX Runtime build's providers "
        "(DirectML is Windows-only)",
        "not_implemented",
        required_ort_provider="DmlExecutionProvider",
    ),
    DeclaredRuntime(
        "tensorrt", "TensorRT", "tensorrt",
        "the tensorrt package is not installed", "not_implemented",
        "ONNX Runtime lists TensorrtExecutionProvider, but building an engine additionally "
        "requires the TensorRT libraries; engine build time is modelled as a cold-start phase.",
    ),
    DeclaredRuntime(
        "openvino", "OpenVINO", "openvino",
        "the openvino package is not installed", "not_implemented",
    ),
    DeclaredRuntime(
        "coreml", "Core ML", "coremltools",
        "Core ML is available on macOS only", "not_implemented",
    ),
    DeclaredRuntime(
        "tflite", "TensorFlow Lite", "tflite_runtime",
        "the tflite_runtime package is not installed", "not_implemented",
    ),
    DeclaredRuntime(
        "mlx", "MLX", "mlx",
        "MLX requires Apple silicon", "not_implemented",
    ),
    DeclaredRuntime(
        "llama-cpp", "llama.cpp", "llama_cpp",
        "the llama-cpp-python package is not installed", "not_implemented",
    ),
    DeclaredRuntime(
        "vllm", "vLLM", "vllm",
        "the vllm package is not installed", "not_implemented",
        "vLLM needs a CUDA torch build and several GB of RAM; neither is present here.",
    ),
    DeclaredRuntime(
        "transformers", "HuggingFace Transformers", "transformers",
        "the transformers package is not installed", "not_implemented",
    ),
    DeclaredRuntime(
        "tgi", "Text Generation Inference", None,
        "no TGI endpoint is configured", "not_implemented",
    ),
    DeclaredRuntime(
        "browser-webgpu", "Browser WebGPU", None,
        "runs in the browser, not in this process", "not_implemented",
        "Probed client-side; the server cannot answer for it.",
    ),
    DeclaredRuntime(
        "browser-wasm", "Browser WASM", None,
        "runs in the browser, not in this process", "not_implemented",
        "Probed client-side; the server cannot answer for it.",
    ),
    DeclaredRuntime(
        "remote-http", "Remote HTTP inference", None,
        "no remote endpoint is configured", "not_implemented",
    ),
    DeclaredRuntime(
        "remote-streaming", "Remote streaming inference", None,
        "no remote endpoint is configured", "not_implemented",
    ),
)


def get_adapter(runtime_id: str) -> RuntimeAdapter:
    """Return the adapter for ``runtime_id``, or raise if it is not implemented."""
    cls = _IMPLEMENTED.get(runtime_id)
    if cls is None:
        known = {d.runtime_id for d in _DECLARED}
        if runtime_id in known:
            raise NotImplementedError(
                f"runtime '{runtime_id}' is declared but has no execution adapter in this build"
            )
        raise KeyError(f"unknown runtime '{runtime_id}'")
    return cls()


def _module_present(module_name: str | None) -> bool:
    if module_name is None:
        return False
    try:
        __import__(module_name)
        return True
    except Exception:  # noqa: BLE001
        return False


def _ort_provider_present(provider: str) -> bool:
    try:
        import onnxruntime as ort

        return provider in ort.get_available_providers()
    except Exception:  # noqa: BLE001
        return False


def probe_declared(entry: DeclaredRuntime) -> RuntimeCapability:
    """Probe a declared-but-unimplemented runtime.

    Distinguishes "dependency missing" from "we never built this", because the two
    imply completely different next steps for anyone reading the matrix.
    """
    installed = _module_present(entry.probe_module)
    if installed and entry.required_ort_provider is not None:
        # The module being importable is not enough — the specific execution provider
        # has to actually be offered by this build.
        installed = _ort_provider_present(entry.required_ort_provider)
    if not installed:
        reason = entry.reason_when_missing
    else:
        reason = (
            f"{entry.display_name} is installed, but InferenceLab has no execution adapter "
            "for it in this build"
        )
    notes = [entry.notes] if entry.notes else []
    return RuntimeCapability(
        runtime_id=entry.runtime_id,
        available=False,
        unavailable_reason=reason,
        notes=notes,
    )


def probe_all() -> list[RuntimeCapability]:
    """Probe every known runtime. Never raises; a failed probe becomes a reason."""
    results: list[RuntimeCapability] = []

    for runtime_id, cls in _IMPLEMENTED.items():
        try:
            results.append(cls().probe())
        except Exception as exc:  # noqa: BLE001 - a probe must not take the app down
            results.append(
                RuntimeCapability(
                    runtime_id=runtime_id,
                    available=False,
                    unavailable_reason=f"probe raised {type(exc).__name__}: {exc}",
                )
            )

    results.extend(probe_declared(entry) for entry in _DECLARED)
    return results


def available_runtime_ids() -> list[str]:
    """Only runtimes whose probe actually succeeded. This is what the UI may offer."""
    return [c.runtime_id for c in probe_all() if c.available]


def capability_matrix(
    devices: tuple[DeviceKind, ...] = (DeviceKind.CPU, DeviceKind.CUDA),
    precisions: tuple[Precision, ...] = (Precision.FP32, Precision.FP16, Precision.INT8),
) -> list[dict]:
    """Runtime x device x precision, with a reason attached to every unsupported cell.

    Feeds the System page's capability matrix (§23), where an empty cell without an
    explanation would be worse than no matrix at all.
    """
    rows: list[dict] = []
    for cap in probe_all():
        for device in devices:
            for precision in precisions:
                if not cap.available:
                    supported, reason = False, cap.unavailable_reason
                elif device not in cap.devices:
                    supported, reason = False, (
                        f"{cap.runtime_id} reported no support for device '{device.value}' here"
                    )
                elif precision not in cap.precisions_for(device):
                    supported, reason = False, (
                        f"{cap.runtime_id} reported no support for precision "
                        f"'{precision.value}' on '{device.value}'"
                    )
                else:
                    supported, reason = True, None
                rows.append({
                    "runtime_id": cap.runtime_id,
                    "device": device.value,
                    "precision": precision.value,
                    "supported": supported,
                    "reason": reason,
                })
    return rows
