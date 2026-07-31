# Runtime adapters

A runtime adapter owns execution: creating a session, placing it on a device,
configuring threads and precision, running tensors, and synchronizing the device
before anyone stops a clock. It knows nothing about what the tensors mean.

## Contract

`app/runtimes/base.py::RuntimeAdapter`

```python
class RuntimeAdapter(Protocol):
    runtime_id: str
    def probe(self) -> RuntimeCapability: ...          # must never raise
    def create_session(self, config: SessionConfig) -> SessionHandle: ...
    def run(self, handle, inputs) -> list[np.ndarray]: ...
    def synchronize(self, handle) -> None: ...
    def release(self, handle) -> None: ...
```

## Two invariants

**Nothing is claimed without a probe.** `probe()` is the only thing that may report a
runtime as available, and it does so by importing the package and asking it what it
can do. It never raises — a broken optional dependency is a capability answer, not an
exception, because the platform must keep working with every optional runtime absent.
A `RuntimeCapability` with `available=False` and no reason is rejected at construction.

**No silent fallback.** `SessionHandle.honored` compares the effective device against
the requested one, computed once so no caller can forget to check. The mismatch
message says explicitly that adopting the session would misattribute every
measurement.

## Capability states

Every runtime reports one of three states, and the distinction is shown in the UI
because they imply completely different next steps:

| State | Meaning |
|---|---|
| **available** | An adapter exists and its probe succeeded here |
| **declared, not installed** | An adapter exists; the dependency is absent |
| **declared, no adapter** | The platform models this runtime but has not implemented execution |

The third state is normally hidden behind an unexplained greyed-out checkbox. It is
spelled out instead.

## Status on the reference machine

| Runtime | State | Reason |
|---|---|---|
| `onnxruntime` | **available** 1.20.1 | CPU + CUDA providers listed |
| `pytorch`, `pytorch-compile`, `torchscript` | declared, no adapter | torch installed (CPU-only build), execution not implemented |
| `onnxruntime-directml` | declared, not installed | `DmlExecutionProvider` absent (Windows-only) |
| `tensorrt` | declared, not installed | `tensorrt` package absent |
| `openvino`, `coreml`, `tflite`, `mlx` | declared, not installed | package absent / wrong platform |
| `llama-cpp`, `vllm`, `transformers`, `tgi` | declared, not installed | package absent |
| `browser-webgpu`, `browser-wasm` | declared, no adapter | Runs in the browser; the server cannot answer for it |
| `remote-http`, `remote-streaming` | declared, no adapter | No endpoint configured |

## ONNX Runtime specifics

**Precision support is per-device, not global.** fp16 is offered on CUDA and refused on
CPU: ORT will run an fp16 graph on CPU by inserting cast nodes around fp32 kernels,
which is slower than fp32 with no memory benefit, so advertising it would send users
toward a strictly worse configuration.

**`session.run()` is blocking and returns host-resident arrays.** There is no
outstanding device work when it returns, so spans around it are correctly marked
synchronized — but the measured time *includes the device-to-host copy* and is not
pure kernel time. The probe states this and the note travels with the measurement.

**Listing a provider is not the same as being able to use it.** On the reference
machine ORT lists `CUDAExecutionProvider`, but session creation fails with
`libcublasLt.so.12: cannot open shared object file` and falls back to CPU. The probe
notes say listing does not guarantee session creation; the `honored` check catches
the fallback per load.

## Capability matrix

`capability_matrix()` produces runtime × device × precision with a reason on **every**
unsupported cell. An empty cell with no explanation would be worse than no matrix at
all. Surfaced on the System page and via `inference-lab matrix`.

## Adding a runtime adapter

1. Implement the protocol in `app/runtimes/<name>_adapter.py`.
2. `probe()` must import defensively and return reasons, never raise.
3. Populate `precisions_by_device`, not a flat list.
4. If execution is asynchronous, implement `synchronize()` properly — or return
   `supports_device_synchronization=False` so runs are flagged.
5. Register in `app/runtimes/registry.py::_IMPLEMENTED` and remove the `_DECLARED` entry.
6. Test: probe honesty, an unsupported combination's reason, and the device-mismatch path.
