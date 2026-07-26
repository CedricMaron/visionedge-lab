"""The runtime adapter contract and its capability-probe discipline.

A runtime adapter owns execution: creating a session, placing it on a device,
configuring threads and precision, running tensors through it, and — critically —
synchronizing the device before anyone stops a clock.

Two invariants are enforced here rather than trusted:

1. **Nothing is claimed without a probe.** :meth:`RuntimeAdapter.probe` is the only
   thing that may report a runtime as available, and it does so by actually
   importing the package and asking it what it can do. A runtime the platform has
   heard of but cannot verify reports ``available=False`` with the reason.

2. **No silent fallback.** If CUDA was requested and the session came up on CPU,
   :meth:`create_session` reports the effective device honestly and the caller
   fails the load. ONNX Runtime falls back by default; this layer refuses to let
   that fallback go unnoticed, because every downstream number would then be
   attributed to hardware that did no work.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from app.schemas.enums import DeviceKind, Precision


@dataclass(slots=True)
class RuntimeCapability:
    """The result of asking a runtime what it can actually do, right now, here."""

    runtime_id: str
    available: bool
    unavailable_reason: str | None = None
    version: str | None = None
    execution_providers: list[str] = field(default_factory=list)
    devices: list[DeviceKind] = field(default_factory=list)
    #: Precision support is per-device, not global: ONNX Runtime offers fp16 on CUDA
    #: but not usefully on CPU, and a flat list would let the capability matrix claim
    #: a combination that does not exist.
    precisions_by_device: dict[DeviceKind, list[Precision]] = field(default_factory=dict)
    supports_device_synchronization: bool = False
    supports_profiling: bool = False
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.available and not self.unavailable_reason:
            raise ValueError(
                f"runtime '{self.runtime_id}' reported unavailable without a reason — "
                "the UI displays this reason, so it may not be empty"
            )

    def precisions_for(self, device: DeviceKind) -> list[Precision]:
        """Precisions this runtime supports on ``device``. Empty when the device is unsupported."""
        return self.precisions_by_device.get(device, [])


@dataclass(slots=True)
class SessionConfig:
    """Everything that changes how a session executes, and therefore how it measures."""

    model_path: str
    device: DeviceKind = DeviceKind.CPU
    device_index: int = 0
    precision: Precision = Precision.FP32
    intra_op_threads: int | None = None
    inter_op_threads: int | None = None
    graph_optimization_level: str | None = None
    enable_profiling: bool = False
    backend_options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SessionHandle:
    """A loaded session plus what it truthfully turned out to be.

    ``effective_device`` is what the runtime reports after creation, not what was
    asked for. ``honored`` is the comparison, computed once, so no caller has to
    re-derive it and none can forget to.
    """

    session: Any
    runtime_id: str
    runtime_version: str | None
    execution_provider: str | None
    effective_device: DeviceKind
    effective_precision: Precision
    requested_device: DeviceKind
    input_names: list[str] = field(default_factory=list)
    output_names: list[str] = field(default_factory=list)
    load_ms: float = 0.0
    thread_config: dict[str, int] = field(default_factory=dict)

    @property
    def honored(self) -> bool:
        """False when the session did not land on the requested device."""
        return self.effective_device is self.requested_device

    def mismatch_message(self) -> str:
        return (
            f"runtime '{self.runtime_id}' was asked for device "
            f"'{self.requested_device.value}' but the session was created on "
            f"'{self.effective_device.value}' (provider={self.execution_provider}). "
            "Reporting this run as if it used the requested device would misattribute "
            "every measurement, so the load is refused."
        )


@runtime_checkable
class RuntimeAdapter(Protocol):
    """Execution backend, independent of any particular model."""

    runtime_id: str

    def probe(self) -> RuntimeCapability:
        """Ask this runtime what it can do here. Must never raise; failures become reasons."""
        ...

    def create_session(self, config: SessionConfig) -> SessionHandle: ...

    def run(self, handle: SessionHandle, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        """Execute. Returns outputs in ``handle.output_names`` order."""
        ...

    def synchronize(self, handle: SessionHandle) -> None:
        """Block until queued device work has completed.

        Called by the benchmark engine before stopping the model-execution clock.
        On a synchronous CPU runtime this is a no-op, and the corresponding span is
        marked ``device_synchronized=True`` because there is genuinely nothing
        outstanding. On an asynchronous device runtime that cannot synchronize, the
        span is marked ``False`` and the run carries a warning that its
        model-execution time reflects dispatch rather than completion.
        """
        ...

    def release(self, handle: SessionHandle) -> None: ...


class BaseRuntimeAdapter:
    """Shared helpers. Subclasses implement the protocol methods."""

    runtime_id: str = "base"

    def probe(self) -> RuntimeCapability:  # pragma: no cover - subclasses override
        return RuntimeCapability(
            runtime_id=self.runtime_id,
            available=False,
            unavailable_reason="probe() not implemented for this runtime",
        )

    @staticmethod
    def _import_version(module_name: str) -> tuple[Any | None, str | None, str | None]:
        """Import a module and read its version.

        Returns ``(module, version, error)``. Never raises — a missing or broken
        dependency is a capability answer, not an exception, because the platform
        must keep working with every optional runtime absent.
        """
        try:
            module = __import__(module_name)
        except Exception as exc:  # noqa: BLE001 - any import failure is a valid "unavailable"
            return None, None, f"{type(exc).__name__}: {exc}"
        version = getattr(module, "__version__", None)
        return module, version, None

    def synchronize(self, handle: SessionHandle) -> None:
        """Default: nothing is outstanding on a synchronous backend."""
        return None

    def release(self, handle: SessionHandle) -> None:
        handle.session = None
