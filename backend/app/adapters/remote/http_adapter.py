"""Remote inference over HTTP, with distributed timing.

The point of this adapter is not to run a model — it is to *decompose* what a remote
call costs, separating what the client spent, what travelled, and what the server
spent, without any of the three being inferred from the others.

How the split is obtained honestly:

* The **client** measures serialization, round-trip time and parsing on its own clock.
* The **server** measures queueing, preprocessing, inference, postprocessing and
  response serialization on *its* clock, and returns them in the response envelope.
* Their difference is a duration minus a duration, so no clock synchronization is
  required and the result is exact up to timer resolution.

What that difference contains — uplink, downlink, TLS framing, socket queueing, and
any client-side scheduling — cannot be separated further without an NTP-style
handshake. It is therefore reported as a single bucket named
``Phase.RESIDUAL_OVERHEAD``, never as "network". Section 18 of the brief forbids
calling a residual by the name of one of its components, and doing so would present a
guess as a measurement.

Correlation: every request carries an ``X-InferenceLab-Request-Id`` header which the
server echoes, so a client-side timing can be tied to a server-side one in logs.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any

import numpy as np

from app.adapters.base import (
    HardwareRequirement,
    InferenceOutput,
    InferenceRequest,
    LoadConfig,
    LoadResult,
    ModelMetadata,
    PreparedInput,
    RawOutput,
    ReferenceOutput,
)
from app.core.errors import ConfigInvalidError, InferenceLabError, ModelLoadError
from app.core.logging import get_logger
from app.schemas.enums import DeviceKind, Modality, Phase, Precision, Task
from app.schemas.measurement import Measurement
from app.schemas.quality import QualityMetrics

log = get_logger("adapters.remote")

REQUEST_ID_HEADER = "X-InferenceLab-Request-Id"

#: Server-reported timing fields mapped onto timeline phases. A field the server does
#: not report is simply absent from the timeline rather than recorded as zero.
_SERVER_PHASE_FIELDS: dict[str, Phase] = {
    "queue_ms": Phase.SERVER_QUEUE,
    "preprocess_ms": Phase.SERVER_PREPROCESSING,
    "inference_ms": Phase.SERVER_MODEL_EXECUTION,
    "postprocess_ms": Phase.SERVER_POSTPROCESSING,
    "serialization_ms": Phase.RESPONSE_SERIALIZATION,
}


class RemoteTimings:
    """Client-measured and server-reported durations for one remote call."""

    __slots__ = (
        "request_id", "serialize_ms", "round_trip_ms", "parse_ms",
        "server_phases", "server_total_ms",
    )

    def __init__(self) -> None:
        self.request_id: str = ""
        self.serialize_ms: float = 0.0
        self.round_trip_ms: float = 0.0
        self.parse_ms: float = 0.0
        self.server_phases: dict[Phase, float] = {}
        self.server_total_ms: float | None = None

    @property
    def transport_residual_ms(self) -> float | None:
        """Round trip minus everything the server accounted for.

        Contains uplink, downlink, TLS framing and socket queueing, which cannot be
        separated without clock synchronization. Deliberately not named "network".
        Clamped at zero: a small negative arises when both figures sit near timer
        resolution, and a negative duration is not a measurement.
        """
        if self.server_total_ms is None:
            return None
        return max(0.0, self.round_trip_ms - self.server_total_ms)

    @property
    def server_unaccounted_ms(self) -> float | None:
        """Server total minus the server phases it itemized."""
        if self.server_total_ms is None or not self.server_phases:
            return None
        return max(0.0, self.server_total_ms - sum(self.server_phases.values()))


class RemoteHttpAdapter:
    """Runs inference on another machine and decomposes what that costs.

    ``endpoint`` is read from configuration, never from user input at request time
    (§25): a caller-supplied URL would turn this into a server-side request forgery
    primitive.
    """

    preprocess_phase = Phase.REQUEST_SERIALIZATION

    def __init__(
        self,
        endpoint: str,
        model_id: str = "remote-detection",
        display_name: str = "Remote detection endpoint",
        timeout_s: float = 30.0,
        api_key: str | None = None,
        task: Task = Task.OBJECT_DETECTION,
    ) -> None:
        if not endpoint.startswith(("http://", "https://")):
            raise ConfigInvalidError(
                f"remote endpoint must be an http(s) URL, got {endpoint!r}"
            )
        self.endpoint = endpoint.rstrip("/")
        self.timeout_s = timeout_s
        # Held in memory only, taken from configuration, and never serialized into a
        # benchmark record or a log line.
        self._api_key = api_key
        self._client: Any = None
        self.last_timings = RemoteTimings()

        self.metadata = ModelMetadata(
            model_id=model_id,
            display_name=display_name,
            family="remote",
            task=task,
            modality=Modality.IMAGE,
            source_repository=None,
            model_license="unknown — the remote endpoint owns the model",
            weights_license="unknown — the remote endpoint owns the weights",
            commercial_use_permitted=None,
            auto_download_permitted=False,
            supported_precisions=[Precision.FP32],
            supported_devices=[DeviceKind.CPU],
            supported_runtimes=["remote-http"],
            input_format="JPEG-encoded image uploaded as multipart/form-data",
            output_format="server-defined JSON, parsed into the shared Detection shape",
            dynamic_input_supported=True,
            streaming_supported=False,
            batch_supported=False,
            hardware_requirements=HardwareRequirement(
                min_ram_mb=64,
                note="Compute happens remotely; local requirements cover encoding only.",
            ),
            known_limitations=[
                "Uplink and downlink cannot be separated without clock synchronization; "
                "they are reported together as transport residual.",
                "The remote model's licence and hardware are not knowable from here and "
                "are reported as unknown rather than assumed.",
                "Server-side phase timings are only as trustworthy as the server "
                "reporting them.",
            ],
        )

    # --- lifecycle -------------------------------------------------------

    def load(self, config: LoadConfig) -> LoadResult:
        """Open a client and verify the endpoint answers.

        A remote 'load' is a reachability check, not a weight load, so its duration is
        recorded as connection cost rather than as model load time.
        """
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - httpx is a base dependency
            raise ModelLoadError("httpx is required for remote inference") from exc

        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        t0 = time.perf_counter()
        self._client = httpx.Client(timeout=self.timeout_s, headers=headers)
        try:
            response = self._client.get(f"{self.endpoint}/health")
            reachable = response.status_code < 500
        except Exception as exc:  # noqa: BLE001
            self._client.close()
            self._client = None
            raise ModelLoadError(
                f"remote endpoint {self.endpoint} is not reachable: {type(exc).__name__}: {exc}"
            ) from exc
        connect_ms = (time.perf_counter() - t0) * 1000.0

        if not reachable:
            self._client.close()
            self._client = None
            raise ModelLoadError(
                f"remote endpoint {self.endpoint} answered {response.status_code}"
            )

        return LoadResult(
            ok=True,
            effective_device=DeviceKind.CPU,
            effective_precision=Precision.FP32,
            execution_provider="remote-http",
            runtime_version=None,
            load_ms=connect_ms,
            message=f"connected to {self.endpoint}",
        )

    def unload(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    # --- execution -------------------------------------------------------

    def preprocess(self, request: InferenceRequest) -> PreparedInput:
        """Encode the frame for transport. This IS the client-side work."""
        if not request.images:
            raise ConfigInvalidError("RemoteHttpAdapter requires at least one image")
        if len(request.images) > 1:
            raise ConfigInvalidError("this adapter sends one image per request")

        import cv2

        ok, buffer = cv2.imencode(".jpg", request.images[0], [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise ConfigInvalidError("failed to JPEG-encode the frame for transport")

        payload = buffer.tobytes()
        return PreparedInput(
            tensors={},
            context={
                "payload": payload,
                "payload_bytes": len(payload),
                "confidence": request.confidence,
                "iou": request.iou,
            },
        )

    def infer(self, prepared: PreparedInput) -> RawOutput:
        """Send the request and record client-measured round-trip time."""
        if self._client is None:
            raise ModelLoadError("adapter is not loaded")

        timings = RemoteTimings()
        timings.request_id = uuid.uuid4().hex
        timings.serialize_ms = 0.0  # encoding was charged to preprocess

        params: dict[str, str] = {}
        if prepared.context.get("confidence") is not None:
            params["confidence"] = str(prepared.context["confidence"])
        if prepared.context.get("iou") is not None:
            params["iou"] = str(prepared.context["iou"])

        t0 = time.perf_counter()
        response = self._client.post(
            f"{self.endpoint}/api/infer",
            files={"file": ("frame.jpg", prepared.context["payload"], "image/jpeg")},
            params=params,
            headers={REQUEST_ID_HEADER: timings.request_id},
        )
        timings.round_trip_ms = (time.perf_counter() - t0) * 1000.0

        if response.status_code >= 400:
            raise InferenceLabError(
                f"remote inference failed with {response.status_code}",
                user_message=f"The remote endpoint returned {response.status_code}.",
            )

        t1 = time.perf_counter()
        body = response.json()
        timings.parse_ms = (time.perf_counter() - t1) * 1000.0

        self._read_server_timings(body, response, timings)
        self.last_timings = timings

        return RawOutput(tensors=[], names=["remote_json"], payload=body)

    @staticmethod
    def _read_server_timings(body: dict, response: Any, timings: RemoteTimings) -> None:
        """Adopt server-reported phase timings, ignoring anything unrecognized.

        A server that reports nothing leaves ``server_total_ms`` as None, which makes
        the transport residual unavailable rather than equal to the whole round trip —
        attributing all of it to the network would be a fabrication.
        """
        reported = body.get("timings")
        if not isinstance(reported, dict):
            return
        for field, phase in _SERVER_PHASE_FIELDS.items():
            value = reported.get(field)
            if isinstance(value, (int, float)) and value >= 0:
                timings.server_phases[phase] = float(value)

        total = reported.get("server_total_ms", reported.get("end_to_end_ms"))
        if isinstance(total, (int, float)) and total >= 0:
            timings.server_total_ms = float(total)
        elif timings.server_phases:
            # Fall back to the sum of itemized phases, and say so by leaving
            # server_unaccounted at zero rather than inventing a larger total.
            timings.server_total_ms = sum(timings.server_phases.values())

    def postprocess(self, raw: RawOutput, prepared: PreparedInput) -> InferenceOutput:
        """Parse the server's payload into the shared output shape."""
        detections = raw.payload.get("detections", []) if isinstance(raw.payload, dict) else []
        return InferenceOutput(
            detections=detections,
            extra={
                "request_id": self.last_timings.request_id,
                "payload_bytes": prepared.context.get("payload_bytes"),
                "remote": True,
            },
        )

    def synchronize(self) -> None:
        """Nothing is outstanding: the HTTP call is synchronous and fully returned."""
        return None

    # --- timeline contribution -------------------------------------------

    def record_remote_phases(self, timeline) -> None:
        """Write the distributed breakdown into a Timeline.

        Called by the engine after ``infer``. Server phases are recorded, not timed
        here, because this process did not measure them — :meth:`Timeline.record`
        exists precisely to keep locally-measured and remotely-reported durations
        distinguishable in the code.
        """
        t = self.last_timings

        for phase, duration_ms in t.server_phases.items():
            timeline.record(
                phase, duration_ms,
                note="reported by the remote server, measured on its clock",
            )

        unaccounted = t.server_unaccounted_ms
        if unaccounted:
            timeline.record(
                Phase.RESIDUAL_OVERHEAD, unaccounted, label="server unaccounted",
                note="server total minus the phases the server itemized",
            )

        residual = t.transport_residual_ms
        if residual is not None:
            timeline.record(
                Phase.RESIDUAL_OVERHEAD, residual, label="transport",
                note="client round-trip minus server-reported total. Contains uplink, "
                     "downlink, TLS framing and socket queueing, which cannot be "
                     "separated without clock synchronization. Not labelled 'network' "
                     "because it is not known to be only network.",
            )

        if t.parse_ms:
            timeline.record(Phase.CLIENT_PARSING, t.parse_ms)

    def timing_measurements(self) -> dict[str, Measurement]:
        """The distributed split, as provenance-carrying measurements."""
        t = self.last_timings
        out: dict[str, Measurement] = {
            "round_trip_ms": Measurement[float].of(
                t.round_trip_ms, "ms", "client time.perf_counter around the HTTP call"
            ),
            "client_parse_ms": Measurement[float].of(
                t.parse_ms, "ms", "client time.perf_counter around JSON parsing"
            ),
        }
        if t.server_total_ms is not None:
            out["server_total_ms"] = Measurement[float].of(
                t.server_total_ms, "ms", "reported by the remote server",
                note="measured on the server's clock; trustworthy only insofar as the "
                     "server is",
            )
            out["transport_residual_ms"] = Measurement[float].derived(
                t.transport_residual_ms or 0.0, "ms",
                "client round-trip minus server-reported total",
                note="uplink + downlink + TLS + socket queueing, inseparable without "
                     "clock synchronization",
            )
        else:
            reason = (
                "the remote endpoint reported no server-side timings, so the round trip "
                "cannot be split between transport and remote compute"
            )
            out["server_total_ms"] = Measurement[float].unavailable(reason, "ms")
            out["transport_residual_ms"] = Measurement[float].unavailable(reason, "ms")
        return out

    def synthetic_request(self, batch_size: int = 1) -> InferenceRequest:
        frame = np.full((640, 640, 3), 128, dtype=np.uint8)
        return InferenceRequest(images=[frame], confidence=0.25, iou=0.45)

    def evaluate(
        self,
        predictions: list[InferenceOutput],
        references: list[ReferenceOutput],
    ) -> QualityMetrics:
        # Quality of a remote model is not knowable from here: this process does not
        # know which model answered, and has no reference data to score it against.
        return QualityMetrics(reference_dataset=None, sample_count=len(references))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"RemoteHttpAdapter(endpoint={self.endpoint!r})"


def parse_server_envelope(text: str) -> dict:
    """Parse a server response envelope, tolerating a non-JSON body."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}
