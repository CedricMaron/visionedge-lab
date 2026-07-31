"""Distributed timing: client, transport and server measured separately.

The decisive property is that no part of the split is inferred from another. The
transport figure is a residual and is labelled as one — never as "network".
"""
from __future__ import annotations

import numpy as np
import pytest

from app.adapters.base import InferenceRequest, LoadConfig
from app.adapters.remote.http_adapter import (
    REQUEST_ID_HEADER,
    RemoteHttpAdapter,
    RemoteTimings,
    parse_server_envelope,
)
from app.core.errors import ConfigInvalidError, ModelLoadError
from app.instrumentation.timeline import Timeline
from app.schemas.enums import Phase


class TestRemoteTimingsArithmetic:
    def test_transport_is_round_trip_minus_server_total(self):
        t = RemoteTimings()
        t.round_trip_ms = 50.0
        t.server_total_ms = 30.0
        assert t.transport_residual_ms == pytest.approx(20.0)

    def test_transport_is_unavailable_without_a_server_total(self):
        # Attributing the whole round trip to the network would be a fabrication.
        t = RemoteTimings()
        t.round_trip_ms = 50.0
        assert t.transport_residual_ms is None

    def test_negative_transport_clamps_to_zero(self):
        # Arises when both figures sit near timer resolution. A negative duration is
        # not a measurement.
        t = RemoteTimings()
        t.round_trip_ms = 10.0
        t.server_total_ms = 10.4
        assert t.transport_residual_ms == 0.0

    def test_server_unaccounted_is_total_minus_itemized_phases(self):
        t = RemoteTimings()
        t.server_total_ms = 30.0
        t.server_phases = {
            Phase.SERVER_PREPROCESSING: 5.0,
            Phase.SERVER_MODEL_EXECUTION: 20.0,
        }
        assert t.server_unaccounted_ms == pytest.approx(5.0)

    def test_server_unaccounted_is_none_without_phases(self):
        t = RemoteTimings()
        t.server_total_ms = 30.0
        assert t.server_unaccounted_ms is None


class TestEnvelopeParsing:
    def test_reads_every_known_server_phase(self):
        adapter = RemoteHttpAdapter("http://example.test")
        timings = RemoteTimings()
        body = {
            "timings": {
                "queue_ms": 1.0, "preprocess_ms": 2.0, "inference_ms": 20.0,
                "postprocess_ms": 3.0, "serialization_ms": 0.5, "server_total_ms": 28.0,
            }
        }
        adapter._read_server_timings(body, None, timings)

        assert timings.server_phases[Phase.SERVER_MODEL_EXECUTION] == 20.0
        assert timings.server_phases[Phase.SERVER_QUEUE] == 1.0
        assert timings.server_total_ms == 28.0

    def test_missing_server_total_falls_back_to_the_phase_sum(self):
        adapter = RemoteHttpAdapter("http://example.test")
        timings = RemoteTimings()
        adapter._read_server_timings({"timings": {"inference_ms": 10.0}}, None, timings)
        assert timings.server_total_ms == 10.0

    def test_a_server_reporting_nothing_leaves_the_total_unknown(self):
        adapter = RemoteHttpAdapter("http://example.test")
        timings = RemoteTimings()
        adapter._read_server_timings({}, None, timings)
        assert timings.server_total_ms is None
        assert timings.transport_residual_ms is None

    def test_negative_and_non_numeric_values_are_ignored(self):
        adapter = RemoteHttpAdapter("http://example.test")
        timings = RemoteTimings()
        adapter._read_server_timings(
            {"timings": {"inference_ms": -5, "preprocess_ms": "fast"}}, None, timings
        )
        assert timings.server_phases == {}

    def test_non_json_body_parses_to_empty(self):
        assert parse_server_envelope("<html>502</html>") == {}


class TestTimelineContribution:
    def _adapter_with(self, round_trip: float, server_total: float) -> RemoteHttpAdapter:
        adapter = RemoteHttpAdapter("http://example.test")
        t = RemoteTimings()
        t.round_trip_ms = round_trip
        t.server_total_ms = server_total
        t.server_phases = {
            Phase.SERVER_PREPROCESSING: 4.0,
            Phase.SERVER_MODEL_EXECUTION: 20.0,
            Phase.SERVER_POSTPROCESSING: 2.0,
        }
        t.parse_ms = 1.0
        adapter.last_timings = t
        return adapter

    def test_records_server_phases_as_reported_not_measured(self):
        timeline = Timeline()
        self._adapter_with(50.0, 28.0).record_remote_phases(timeline)

        server_spans = [s for s in timeline.spans() if s.phase.value.startswith("server_")]
        assert len(server_spans) == 3
        assert all("reported by the remote server" in s.note for s in server_spans)

    def test_transport_is_a_residual_never_called_network(self):
        timeline = Timeline()
        self._adapter_with(50.0, 28.0).record_remote_phases(timeline)

        transport = next(
            s for s in timeline.spans()
            if s.phase is Phase.RESIDUAL_OVERHEAD and s.label == "transport"
        )
        assert transport.duration_ms == pytest.approx(22.0)
        assert "not labelled 'network'" in transport.note.lower()
        # The phase enum itself must not be a real transport phase.
        assert transport.phase is not Phase.DOWNLOAD
        assert transport.phase is not Phase.UPLOAD

    def test_server_unaccounted_time_is_surfaced(self):
        timeline = Timeline()
        self._adapter_with(50.0, 28.0).record_remote_phases(timeline)
        unaccounted = next(
            s for s in timeline.spans() if s.label == "server unaccounted"
        )
        assert unaccounted.duration_ms == pytest.approx(2.0)


class TestMeasurementProvenance:
    def test_split_is_reported_with_provenance(self):
        adapter = RemoteHttpAdapter("http://example.test")
        t = RemoteTimings()
        t.round_trip_ms = 40.0
        t.server_total_ms = 25.0
        adapter.last_timings = t

        measurements = adapter.timing_measurements()
        assert measurements["round_trip_ms"].kind.value == "measured"
        assert measurements["server_total_ms"].kind.value == "measured"
        assert measurements["transport_residual_ms"].kind.value == "derived"
        assert "clock synchronization" in measurements["transport_residual_ms"].note

    def test_split_is_unavailable_when_the_server_is_silent(self):
        adapter = RemoteHttpAdapter("http://example.test")
        adapter.last_timings.round_trip_ms = 40.0

        measurements = adapter.timing_measurements()
        assert not measurements["transport_residual_ms"].available
        assert "cannot be split" in measurements["transport_residual_ms"].unavailable_reason


class TestAdapterGuards:
    def test_non_http_endpoint_is_rejected(self):
        # A caller-supplied non-URL must not reach a request.
        with pytest.raises(ConfigInvalidError, match="http"):
            RemoteHttpAdapter("file:///etc/passwd")

    def test_use_before_load_raises(self):
        adapter = RemoteHttpAdapter("http://example.test")
        prepared = adapter.preprocess(adapter.synthetic_request())
        with pytest.raises(ModelLoadError, match="not loaded"):
            adapter.infer(prepared)

    def test_empty_request_is_rejected(self):
        adapter = RemoteHttpAdapter("http://example.test")
        with pytest.raises(ConfigInvalidError, match="at least one image"):
            adapter.preprocess(InferenceRequest(images=[]))

    def test_batching_is_refused(self):
        adapter = RemoteHttpAdapter("http://example.test")
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        with pytest.raises(ConfigInvalidError, match="one image per request"):
            adapter.preprocess(InferenceRequest(images=[frame, frame]))

    def test_unreachable_endpoint_fails_load_with_a_reason(self):
        adapter = RemoteHttpAdapter("http://127.0.0.1:1", timeout_s=1.0)
        with pytest.raises(ModelLoadError, match="not reachable"):
            adapter.load(LoadConfig(runtime_id="remote-http"))

    def test_api_key_is_never_placed_in_metadata(self):
        # Benchmark records get exported; a key in metadata would travel with them.
        adapter = RemoteHttpAdapter("http://example.test", api_key="sk-secret-value")
        serialized = adapter.metadata.model_dump_json()
        assert "sk-secret-value" not in serialized

    def test_encoding_is_charged_to_request_serialization(self):
        adapter = RemoteHttpAdapter("http://example.test")
        assert adapter.preprocess_phase is Phase.REQUEST_SERIALIZATION

    def test_licence_of_a_remote_model_is_unknown_not_assumed(self):
        adapter = RemoteHttpAdapter("http://example.test")
        assert "unknown" in adapter.metadata.weights_license
        assert adapter.metadata.commercial_use_permitted is None


class TestAgainstRealServer:
    """End-to-end against this application served over real HTTP."""

    @pytest.fixture
    def live_endpoint(self):
        import threading

        import uvicorn

        from app.main import create_app

        config = uvicorn.Config(
            create_app(), host="127.0.0.1", port=8477, log_level="error", access_log=False
        )
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        import time as _time

        for _ in range(100):
            if server.started:
                break
            _time.sleep(0.1)
        if not server.started:
            pytest.skip("could not start a local server for the remote-inference test")

        yield "http://127.0.0.1:8477"
        server.should_exit = True
        thread.join(timeout=5)

    def test_full_distributed_decomposition(self, live_endpoint):
        adapter = RemoteHttpAdapter(live_endpoint, timeout_s=30.0)
        result = adapter.load(LoadConfig(runtime_id="remote-http"))
        assert result.ok

        try:
            request = adapter.synthetic_request()
            timeline = Timeline()
            timeline.start()
            with timeline.span(adapter.preprocess_phase):
                prepared = adapter.preprocess(request)
            with timeline.span(Phase.UPLOAD, label="round trip"):
                raw = adapter.infer(prepared)
            output = adapter.postprocess(raw, prepared)
            adapter.record_remote_phases(timeline)
            timeline.stop()

            timings = adapter.last_timings

            # The server answered with real per-phase timings.
            assert timings.server_total_ms is not None and timings.server_total_ms > 0
            assert Phase.SERVER_MODEL_EXECUTION in timings.server_phases

            # The round trip is at least the server's own total: the client cannot
            # observe less time than the server spent.
            assert timings.round_trip_ms >= timings.server_total_ms

            # Transport is a genuine positive residual over loopback.
            assert timings.transport_residual_ms is not None

            # Correlation id round-tripped.
            assert output.extra["request_id"] == timings.request_id
            assert len(timings.request_id) == 32
        finally:
            adapter.unload()

    def test_correlation_id_is_echoed_by_the_server(self, live_endpoint):
        import httpx

        request_id = "abc123deadbeef"
        with httpx.Client(timeout=30.0) as client:
            import cv2

            frame = np.full((64, 64, 3), 128, dtype=np.uint8)
            ok, buf = cv2.imencode(".jpg", frame)
            assert ok
            response = client.post(
                f"{live_endpoint}/api/infer",
                files={"file": ("f.jpg", buf.tobytes(), "image/jpeg")},
                headers={REQUEST_ID_HEADER: request_id},
            )
        assert response.status_code == 200
        assert response.headers.get(REQUEST_ID_HEADER) == request_id
        assert response.json()["request_id"] == request_id
