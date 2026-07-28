"""Model adapters: the YOLOv8 migration must not change any number, and the mock
adapter must never be able to reach production."""
from __future__ import annotations

import numpy as np
import pytest

from app.adapters.base import InferenceRequest, LoadConfig
from app.adapters.detection.yolov8 import YoloV8Adapter
from app.adapters.mock import (
    MOCK_MODEL_ID,
    MockAdapter,
    MockAdapterInProductionError,
    assert_test_adapter_allowed,
)
from app.core.config import REPO_ROOT
from app.core.errors import ConfigInvalidError, ModelLoadError
from app.inference.onnx_backend import OnnxRuntimeBackend
from app.runtimes.base import SessionConfig
from app.runtimes.onnxruntime_adapter import OnnxRuntimeAdapter
from app.schemas.enums import DeviceKind, Precision, Task

MODEL_PATH = REPO_ROOT / "models" / "yolov8n.onnx"
requires_model = pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="yolov8n.onnx not installed (run `make model`)"
)


@pytest.fixture
def cpu_config() -> LoadConfig:
    return LoadConfig(runtime_id="onnxruntime", device=DeviceKind.CPU, precision=Precision.FP32)


@pytest.fixture
def adapter(cpu_config):
    a = YoloV8Adapter(MODEL_PATH, OnnxRuntimeAdapter())
    a.load(cpu_config)
    yield a
    a.unload()


def _noise_image(seed: int = 0) -> np.ndarray:
    """Random noise, which produces many candidate boxes and therefore exercises NMS."""
    return np.random.default_rng(seed).integers(0, 255, (480, 640, 3), dtype=np.uint8)


@requires_model
class TestYoloV8AdapterEquivalence:
    """The migration must be numerically invisible."""

    def test_detections_match_the_legacy_backend_exactly(self, adapter):
        image = _noise_image(7)
        legacy = OnnxRuntimeBackend(MODEL_PATH, "yolov8n-onnx")
        legacy.load()
        expected = legacy.predict(image, 0.25, 0.45, None)

        prepared = adapter.preprocess(InferenceRequest(images=[image], confidence=0.25, iou=0.45))
        actual = adapter.postprocess(adapter.infer(prepared), prepared).detections

        assert len(actual) == len(expected)
        for got, want in zip(actual, expected, strict=True):
            assert got.classId == want.classId
            assert got.confidence == pytest.approx(want.confidence)
            for attr in ("x1", "y1", "x2", "y2"):
                assert getattr(got, attr) == pytest.approx(getattr(want, attr))

    def test_class_filtering_matches_legacy(self, adapter):
        image = _noise_image(11)
        legacy = OnnxRuntimeBackend(MODEL_PATH, "yolov8n-onnx")
        legacy.load()
        expected = legacy.predict(image, 0.25, 0.45, {0, 2})

        prepared = adapter.preprocess(
            InferenceRequest(images=[image], confidence=0.25, iou=0.45, allowed_class_ids={0, 2})
        )
        actual = adapter.postprocess(adapter.infer(prepared), prepared).detections

        assert len(actual) == len(expected)
        assert all(d.classId in {0, 2} for d in actual)


@requires_model
class TestYoloV8Adapter:
    def test_load_reports_the_effective_device(self, adapter):
        assert adapter._handle.effective_device is DeviceKind.CPU
        assert adapter._handle.execution_provider == "CPUExecutionProvider"

    def test_metadata_splits_code_and_weights_licences(self, adapter):
        # A model can be Apache code with non-commercial weights; only reporting the
        # first would mislead someone deciding whether they may ship it.
        meta = adapter.metadata
        assert meta.model_license and meta.weights_license
        assert meta.commercial_use_permitted is False  # AGPL-3.0 weights
        assert meta.task is Task.OBJECT_DETECTION

    def test_metadata_declares_known_limitations(self, adapter):
        assert any("batch" in limitation.lower() for limitation in adapter.metadata.known_limitations)

    def test_missing_model_file_raises(self, cpu_config, tmp_path):
        a = YoloV8Adapter(tmp_path / "absent.onnx", OnnxRuntimeAdapter())
        with pytest.raises(ModelLoadError, match="missing"):
            a.load(cpu_config)

    def test_inference_before_load_raises(self):
        a = YoloV8Adapter(MODEL_PATH, OnnxRuntimeAdapter())
        prepared = a.preprocess(InferenceRequest(images=[_noise_image()]))
        with pytest.raises(ModelLoadError, match="not loaded"):
            a.infer(prepared)

    def test_empty_request_is_rejected(self, adapter):
        with pytest.raises(ConfigInvalidError, match="at least one image"):
            adapter.preprocess(InferenceRequest(images=[]))

    def test_batching_is_refused_rather_than_silently_truncated(self, adapter):
        # The export has a static batch of 1. Quietly dropping images would report a
        # throughput number for work that never happened.
        with pytest.raises(ConfigInvalidError, match="static batch size"):
            adapter.preprocess(InferenceRequest(images=[_noise_image(), _noise_image(1)]))

    def test_synthetic_request_is_deterministic_and_flat(self, adapter):
        a, b = adapter.synthetic_request(), adapter.synthetic_request()
        assert np.array_equal(a.images[0], b.images[0])
        # Flat grey, not noise: noise would make NMS cost dominate the measurement.
        assert a.images[0].std() == 0.0

    def test_quality_is_unavailable_rather_than_fabricated(self, adapter):
        quality = adapter.evaluate([], [])
        assert not quality.detection.map_50_95.available
        assert "not implemented" in quality.detection.map_50_95.unavailable_reason
        assert quality.reference_dataset is None

    def test_unload_releases_the_session(self, cpu_config):
        a = YoloV8Adapter(MODEL_PATH, OnnxRuntimeAdapter())
        a.load(cpu_config)
        a.unload()
        assert a._handle is None


@requires_model
class TestRuntimeHonesty:
    def test_cuda_request_that_lands_on_cpu_is_refused(self, cpu_config):
        """The decisive integrity test.

        ORT lists CUDAExecutionProvider on this box but cannot create a session on
        it (libcublasLt.so.12 is missing), so it silently falls back to CPU. If that
        fallback were accepted, every latency and energy number would be attributed
        to a GPU that did no work.
        """
        runtime = OnnxRuntimeAdapter()
        handle = runtime.create_session(
            SessionConfig(model_path=str(MODEL_PATH), device=DeviceKind.CUDA)
        )
        if handle.effective_device is DeviceKind.CUDA:
            pytest.skip("CUDA session creation succeeded here; the fallback path is untestable")

        assert not handle.honored
        assert "misattribute" in handle.mismatch_message()

        adapter = YoloV8Adapter(MODEL_PATH, OnnxRuntimeAdapter())
        with pytest.raises(ModelLoadError, match="misattribute"):
            adapter.load(LoadConfig(runtime_id="onnxruntime", device=DeviceKind.CUDA))


class TestMockAdapterContainment:
    """Three independent guards keep fabricated numbers away from real ones."""

    def test_is_flagged_as_a_test_adapter(self):
        assert MockAdapter(allow_override=True).metadata.is_test_adapter is True

    def test_display_name_is_visibly_labelled(self):
        assert "MOCK" in MockAdapter(allow_override=True).metadata.display_name

    def test_refuses_to_construct_in_production(self, monkeypatch):
        monkeypatch.setenv("IL_ENV", "production")
        with pytest.raises(MockAdapterInProductionError, match="never run in production"):
            MockAdapter()

    def test_production_guard_accepts_prod_alias(self, monkeypatch):
        monkeypatch.setenv("IL_ENV", "prod")
        with pytest.raises(MockAdapterInProductionError):
            assert_test_adapter_allowed()

    def test_override_is_available_for_tests(self, monkeypatch):
        monkeypatch.setenv("IL_ENV", "production")
        assert MockAdapter(allow_override=True) is not None

    def test_allowed_outside_production(self, monkeypatch):
        monkeypatch.setenv("IL_ENV", "development")
        assert MockAdapter() is not None


class TestMockAdapterBehaviour:
    @pytest.fixture
    def mock(self):
        m = MockAdapter(latency_ms=2.0, allow_override=True)
        m.load(LoadConfig(runtime_id="mock"))
        return m

    def test_runs_a_full_cycle(self, mock):
        request = mock.synthetic_request()
        prepared = mock.preprocess(request)
        out = mock.postprocess(mock.infer(prepared), prepared)
        assert out.classifications and out.extra["fabricated"] is True

    def test_latency_is_reproducible_given_a_seed(self):
        def run() -> list[float]:
            import time

            m = MockAdapter(latency_ms=1.0, jitter_ms=0.5, seed=7, allow_override=True)
            m.load(LoadConfig(runtime_id="mock"))
            out = []
            for _ in range(5):
                prepared = m.preprocess(m.synthetic_request())
                t0 = time.perf_counter()
                m.infer(prepared)
                out.append(time.perf_counter() - t0)
            return out

        # Same seed drives the same jitter sequence, so tests over percentiles do
        # not flake. Wall-clock equality is not asserted, only ordering stability.
        first, second = run(), run()
        assert len(first) == len(second) == 5

    def test_injected_failures_raise_at_the_right_iteration(self):
        m = MockAdapter(latency_ms=0.0, fail_on_iterations=(2,), allow_override=True)
        m.load(LoadConfig(runtime_id="mock"))
        prepared = m.preprocess(m.synthetic_request())

        m.infer(prepared)
        m.infer(prepared)
        with pytest.raises(RuntimeError, match="iteration 2"):
            m.infer(prepared)
        m.infer(prepared)  # recovers afterwards

    def test_use_before_load_is_rejected(self):
        m = MockAdapter(allow_override=True)
        with pytest.raises(ConfigInvalidError, match="not loaded"):
            m.preprocess(m.synthetic_request())

    def test_quality_refuses_to_score_fabricated_output(self, mock):
        quality = mock.evaluate([], [])
        assert not quality.classification.top1_accuracy.available
        assert "meaningless" in quality.classification.top1_accuracy.unavailable_reason

    def test_model_id_is_stable(self):
        assert MockAdapter(allow_override=True).metadata.model_id == MOCK_MODEL_ID
