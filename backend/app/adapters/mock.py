"""Deterministic mock adapter, for integration testing only.

Required by §26 of the brief so the benchmark engine can be tested end to end
without loading a real model. Everything it returns is fabricated, which makes it
the single most dangerous object in the codebase: a fabricated benchmark that
reached a results page would undermine every honest number beside it.

Three independent guards keep it contained:

1. ``metadata.is_test_adapter`` is True, and the model registry filters those out
   of production listings.
2. :func:`assert_test_adapter_allowed` refuses to construct one when
   ``IL_ENV=production`` unless explicitly overridden.
3. Its ``display_name`` and every result it produces are labelled ``MOCK``.

The latency it simulates is deterministic given a seed, so tests can assert on
percentiles without flaking.
"""
from __future__ import annotations

import os
import time

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
from app.core.errors import ConfigInvalidError
from app.schemas.enums import DeviceKind, Modality, Precision, Task
from app.schemas.measurement import Measurement
from app.schemas.quality import ClassificationQuality, QualityMetrics

MOCK_MODEL_ID = "mock-adapter"


class MockAdapterInProductionError(RuntimeError):
    """Raised when a fabricating adapter is constructed in a production process."""


def assert_test_adapter_allowed(allow_override: bool = False) -> None:
    """Refuse to build a mock adapter in production.

    Checked at construction rather than at listing time, so no code path can reach
    a fabricated measurement even by importing the class directly.
    """
    if allow_override:
        return
    env = os.environ.get("IL_ENV", "").strip().lower()
    if env in ("production", "prod"):
        raise MockAdapterInProductionError(
            "the mock adapter fabricates measurements and must never run in production "
            f"(IL_ENV={env!r}). Pass allow_override=True only in tests."
        )


class MockAdapter:
    """Fabricates deterministic results for testing the engine, not a model."""

    def __init__(
        self,
        latency_ms: float = 5.0,
        jitter_ms: float = 0.0,
        seed: int = 42,
        fail_on_iterations: frozenset[int] | tuple[int, ...] = (),
        load_ms: float = 20.0,
        allow_override: bool = False,
    ) -> None:
        assert_test_adapter_allowed(allow_override)

        self.latency_ms = latency_ms
        self.jitter_ms = jitter_ms
        self.load_ms = load_ms
        #: Iteration indices that must raise, so failure handling can be tested.
        self.fail_on_iterations = frozenset(fail_on_iterations)
        self._rng = np.random.default_rng(seed)
        self._call_count = 0
        self._loaded = False

        self.metadata = ModelMetadata(
            model_id=MOCK_MODEL_ID,
            display_name="MOCK adapter (deterministic, test only)",
            family="mock",
            task=Task.IMAGE_CLASSIFICATION,
            modality=Modality.IMAGE,
            source_repository=None,
            model_license="MIT (this project)",
            weights_license="not applicable — this adapter has no weights",
            commercial_use_permitted=True,
            auto_download_permitted=False,
            parameters_millions=0.0,
            model_size_bytes=0,
            supported_precisions=[Precision.FP32],
            supported_devices=[DeviceKind.CPU],
            supported_runtimes=["mock"],
            input_format="ignored",
            output_format="fabricated classification scores",
            dynamic_input_supported=True,
            streaming_supported=False,
            batch_supported=True,
            hardware_requirements=HardwareRequirement(min_ram_mb=0),
            known_limitations=[
                "Every output is fabricated. Results are meaningless as measurements "
                "of any real model and exist only to exercise the benchmark engine.",
            ],
            is_test_adapter=True,
        )

    # --- lifecycle -------------------------------------------------------

    def load(self, config: LoadConfig) -> LoadResult:
        time.sleep(self.load_ms / 1000.0)
        self._loaded = True
        return LoadResult(
            ok=True,
            effective_device=DeviceKind.CPU,
            effective_precision=Precision.FP32,
            execution_provider="mock",
            runtime_version="0",
            load_ms=self.load_ms,
            message="MOCK adapter loaded — all results are fabricated",
        )

    def unload(self) -> None:
        self._loaded = False

    # --- execution -------------------------------------------------------

    def preprocess(self, request: InferenceRequest) -> PreparedInput:
        if not self._loaded:
            raise ConfigInvalidError("mock adapter is not loaded")
        return PreparedInput(tensors={"x": np.zeros((1, 1), dtype=np.float32)}, context={})

    def infer(self, prepared: PreparedInput) -> RawOutput:
        index = self._call_count
        self._call_count += 1
        if index in self.fail_on_iterations:
            raise RuntimeError(f"MOCK failure injected at iteration {index}")

        delay = self.latency_ms
        if self.jitter_ms:
            delay += float(self._rng.normal(0.0, self.jitter_ms))
        time.sleep(max(0.0, delay) / 1000.0)
        return RawOutput(tensors=[np.array([[0.1, 0.7, 0.2]], dtype=np.float32)], names=["logits"])

    def synchronize(self) -> None:
        return None

    def postprocess(self, raw: RawOutput, prepared: PreparedInput) -> InferenceOutput:
        scores = raw.tensors[0][0]
        order = np.argsort(scores)[::-1]
        return InferenceOutput(
            classifications=[(int(i), f"MOCK_class_{i}", float(scores[i])) for i in order],
            extra={"fabricated": True},
        )

    def synthetic_request(self, batch_size: int = 1) -> InferenceRequest:
        return InferenceRequest(images=[np.zeros((8, 8, 3), dtype=np.uint8)] * batch_size)

    def evaluate(
        self,
        predictions: list[InferenceOutput],
        references: list[ReferenceOutput],
    ) -> QualityMetrics:
        reason = "the mock adapter fabricates outputs, so accuracy against them is meaningless"
        return QualityMetrics(
            classification=ClassificationQuality(
                top1_accuracy=Measurement[float].unavailable(reason),
                top5_accuracy=Measurement[float].unavailable(reason),
                f1_macro=Measurement[float].unavailable(reason),
            ),
            reference_dataset=None,
            sample_count=len(references),
        )

    @property
    def call_count(self) -> int:
        return self._call_count
