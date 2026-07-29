"""Benchmark engine integrity rules.

Uses the mock adapter so these run fast and deterministically; the real-model path
is covered by tests/test_adapters.py.
"""
from __future__ import annotations

import threading

import pytest

from app.adapters.base import LoadConfig
from app.adapters.mock import MockAdapter
from app.benchmark import BenchmarkEngine, EngineOptions
from app.schemas.enums import (
    BenchmarkMode,
    DeviceKind,
    IterationPhaseGroup,
    Precision,
    RunStatus,
    Task,
)
from app.schemas.environment import RuntimeReference
from app.schemas.scenario import ScenarioSpec


@pytest.fixture
def runtime_ref() -> RuntimeReference:
    return RuntimeReference(
        runtime_id="mock", runtime_version="0", execution_provider="mock",
        device=DeviceKind.CPU, precision=Precision.FP32,
    )


@pytest.fixture
def engine():
    e = BenchmarkEngine(EngineOptions(enable_sampler=False))
    yield e
    e.close()


def scenario(**kwargs) -> ScenarioSpec:
    defaults = dict(
        id="mock-scenario",
        task=Task.IMAGE_CLASSIFICATION,
        warmup_iterations=2,
        measured_iterations=10,
    )
    return ScenarioSpec(**{**defaults, **kwargs})


def run_mock(engine, runtime_ref, adapter=None, **scenario_kwargs):
    adapter = adapter or MockAdapter(latency_ms=1.0, load_ms=1.0, allow_override=True)
    return engine.run(
        adapter, scenario(**scenario_kwargs), LoadConfig(runtime_id="mock"), runtime_ref
    )


class TestWarmupExclusion:
    def test_warmup_iterations_are_recorded_but_not_counted(self, engine, runtime_ref):
        run = run_mock(engine, runtime_ref, warmup_iterations=3, measured_iterations=5)

        assert run.warmup_iterations_run == 3
        assert run.successful_iterations == 5
        assert len(run.iterations) == 8  # every iteration retained
        assert run.timings.total.n == 5  # only measured ones summarized

    def test_warmup_latency_does_not_pollute_statistics(self, engine, runtime_ref):
        # A slow first iteration is exactly what warm-up exists to absorb.
        adapter = MockAdapter(latency_ms=1.0, load_ms=1.0, allow_override=True)
        run = run_mock(engine, runtime_ref, adapter=adapter, warmup_iterations=2,
                       measured_iterations=10)
        warmups = [i for i in run.iterations if i.group is IterationPhaseGroup.WARMUP]
        assert len(warmups) == 2
        assert run.timings.total.n == 10

    def test_zero_warmup_is_allowed(self, engine, runtime_ref):
        run = run_mock(engine, runtime_ref, warmup_iterations=0, measured_iterations=3)
        assert run.warmup_iterations_run == 0


class TestFailureHandling:
    def test_failed_iterations_are_never_silently_dropped(self, engine, runtime_ref):
        adapter = MockAdapter(
            latency_ms=0.5, load_ms=1.0, fail_on_iterations=(3, 5), allow_override=True
        )
        run = run_mock(engine, runtime_ref, adapter=adapter, warmup_iterations=1,
                       measured_iterations=8)

        assert run.status is RunStatus.PARTIAL
        assert run.failed_iterations == 2
        assert run.errors.failure_count == 2
        assert all(f.error_type == "RuntimeError" for f in run.errors.failures)
        assert "injected" in run.errors.failures[0].error_message

    def test_result_states_that_statistics_exclude_failures(self, engine, runtime_ref):
        adapter = MockAdapter(latency_ms=0.5, load_ms=1.0, fail_on_iterations=(2,),
                              allow_override=True)
        run = run_mock(engine, runtime_ref, adapter=adapter, warmup_iterations=0,
                       measured_iterations=6)

        assert run.errors.statistics_exclude_failures is True
        assert run.timings.total.n == run.successful_iterations
        assert any("failed" in w for w in run.warnings)

    def test_a_load_failure_marks_the_run_failed(self, engine, runtime_ref):
        class BrokenAdapter(MockAdapter):
            def load(self, config):
                raise RuntimeError("weights are corrupt")

        run = run_mock(engine, runtime_ref, adapter=BrokenAdapter(allow_override=True))
        assert run.status is RunStatus.FAILED
        assert "corrupt" in run.errors.failures[0].error_message


class TestCancellationAndTimeout:
    def test_cancellation_marks_the_run_rather_than_returning_a_short_success(
        self, engine, runtime_ref
    ):
        cancel = threading.Event()
        adapter = MockAdapter(latency_ms=5.0, load_ms=1.0, allow_override=True)

        def cancel_soon():
            import time

            time.sleep(0.05)
            cancel.set()

        thread = threading.Thread(target=cancel_soon)
        thread.start()
        run = engine.run(
            adapter, scenario(measured_iterations=200), LoadConfig(runtime_id="mock"),
            runtime_ref, cancel=cancel,
        )
        thread.join()

        assert run.status is RunStatus.CANCELLED
        assert any("cancelled" in w for w in run.warnings)
        # Partial data is still returned, clearly labelled.
        assert run.successful_iterations < 200

    def test_timeout_is_marked(self, engine, runtime_ref):
        adapter = MockAdapter(latency_ms=5.0, load_ms=1.0, allow_override=True)
        run = engine.run(
            adapter, scenario(measured_iterations=500, timeout_seconds=0.2),
            LoadConfig(runtime_id="mock"), runtime_ref,
        )
        assert run.status is RunStatus.TIMED_OUT
        assert any("timeout" in w for w in run.warnings)


class TestIntegrityWarnings:
    def test_thin_sample_is_flagged(self, engine, runtime_ref):
        run = run_mock(engine, runtime_ref, measured_iterations=3)
        assert any("statistically meaningful" in w for w in run.warnings)

    def test_adequate_sample_is_not_flagged(self, engine, runtime_ref):
        run = run_mock(engine, runtime_ref, measured_iterations=12)
        assert not any("statistically meaningful" in w for w in run.warnings)

    def test_synthetic_input_is_disclosed(self, engine, runtime_ref):
        run = run_mock(engine, runtime_ref)
        assert any("synthetic input" in w for w in run.warnings)

    def test_non_standard_mode_is_marked_incomparable(self, engine, runtime_ref):
        run = run_mock(engine, runtime_ref, mode=BenchmarkMode.PROFILER)
        assert any("not comparable with standard-mode" in w for w in run.warnings)


class TestRawSampleRetention:
    def test_every_iteration_is_persisted_not_just_averages(self, engine, runtime_ref):
        run = run_mock(engine, runtime_ref, warmup_iterations=2, measured_iterations=10)
        assert len(run.iterations) == 12
        assert all(i.total_ms is not None for i in run.iterations if i.succeeded)

    def test_percentiles_are_recomputable_from_raw_samples(self, engine, runtime_ref):
        from app.schemas.timing import DurationStats

        run = run_mock(engine, runtime_ref, measured_iterations=15)
        raw = [i.total_ms for i in run.iterations if i.counts_toward_statistics]
        assert DurationStats.from_samples(raw).p95_ms == pytest.approx(run.timings.total.p95_ms)


class TestColdVersusWarm:
    def test_cold_start_is_separated_from_steady_state(self, engine, runtime_ref):
        adapter = MockAdapter(latency_ms=1.0, load_ms=30.0, allow_override=True)
        run = run_mock(engine, runtime_ref, adapter=adapter, warmup_iterations=2,
                       measured_iterations=10)

        assert run.cold_warm.model_load_ms >= 29.0
        assert run.cold_warm.cold_start_total_ms >= run.cold_warm.model_load_ms
        # Steady state must not include the 30 ms load.
        assert run.cold_warm.warm_inference.p50_ms < run.cold_warm.model_load_ms


class TestReproducibilityMetadata:
    def test_fingerprint_is_stable_across_identical_configurations(self, engine, runtime_ref):
        a = run_mock(engine, runtime_ref, measured_iterations=3)
        b = run_mock(engine, runtime_ref, measured_iterations=3)
        assert a.fingerprint.digest == b.fingerprint.digest

    def test_fingerprint_changes_with_precision(self, engine, runtime_ref):
        a = run_mock(engine, runtime_ref, measured_iterations=3)
        other = RuntimeReference(
            runtime_id="mock", runtime_version="0", execution_provider="mock",
            device=DeviceKind.CPU, precision=Precision.INT8,
        )
        b = run_mock(engine, other, measured_iterations=3)
        assert a.fingerprint.digest != b.fingerprint.digest

    def test_environment_is_captured(self, engine, runtime_ref):
        run = run_mock(engine, runtime_ref, measured_iterations=3)
        assert run.hardware.cpu_cores_logical >= 1
        assert run.software.python_version
        assert run.reproducibility.random_seed == 42


class TestComparability:
    def test_identical_scenarios_are_comparable(self, engine, runtime_ref):
        a = run_mock(engine, runtime_ref, measured_iterations=3)
        b = run_mock(engine, runtime_ref, measured_iterations=3)
        ok, reasons = a.is_comparable_to(b)
        assert ok and not reasons

    def test_different_batch_sizes_are_refused_with_a_reason(self, engine, runtime_ref):
        a = run_mock(engine, runtime_ref, measured_iterations=3, batch_size=1)
        b = run_mock(engine, runtime_ref, measured_iterations=3, batch_size=4)
        ok, reasons = a.is_comparable_to(b)
        assert not ok
        assert any("batch size" in r for r in reasons)

    def test_different_modes_are_refused(self, engine, runtime_ref):
        a = run_mock(engine, runtime_ref, measured_iterations=3, mode=BenchmarkMode.STANDARD)
        b = run_mock(engine, runtime_ref, measured_iterations=3, mode=BenchmarkMode.PROFILER)
        ok, reasons = a.is_comparable_to(b)
        assert not ok
        assert any("instrumentation modes" in r for r in reasons)


class TestThroughput:
    def test_rates_are_derived_not_separately_timed(self, engine, runtime_ref):
        run = run_mock(engine, runtime_ref, measured_iterations=10)
        assert run.throughput.requests_per_second.kind.value == "derived"

    def test_inapplicable_metrics_say_why(self, engine, runtime_ref):
        run = run_mock(engine, runtime_ref, measured_iterations=5)
        rtf = run.throughput.real_time_factor
        assert not rtf.available and "not applicable" in rtf.unavailable_reason


class TestSamplerDisabled:
    def test_disabled_sampler_is_reported_as_such(self, engine, runtime_ref):
        run = run_mock(engine, runtime_ref, measured_iterations=3)
        assert "sampler" in run.utilization.unavailable
        assert not run.energy.total_energy_j.available
