"""The provenance rules are the platform's core promise, so they are tested first."""
from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from app.schemas.enums import IterationPhaseGroup, MetricKind, Phase
from app.schemas.measurement import Measurement
from app.schemas.timing import (
    DurationStats,
    IterationSample,
    PhaseBreakdown,
    PhaseSpan,
    percentile,
)


class TestMeasurementProvenance:
    def test_measured_value_is_available(self):
        m = Measurement[float].of(41.7, "ms", "time.perf_counter")
        assert m.available and m.value == 41.7
        assert m.kind is MetricKind.MEASURED

    def test_missing_value_requires_a_reason(self):
        # This is the rule that stops a metric from silently vanishing.
        with pytest.raises(ValidationError, match="unavailable_reason"):
            Measurement[float](value=None)

    def test_unavailable_carries_the_reason(self):
        m = Measurement[float].unavailable("no NVML on this host", unit="W")
        assert not m.available
        assert m.unavailable_reason == "no NVML on this host"

    def test_value_and_reason_are_mutually_exclusive(self):
        with pytest.raises(ValidationError):
            Measurement[float](value=1.0, unavailable_reason="but also missing?")

    def test_estimate_without_methodology_is_rejected(self):
        # Section 11: never report an estimate whose methodology is undocumented.
        with pytest.raises(ValidationError, match="methodology"):
            Measurement[float](value=1.0, kind=MetricKind.ESTIMATED)

    def test_estimate_with_methodology_is_accepted(self):
        m = Measurement[float].estimated(2.5, "trapezoidal integral of NVML power", "J")
        assert m.kind is MetricKind.ESTIMATED and m.note

    def test_derived_is_distinct_from_measured(self):
        m = Measurement[float].derived(18.0, "ms", note="rtt - server_total")
        assert m.kind is MetricKind.DERIVED

    def test_round_trips_through_json(self):
        m = Measurement[float].unavailable("probe absent", unit="W")
        assert Measurement[float].model_validate_json(m.model_dump_json()) == m


class TestPercentiles:
    @pytest.mark.parametrize("q", [0, 25, 50, 90, 95, 99, 100])
    def test_matches_numpy_linear_interpolation(self, q):
        xs = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0]
        assert percentile(sorted(xs), q) == pytest.approx(float(np.percentile(xs, q)))

    def test_even_and_odd_sample_counts(self):
        assert percentile([1.0, 2.0, 3.0, 4.0], 50) == pytest.approx(2.5)
        assert percentile([1.0, 2.0, 3.0], 50) == pytest.approx(2.0)

    def test_single_sample(self):
        assert percentile([7.0], 99) == 7.0

    def test_empty_is_none_not_zero(self):
        # Zero would be a claim about latency; None is the absence of a claim.
        assert percentile([], 50) is None

    def test_out_of_range_q_raises(self):
        with pytest.raises(ValueError):
            percentile([1.0, 2.0], 101)


class TestDurationStats:
    def test_mean_never_travels_without_n_and_percentiles(self):
        s = DurationStats.from_samples([10.0, 12.0, 11.0, 50.0])
        assert s.n == 4
        for field in ("mean_ms", "p50_ms", "p90_ms", "p95_ms", "p99_ms", "min_ms", "max_ms"):
            assert getattr(s, field) is not None

    def test_single_sample_has_no_stddev(self):
        # One sample has no spread; reporting 0.0 would imply perfect consistency.
        s = DurationStats.from_samples([5.0])
        assert s.n == 1 and s.stddev_ms is None and s.coefficient_of_variation is None

    def test_empty_yields_no_statistics(self):
        s = DurationStats.from_samples([])
        assert s.n == 0 and s.mean_ms is None

    def test_coefficient_of_variation_flags_instability(self):
        stable = DurationStats.from_samples([10.0, 10.1, 9.9, 10.0])
        volatile = DurationStats.from_samples([10.0, 90.0, 12.0, 85.0])
        assert stable.coefficient_of_variation < volatile.coefficient_of_variation


class TestPhaseBreakdown:
    @staticmethod
    def _iteration(index, group, total, spans):
        return IterationSample(
            index=index,
            group=group,
            total_ms=total,
            spans=[PhaseSpan(phase=p, duration_ms=d) for p, d in spans],
        )

    def test_warmup_is_retained_but_excluded_from_statistics(self):
        iterations = [
            self._iteration(0, IterationPhaseGroup.WARMUP, 500.0, [(Phase.MODEL_EXECUTION, 500.0)]),
            self._iteration(1, IterationPhaseGroup.MEASURED, 10.0, [(Phase.MODEL_EXECUTION, 10.0)]),
            self._iteration(2, IterationPhaseGroup.MEASURED, 12.0, [(Phase.MODEL_EXECUTION, 12.0)]),
        ]
        b = PhaseBreakdown.from_iterations(iterations)
        # The 500 ms warm-up would have tripled the mean had it leaked in.
        assert b.total.n == 2
        assert b.total.mean_ms == pytest.approx(11.0)

    def test_failed_iterations_do_not_pollute_statistics(self):
        good = self._iteration(0, IterationPhaseGroup.MEASURED, 10.0, [(Phase.MODEL_EXECUTION, 10.0)])
        bad = IterationSample(
            index=1, group=IterationPhaseGroup.MEASURED, total_ms=0.5,
            succeeded=False, error_type="RuntimeError", error_message="boom",
        )
        b = PhaseBreakdown.from_iterations([good, bad])
        assert b.total.n == 1 and b.total.mean_ms == pytest.approx(10.0)

    def test_residual_is_reported_not_hidden(self):
        # Total exceeds the sum of its phases; that gap must surface as residual
        # overhead rather than being charged to whichever phase is convenient.
        it = self._iteration(
            0, IterationPhaseGroup.MEASURED, 20.0,
            [(Phase.PREPROCESSING, 5.0), (Phase.MODEL_EXECUTION, 10.0)],
        )
        b = PhaseBreakdown.from_iterations([it])
        assert b.residual_ms == pytest.approx(5.0)

    def test_sub_spans_do_not_double_count(self):
        it = IterationSample(
            index=0, group=IterationPhaseGroup.MEASURED, total_ms=10.0,
            spans=[
                PhaseSpan(phase=Phase.PREPROCESSING, duration_ms=10.0),
                PhaseSpan(phase=Phase.PREPROCESSING, duration_ms=6.0,
                          parent=Phase.PREPROCESSING, label="resize"),
                PhaseSpan(phase=Phase.PREPROCESSING, duration_ms=4.0,
                          parent=Phase.PREPROCESSING, label="normalize"),
            ],
        )
        b = PhaseBreakdown.from_iterations([it])
        # 10, not 20: the child spans are contained within the parent.
        assert b.phases[Phase.PREPROCESSING].mean_ms == pytest.approx(10.0)
        assert b.residual_ms == pytest.approx(0.0)
