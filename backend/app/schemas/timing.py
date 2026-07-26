"""Timing schemas: phase spans, per-iteration samples, and aggregate statistics.

Two rules from the brief are encoded structurally here rather than left to
discipline:

* An average never travels alone (§7). ``DurationStats`` cannot be constructed
  without the sample count and the full percentile set, so a template that renders
  a mean always has ``n`` and ``p95`` in hand.
* Warm-up samples are retained, not discarded (§14), but are excluded from the
  statistics. Both facts are visible in the same object.
"""
from __future__ import annotations

import math
import statistics

from pydantic import BaseModel, Field

from app.schemas.enums import IterationPhaseGroup, Phase


class PhaseSpan(BaseModel):
    """One measured phase of a single inference.

    Durations come from a monotonic clock (``time.perf_counter``); wall-clock time
    is never used to compute a duration. ``parent`` allows a hierarchy such as
    ``preprocessing -> resize``.
    """

    phase: Phase
    duration_ms: float = Field(ge=0.0)
    parent: Phase | None = None
    label: str | None = Field(
        default=None,
        description="Sub-span name when several spans share a phase, e.g. 'resize', 'normalize'.",
    )
    device_synchronized: bool = Field(
        default=False,
        description="True when the device was synchronized before the clock stopped. "
                    "False on a GPU path means this measures dispatch, not execution.",
    )
    note: str | None = None


class IterationSample(BaseModel):
    """One complete execution. Raw samples are persisted; §19 forbids storing only averages."""

    index: int = Field(ge=0)
    group: IterationPhaseGroup
    total_ms: float | None = Field(default=None, ge=0.0)
    spans: list[PhaseSpan] = Field(default_factory=list)
    succeeded: bool = True
    error_type: str | None = None
    error_message: str | None = None
    # Populated for generative workloads; None where the concept does not apply.
    time_to_first_token_ms: float | None = None
    inter_token_latency_ms: list[float] = Field(default_factory=list)
    output_token_count: int | None = None
    prompt_token_count: int | None = None

    @property
    def counts_toward_statistics(self) -> bool:
        return self.succeeded and self.group is IterationPhaseGroup.MEASURED


class DurationStats(BaseModel):
    """Aggregate statistics over a set of durations.

    Build with :meth:`from_samples` rather than by hand — it is the only path that
    guarantees the percentile set and ``n`` agree with the data.
    """

    n: int = Field(ge=0, description="Number of samples these statistics summarize.")
    min_ms: float | None = None
    max_ms: float | None = None
    mean_ms: float | None = None
    median_ms: float | None = None
    stddev_ms: float | None = None
    p50_ms: float | None = None
    p90_ms: float | None = None
    p95_ms: float | None = None
    p99_ms: float | None = None
    coefficient_of_variation: float | None = Field(
        default=None,
        description="stddev / mean. Dimensionless. A high value means the mean is a poor summary.",
    )

    @classmethod
    def from_samples(cls, samples: list[float]) -> DurationStats:
        """Compute the full statistic set. An empty input yields n=0 and all-None."""
        if not samples:
            return cls(n=0)
        ordered = sorted(samples)
        mean = statistics.fmean(ordered)
        # Sample standard deviation needs n >= 2; a single sample has no spread to report.
        stddev = statistics.stdev(ordered) if len(ordered) > 1 else None
        return cls(
            n=len(ordered),
            min_ms=ordered[0],
            max_ms=ordered[-1],
            mean_ms=mean,
            median_ms=statistics.median(ordered),
            stddev_ms=stddev,
            p50_ms=percentile(ordered, 50),
            p90_ms=percentile(ordered, 90),
            p95_ms=percentile(ordered, 95),
            p99_ms=percentile(ordered, 99),
            coefficient_of_variation=(stddev / mean) if (stddev is not None and mean > 0) else None,
        )


def percentile(ordered: list[float], q: float) -> float | None:
    """Linear-interpolation percentile over an already-sorted list.

    Matches ``numpy.percentile``'s default 'linear' method, so results agree with
    the NumPy-based paths elsewhere in the codebase. ``q`` is in 0..100.
    """
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"percentile q must be within 0..100, got {q}")
    rank = (q / 100.0) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[int(rank)]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


class PhaseBreakdown(BaseModel):
    """Per-phase statistics across all measured iterations, plus the residual.

    ``residual_ms`` is the part of the measured total that no phase accounted for.
    It is reported explicitly instead of being folded into whichever phase is
    convenient — §18 of the brief specifically forbids charging unattributed time
    to 'network'.
    """

    phases: dict[Phase, DurationStats] = Field(default_factory=dict)
    total: DurationStats = Field(default_factory=lambda: DurationStats(n=0))
    residual_ms: float | None = Field(
        default=None,
        description="mean(total) - sum(mean(phase)). Labelled 'residual overhead' in the UI.",
    )

    @classmethod
    def from_iterations(cls, iterations: list[IterationSample]) -> PhaseBreakdown:
        measured = [it for it in iterations if it.counts_toward_statistics]
        totals = [it.total_ms for it in measured if it.total_ms is not None]

        by_phase: dict[Phase, list[float]] = {}
        for it in measured:
            for span in it.spans:
                # Only top-level spans contribute to the breakdown; sub-spans are
                # already contained within their parent and would double-count.
                if span.parent is None:
                    by_phase.setdefault(span.phase, []).append(span.duration_ms)

        phases = {p: DurationStats.from_samples(v) for p, v in by_phase.items()}
        total = DurationStats.from_samples(totals)

        residual: float | None = None
        if total.mean_ms is not None and phases:
            accounted = sum(s.mean_ms for s in phases.values() if s.mean_ms is not None)
            residual = total.mean_ms - accounted

        return cls(phases=phases, total=total, residual_ms=residual)
