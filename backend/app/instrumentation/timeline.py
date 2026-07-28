"""Hierarchical phase timing on a monotonic clock.

Three properties this module guarantees, each of which is a way benchmarks are
commonly wrong:

**Monotonic only.** Durations come from :func:`time.perf_counter`, never from
``time.time``. Wall-clock is subject to NTP steps and DST; a benchmark that
straddles an adjustment would report a negative or wildly inflated phase.

**Device synchronization is explicit.** A span over GPU work must record whether
the device was synchronized before the clock stopped. Unsynchronized, the number
is kernel-launch time — often 100x smaller than execution — and looks like a
spectacular result. :meth:`Timeline.span` takes a ``synchronize`` callable and
marks the resulting span accordingly, so a reader can always tell which they got.

**Nesting cannot double-count.** Sub-spans record their parent, and the aggregation
in :class:`~app.schemas.timing.PhaseBreakdown` only sums top-level spans.

Usage::

    tl = Timeline()
    with tl.span(Phase.PREPROCESSING):
        with tl.span(Phase.PREPROCESSING, label="resize"):
            ...
    with tl.span(Phase.MODEL_EXECUTION, synchronize=runtime_sync):
        ...
    spans = tl.spans()
"""
from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from app.schemas.enums import Phase
from app.schemas.timing import PhaseSpan


@dataclass(slots=True)
class _OpenSpan:
    phase: Phase
    label: str | None
    parent: Phase | None
    started_at: float


@dataclass
class Timeline:
    """Collects phase spans for one inference.

    Not thread-safe by design: one Timeline belongs to one iteration on one thread.
    Sharing it across threads would interleave the parent stack and produce a
    nonsense hierarchy, so concurrent workloads build one Timeline per request.
    """

    _spans: list[PhaseSpan] = field(default_factory=list)
    _stack: list[_OpenSpan] = field(default_factory=list)
    _t_origin: float | None = None
    _t_end: float | None = None

    def start(self) -> None:
        """Mark the beginning of the measured region."""
        self._t_origin = time.perf_counter()

    def stop(self) -> None:
        """Mark the end of the measured region."""
        self._t_end = time.perf_counter()

    @contextmanager
    def span(
        self,
        phase: Phase,
        label: str | None = None,
        synchronize: Callable[[], None] | None = None,
        note: str | None = None,
    ) -> Iterator[None]:
        """Time a phase.

        ``synchronize`` is invoked *inside* the timed region, immediately before the
        clock stops, so the cost of waiting for the device is attributed to the phase
        that queued the work rather than disappearing into the next one.
        """
        if self._t_origin is None:
            self.start()

        parent = self._stack[-1].phase if self._stack else None
        open_span = _OpenSpan(phase=phase, label=label, parent=parent, started_at=time.perf_counter())
        self._stack.append(open_span)
        try:
            yield
        finally:
            synchronized = False
            if synchronize is not None:
                synchronize()
                synchronized = True
            elapsed_ms = (time.perf_counter() - open_span.started_at) * 1000.0
            self._stack.pop()
            self._spans.append(
                PhaseSpan(
                    phase=phase,
                    duration_ms=elapsed_ms,
                    parent=open_span.parent,
                    label=label,
                    device_synchronized=synchronized,
                    note=note,
                )
            )

    def record(
        self,
        phase: Phase,
        duration_ms: float,
        parent: Phase | None = None,
        label: str | None = None,
        note: str | None = None,
        device_synchronized: bool = False,
    ) -> None:
        """Record a phase measured elsewhere.

        Used for durations this process did not time itself — a server-reported
        phase arriving in a response, or a client-side figure sent up from the
        browser. Kept separate from :meth:`span` so it is obvious in the code which
        numbers are locally measured.
        """
        self._spans.append(
            PhaseSpan(
                phase=phase,
                duration_ms=duration_ms,
                parent=parent,
                label=label,
                device_synchronized=device_synchronized,
                note=note,
            )
        )

    def spans(self) -> list[PhaseSpan]:
        if self._stack:
            raise RuntimeError(
                f"{len(self._stack)} span(s) still open: "
                f"{[s.phase.value for s in self._stack]} — a span was never exited"
            )
        return list(self._spans)

    @property
    def total_ms(self) -> float | None:
        """Measured wall duration of the whole region, if start/stop were called."""
        if self._t_origin is None or self._t_end is None:
            return None
        return (self._t_end - self._t_origin) * 1000.0

    def top_level_total_ms(self) -> float:
        """Sum of top-level spans. Compare against :attr:`total_ms` to find the residual."""
        return sum(s.duration_ms for s in self._spans if s.parent is None)

    def residual_ms(self) -> float | None:
        """Unattributed time inside the measured region.

        Never labelled as any real phase — §18 of the brief specifically forbids
        charging unattributed time to 'network'.
        """
        total = self.total_ms
        if total is None:
            return None
        return total - self.top_level_total_ms()


@contextmanager
def measure_ms() -> Iterator[Callable[[], float]]:
    """Time a block without a Timeline.

    Yields a callable returning elapsed milliseconds; readable after the block, and
    live during it::

        with measure_ms() as elapsed:
            work()
        print(elapsed())
    """
    start = time.perf_counter()
    end: float | None = None

    def elapsed() -> float:
        stop = end if end is not None else time.perf_counter()
        return (stop - start) * 1000.0

    try:
        yield elapsed
    finally:
        end = time.perf_counter()
