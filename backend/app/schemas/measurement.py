"""``Measurement`` — the platform's rule that no number exists without provenance.

Every metric InferenceLab reports is wrapped in this type. A metric therefore
cannot be serialized without declaring:

* its unit,
* whether it was measured, derived or estimated,
* which instrumentation source produced it, and
* if it is missing, *why* it is missing.

This is what turns "do not fabricate metrics" from a code-review convention into
something the type system enforces. A caller that has no value cannot simply omit
the field: it must construct ``Measurement.unavailable(reason=...)`` and say why.
"""
from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, Field, model_validator

from app.schemas.enums import MetricKind

T = TypeVar("T", int, float, str, bool)


class Measurement(BaseModel, Generic[T]):
    """One metric, with the provenance needed to interpret it.

    ``value is None`` means unavailable, and ``unavailable_reason`` is then required.
    """

    value: T | None = None
    unit: str = ""
    kind: MetricKind = MetricKind.MEASURED
    source: str = Field(
        default="",
        description="Instrumentation that produced this value, e.g. 'psutil.Process.memory_info().rss' "
                    "or 'NVML nvmlDeviceGetPowerUsage'. Free text, but must name a real API.",
    )
    unavailable_reason: str | None = Field(
        default=None,
        description="Required when value is None. Shown verbatim in the UI.",
    )
    note: str | None = Field(
        default=None,
        description="Caveats that survive into the UI, e.g. 'CPU dispatch only, device not synchronized'.",
    )

    @property
    def available(self) -> bool:
        return self.value is not None

    @model_validator(mode="after")
    def _require_reason_when_absent(self) -> Measurement[T]:
        if self.value is None and not self.unavailable_reason:
            raise ValueError(
                "a Measurement without a value must carry an unavailable_reason — "
                "silently missing metrics are not permitted"
            )
        if self.value is not None and self.unavailable_reason:
            raise ValueError("a Measurement with a value must not carry an unavailable_reason")
        return self

    @model_validator(mode="after")
    def _estimated_needs_methodology(self) -> Measurement[T]:
        # Section 11 of the brief: never report an estimate whose methodology is undocumented.
        if self.value is not None and self.kind is MetricKind.ESTIMATED and not self.note:
            raise ValueError(
                "an ESTIMATED measurement must document its methodology in `note` "
                "(see docs/BENCHMARK_METHODOLOGY.md)"
            )
        return self

    # --- constructors -----------------------------------------------------

    @classmethod
    def of(
        cls,
        value: T,
        unit: str = "",
        source: str = "",
        kind: MetricKind = MetricKind.MEASURED,
        note: str | None = None,
    ) -> Measurement[T]:
        """A metric that was successfully obtained."""
        return cls(value=value, unit=unit, source=source, kind=kind, note=note)

    @classmethod
    def derived(cls, value: T, unit: str = "", source: str = "", note: str | None = None) -> Measurement[T]:
        """A metric computed exactly from other measured values."""
        return cls(value=value, unit=unit, source=source, kind=MetricKind.DERIVED, note=note)

    @classmethod
    def estimated(cls, value: T, methodology: str, unit: str = "", source: str = "") -> Measurement[T]:
        """A modelled metric. ``methodology`` is mandatory and is shown to the user."""
        return cls(
            value=value, unit=unit, source=source, kind=MetricKind.ESTIMATED, note=methodology
        )

    @classmethod
    def unavailable(cls, reason: str, unit: str = "", source: str = "") -> Measurement[T]:
        """A metric this environment cannot provide, and the reason why."""
        return cls(value=None, unit=unit, source=source, unavailable_reason=reason)


# Concrete parameterizations. Pydantic needs the generic resolved to build a schema,
# and naming them keeps annotations across the codebase short and consistent.
FloatMeasurement = Measurement[float]
IntMeasurement = Measurement[int]
StrMeasurement = Measurement[str]
BoolMeasurement = Measurement[bool]


def unavailable_float(reason: str, unit: str = "", source: str = "") -> FloatMeasurement:
    """Shorthand for the most common unavailable case."""
    return Measurement[float].unavailable(reason, unit=unit, source=source)


def unavailable_int(reason: str, unit: str = "", source: str = "") -> IntMeasurement:
    return Measurement[int].unavailable(reason, unit=unit, source=source)
