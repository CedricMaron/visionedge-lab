"""Versioned data contracts for InferenceLab.

Import from this package rather than from the individual modules, so a future
schema reorganization stays internal.
"""
from __future__ import annotations

from app.schemas.enums import (
    BenchmarkMode,
    DeviceKind,
    ExecutionLocation,
    IterationPhaseGroup,
    MetricKind,
    Modality,
    Phase,
    Precision,
    RunStatus,
    Task,
)
from app.schemas.environment import (
    EnvironmentFingerprint,
    GpuDescriptor,
    HardwareInfo,
    ModelReference,
    Reproducibility,
    RuntimeReference,
    SoftwareEnvironment,
    ThermalAndLoadState,
)
from app.schemas.measurement import (
    BoolMeasurement,
    FloatMeasurement,
    IntMeasurement,
    Measurement,
    StrMeasurement,
    unavailable_float,
    unavailable_int,
)
from app.schemas.quality import (
    ClassificationQuality,
    DetectionQuality,
    EmbeddingQuality,
    QualityMetrics,
    SubjectiveEvaluation,
)
from app.schemas.resources import (
    EnergyMetrics,
    MemoryMetrics,
    MemorySnapshot,
    ThroughputMetrics,
    UtilizationSample,
    UtilizationSeries,
)
from app.schemas.run import (
    RESULT_SCHEMA_VERSION,
    BenchmarkRun,
    ColdWarmSplit,
    IterationFailure,
    ProfilerArtifact,
    RunErrors,
    RunIdentity,
)
from app.schemas.scenario import GenerationSettings, ScenarioSpec
from app.schemas.timing import (
    DurationStats,
    IterationSample,
    PhaseBreakdown,
    PhaseSpan,
    percentile,
)

__all__ = [
    "RESULT_SCHEMA_VERSION",
    "BenchmarkMode",
    "BenchmarkRun",
    "BoolMeasurement",
    "ClassificationQuality",
    "ColdWarmSplit",
    "DetectionQuality",
    "DeviceKind",
    "DurationStats",
    "EmbeddingQuality",
    "EnergyMetrics",
    "EnvironmentFingerprint",
    "ExecutionLocation",
    "FloatMeasurement",
    "GenerationSettings",
    "GpuDescriptor",
    "HardwareInfo",
    "IntMeasurement",
    "IterationFailure",
    "IterationPhaseGroup",
    "IterationSample",
    "Measurement",
    "MemoryMetrics",
    "MemorySnapshot",
    "MetricKind",
    "Modality",
    "ModelReference",
    "Phase",
    "PhaseBreakdown",
    "PhaseSpan",
    "Precision",
    "ProfilerArtifact",
    "QualityMetrics",
    "Reproducibility",
    "RunErrors",
    "RunIdentity",
    "RunStatus",
    "RuntimeReference",
    "ScenarioSpec",
    "SoftwareEnvironment",
    "StrMeasurement",
    "SubjectiveEvaluation",
    "Task",
    "ThermalAndLoadState",
    "ThroughputMetrics",
    "UtilizationSample",
    "UtilizationSeries",
    "percentile",
    "unavailable_float",
    "unavailable_int",
]
