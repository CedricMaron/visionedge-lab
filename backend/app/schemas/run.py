"""``BenchmarkRun`` — the versioned top-level result record.

One run is one (scenario x model x runtime x hardware) measurement. It carries both
aggregate statistics *and* every raw iteration, because §19 of the brief forbids
storing only averages: without the raw samples nobody can re-derive a percentile,
spot a bimodal distribution, or see that the mean was dragged by one outlier.

Schema versioning: ``schema_version`` is bumped on any breaking change, and the
persistence layer migrates old rows forward rather than reinterpreting them.
"""
from __future__ import annotations

import time
import uuid

from pydantic import BaseModel, Field, computed_field

from app.schemas.enums import (
    BenchmarkMode,
    ExecutionLocation,
    IterationPhaseGroup,
    RunStatus,
    Task,
)
from app.schemas.environment import (
    EnvironmentFingerprint,
    HardwareInfo,
    ModelReference,
    Reproducibility,
    RuntimeReference,
    SoftwareEnvironment,
    ThermalAndLoadState,
)
from app.schemas.quality import QualityMetrics
from app.schemas.resources import EnergyMetrics, MemoryMetrics, ThroughputMetrics, UtilizationSeries
from app.schemas.scenario import ScenarioSpec
from app.schemas.timing import DurationStats, IterationSample, PhaseBreakdown

RESULT_SCHEMA_VERSION = 1


class RunIdentity(BaseModel):
    run_id: str = Field(default_factory=lambda: f"run_{uuid.uuid4().hex[:12]}")
    created_at: float = Field(default_factory=time.time, description="Wall clock, for ordering only.")
    label: str | None = None
    tags: list[str] = Field(default_factory=list)


class IterationFailure(BaseModel):
    index: int
    error_type: str
    error_message: str
    phase: str | None = None


class RunErrors(BaseModel):
    """Failed iterations are never silently dropped (§14)."""

    failures: list[IterationFailure] = Field(default_factory=list)
    statistics_exclude_failures: bool = Field(
        default=True,
        description="Always true in the current engine, recorded explicitly so a reader "
                    "never has to guess whether a failed run polluted the mean.",
    )

    @property
    def failure_count(self) -> int:
        return len(self.failures)


class ColdWarmSplit(BaseModel):
    """Cold-start costs, separated from steady-state (§12)."""

    dependency_init_ms: float | None = None
    model_download_ms: float | None = None
    model_load_ms: float | None = None
    graph_compilation_ms: float | None = None
    engine_build_ms: float | None = None
    kernel_warmup_ms: float | None = None
    first_inference_ms: float | None = None
    cold_start_total_ms: float | None = None
    warm_inference: DurationStats = Field(default_factory=lambda: DurationStats(n=0))


class ProfilerArtifact(BaseModel):
    """Large profiling outputs are referenced, not embedded (§20)."""

    kind: str = Field(description="e.g. 'chrome_trace', 'ort_profile', 'torch_profiler'")
    path: str
    size_bytes: int
    note: str | None = None


class BenchmarkRun(BaseModel):
    """A complete, self-describing benchmark result."""

    schema_version: int = RESULT_SCHEMA_VERSION

    identity: RunIdentity = Field(default_factory=RunIdentity)
    status: RunStatus = RunStatus.PENDING

    scenario: ScenarioSpec
    model: ModelReference
    runtime: RuntimeReference
    execution_location: ExecutionLocation = ExecutionLocation.IN_PROCESS
    task: Task
    mode: BenchmarkMode = BenchmarkMode.STANDARD

    hardware: HardwareInfo
    software: SoftwareEnvironment
    fingerprint: EnvironmentFingerprint
    thermal_and_load: ThermalAndLoadState = Field(default_factory=ThermalAndLoadState)
    reproducibility: Reproducibility = Field(default_factory=Reproducibility)

    # --- measurements ---
    timings: PhaseBreakdown = Field(default_factory=PhaseBreakdown)
    cold_warm: ColdWarmSplit = Field(default_factory=ColdWarmSplit)
    throughput: ThroughputMetrics
    memory: MemoryMetrics
    utilization: UtilizationSeries
    energy: EnergyMetrics
    quality: QualityMetrics = Field(default_factory=QualityMetrics)

    # --- raw evidence ---
    iterations: list[IterationSample] = Field(
        default_factory=list,
        description="Every iteration, including warm-up and failures. Never truncated to averages.",
    )
    errors: RunErrors = Field(default_factory=RunErrors)
    warnings: list[str] = Field(
        default_factory=list,
        description="Integrity warnings: thin sample, background load, thermal throttling, "
                    "unsynchronized device timing. Surfaced next to the numbers, not buried.",
    )
    artifacts: list[ProfilerArtifact] = Field(default_factory=list)

    instrumentation_overhead_ms: float | None = Field(
        default=None,
        description="Measured cost the instrumentation itself added per iteration (§27).",
    )
    duration_s: float | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def successful_iterations(self) -> int:
        return sum(
            1 for it in self.iterations
            if it.succeeded and it.group is IterationPhaseGroup.MEASURED
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def failed_iterations(self) -> int:
        return sum(1 for it in self.iterations if not it.succeeded)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def warmup_iterations_run(self) -> int:
        return sum(1 for it in self.iterations if it.group is IterationPhaseGroup.WARMUP)

    def is_comparable_to(self, other: BenchmarkRun) -> tuple[bool, list[str]]:
        """Whether two runs may be placed side by side, and what differs if not.

        Comparison is refused rather than silently normalized when the runs measured
        materially different things. The reasons are returned so the UI can explain
        the refusal instead of just greying a button out.
        """
        reasons: list[str] = []
        if self.task is not other.task:
            reasons.append(f"different tasks: {self.task.value} vs {other.task.value}")
        if self.scenario.id != other.scenario.id:
            reasons.append(f"different scenarios: {self.scenario.id} vs {other.scenario.id}")
        if self.mode is not other.mode:
            reasons.append(
                f"different instrumentation modes: {self.mode.value} vs {other.mode.value} "
                "— profiler and detailed modes perturb timing and are not comparable to standard"
            )
        if self.scenario.batch_size != other.scenario.batch_size:
            reasons.append(
                f"different batch sizes: {self.scenario.batch_size} vs {other.scenario.batch_size}"
            )
        if self.scenario.concurrency != other.scenario.concurrency:
            reasons.append(
                f"different concurrency: {self.scenario.concurrency} vs {other.scenario.concurrency}"
            )
        if self.scenario.input_size != other.scenario.input_size:
            reasons.append(
                f"different input sizes: {self.scenario.input_size} vs {other.scenario.input_size}"
            )
        if self.scenario.sequence_length != other.scenario.sequence_length:
            reasons.append(
                f"different sequence lengths: {self.scenario.sequence_length} "
                f"vs {other.scenario.sequence_length}"
            )
        return (not reasons), reasons
