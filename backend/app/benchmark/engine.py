"""The benchmark engine.

One engine, used by both the HTTP API and the CLI (§24), so a benchmark cannot
mean two different things depending on how it was launched.

The measured lifecycle, in order:

    collect environment
    snapshot memory (before_load)
    [cold start] load -> warm-up -> first inference
    snapshot memory (after_load)
    start hardware sampler
    warm-up iterations        (recorded, excluded from statistics)
    measured iterations       (recorded, included)
    stop sampler
    snapshot memory (after_run)
    cooldown
    derive statistics, throughput, memory, energy
    attach integrity warnings

Integrity rules enforced here rather than left to the caller:

* Warm-up samples are kept but never counted (§12, §14).
* A failed iteration is recorded with its error and excluded from statistics; the
  result reports how many failed and states that exclusion explicitly (§14).
* A run with too few samples for meaningful percentiles is flagged.
* Background CPU load and GPU thermal throttling during the run are detected and
  flagged, because they invalidate comparisons rather than the run itself.
* Cancellation and timeouts stop the run and mark it, rather than silently
  returning a short result that looks complete.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from app.adapters.base import InferenceRequest, LoadConfig, ModelAdapter
from app.benchmark.throughput import build_throughput
from app.core.logging import get_logger
from app.instrumentation.energy import integrate_energy
from app.instrumentation.environment import (
    collect_hardware,
    collect_reproducibility,
    collect_software,
)
from app.instrumentation.memory import build_memory_metrics, snapshot
from app.instrumentation.probes.gpu import NvmlProbe
from app.instrumentation.probes.system import SystemProbe
from app.instrumentation.sampler import INTERVAL_MS_BY_MODE, HardwareSampler, SamplingDetail
from app.instrumentation.timeline import Timeline
from app.schemas.enums import (
    ExecutionLocation,
    IterationPhaseGroup,
    Phase,
    RunStatus,
    Task,
)
from app.schemas.environment import EnvironmentFingerprint, ModelReference, RuntimeReference
from app.schemas.run import BenchmarkRun, ColdWarmSplit, IterationFailure, RunErrors, RunIdentity
from app.schemas.scenario import ScenarioSpec
from app.schemas.timing import IterationSample, PhaseBreakdown

log = get_logger("benchmark.engine")


class BenchmarkCancelled(RuntimeError):
    """Raised internally when a run is cancelled; surfaces as RunStatus.CANCELLED."""


@dataclass
class EngineOptions:
    """Engine-level knobs that are not part of the scenario definition."""

    gpu_index: int = 0
    enable_sampler: bool = True
    reproduction_command: str | None = None
    label: str | None = None
    tags: list[str] = field(default_factory=list)


class BenchmarkEngine:
    """Executes a scenario against a loaded-or-loadable adapter."""

    def __init__(self, options: EngineOptions | None = None) -> None:
        self.options = options or EngineOptions()
        self._gpu = NvmlProbe()

    def close(self) -> None:
        self._gpu.shutdown()

    # --- main entry point -------------------------------------------------

    def run(
        self,
        adapter: ModelAdapter,
        scenario: ScenarioSpec,
        load_config: LoadConfig,
        runtime_ref: RuntimeReference,
        execution_location: ExecutionLocation = ExecutionLocation.IN_PROCESS,
        request: InferenceRequest | None = None,
        cancel: threading.Event | None = None,
    ) -> BenchmarkRun:
        cancel = cancel or threading.Event()
        started_wall = time.perf_counter()
        warnings: list[str] = []

        system = SystemProbe()
        hardware = collect_hardware(self._gpu)
        software = collect_software()

        snapshots = [snapshot("before_load", system, self._gpu, self.options.gpu_index)]
        gpu_temp_start = self._gpu.temperature_c(self.options.gpu_index)
        throttling_start = self._gpu.is_throttling(self.options.gpu_index)

        cold = ColdWarmSplit()
        iterations: list[IterationSample] = []
        failures: list[IterationFailure] = []
        status = RunStatus.RUNNING
        peak_rss = system.process_rss_mb()
        detected_objects = 0
        postprocess_total_ms = 0.0

        sampler: HardwareSampler | None = None

        try:
            # --- load (cold start) ---
            load_result = adapter.load(load_config)
            cold.model_load_ms = load_result.load_ms
            snapshots.append(snapshot("after_load", system, self._gpu, self.options.gpu_index))

            if not load_result.ok:
                raise RuntimeError(load_result.message or "adapter load failed")

            payload = request or adapter.synthetic_request(scenario.batch_size)
            if request is None:
                warnings.append(
                    "measured against the adapter's deterministic synthetic input, not a dataset; "
                    "latency reflects the runtime path rather than representative content"
                )

            # --- sampler ---
            if self.options.enable_sampler:
                sampler = HardwareSampler(
                    interval_ms=INTERVAL_MS_BY_MODE.get(scenario.mode.value, 1000.0),
                    detail=SamplingDetail.for_mode(scenario.mode.value),
                    gpu_index=self.options.gpu_index,
                    gpu_probe=self._gpu,
                )
                sampler.start()

            deadline = time.perf_counter() + scenario.timeout_seconds

            # --- warm-up: recorded, never counted ---
            for i in range(scenario.warmup_iterations):
                self._check_stop(cancel, deadline)
                sample = self._one_iteration(adapter, payload, i, IterationPhaseGroup.WARMUP)
                iterations.append(sample)
                if i == 0 and sample.total_ms is not None:
                    cold.first_inference_ms = sample.total_ms
                    cold.kernel_warmup_ms = sample.total_ms

            cold.cold_start_total_ms = (cold.model_load_ms or 0.0) + (cold.first_inference_ms or 0.0)

            # --- measured ---
            for i in range(scenario.measured_iterations):
                self._check_stop(cancel, deadline)
                sample = self._one_iteration(
                    adapter, payload, scenario.warmup_iterations + i, IterationPhaseGroup.MEASURED
                )
                iterations.append(sample)

                if sample.succeeded:
                    detected_objects += int(sample.output_token_count or 0)
                    postprocess_total_ms += sum(
                        s.duration_ms for s in sample.spans if s.phase is Phase.POSTPROCESSING
                    )
                else:
                    failures.append(
                        IterationFailure(
                            index=sample.index,
                            error_type=sample.error_type or "Unknown",
                            error_message=sample.error_message or "",
                        )
                    )
                rss = system.process_rss_mb()
                if rss and (peak_rss is None or rss > peak_rss):
                    peak_rss = rss

            status = RunStatus.PARTIAL if failures else RunStatus.COMPLETED

        except BenchmarkCancelled:
            status = RunStatus.CANCELLED
            warnings.append(
                f"run cancelled after {len([i for i in iterations if i.counts_toward_statistics])} "
                "measured iterations; statistics below summarize only what completed"
            )
        except TimeoutError:
            status = RunStatus.TIMED_OUT
            warnings.append(
                f"run exceeded its {scenario.timeout_seconds:.0f}s timeout and was stopped; "
                "statistics summarize only the iterations that completed"
            )
        except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
            status = RunStatus.FAILED
            failures.append(
                IterationFailure(index=-1, error_type=type(exc).__name__, error_message=str(exc))
            )
            log.warning("benchmark_run_failed", error=str(exc), model=adapter.metadata.model_id)
        finally:
            if sampler is not None:
                sampler.stop()

        snapshots.append(snapshot("after_run", system, self._gpu, self.options.gpu_index))

        if scenario.cooldown_seconds > 0:
            time.sleep(scenario.cooldown_seconds)

        # --- derive ---
        series = sampler.series() if sampler is not None else _empty_series()
        breakdown = PhaseBreakdown.from_iterations(iterations)
        measured = [i for i in iterations if i.counts_toward_statistics]
        measured_seconds = sum((i.total_ms or 0.0) for i in measured) / 1000.0

        throughput = build_throughput(
            adapter.metadata.task,
            successful_iterations=len(measured),
            total_measured_seconds=measured_seconds,
            batch_size=scenario.batch_size,
            mean_latency_ms=breakdown.total.mean_ms,
            concurrency=scenario.concurrency,
            detected_objects=detected_objects if adapter.metadata.task in _OBJECT_TASKS else None,
            postprocess_total_ms=postprocess_total_ms,
            frames_processed=len(measured) * scenario.batch_size,
        )

        energy = integrate_energy(
            series,
            request_count=len(measured),
            images=len(measured) * scenario.batch_size
            if adapter.metadata.modality.value == "image"
            else None,
        )

        warnings.extend(
            self._integrity_warnings(
                scenario, iterations, failures, system, gpu_temp_start, throttling_start, series
            )
        )

        model_ref = ModelReference(
            model_id=adapter.metadata.model_id,
            display_name=adapter.metadata.display_name,
            revision=adapter.metadata.revision,
            weights_checksum_sha256=adapter.metadata.weights_checksum_sha256,
            parameters_millions=adapter.metadata.parameters_millions,
            file_size_bytes=adapter.metadata.model_size_bytes,
        )
        reproducibility = collect_reproducibility(
            seed=scenario.random_seed,
            deterministic=scenario.deterministic,
            dataset_revision=scenario.input_dataset,
            reproduction_command=self.options.reproduction_command,
        )
        if reproducibility.git_dirty:
            warnings.append(
                "the working tree had uncommitted changes, so this run is not reproducible "
                "from its commit alone"
            )

        thermal = self._thermal_state(system, gpu_temp_start, throttling_start)

        return BenchmarkRun(
            identity=RunIdentity(label=self.options.label, tags=list(self.options.tags)),
            status=status,
            scenario=scenario,
            model=model_ref,
            runtime=runtime_ref,
            execution_location=execution_location,
            task=adapter.metadata.task,
            mode=scenario.mode,
            hardware=hardware,
            software=software,
            fingerprint=EnvironmentFingerprint.compute(
                hardware, software, model_ref, runtime_ref, scenario.mode
            ),
            thermal_and_load=thermal,
            reproducibility=reproducibility,
            timings=breakdown,
            cold_warm=ColdWarmSplit(
                model_load_ms=cold.model_load_ms,
                kernel_warmup_ms=cold.kernel_warmup_ms,
                first_inference_ms=cold.first_inference_ms,
                cold_start_total_ms=cold.cold_start_total_ms,
                warm_inference=breakdown.total,
            ),
            throughput=throughput,
            memory=build_memory_metrics(snapshots, peak_rss),
            utilization=series,
            energy=energy,
            iterations=iterations,
            errors=RunErrors(failures=failures, statistics_exclude_failures=True),
            warnings=warnings,
            instrumentation_overhead_ms=(
                series.sampler_overhead_ms.value if series.sampler_overhead_ms.available else None
            ),
            duration_s=time.perf_counter() - started_wall,
        )

    # --- internals --------------------------------------------------------

    @staticmethod
    def _check_stop(cancel: threading.Event, deadline: float) -> None:
        if cancel.is_set():
            raise BenchmarkCancelled
        if time.perf_counter() > deadline:
            raise TimeoutError

    def _one_iteration(
        self,
        adapter: ModelAdapter,
        request: InferenceRequest,
        index: int,
        group: IterationPhaseGroup,
    ) -> IterationSample:
        """Execute and time one inference. Failures are captured, never raised out."""
        timeline = Timeline()
        timeline.start()
        synchronize = getattr(adapter, "synchronize", None)

        # Text adapters report TOKENIZATION here; image adapters PREPROCESSING. Using a
        # single hardcoded phase would have hidden tokenization cost inside a generic
        # bucket for every text workload.
        preprocess_phase = getattr(adapter, "preprocess_phase", Phase.PREPROCESSING)

        try:
            with timeline.span(preprocess_phase):
                prepared = adapter.preprocess(request)
            with timeline.span(Phase.MODEL_EXECUTION, synchronize=synchronize):
                raw = adapter.infer(prepared)
            with timeline.span(Phase.POSTPROCESSING):
                output = adapter.postprocess(raw, prepared)
            timeline.stop()

            # Object count rides in output_token_count so detection throughput can be
            # derived without a second task-specific field on IterationSample.
            produced = len(output.detections) if output.detections is not None else None

            return IterationSample(
                index=index,
                group=group,
                total_ms=timeline.total_ms,
                spans=timeline.spans(),
                succeeded=True,
                output_token_count=produced,
            )
        except Exception as exc:  # noqa: BLE001 - a failed iteration is data, not a crash
            timeline.stop()
            log.warning(
                "iteration_failed", index=index, error=str(exc), error_type=type(exc).__name__
            )
            return IterationSample(
                index=index,
                group=group,
                total_ms=timeline.total_ms,
                spans=[],
                succeeded=False,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

    def _integrity_warnings(
        self,
        scenario: ScenarioSpec,
        iterations: list[IterationSample],
        failures: list[IterationFailure],
        system: SystemProbe,
        gpu_temp_start: float | None,
        throttling_start: bool | None,
        series,
    ) -> list[str]:
        out: list[str] = []
        measured = [i for i in iterations if i.counts_toward_statistics]

        if len(measured) < 10:
            out.append(
                f"only {len(measured)} measured iteration(s) completed; percentiles above P50 "
                "are not statistically meaningful below about 10 samples"
            )
        if failures:
            out.append(
                f"{len(failures)} of {scenario.measured_iterations} measured iteration(s) failed; "
                "statistics exclude them, so the reported rate reflects successful work only"
            )
        if any(
            not span.device_synchronized
            for it in measured
            for span in it.spans
            if span.phase is Phase.MODEL_EXECUTION
        ):
            out.append(
                "model execution was timed without device synchronization; on an asynchronous "
                "device this measures dispatch rather than completion"
            )

        busy, detail = system.detect_background_load()
        if busy:
            out.append(f"background load detected during the run: {detail}")

        throttling_end = self._gpu.is_throttling(self.options.gpu_index)
        if throttling_end or throttling_start:
            out.append(
                "the GPU reported thermal or power throttling during this run; results reflect "
                "a throttled device and are not comparable with an unthrottled one"
            )
        gpu_temp_end = self._gpu.temperature_c(self.options.gpu_index)
        if gpu_temp_start is not None and gpu_temp_end is not None:
            if gpu_temp_end - gpu_temp_start > 15.0:
                out.append(
                    f"GPU temperature rose {gpu_temp_end - gpu_temp_start:.0f}C during the run "
                    f"({gpu_temp_start:.0f}C to {gpu_temp_end:.0f}C); later iterations may have "
                    "been clocked lower than earlier ones"
                )
        if scenario.mode.value != "standard":
            out.append(
                f"measured in '{scenario.mode.value}' mode, which adds instrumentation overhead; "
                "these results are not comparable with standard-mode results"
            )
        return out

    def _thermal_state(self, system: SystemProbe, temp_start, throttling_start):
        from app.schemas.environment import ThermalAndLoadState

        busy, detail = system.detect_background_load()
        return ThermalAndLoadState(
            gpu_temperature_start_c=temp_start,
            gpu_temperature_end_c=self._gpu.temperature_c(self.options.gpu_index),
            thermal_throttling_detected=(
                None if throttling_start is None
                else bool(throttling_start or self._gpu.is_throttling(self.options.gpu_index))
            ),
            system_load_average_1m=system.load_average_1m(),
            concurrent_workload_detected=busy,
            concurrent_workload_detail=detail,
        )


_OBJECT_TASKS = (Task.OBJECT_DETECTION, Task.IMAGE_SEGMENTATION)


def _empty_series():
    from app.schemas.measurement import Measurement
    from app.schemas.resources import UtilizationSeries

    return UtilizationSeries(
        samples=[],
        sample_interval_ms=0.0,
        sampler_overhead_ms=Measurement[float].unavailable(
            "hardware sampling was disabled for this run", "ms"
        ),
        sources=[],
        unavailable={"sampler": "disabled via EngineOptions.enable_sampler=False"},
    )
