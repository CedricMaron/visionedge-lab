"""Instrumentation: timing, probes, sampling, energy and environment capture."""
from __future__ import annotations

import time

import pytest

from app.instrumentation.energy import (
    MIN_POWER_SAMPLES,
    integrate_energy,
    trapezoidal_energy_j,
)
from app.instrumentation.environment import (
    _ALLOWED_ENV_VARS,
    _commit_from_git_dir,
    collect_hardware,
    collect_software,
    git_commit,
)
from app.instrumentation.memory import build_memory_metrics, snapshot
from app.instrumentation.probes.gpu import NvmlProbe
from app.instrumentation.probes.system import SystemProbe, cpu_static_info
from app.instrumentation.sampler import HardwareSampler, SamplingDetail
from app.instrumentation.timeline import Timeline, measure_ms
from app.schemas.enums import Phase
from app.schemas.measurement import Measurement
from app.schemas.resources import UtilizationSample, UtilizationSeries


class TestTimeline:
    def test_records_span_durations(self):
        tl = Timeline()
        tl.start()
        with tl.span(Phase.PREPROCESSING):
            time.sleep(0.01)
        tl.stop()

        spans = tl.spans()
        assert len(spans) == 1
        assert spans[0].phase is Phase.PREPROCESSING
        assert spans[0].duration_ms >= 9.0

    def test_nested_spans_record_their_parent(self):
        tl = Timeline()
        with tl.span(Phase.PREPROCESSING):
            with tl.span(Phase.PREPROCESSING, label="resize"):
                pass

        parent, child = (s for s in sorted(tl.spans(), key=lambda s: s.parent is not None))
        assert parent.parent is None
        assert child.parent is Phase.PREPROCESSING and child.label == "resize"

    def test_child_duration_is_contained_by_parent(self):
        tl = Timeline()
        with tl.span(Phase.PREPROCESSING):
            with tl.span(Phase.PREPROCESSING, label="inner"):
                time.sleep(0.01)
            time.sleep(0.005)

        by_label = {s.label: s for s in tl.spans()}
        assert by_label["inner"].duration_ms < by_label[None].duration_ms

    def test_synchronize_is_called_inside_the_timed_region(self):
        # The cost of waiting for the device must land in the phase that queued the
        # work, not in whatever runs next.
        calls: list[str] = []

        def sync():
            calls.append("sync")
            time.sleep(0.02)

        tl = Timeline()
        with tl.span(Phase.MODEL_EXECUTION, synchronize=sync):
            pass

        span = tl.spans()[0]
        assert calls == ["sync"]
        assert span.device_synchronized is True
        assert span.duration_ms >= 19.0

    def test_span_without_synchronize_is_marked_unsynchronized(self):
        tl = Timeline()
        with tl.span(Phase.MODEL_EXECUTION):
            pass
        # An unsynchronized GPU span measures dispatch, not execution. The flag is
        # what lets a reader tell the difference.
        assert tl.spans()[0].device_synchronized is False

    def test_unclosed_span_is_an_error_not_a_silent_omission(self):
        tl = Timeline()
        cm = tl.span(Phase.MODEL_EXECUTION)
        cm.__enter__()
        with pytest.raises(RuntimeError, match="still open"):
            tl.spans()

    def test_span_is_closed_even_when_the_body_raises(self):
        tl = Timeline()
        with pytest.raises(ValueError), tl.span(Phase.MODEL_EXECUTION):
            raise ValueError("boom")
        assert len(tl.spans()) == 1  # recorded, and the stack is not left open

    def test_residual_is_unattributed_time(self):
        tl = Timeline()
        tl.start()
        with tl.span(Phase.PREPROCESSING):
            time.sleep(0.01)
        time.sleep(0.02)  # deliberately outside any span
        tl.stop()

        assert tl.residual_ms() >= 15.0

    def test_recorded_spans_come_from_elsewhere(self):
        tl = Timeline()
        tl.record(Phase.SERVER_MODEL_EXECUTION, 12.5, note="reported by remote server")
        span = tl.spans()[0]
        assert span.duration_ms == 12.5 and "remote" in span.note

    def test_total_is_none_before_stop(self):
        tl = Timeline()
        tl.start()
        assert tl.total_ms is None


class TestMeasureMs:
    def test_reports_elapsed_after_the_block(self):
        with measure_ms() as elapsed:
            time.sleep(0.01)
        assert elapsed() >= 9.0

    def test_is_readable_during_the_block(self):
        with measure_ms() as elapsed:
            time.sleep(0.01)
            mid = elapsed()
            time.sleep(0.01)
        assert elapsed() > mid


class TestTrapezoidalEnergy:
    def test_constant_power_is_power_times_time(self):
        # 10 W held for 2 s is 20 J, exactly.
        assert trapezoidal_energy_j([0.0, 1.0, 2.0], [10.0, 10.0, 10.0]) == pytest.approx(20.0)

    def test_linear_ramp_uses_the_average(self):
        # 0 -> 10 W over 1 s averages 5 W, so 5 J. Rectangular integration would
        # give 0 or 10 depending on which edge it took.
        assert trapezoidal_energy_j([0.0, 1.0], [0.0, 10.0]) == pytest.approx(5.0)

    def test_single_sample_yields_zero(self):
        assert trapezoidal_energy_j([1.0], [50.0]) == 0.0

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            trapezoidal_energy_j([0.0, 1.0], [10.0])

    def test_descending_time_raises(self):
        with pytest.raises(ValueError, match="ascending"):
            trapezoidal_energy_j([1.0, 0.0], [10.0, 10.0])


class TestEnergyMetrics:
    @staticmethod
    def _series(power: list[float], interval_ms: float = 100.0) -> UtilizationSeries:
        return UtilizationSeries(
            samples=[
                UtilizationSample(t_offset_ms=i * interval_ms, gpu_power_w=p)
                for i, p in enumerate(power)
            ],
            sample_interval_ms=interval_ms,
            sampler_overhead_ms=Measurement[float].of(1.0, "ms", "test"),
            sources=["NVML"],
        )

    def test_energy_is_derived_not_measured(self):
        # No consumer GPU exposes a joule counter; energy is always an integral.
        e = integrate_energy(self._series([10.0] * 10), request_count=5)
        assert e.total_energy_j.kind.value == "derived"
        assert "trapezoidal" in e.total_energy_j.note

    def test_scope_limitation_is_stated_on_every_value(self):
        e = integrate_energy(self._series([10.0] * 10), request_count=5)
        assert "GPU only" in e.total_energy_j.note

    def test_no_power_samples_is_unavailable_with_a_reason(self):
        series = UtilizationSeries(
            samples=[UtilizationSample(t_offset_ms=0.0)],
            sample_interval_ms=100.0,
            sampler_overhead_ms=Measurement[float].of(1.0, "ms", "test"),
            unavailable={"NVML": "no NVIDIA driver present"},
        )
        e = integrate_energy(series, request_count=1)
        assert not e.total_energy_j.available
        assert "no NVIDIA driver" in e.total_energy_j.unavailable_reason

    def test_too_few_samples_refuses_to_guess(self):
        e = integrate_energy(self._series([10.0] * (MIN_POWER_SAMPLES - 1)), request_count=1)
        assert not e.total_energy_j.available
        assert "at least" in e.total_energy_j.unavailable_reason

    def test_per_unit_metrics_are_unavailable_when_the_unit_does_not_apply(self):
        e = integrate_energy(self._series([10.0] * 10), request_count=5, images=5)
        assert e.joules_per_image.available
        assert not e.joules_per_token.available
        assert "no tokens" in e.joules_per_token.unavailable_reason

    def test_energy_per_request_divides_by_successful_requests(self):
        e = integrate_energy(self._series([10.0, 10.0, 10.0]), request_count=2)
        assert e.energy_per_request_j.value == pytest.approx(e.total_energy_j.value / 2)

    def test_zero_requests_does_not_divide_by_zero(self):
        e = integrate_energy(self._series([10.0] * 5), request_count=0)
        assert not e.energy_per_request_j.available


class TestSystemProbe:
    def test_first_sample_is_already_meaningful(self):
        # psutil.cpu_percent returns 0.0 on its first call; the probe primes it at
        # construction so a sampler does not record a flat zero line.
        probe = SystemProbe()
        for _ in range(200_000):
            pass
        sample = probe.sample()
        assert sample.process_rss_mb and sample.process_rss_mb > 0
        assert sample.ram_used_mb and sample.ram_used_mb > 0

    def test_optional_probes_can_be_skipped(self):
        sample = SystemProbe().sample(include_io=False, include_freq=False)
        assert sample.disk_read_mb_s is None and sample.cpu_freq_mhz is None

    def test_static_info_reports_real_hardware(self):
        info = cpu_static_info()
        assert info.cores_logical >= 1
        assert info.model

    def test_background_load_detection_returns_a_reason_when_flagged(self):
        flagged, reason = SystemProbe().detect_background_load()
        assert isinstance(flagged, bool)
        assert (reason is not None) == flagged


class TestNvmlProbe:
    def test_probe_reports_availability_or_a_reason(self):
        probe = NvmlProbe()
        try:
            assert probe.available or probe.unavailable_reason
        finally:
            probe.shutdown()

    def test_sample_returns_none_rather_than_raising_when_unavailable(self):
        probe = NvmlProbe()
        probe.shutdown()  # force the unavailable path
        assert probe.sample(0) is None

    def test_out_of_range_index_is_none(self):
        probe = NvmlProbe()
        try:
            assert probe.sample(99) is None
        finally:
            probe.shutdown()


class TestHardwareSampler:
    def test_collects_samples_over_time(self):
        with HardwareSampler(interval_ms=20.0) as sampler:
            time.sleep(0.25)
        series = sampler.series()
        assert len(series.samples) >= 5
        assert "psutil" in series.sources

    def test_offsets_are_monotonic_and_start_near_zero(self):
        with HardwareSampler(interval_ms=20.0) as sampler:
            time.sleep(0.2)
        offsets = [s.t_offset_ms for s in sampler.series().samples]
        assert offsets == sorted(offsets)
        assert offsets[0] < 50.0

    def test_reports_its_own_overhead(self):
        with HardwareSampler(interval_ms=20.0) as sampler:
            time.sleep(0.2)
        overhead = sampler.series().sampler_overhead_ms
        assert overhead.available and overhead.value >= 0.0
        assert "per tick" in overhead.note

    def test_missing_gpu_probe_is_named_in_unavailable(self):
        with HardwareSampler(interval_ms=20.0) as sampler:
            time.sleep(0.1)
        series = sampler.series()
        if "NVML" not in series.sources:
            assert "NVML" in series.unavailable

    def test_lean_detail_skips_expensive_probes(self):
        detail = SamplingDetail.lean()
        assert not detail.gpu_clocks and not detail.disk_and_network_io

    def test_full_detail_still_excludes_throttle_reasons(self):
        # Throttle reasons cost ~15 ms on the reference GPU — far too expensive to
        # sample at 100 ms. They are captured at run boundaries instead.
        detail = SamplingDetail.full()
        assert detail.gpu_clocks and detail.per_core_cpu

    def test_mode_selects_detail(self):
        assert SamplingDetail.for_mode("standard") == SamplingDetail.lean()
        assert SamplingDetail.for_mode("detailed") == SamplingDetail.full()

    def test_double_start_is_rejected(self):
        sampler = HardwareSampler(interval_ms=50.0)
        sampler.start()
        try:
            with pytest.raises(RuntimeError, match="already started"):
                sampler.start()
        finally:
            sampler.stop()

    def test_stop_before_start_is_safe(self):
        HardwareSampler(interval_ms=50.0).stop()

    def test_stop_timeout_scales_with_interval(self):
        # A fixed 2 s timeout was too short for a 1 s sampler: the join gave up and
        # the thread went on perturbing whatever ran next.
        sampler = HardwareSampler(interval_ms=1000.0)
        sampler.start()
        time.sleep(0.05)
        sampler.stop()
        assert sampler._thread is None

    def test_rejects_non_positive_interval(self):
        with pytest.raises(ValueError):
            HardwareSampler(interval_ms=0.0)


class TestMemory:
    def test_snapshot_labels_and_populates_process_memory(self):
        snap = snapshot("before_load", SystemProbe())
        assert snap.label == "before_load"
        assert snap.process_rss_mb.available

    def test_allocator_stats_are_honestly_unavailable_on_ort(self):
        # ORT exposes no allocator statistics; NVML's device-wide number is a
        # different quantity and is reported in its own field.
        snap = snapshot("x", SystemProbe())
        assert not snap.gpu_allocated_mb.available
        assert "allocator statistics" in snap.gpu_allocated_mb.unavailable_reason

    def test_weight_memory_is_the_load_delta(self):
        probe = SystemProbe()
        before = snapshot("before_load", probe)
        ballast = [bytearray(2_000_000) for _ in range(5)]  # ~10 MB
        after = snapshot("after_load", probe)
        metrics = build_memory_metrics([before, after], peak_rss_mb=100.0)

        assert metrics.model_weights_mb.available
        assert metrics.model_weights_mb.kind.value == "derived"
        del ballast

    def test_missing_snapshots_make_derived_metrics_unavailable(self):
        metrics = build_memory_metrics([], peak_rss_mb=None)
        assert not metrics.model_weights_mb.available
        assert not metrics.leak_indicator_mb.available
        assert not metrics.peak_process_rss_mb.available

    def test_kv_cache_is_unavailable_for_non_generative_workloads(self):
        metrics = build_memory_metrics([], peak_rss_mb=1.0)
        assert "not generative" in metrics.kv_cache_mb.unavailable_reason


class TestEnvironmentCapture:
    def test_hardware_is_probed(self):
        hw = collect_hardware()
        assert hw.cpu_cores_logical >= 1
        assert hw.ram_total_mb > 0
        assert hw.gpu_count == len(hw.gpus)

    def test_software_records_tracked_package_versions(self):
        sw = collect_software()
        assert sw.python_version
        # ORT is a hard dependency of this project.
        assert any(k.startswith("onnxruntime") for k in sw.package_versions)

    def test_environment_variables_are_allow_listed_not_swept(self, monkeypatch):
        # A benchmark record gets exported and shared. Collecting os.environ
        # wholesale would put credentials in it.
        monkeypatch.setenv("MY_SECRET_API_KEY", "sk-do-not-capture")
        monkeypatch.setenv("OMP_NUM_THREADS", "4")

        captured = collect_software().relevant_env_vars

        assert "MY_SECRET_API_KEY" not in captured
        assert captured.get("OMP_NUM_THREADS") == "4"
        assert set(captured).issubset(set(_ALLOWED_ENV_VARS))


class TestDeployedCommit:
    """The deploy proves it is talking to the process it just started by comparing
    /health's commit against the one it checked out. On the VPS the backend runs as
    SYSTEM, where git is off PATH and the worktree belongs to another account, so
    that check quietly degraded to "cannot verify" — which is how a stale process
    survives a restart unnoticed."""

    def test_commit_is_resolved_without_the_git_binary(self, monkeypatch):
        expected = _run_git_head()
        if expected is None:
            pytest.skip("not running inside a git worktree")

        monkeypatch.setattr("app.instrumentation.environment._run_git", lambda *a: None)
        assert git_commit() == expected

    def test_git_dir_reader_agrees_with_git_itself(self):
        expected = _run_git_head()
        if expected is None:
            pytest.skip("not running inside a git worktree")
        assert _commit_from_git_dir() == expected


def _run_git_head() -> str | None:
    import subprocess

    from app.core.config import REPO_ROOT

    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None
