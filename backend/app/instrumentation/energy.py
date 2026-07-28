"""Energy accounting, integrated from measured GPU power samples.

Energy is not measured directly — no consumer GPU exposes a joule counter. It is
the integral of instantaneous power over time, so it is classified ``DERIVED``
(exact arithmetic on measured inputs) rather than ``MEASURED``.

The methodology, stated once here and repeated in every ``note`` the UI renders:

    E = trapezoidal integral of NVML instantaneous power over the sampled window.

Accuracy is bounded by the sampling interval. NVML's power reading is itself an
average over roughly the last millisecond, and sampling every 100 ms means bursts
between samples are invisible. Reported energy is therefore reliable for runs
lasting many multiples of the sample interval and unreliable for very short ones —
which is why :func:`integrate_energy` refuses to produce a figure below a minimum
sample count instead of returning a confident-looking wrong number.

Scope: **GPU only**. CPU package power would need RAPL via
``/sys/class/powercap/intel-rapl``, which is not readable under WSL2 on the
reference machine, and there is no NVML equivalent for system RAM or storage. So
these figures are the GPU's share of consumption, not whole-system draw, and every
value says so.
"""
from __future__ import annotations

from app.schemas.measurement import FloatMeasurement, Measurement
from app.schemas.resources import EnergyMetrics, UtilizationSeries

#: Below this many power samples, trapezoidal integration over a sparse series is
#: not defensible and the metric is reported unavailable instead.
MIN_POWER_SAMPLES = 3

_SCOPE_NOTE = (
    "GPU only, via NVML. Excludes CPU package, RAM and storage power, which this "
    "platform cannot read on the reference hardware."
)
_METHOD = (
    "trapezoidal integral of NVML instantaneous power over the sampled window; "
    "accuracy is bounded by the sampling interval"
)


def _unavailable(reason: str, unit: str) -> FloatMeasurement:
    return Measurement[float].unavailable(reason, unit=unit, source="NVML")


def trapezoidal_energy_j(times_s: list[float], power_w: list[float]) -> float:
    """Integrate a power series to joules.

    Both lists must be the same length and ``times_s`` ascending. Trapezoidal rather
    than rectangular because power ramps between samples; assuming a step function
    would systematically over- or under-count depending on which edge was taken.
    """
    if len(times_s) != len(power_w):
        raise ValueError("times and power series must be the same length")
    if len(times_s) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(times_s)):
        dt = times_s[i] - times_s[i - 1]
        if dt < 0:
            raise ValueError("time series must be ascending")
        total += (power_w[i] + power_w[i - 1]) / 2.0 * dt
    return total


def integrate_energy(
    series: UtilizationSeries,
    request_count: int,
    *,
    tokens: int | None = None,
    images: int | None = None,
    audio_seconds: float | None = None,
    video_frames: int | None = None,
) -> EnergyMetrics:
    """Derive energy metrics from a sampled utilization series.

    Everything is unavailable-with-a-reason when the power probe produced nothing,
    which is the normal case on the CPU-only production host.
    """
    samples = [s for s in series.samples if s.gpu_power_w is not None]

    if not samples:
        reason = series.unavailable.get(
            "NVML", "no GPU power samples were collected during this run"
        )
        return _all_unavailable(reason)

    if len(samples) < MIN_POWER_SAMPLES:
        reason = (
            f"only {len(samples)} power sample(s) collected; at least {MIN_POWER_SAMPLES} are "
            f"needed to integrate a power series over a {series.sample_interval_ms:.0f} ms interval"
        )
        return _all_unavailable(reason)

    times_s = [s.t_offset_ms / 1000.0 for s in samples]
    power_w = [s.gpu_power_w for s in samples]

    total_j = trapezoidal_energy_j(times_s, power_w)
    window_s = times_s[-1] - times_s[0]
    average_w = (total_j / window_s) if window_s > 0 else power_w[0]
    peak_w = max(power_w)

    power_limit = next(
        (s.gpu_power_limit_w for s in series.samples if s.gpu_power_limit_w), None
    )

    def derived(value: float, unit: str, extra: str = "") -> FloatMeasurement:
        return Measurement[float].derived(
            value, unit=unit, source="NVML nvmlDeviceGetPowerUsage",
            note=f"{_METHOD}. {_SCOPE_NOTE}{(' ' + extra) if extra else ''}",
        )

    def per_unit(count: float | None, unit_name: str, unit: str) -> FloatMeasurement:
        if not count:
            return _unavailable(
                f"this workload produced no {unit_name}, so the metric does not apply", unit
            )
        return derived(total_j / count, unit)

    def inverse(count: float | None, unit_name: str, unit: str) -> FloatMeasurement:
        if not count or total_j <= 0:
            return _unavailable(
                f"this workload produced no {unit_name}, so the metric does not apply", unit
            )
        return derived(count / total_j, unit)

    return EnergyMetrics(
        average_power_w=derived(average_w, "W", f"mean over {window_s:.1f} s of sampling."),
        peak_power_w=Measurement[float].of(
            peak_w, "W", "NVML nvmlDeviceGetPowerUsage",
            note=f"highest of {len(samples)} samples. {_SCOPE_NOTE}",
        ),
        power_limit_w=(
            Measurement[float].of(power_limit, "W", "NVML nvmlDeviceGetPowerManagementLimit")
            if power_limit
            else _unavailable("the device did not report a power management limit", "W")
        ),
        total_energy_j=derived(total_j, "J"),
        energy_per_request_j=(
            derived(total_j / request_count, "J")
            if request_count > 0
            else _unavailable("no successful requests to divide by", "J")
        ),
        joules_per_token=per_unit(tokens, "tokens", "J/token"),
        joules_per_image=per_unit(images, "images", "J/image"),
        joules_per_audio_second=per_unit(audio_seconds, "audio", "J/audio-s"),
        joules_per_video_frame=per_unit(video_frames, "video frames", "J/frame"),
        tokens_per_joule=inverse(tokens, "tokens", "tokens/J"),
        frames_per_joule=inverse(video_frames, "video frames", "frames/J"),
        requests_per_joule=inverse(request_count, "requests", "req/J"),
    )


def _all_unavailable(reason: str) -> EnergyMetrics:
    """Every energy figure absent, all carrying the same explanation."""
    return EnergyMetrics(
        average_power_w=_unavailable(reason, "W"),
        peak_power_w=_unavailable(reason, "W"),
        power_limit_w=_unavailable(reason, "W"),
        energy_per_request_j=_unavailable(reason, "J"),
        total_energy_j=_unavailable(reason, "J"),
        joules_per_token=_unavailable(reason, "J/token"),
        joules_per_image=_unavailable(reason, "J/image"),
        joules_per_audio_second=_unavailable(reason, "J/audio-s"),
        joules_per_video_frame=_unavailable(reason, "J/frame"),
        tokens_per_joule=_unavailable(reason, "tokens/J"),
        frames_per_joule=_unavailable(reason, "frames/J"),
        requests_per_joule=_unavailable(reason, "req/J"),
    )
