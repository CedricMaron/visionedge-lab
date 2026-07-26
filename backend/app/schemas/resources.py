"""Memory, hardware-utilization, energy and throughput schemas.

Memory naming follows §9 of the brief strictly. Four different quantities are
routinely conflated in benchmark reports, and this module keeps them apart:

* **allocated** — bytes the framework's allocator currently holds for live tensors.
* **reserved** — bytes the allocator has taken from the driver, including its own
  free pool. Always >= allocated.
* **process** — RSS of this OS process. Includes weights, code, CUDA context, and
  anything else in the address space.
* **device total** — physical VRAM on the card, and what *other* processes are
  using of it.

Reporting "GPU memory: 1.2 GB" without saying which of these it is makes the number
meaningless, so every field names exactly one of them.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.measurement import FloatMeasurement, IntMeasurement


class MemorySnapshot(BaseModel):
    """Memory state at one named moment (before load, after load, peak, …)."""

    label: str = Field(description="e.g. 'before_load', 'after_load', 'during_inference', 'after_unload'")

    process_rss_mb: FloatMeasurement
    process_vms_mb: FloatMeasurement
    system_used_mb: FloatMeasurement
    system_available_mb: FloatMeasurement

    # GPU. All optional in the sense that they may be `unavailable` with a reason,
    # which is what happens on the CPU-only production host.
    gpu_allocated_mb: FloatMeasurement
    gpu_reserved_mb: FloatMeasurement
    gpu_process_used_mb: FloatMeasurement
    gpu_device_used_mb: FloatMeasurement
    gpu_device_total_mb: FloatMeasurement


class MemoryMetrics(BaseModel):
    """Memory across the whole run, sampled at several points rather than once."""

    snapshots: list[MemorySnapshot] = Field(default_factory=list)

    peak_process_rss_mb: FloatMeasurement
    peak_gpu_allocated_mb: FloatMeasurement
    peak_gpu_reserved_mb: FloatMeasurement

    model_weights_mb: FloatMeasurement = Field(
        description="Delta in device (or process, on CPU) memory attributable to loading weights."
    )
    kv_cache_mb: FloatMeasurement = Field(
        description="Generative models only; unavailable elsewhere with that as the reason."
    )

    leak_indicator_mb: FloatMeasurement = Field(
        description="Process RSS after the final iteration minus RSS after warm-up. "
                    "Persistent growth across iterations suggests a leak; it is reported, "
                    "not diagnosed.",
    )


class UtilizationSample(BaseModel):
    """One tick of the hardware sampler.

    ``t_offset_ms`` is measured from the start of the run on a monotonic clock, so
    samples align with the phase timeline without depending on wall-clock time.
    """

    t_offset_ms: float = Field(ge=0.0)

    cpu_percent: float | None = None
    cpu_per_core_percent: list[float] = Field(default_factory=list)
    process_cpu_percent: float | None = None
    cpu_freq_mhz: float | None = None
    thread_count: int | None = None
    context_switches: int | None = None

    ram_used_mb: float | None = None
    swap_used_mb: float | None = None

    gpu_percent: float | None = None
    gpu_memory_percent: float | None = None
    gpu_memory_used_mb: float | None = None
    gpu_clock_mhz: float | None = None
    gpu_memory_clock_mhz: float | None = None
    gpu_temperature_c: float | None = None
    gpu_power_w: float | None = None
    gpu_encoder_percent: float | None = None
    gpu_decoder_percent: float | None = None

    disk_read_mb_s: float | None = None
    disk_write_mb_s: float | None = None
    net_sent_mb_s: float | None = None
    net_recv_mb_s: float | None = None


class UtilizationSeries(BaseModel):
    """The sampled time series plus the metadata needed to judge it.

    ``sample_interval_ms`` and ``sampler_overhead_ms`` are recorded because a
    time series is uninterpretable without knowing how often it was taken and how
    much taking it cost (§27).
    """

    samples: list[UtilizationSample] = Field(default_factory=list)
    sample_interval_ms: float
    sampler_overhead_ms: FloatMeasurement
    sources: list[str] = Field(
        default_factory=list,
        description="Which probes contributed, e.g. ['psutil', 'NVML']. Empty means no probe was available.",
    )
    unavailable: dict[str, str] = Field(
        default_factory=dict,
        description="Probe name -> reason it could not be used. Rendered in the UI as-is.",
    )


class EnergyMetrics(BaseModel):
    """Energy, integrated from the sampled power series.

    Energy is a *derived* quantity: it is the trapezoidal integral of measured
    instantaneous power over the measured wall duration of the run. It is only
    populated when a real power probe (NVML) was available for the whole run; a
    partially-sampled run yields ``unavailable`` rather than an extrapolation.

    CPU package power is not read on this platform (no RAPL access under WSL2), so
    every figure here is GPU-only and says so in its ``note``.
    """

    average_power_w: FloatMeasurement
    peak_power_w: FloatMeasurement
    power_limit_w: FloatMeasurement
    energy_per_request_j: FloatMeasurement
    total_energy_j: FloatMeasurement

    # Workload-specific efficiency figures. Each is unavailable unless the workload
    # produces the corresponding unit of output.
    joules_per_token: FloatMeasurement
    joules_per_image: FloatMeasurement
    joules_per_audio_second: FloatMeasurement
    joules_per_video_frame: FloatMeasurement
    tokens_per_joule: FloatMeasurement
    frames_per_joule: FloatMeasurement
    requests_per_joule: FloatMeasurement


class ThroughputMetrics(BaseModel):
    """Workload-appropriate throughput. Fields that do not apply are unavailable."""

    # general
    requests_per_second: FloatMeasurement
    samples_per_second: FloatMeasurement
    batches_per_second: FloatMeasurement

    # generative text
    prompt_tokens_per_second: FloatMeasurement
    output_tokens_per_second: FloatMeasurement
    total_tokens_per_second: FloatMeasurement
    prefill_tokens_per_second: FloatMeasurement
    decode_tokens_per_second: FloatMeasurement
    concurrent_requests: IntMeasurement

    # audio
    real_time_factor: FloatMeasurement = Field(
        description="Compute seconds per second of audio. < 1.0 is faster than real time."
    )
    audio_seconds_per_compute_second: FloatMeasurement
    characters_per_second: FloatMeasurement

    # image / video
    images_per_minute: FloatMeasurement
    seconds_per_image: FloatMeasurement
    denoising_steps_per_second: FloatMeasurement
    frames_per_second: FloatMeasurement
    generated_frames_per_second: FloatMeasurement

    # detection / segmentation
    objects_per_second: FloatMeasurement
    postprocess_ms_per_object: FloatMeasurement
