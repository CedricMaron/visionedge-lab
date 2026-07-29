"""Throughput derivation, per task.

Throughput is always derived from measured durations and measured output counts —
never timed independently — so it cannot disagree with the latency figures beside
it. Metrics that do not apply to the workload are unavailable with that as the
stated reason, rather than being reported as zero.
"""
from __future__ import annotations

from app.schemas.enums import Task
from app.schemas.measurement import FloatMeasurement, IntMeasurement, Measurement
from app.schemas.resources import ThroughputMetrics

_SOURCE = "derived from measured iteration durations"


def _na(reason: str, unit: str) -> FloatMeasurement:
    return Measurement[float].unavailable(reason, unit=unit)


def _na_int(reason: str) -> IntMeasurement:
    return Measurement[int].unavailable(reason, unit="")


def _rate(count: float, seconds: float, unit: str, note: str | None = None) -> FloatMeasurement:
    if seconds <= 0:
        return _na("measured duration was zero, so a rate cannot be derived", unit)
    return Measurement[float].derived(count / seconds, unit=unit, source=_SOURCE, note=note)


def build_throughput(
    task: Task,
    *,
    successful_iterations: int,
    total_measured_seconds: float,
    batch_size: int,
    mean_latency_ms: float | None,
    concurrency: int = 1,
    detected_objects: int | None = None,
    postprocess_total_ms: float | None = None,
    prompt_tokens: int | None = None,
    output_tokens: int | None = None,
    prefill_seconds: float | None = None,
    decode_seconds: float | None = None,
    audio_seconds_produced: float | None = None,
    audio_seconds_consumed: float | None = None,
    characters: int | None = None,
    images_produced: int | None = None,
    denoising_steps: int | None = None,
    frames_processed: int | None = None,
    frames_generated: int | None = None,
) -> ThroughputMetrics:
    """Assemble the throughput block for one run.

    ``total_measured_seconds`` is the summed duration of successful measured
    iterations, not wall time — warm-up, cooldown and failed iterations must not
    inflate a rate.
    """
    samples = successful_iterations * batch_size
    not_this_task = f"not applicable to a {task.value} workload"

    generative_text = task in (Task.TEXT_GENERATION, Task.MULTIMODAL_GENERATION)
    audio_out = task is Task.TEXT_TO_SPEECH
    audio_in = task in (Task.SPEECH_TO_TEXT, Task.AUDIO_CLASSIFICATION)
    image_gen = task is Task.IMAGE_GENERATION
    video = task in (Task.VIDEO_UNDERSTANDING, Task.VIDEO_GENERATION)
    boxes_or_masks = task in (Task.OBJECT_DETECTION, Task.IMAGE_SEGMENTATION)
    frame_like = task in (
        Task.OBJECT_DETECTION, Task.IMAGE_SEGMENTATION, Task.IMAGE_CLASSIFICATION,
    )

    return ThroughputMetrics(
        requests_per_second=_rate(successful_iterations, total_measured_seconds, "req/s"),
        samples_per_second=_rate(samples, total_measured_seconds, "samples/s"),
        batches_per_second=_rate(successful_iterations, total_measured_seconds, "batches/s"),

        prompt_tokens_per_second=(
            _rate(prompt_tokens, total_measured_seconds, "tok/s")
            if generative_text and prompt_tokens
            else _na(not_this_task if not generative_text else "no prompt tokens recorded", "tok/s")
        ),
        output_tokens_per_second=(
            _rate(output_tokens, total_measured_seconds, "tok/s")
            if generative_text and output_tokens
            else _na(not_this_task if not generative_text else "no output tokens recorded", "tok/s")
        ),
        total_tokens_per_second=(
            _rate((prompt_tokens or 0) + (output_tokens or 0), total_measured_seconds, "tok/s")
            if generative_text and (prompt_tokens or output_tokens)
            else _na(not_this_task if not generative_text else "no tokens recorded", "tok/s")
        ),
        prefill_tokens_per_second=(
            _rate(prompt_tokens, prefill_seconds, "tok/s")
            if generative_text and prompt_tokens and prefill_seconds
            else _na(
                not_this_task if not generative_text
                else "prefill duration was not measured separately", "tok/s"
            )
        ),
        decode_tokens_per_second=(
            _rate(output_tokens, decode_seconds, "tok/s")
            if generative_text and output_tokens and decode_seconds
            else _na(
                not_this_task if not generative_text
                else "decode duration was not measured separately", "tok/s"
            )
        ),
        concurrent_requests=Measurement[int].of(
            concurrency, "requests", "scenario configuration",
            note="configured concurrency, not a measurement of achieved parallelism",
        ),

        real_time_factor=(
            Measurement[float].derived(
                total_measured_seconds / (audio_seconds_produced or audio_seconds_consumed),
                unit="compute-s/audio-s", source=_SOURCE,
                note="below 1.0 is faster than real time",
            )
            if (audio_out or audio_in) and (audio_seconds_produced or audio_seconds_consumed)
            else _na(
                not_this_task if not (audio_out or audio_in) else "no audio duration recorded",
                "compute-s/audio-s",
            )
        ),
        audio_seconds_per_compute_second=(
            _rate(audio_seconds_produced or audio_seconds_consumed, total_measured_seconds, "audio-s/s")
            if (audio_out or audio_in) and (audio_seconds_produced or audio_seconds_consumed)
            else _na(not_this_task, "audio-s/s")
        ),
        characters_per_second=(
            _rate(characters, total_measured_seconds, "chars/s")
            if audio_out and characters
            else _na(not_this_task if not audio_out else "no character count recorded", "chars/s")
        ),

        images_per_minute=(
            _rate((images_produced or successful_iterations) * 60.0, total_measured_seconds, "img/min")
            if image_gen
            else _na(not_this_task, "img/min")
        ),
        seconds_per_image=(
            Measurement[float].derived(
                total_measured_seconds / (images_produced or successful_iterations),
                unit="s/img", source=_SOURCE,
            )
            if image_gen and (images_produced or successful_iterations)
            else _na(not_this_task, "s/img")
        ),
        denoising_steps_per_second=(
            _rate(denoising_steps, total_measured_seconds, "steps/s")
            if image_gen and denoising_steps
            else _na(
                not_this_task if not image_gen else "denoising step count was not recorded",
                "steps/s",
            )
        ),

        frames_per_second=(
            _rate(frames_processed or samples, total_measured_seconds, "fps")
            if (video or frame_like)
            else _na(not_this_task, "fps")
        ),
        generated_frames_per_second=(
            _rate(frames_generated, total_measured_seconds, "fps")
            if task is Task.VIDEO_GENERATION and frames_generated
            else _na(not_this_task, "fps")
        ),

        objects_per_second=(
            _rate(detected_objects, total_measured_seconds, "obj/s")
            if boxes_or_masks and detected_objects is not None
            else _na(
                not_this_task if not boxes_or_masks else "no objects were detected", "obj/s"
            )
        ),
        postprocess_ms_per_object=(
            Measurement[float].derived(
                postprocess_total_ms / detected_objects, unit="ms/obj", source=_SOURCE,
                note="postprocessing cost amortized over detected objects; dominated by NMS",
            )
            if boxes_or_masks and detected_objects and postprocess_total_ms is not None
            else _na(
                not_this_task if not boxes_or_masks
                else "no objects detected, so per-object cost is undefined",
                "ms/obj",
            )
        ),
    )
