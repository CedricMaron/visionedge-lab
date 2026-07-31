# Metrics reference

Every metric InferenceLab reports, with its unit, source, formula, applicability and
limits.

**Kind** is the load-bearing column:

| Kind | Meaning |
|---|---|
| `measured` | Read directly from an instrumentation source |
| `derived` | Computed exactly from measured values |
| `estimated` | Modelled under stated assumptions — requires a documented methodology or the schema rejects it |

Anything the environment cannot provide is `unavailable` with a reason attached. It is
never a zero, a dash, or a blank.

---

## Latency

All durations use `time.perf_counter()`. See
[BENCHMARK_METHODOLOGY.md §1](BENCHMARK_METHODOLOGY.md).

| Metric | Unit | Kind | Source | Notes |
|---|---|---|---|---|
| `input_loading` | ms | measured | `Timeline.span` | Reading the input from disk or request body |
| `input_decoding` | ms | measured | `Timeline.span` | JPEG/PNG/audio decode |
| `preprocessing` | ms | measured | adapter span | Resize, crop, normalize, tensor conversion |
| `tokenization` | ms | measured | adapter span | Text pipelines report this instead of `preprocessing` |
| `host_to_device` | ms | measured | runtime span | Not separable on the ORT path |
| `queue_wait` | ms | measured | server span | Time a frame waited under backpressure |
| `model_execution` | ms | measured | `Timeline.span` + device sync | **Includes device→host copy on the ORT GPU path** |
| `device_synchronization` | ms | measured | runtime span | Zero on synchronous backends |
| `postprocessing` | ms | measured | adapter span | Decode, NMS, pooling, normalization |
| `output_serialization` | ms | measured | `Timeline.span` | Result → JSON |
| `residual_overhead` | ms | derived | `total − Σ phases` | Unattributed. Never labelled as a real phase |
| `end_to_end` | ms | measured | `Timeline.total_ms` | Whole measured region |

### Aggregates

Computed by `DurationStats.from_samples()` over successful **measured** iterations only.

| Metric | Unit | Kind | Formula | Notes |
|---|---|---|---|---|
| `n` | count | measured | — | Always reported alongside any average |
| `min_ms` / `max_ms` | ms | measured | order statistics | |
| `mean_ms` | ms | derived | `statistics.fmean` | Never shown without `n` and percentiles |
| `median_ms` | ms | derived | `statistics.median` | |
| `stddev_ms` | ms | derived | sample stddev | `None` for n = 1 — one sample has no spread |
| `p50/p90/p95/p99_ms` | ms | derived | linear interpolation | Matches `numpy.percentile` default |
| `coefficient_of_variation` | — | derived | `stddev / mean` | High values mean the mean is a poor summary |

### Streaming (generative workloads)

| Metric | Unit | Kind | Applicability |
|---|---|---|---|
| `time_to_first_token_ms` | ms | measured | text generation |
| `inter_token_latency_ms` | ms | measured | text generation |
| `time_to_first_audio_chunk_ms` | ms | measured | TTS |

Unavailable for non-generative tasks, with `"not applicable to a {task} workload"`.

---

## Cold start vs. steady state

| Metric | Unit | Kind | Notes |
|---|---|---|---|
| `model_load_ms` | ms | measured | Session creation including weight read |
| `graph_compilation_ms` | ms | measured | Unavailable on ORT (no separate compile step exposed) |
| `engine_build_ms` | ms | measured | TensorRT only; not implemented |
| `kernel_warmup_ms` | ms | measured | First warm-up iteration |
| `first_inference_ms` | ms | measured | |
| `cold_start_total_ms` | ms | derived | `model_load + first_inference` |
| `warm_inference` | stats | derived | Steady-state `DurationStats` |

---

## Throughput

Derived from measured durations and measured output counts. `total_measured_seconds`
excludes warm-up, cooldown and failed iterations.

| Metric | Unit | Formula | Applicability |
|---|---|---|---|
| `requests_per_second` | req/s | `successful / seconds` | all |
| `samples_per_second` | samples/s | `successful × batch / seconds` | all |
| `batches_per_second` | batches/s | `successful / seconds` | all |
| `frames_per_second` | fps | `frames / seconds` | detection, segmentation, classification, video |
| `objects_per_second` | obj/s | `detected / seconds` | detection, segmentation |
| `postprocess_ms_per_object` | ms/obj | `postprocess_total / detected` | detection, segmentation — dominated by NMS |
| `prompt/output/total_tokens_per_second` | tok/s | `tokens / seconds` | text generation |
| `prefill_tokens_per_second` | tok/s | `prompt_tokens / prefill_seconds` | text generation |
| `decode_tokens_per_second` | tok/s | `output_tokens / decode_seconds` | text generation |
| `real_time_factor` | compute-s/audio-s | `compute / audio_duration` | STT, TTS — below 1.0 is faster than real time |
| `characters_per_second` | chars/s | `characters / seconds` | TTS |
| `images_per_minute` | img/min | `images × 60 / seconds` | image generation |
| `seconds_per_image` | s/img | `seconds / images` | image generation |
| `denoising_steps_per_second` | steps/s | `steps / seconds` | image generation |
| `concurrent_requests` | count | configuration | **Configured, not achieved** — the engine runs serially |

---

## Memory

Four distinct quantities. See [BENCHMARK_METHODOLOGY.md §6](BENCHMARK_METHODOLOGY.md).

| Metric | Unit | Kind | Source | Notes |
|---|---|---|---|---|
| `process_rss_mb` | MB | measured | `psutil.Process.memory_info().rss` | Whole process |
| `process_vms_mb` | MB | measured | `psutil` | Virtual size |
| `system_used_mb` / `system_available_mb` | MB | measured | `psutil.virtual_memory` | System-wide |
| `gpu_allocated_mb` | MB | measured | framework allocator | **Unavailable on ORT** — no allocator stats exposed |
| `gpu_reserved_mb` | MB | measured | framework allocator | **Unavailable on ORT** |
| `gpu_process_used_mb` | MB | measured | `nvmlDeviceGetComputeRunningProcesses` | This process's VRAM |
| `gpu_device_used_mb` | MB | measured | `nvmlDeviceGetMemoryInfo` | **All** processes on the card |
| `gpu_device_total_mb` | MB | measured | NVML | Physical VRAM |
| `peak_process_rss_mb` | MB | measured | sampled max | |
| `model_weights_mb` | MB | derived | `after_load − before_load` | Includes CUDA context on first GPU load |
| `kv_cache_mb` | MB | measured | runtime allocator | Generative only |
| `leak_indicator_mb` | MB | derived | `after_run − before_load` | Non-zero is normal; *growth across runs* is the signal |

---

## Hardware utilization

Time series. Interval and sampler cost are recorded with the series.

| Metric | Unit | Source | Notes |
|---|---|---|---|
| `cpu_percent` | % | `psutil.cpu_percent` | System-wide; probe primed at construction |
| `process_cpu_percent` | % | `psutil.Process.cpu_percent` | May exceed 100% on multi-core |
| `cpu_per_core_percent` | % | `psutil` | Detailed mode only |
| `cpu_freq_mhz` | MHz | `psutil.cpu_freq` | Detailed mode only; unavailable in many containers |
| `thread_count`, `context_switches` | count | `psutil` | |
| `ram_used_mb`, `swap_used_mb` | MB | `psutil` | |
| `gpu_percent` | % | `nvmlDeviceGetUtilizationRates` | Fraction of time any kernel was resident — **not** occupancy |
| `gpu_memory_percent` | % | NVML | Memory-controller activity, not fill level |
| `gpu_memory_used_mb` | MB | NVML | Device-wide |
| `gpu_clock_mhz`, `gpu_memory_clock_mhz` | MHz | NVML | Detailed mode only (2.7 ms/call) |
| `gpu_temperature_c` | °C | NVML | |
| `gpu_power_w` | W | `nvmlDeviceGetPowerUsage` | ~1 ms averaging window |
| `gpu_encoder/decoder_percent` | % | NVML | Video workloads only (opt-in) |
| `disk_read/write_mb_s`, `net_sent/recv_mb_s` | MB/s | `psutil` counters | Rates derived from cumulative counters |
| `sampler_overhead_ms` | ms | self-timing | Mean cost per tick — required context for the series |

---

## Energy

All **derived** by integration; GPU-only. See
[BENCHMARK_METHODOLOGY.md §7](BENCHMARK_METHODOLOGY.md).

| Metric | Unit | Formula |
|---|---|---|
| `total_energy_j` | J | trapezoidal ∫ power dt |
| `average_power_w` | W | `total_energy / window_seconds` |
| `peak_power_w` | W | max sample (measured, not derived) |
| `power_limit_w` | W | `nvmlDeviceGetPowerManagementLimit` (measured) |
| `energy_per_request_j` | J | `total_energy / successful_requests` |
| `joules_per_token/image/audio_second/video_frame` | J/unit | `total_energy / units` |
| `tokens/frames/requests_per_joule` | unit/J | `units / total_energy` |

Requires ≥ 3 power samples; below that, unavailable with the sample count in the reason.

---

## Quality

Separated from performance so a model that is fast and wrong is visibly both.
Objective metrics require a reference dataset; without one they are unavailable.

| Task | Metrics | Status |
|---|---|---|
| Detection | mAP@50-95, AP50, AP75, precision, recall, per-class AP | **Not implemented** — needs COCO protocol |
| Classification | top-1, top-5 | **Implemented** — computed when references are supplied |
| Classification | macro F1, confusion matrix | Not implemented — needs full-dataset aggregation |
| Segmentation | IoU, mIoU, Dice, boundary F1 | Not implemented |
| Text generation | exact match, perplexity, output validity, schema compliance | Not implemented |
| Embedding | dimensionality, bytes per vector | **Implemented** — properties of the output |
| Embedding | recall@k, MRR, NDCG | Not implemented — needs a labelled corpus |
| STT | WER, CER | Not implemented |
| TTS | duration, sample rate, clipping ratio, silence ratio | Not implemented |
| Image generation | dimensions, steps, CLIP similarity, seed reproducibility | Not implemented |

`SubjectiveEvaluation` (MOS, rater count) is **never** auto-populated. An unrated
result reports zero raters, not a score.

---

## Run integrity

| Field | Meaning |
|---|---|
| `successful_iterations` | Measured iterations that completed |
| `failed_iterations` | Iterations that raised; excluded from statistics |
| `warmup_iterations_run` | Retained but never counted |
| `statistics_exclude_failures` | Always true; recorded so a reader need not guess |
| `warnings[]` | Integrity conditions (see methodology §12) |
| `instrumentation_overhead_ms` | Measured cost the platform itself added |
| `fingerprint.digest` | Environment identity for grouping |
| `git_dirty` | Working tree had uncommitted changes → not reproducible from the commit |
