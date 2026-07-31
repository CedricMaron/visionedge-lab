# Benchmark methodology

How InferenceLab measures, what each choice costs, and where the numbers stop being
trustworthy. Written against the shipped code, not against an intention.

The governing rule: **a number is either measured, exactly derived from measured
values, or absent with a stated reason.** There is no fourth category.

---

## 1. Clocks

All durations come from `time.perf_counter()` — a monotonic, high-resolution clock.
`time.time()` is never used to compute a duration.

Why it matters: wall-clock time can be stepped backwards by NTP or a DST change. A
benchmark that straddled an adjustment would report a negative phase or one inflated
by an hour. Wall-clock appears in exactly one place, `RunIdentity.created_at`, purely
for ordering runs in a list.

Timer resolution on Linux is ~1 ns nominal, but the practical floor for a Python-level
span is ~100 ns of call overhead. Phases faster than roughly 1 µs are not meaningfully
resolvable and are reported as measured but should not be compared at that scale.

## 2. Device synchronization

**This is the single most common way a GPU benchmark lies.**

GPU work is queued asynchronously. A naive timer around a launch measures the time to
*enqueue* kernels, not to run them — often 100× smaller than the real execution time,
producing a spectacular and entirely fictional result.

`Timeline.span()` therefore takes a `synchronize` callable and invokes it *inside* the
timed region, immediately before stopping the clock, so the cost of waiting for the
device is charged to the phase that queued the work rather than leaking into the next
one. Every `PhaseSpan` records `device_synchronized`, and any run containing an
unsynchronized `model_execution` span carries a warning that its figure reflects
dispatch rather than completion.

For the ONNX Runtime path specifically: `session.run()` is blocking and returns
host-resident NumPy arrays, so there is no outstanding device work when it returns.
That is genuinely synchronized timing — but it *includes the device-to-host copy*, so
it is not pure kernel time. The runtime probe states this in its notes, and the note
travels with the measurement.

## 3. Warm-up, cold start and steady state

Three different things, kept separate:

| Quantity | Definition |
|---|---|
| **Cold start** | Model load + graph preparation + first inference. What a request costs after a deploy or a scale-from-zero. |
| **Warm-up** | Iterations run to reach steady state. Recorded in full, **never counted** in statistics. |
| **Steady state** | Measured iterations after warm-up. This is what "latency" means unqualified. |

Warm-up samples are retained rather than discarded, so a reader can see how long the
model took to stabilize; they are excluded from every statistic by
`IterationSample.counts_toward_statistics`. The `detection-cold-start` scenario sets
`warmup_iterations: 0` deliberately — warming up would destroy the thing it measures.

## 4. Statistics

`DurationStats.from_samples()` is the only constructor, and it cannot produce a mean
without also producing `n`, min, max, stddev, and p50/p90/p95/p99. A template that
renders an average therefore always has the sample count and spread available.

- **Percentiles** use linear interpolation between order statistics, matching
  `numpy.percentile`'s default method. Verified against NumPy in
  `test_schemas_measurement.py` across n = 1, even and odd counts.
- **Standard deviation** is the sample stddev and is `None` for n = 1. Reporting 0.0
  would imply perfect consistency from a single observation.
- **Coefficient of variation** (stddev / mean) is reported because it answers "is this
  mean a useful summary?" better than stddev alone.
- **Below ~10 measured samples**, percentiles above p50 are not meaningful and the run
  is flagged with an integrity warning.

## 5. Sampling hardware

A background thread samples CPU, memory and GPU state at a fixed interval. Three
properties matter:

**Cost is measured, not assumed.** Each tick times itself and the aggregate is
reported as `sampler_overhead_ms` alongside the series. A series without its sampling
cost is uninterpretable.

**Not all probes cost the same.** Measured on the reference RTX 2060:

| NVML call | Cost |
|---|---|
| current clock throttle reasons | **15.0 ms** |
| graphics + memory clocks | 2.7 ms |
| power usage | 1.4 ms |
| encoder + decoder utilization | 1.1 ms |
| utilization rates | 0.7 ms |
| per-process memory | 0.4 ms |
| memory info | 0.15 ms |
| temperature | 0.13 ms |

Sampling everything cost 28.9 ms per tick — 29% of a 100 ms interval, enough to
distort the workload being measured. Throttle reasons are therefore captured at run
boundaries only (they change on a thermal timescale, not a per-tick one), and codec
utilization is excluded outside video workloads. `SamplingDetail.lean()` and `.full()`
encode the two resulting profiles.

**Drift is corrected.** Each tick's deadline is computed from a fixed origin rather
than from the previous wake-up, so a 250 ms sampler stays at 250 ms instead of sliding.

Default intervals: standard 250 ms, detailed and profiler 100 ms. Standard was 1 s
until it was found that short runs collected only two power samples — below the
minimum needed to integrate a power series, so energy was never available in the
default mode.

### Measured perturbation

Sampling cost is not the same as workload perturbation. Measured by interleaving
conditions over 150 iterations each:

| Condition | p50 | vs. baseline |
|---|---|---|
| no sampler | 71.36 ms | — |
| standard (lean @ 250 ms) | 69.27 ms | within noise |
| detailed (full @ 100 ms) | 68.83 ms | within noise |

**A methodological warning from this measurement.** The first attempt ran the three
conditions sequentially and reported +58% perturbation for the *less* frequent
sampler — an impossible result. The cause was laptop CPU thermal drift across the
sequence, not the sampler. Interleaving the conditions removed it entirely. Any A/B
comparison on thermally-unstable hardware must interleave; sequential runs measure
drift as much as they measure the change.

## 6. Memory

Four quantities are routinely conflated and are kept strictly apart:

| Term | Meaning |
|---|---|
| **allocated** | Bytes the framework allocator holds for live tensors |
| **reserved** | Bytes the allocator took from the driver, including its own free pool (≥ allocated) |
| **process** | RSS of this OS process — weights, code, CUDA context, everything |
| **device total** | Physical VRAM, including what *other* processes use |

ONNX Runtime exposes no allocator statistics, so `gpu_allocated_mb` and
`gpu_reserved_mb` are reported **unavailable with that reason** on the ORT path rather
than being approximated from NVML — NVML's device-wide figure is a different quantity
and lives in its own field.

Snapshots are taken at `before_load`, `after_load` and `after_run`. Weight footprint is
the load delta (device memory where available, process RSS otherwise). The leak
indicator is RSS after the run minus RSS before load; non-zero is expected since the
model is still resident, and it is *growth across repeated identical runs* that is
worth investigating.

## 7. Energy

Energy is classified **derived**, never measured: no consumer GPU exposes a joule
counter.

```
E = trapezoidal integral of NVML instantaneous power over the sampled window
```

Trapezoidal rather than rectangular because power ramps between samples; assuming a
step function would systematically over- or under-count depending on which edge was
taken.

Bounds on accuracy:

- NVML's power reading is itself an average over roughly the last millisecond.
- Bursts between samples are invisible. Energy is reliable for runs lasting many
  multiples of the sample interval and unreliable for very short ones — below three
  power samples, `integrate_energy` refuses to produce a figure rather than returning
  a confident-looking wrong one.
- **Scope is GPU only.** CPU package power would need RAPL via
  `/sys/class/powercap/intel-rapl`, which is not readable under WSL2 on the reference
  machine, and there is no equivalent for system RAM or storage. Every energy value
  states this scope in its note.

## 8. Throughput

Always derived from measured durations and measured output counts, never timed
separately — so it cannot disagree with the latency figures beside it.

`total_measured_seconds` is the summed duration of *successful measured* iterations,
not wall time. Warm-up, cooldown and failed iterations must not inflate a rate.

Metrics that do not apply to a workload are unavailable with that as the reason
(`"not applicable to a text_embedding workload"`), never reported as zero.

## 9. Residual overhead

`residual = measured total − Σ(measured top-level phases)`.

It is reported as its own quantity and rendered as a distinct hatched row. It is
**never** folded into a neighbouring phase, and never labelled "network" — §18 of the
brief forbids exactly that, and it would be a guess dressed as a measurement.

Nested spans record a parent and are excluded from the top-level sum, so a
`preprocessing → resize` sub-span cannot double-count against its parent.

## 10. Network measurement

For remote inference, the client measures round-trip time on its own clock and the
server measures its total on its own clock. Their difference is a *duration* minus a
*duration*, so no clock synchronization is required and the result is exact up to
timer resolution.

What that difference contains — uplink, downlink, TLS framing, socket queueing —
cannot be separated without a clock-sync handshake. It is therefore presented as one
bucket, and the UI states why it is not split. A negative value (possible when both
figures are near timer resolution) clamps to zero and displays as "below measurement
resolution" rather than as a number.

## 11. Failure handling

A failed iteration is data, not a crash:

- recorded with its error type and message,
- excluded from statistics,
- counted in `failed_iterations`,
- and the exclusion is stated explicitly via `RunErrors.statistics_exclude_failures`.

A run is never silently shortened. Cancellation yields `RunStatus.CANCELLED` and a
timeout yields `TIMED_OUT`, each with a warning saying the statistics summarize only
what completed.

## 12. Integrity warnings

Attached automatically, surfaced above the numbers in the UI:

| Warning | Trigger |
|---|---|
| thin sample | fewer than 10 measured iterations |
| failed iterations | any measured iteration raised |
| unsynchronized timing | a `model_execution` span with `device_synchronized = false` |
| background load | non-benchmark CPU activity above 25% during the run |
| thermal throttling | NVML reported a thermal or power throttle |
| temperature rise | GPU rose more than 15 °C during the run |
| dirty working tree | uncommitted changes, so the run is not reproducible from its commit |
| synthetic input | measured against the adapter's deterministic input, not a dataset |
| non-standard mode | detailed or profiler mode, which is not comparable with standard |

## 13. Reproducibility and comparability

Every run records a stable `EnvironmentFingerprint` — a hash over CPU, RAM, GPU, CUDA,
OS, package versions, model id and revision, weights checksum, runtime, execution
provider, device, precision, quantization and benchmark mode. Timestamps, run ids and
thermal state are deliberately excluded: they vary between two otherwise identical
runs and belong to the result, not to the identity of the configuration.

- **Same fingerprint** → produced on equivalent configurations, may be pooled.
- **Different fingerprint** → may be compared, never averaged together.

`BenchmarkRun.is_comparable_to()` refuses a comparison outright when the runs differ in
task, scenario, instrumentation mode, batch size, concurrency, input size or sequence
length, and returns the specific differences so the UI can explain the refusal rather
than greying out a button.

## 14. Instrumentation modes

| Mode | Sampling | Intended use |
|---|---|---|
| **standard** | lean @ 250 ms | Comparable aggregate metrics. The default. |
| **detailed** | full @ 100 ms | Utilization and power time series. |
| **profiler** | full @ 100 ms + framework profilers | Kernel-level attribution. |

Results from different modes are **not comparable**, every run records its mode, and
non-standard runs carry a warning saying so.

## 15. Known limitations

- CPU package energy is not measurable on the reference hardware (no RAPL under WSL2).
- ORT exposes no allocator statistics, so framework-level GPU allocated/reserved
  memory is unavailable on the only implemented GPU path.
- Quality metrics (mAP, WER, retrieval recall) require reference datasets that are not
  wired up; they are reported unavailable rather than approximated.
- Concurrency above 1 is accepted by the scenario schema but the engine executes
  iterations serially; `concurrent_requests` is labelled as configuration, not as a
  measurement of achieved parallelism.
- The reference development machine has ~1 GB of free RAM, which bounds which models
  can be benchmarked locally at all.
