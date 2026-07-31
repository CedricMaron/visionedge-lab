// Types mirroring the InferenceLab backend contract exactly.
// See backend/app/schemas/ — these track those Pydantic models field for field.

/**
 * Every metric carries its own provenance. `value === null` means unavailable, and
 * `unavailable_reason` is then always populated (the backend refuses to construct a
 * valueless Measurement without one), so the UI can explain absence instead of
 * rendering a dash or, worse, a zero.
 */
export interface Measurement<T = number> {
  value: T | null;
  unit: string;
  kind: 'measured' | 'derived' | 'estimated';
  source: string;
  unavailable_reason: string | null;
  note: string | null;
}

export interface DurationStats {
  n: number;
  min_ms: number | null;
  max_ms: number | null;
  mean_ms: number | null;
  median_ms: number | null;
  stddev_ms: number | null;
  p50_ms: number | null;
  p90_ms: number | null;
  p95_ms: number | null;
  p99_ms: number | null;
  coefficient_of_variation: number | null;
}

export interface PhaseSpan {
  phase: string;
  duration_ms: number;
  parent: string | null;
  label: string | null;
  device_synchronized: boolean;
  note: string | null;
}

export interface PhaseBreakdown {
  phases: Record<string, DurationStats>;
  total: DurationStats;
  /** Measured total minus the sum of measured phases. Never charged to a real phase. */
  residual_ms: number | null;
}

export interface IterationSample {
  index: number;
  group: 'warmup' | 'measured' | 'cold_start';
  total_ms: number | null;
  spans: PhaseSpan[];
  succeeded: boolean;
  error_type: string | null;
  error_message: string | null;
}

export interface ColdWarmSplit {
  model_load_ms: number | null;
  graph_compilation_ms: number | null;
  kernel_warmup_ms: number | null;
  first_inference_ms: number | null;
  cold_start_total_ms: number | null;
  warm_inference: DurationStats;
}

export interface GpuDescriptor {
  index: number;
  name: string;
  memory_total_mb: number | null;
  driver_version: string | null;
  compute_capability: string | null;
  power_limit_w: number | null;
}

export interface HardwareInfo {
  cpu_model: string;
  cpu_cores_physical: number | null;
  cpu_cores_logical: number;
  cpu_instruction_sets: string[];
  cpu_max_freq_mhz: number | null;
  ram_total_mb: number;
  gpus: GpuDescriptor[];
  gpu_count: number;
  cuda_version: string | null;
  cudnn_version: string | null;
  nvml_available: boolean;
}

export interface SoftwareEnvironment {
  os: string;
  os_version: string;
  kernel_version: string | null;
  python_version: string;
  package_versions: Record<string, string>;
  relevant_env_vars: Record<string, string>;
}

export interface UtilizationSample {
  t_offset_ms: number;
  cpu_percent: number | null;
  process_cpu_percent: number | null;
  ram_used_mb: number | null;
  gpu_percent: number | null;
  gpu_memory_used_mb: number | null;
  gpu_temperature_c: number | null;
  gpu_power_w: number | null;
  gpu_clock_mhz: number | null;
}

export interface UtilizationSeries {
  samples: UtilizationSample[];
  sample_interval_ms: number;
  sampler_overhead_ms: Measurement;
  sources: string[];
  unavailable: Record<string, string>;
}

export type MeasurementMap = Record<string, Measurement>;

export interface BenchmarkRun {
  schema_version: number;
  identity: { run_id: string; created_at: number; label: string | null; tags: string[] };
  status: 'pending' | 'running' | 'completed' | 'partial' | 'failed' | 'cancelled' | 'timed_out';
  scenario: {
    id: string;
    task: string;
    description: string;
    warmup_iterations: number;
    measured_iterations: number;
    batch_size: number;
    concurrency: number;
    input_size: number | null;
    sequence_length: number | null;
    mode: string;
    random_seed: number | null;
  };
  model: {
    model_id: string;
    display_name: string;
    revision: string | null;
    parameters_millions: number | null;
    file_size_bytes: number | null;
  };
  runtime: {
    runtime_id: string;
    runtime_version: string | null;
    execution_provider: string | null;
    device: string;
    precision: string;
    thread_config: Record<string, number>;
  };
  task: string;
  mode: string;
  execution_location: string;
  hardware: HardwareInfo;
  software: SoftwareEnvironment;
  fingerprint: { digest: string; components: Record<string, string> };
  thermal_and_load: {
    gpu_temperature_start_c: number | null;
    gpu_temperature_end_c: number | null;
    thermal_throttling_detected: boolean | null;
    concurrent_workload_detected: boolean;
    concurrent_workload_detail: string | null;
  };
  reproducibility: {
    git_commit: string | null;
    git_dirty: boolean | null;
    random_seed: number | null;
    deterministic_mode: boolean;
    reproduction_command: string | null;
  };
  timings: PhaseBreakdown;
  cold_warm: ColdWarmSplit;
  throughput: MeasurementMap;
  memory: MeasurementMap & { snapshots: unknown[] };
  utilization: UtilizationSeries;
  energy: MeasurementMap;
  iterations: IterationSample[];
  errors: {
    failures: { index: number; error_type: string; error_message: string }[];
    statistics_exclude_failures: boolean;
  };
  warnings: string[];
  instrumentation_overhead_ms: number | null;
  duration_s: number | null;
  successful_iterations: number;
  failed_iterations: number;
  warmup_iterations_run: number;
}

export interface RunSummary {
  run_id: string;
  created_at: number;
  status: string;
  task: string;
  scenario_id: string;
  model_id: string;
  runtime_id: string;
  device: string;
  precision: string;
  mode: string;
  fingerprint: string;
  batch_size: number;
  measured_iterations: number;
  failed_iterations: number;
  latency_p50_ms: number | null;
  latency_p95_ms: number | null;
  throughput_per_s: number | null;
  peak_rss_mb: number | null;
}

export interface RuntimeCapability {
  runtime_id: string;
  available: boolean;
  unavailable_reason: string | null;
  version: string | null;
  execution_providers: string[];
  devices: string[];
  precisions_by_device: Record<string, string[]>;
  supports_device_synchronization: boolean;
  supports_profiling: boolean;
  notes: string[];
}

export interface CapabilityCell {
  runtime_id: string;
  device: string;
  precision: string;
  supported: boolean;
  reason: string | null;
}

export interface LabModel {
  model_id: string;
  display_name: string;
  family: string;
  task: string;
  modality: string;
  adapter: string;
  source_repository: string | null;
  paper_url: string | null;
  model_license: string;
  weights_license: string;
  commercial_use_permitted: boolean | null;
  parameters_millions: number | null;
  file_size_bytes: number | null;
  local_path: string;
  companion_files: { file_name: string; purpose: string }[];
  input_size: number | null;
  supported_runtimes: string[];
  supported_devices: string[];
  supported_precisions: string[];
  deployment_status: 'installed' | 'not_installed' | 'incomplete' | 'missing';
  not_installed_reason: string | null;
  install_hint: string | null;
  notes: string;
}

export interface LabScenario {
  id: string;
  task: string;
  description: string;
  warmup_iterations: number;
  measured_iterations: number;
  batch_size: number;
  concurrency: number;
  input_size: number | null;
  sequence_length: number | null;
  mode: string;
  has_sufficient_samples: boolean;
}

export interface Overview {
  recent_runs: RunSummary[];
  leaders_by_task: Record<
    string,
    { fastest: RunSummary | null; most_memory_efficient: RunSummary | null; run_count: number }
  >;
  total_runs: number;
  recent_failures: RunSummary[];
  runtimes_available: string[];
  runtimes_unavailable: number;
}

export interface CompareResult {
  baseline_run_id: string;
  all_comparable: boolean;
  comparisons: { run_id: string; comparable: boolean; blocking_differences: string[] }[];
  warning: string | null;
  runs: BenchmarkRun[];
}
