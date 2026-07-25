// Shared TypeScript types mirroring the VisionEdge Lab FastAPI backend contract.
// These interfaces intentionally track the backend field names EXACTLY.

/* ------------------------------------------------------------------ */
/* Health                                                              */
/* ------------------------------------------------------------------ */

export interface HealthResponse {
  status: string;
  detection_health: string;
  warnings: string[];
}

/* ------------------------------------------------------------------ */
/* Capabilities                                                        */
/* ------------------------------------------------------------------ */

export interface GpuInfo {
  name: string;
  memory_total_mb: number;
  memory_used_mb: number;
  driver_version: string;
}

export interface Runtimes {
  onnxruntime: boolean;
  onnxruntime_providers: string[];
  onnxruntime_cuda: boolean;
  pytorch: boolean;
  pytorch_cuda: boolean;
  cuda_version: string | null;
  openvino: boolean;
  tensorrt: boolean;
}

export interface Capabilities {
  os: string;
  os_version: string;
  python_version: string;
  cpu_model: string;
  cpu_cores_physical: number;
  cpu_cores_logical: number;
  ram_total_mb: number;
  ram_available_mb: number;
  gpus: GpuInfo[];
  nvidia_gpu_present: boolean;
  runtimes: Runtimes;
  supported_precisions: string[];
}

/* ------------------------------------------------------------------ */
/* Models                                                              */
/* ------------------------------------------------------------------ */

export interface ModelEntry {
  model_id: string;
  display_name: string;
  family: string;
  size: string;
  architecture: string;
  version: string;
  format: string;
  precision: string;
  input_size: number | number[];
  supported_runtimes: string[];
  supported_devices: string[];
  labels: string;
  file_size_bytes: number;
  checksum_sha256: string;
  expected_memory_mb: number;
  deployment_status: string;
  speed_category: string;
  quality_category: string;
  license: string;
  notes: string;
}

export interface VlmModelEntry {
  model_id: string;
  display_name: string;
  family?: string;
  size?: string;
  version?: string;
  format?: string;
  precision?: string;
  supported_runtimes?: string[];
  supported_devices?: string[];
  deployment_status?: string;
  license?: string;
  notes?: string;
  [key: string]: unknown;
}

export interface ModelsResponse {
  detection_models: ModelEntry[];
  vlm_models: VlmModelEntry[];
}

/* ------------------------------------------------------------------ */
/* Classes                                                             */
/* ------------------------------------------------------------------ */

export interface ClassEntry {
  id: number;
  name: string;
}

export interface ClassGroups {
  people: string[];
  vehicles: string[];
  animals: string[];
  indoor: string[];
  [group: string]: string[];
}

export interface ClassesResponse {
  classes: ClassEntry[];
  groups: ClassGroups;
}

/* ------------------------------------------------------------------ */
/* Detection / inference                                               */
/* ------------------------------------------------------------------ */

// EXACT shape returned per detection by the backend.
export interface Detection {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  confidence: number;
  classId: number;
  className: string;
  inferenceBackend: string;
  modelName: string;
  modelVersion: string;
}

// Real per-stage split measured by predict_timed on the backend — these are four
// distinct measurements, not one value repeated.
export interface InferTimings {
  preprocess_ms: number;
  inference_ms: number;
  postprocess_ms: number;
  end_to_end_ms: number;
}

export interface InferResponse {
  detections: Detection[];
  timings: InferTimings;
  backend: string;
  count: number;
}

// WebSocket frame result.
// `dropped` is the server's CUMULATIVE drop count for the session, not a per-frame
// flag — the backend increments it whenever a frame is superseded before inference.
export interface WsDetectResult {
  frame_id: number;
  detections: Detection[];
  timings: { inference_ms: number };
  dropped: number;
  backend: string;
}

/* ------------------------------------------------------------------ */
/* Inference configuration / switching                                 */
/* ------------------------------------------------------------------ */

// Mirrors backend/app/core/types.py::ExecutionLocation. The backend validates this
// against its enum, so any other value is rejected with 422 before a switch runs.
export type ExecutionLocation = 'pc_local' | 'phone_local' | 'local_server' | 'remote_server';

export interface InferenceConfig {
  model_id: string;
  runtime: string;
  input_size: number;
  confidence: number;
  iou: number;
  execution_location: ExecutionLocation;
  allowed_class_ids: number[];
}

export interface SwitchResponse {
  ok: boolean;
  config: InferenceConfig;
  message: string;
  rolled_back: boolean;
}

/* ------------------------------------------------------------------ */
/* Metrics / status                                                    */
/* ------------------------------------------------------------------ */

export interface Metrics {
  processed_fps: number;
  inference_latency_p50_ms: number;
  inference_latency_p95_ms: number;
  inference_latency_p99_ms: number;
  end_to_end_p50_ms: number;
  processed_frames: number;
  dropped_frames: number;
  [key: string]: number;
}

export interface DetectionEvent {
  timestamp?: string;
  level?: string;
  message?: string;
  [key: string]: unknown;
}

export interface DetectionStatus {
  config: InferenceConfig;
  health: string;
  events: DetectionEvent[];
  metrics: Metrics;
}

export interface RuntimeStatus {
  detection: { config: InferenceConfig; health: string; events: DetectionEvent[] };
  vlm: Record<string, unknown>;
  runtimes: Runtimes;
  metrics: Metrics;
}

/* ------------------------------------------------------------------ */
/* Benchmarks                                                          */
/* ------------------------------------------------------------------ */

export interface BenchmarkResult {
  backend: string;
  model_id: string;
  input_size: number;
  precision: string;
  device: string;
  provider: string;
  runs: number;
  fps: number;
  latency_mean_ms: number;
  latency_p50_ms: number;
  latency_p95_ms: number;
  latency_p99_ms: number;
  memory_rss_mb: number;
  timestamp?: string;
}

export interface BenchmarksResponse {
  benchmarks: BenchmarkResult[];
}

// One configuration's aggregated benchmark history. Median, never best — see
// backend/app/storage/db.py::benchmark_groups.
export interface BenchmarkComparisonRow {
  model_id: string;
  backend: string;
  provider: string | null;
  device: string;
  input_size: number;
  precision: string;
  n: number;
  median_fps: number | null;
  median_p50_ms: number | null;
  latest_ts: number;
  latest_fps: number | null;
  any_concurrent_traffic: boolean;
}

export interface BenchmarkComparisonResponse {
  groups: BenchmarkComparisonRow[];
  note: string;
}

/* ------------------------------------------------------------------ */
/* Sessions                                                            */
/* ------------------------------------------------------------------ */

export interface SessionEntry {
  session_id?: string;
  started_at?: string;
  [key: string]: unknown;
}

export interface SessionsResponse {
  sessions: SessionEntry[];
}

/* ------------------------------------------------------------------ */
/* VLM                                                                 */
/* ------------------------------------------------------------------ */

export interface VLMResponse {
  text: string;
  structured_output: Record<string, unknown> | null;
  model_id: string;
  runtime: string;
  execution_location: string;
  prompt_tokens: number;
  generated_tokens: number;
  time_to_first_token_ms: number;
  generation_latency_ms: number;
  total_latency_ms: number;
  memory_usage_mb: number;
  warnings: string[];
}

export interface VlmModelsResponse {
  models: VlmModelEntry[];
}

// Every /api/vlm answer is wrapped in this envelope: the model's response plus the
// detector grounding it was given, the agreement check against the detector, and the
// standing disclaimer. The VLMResponse itself lives under `response`.
export interface VlmAnswer {
  response: VLMResponse;
  grounding: Record<string, unknown> | null;
  agreement: Record<string, unknown> | null;
  disclaimer: string;
}
