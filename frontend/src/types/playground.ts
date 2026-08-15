// Types mirroring backend/app/api/playground.py exactly.
//
// The playground trace is the single object the Playground renders as a result and
// the Pipeline page renders as a pipeline. A stage whose duration the backend did
// not measure carries `duration_ms: null` and a note — never a substituted number.

import type { Detection } from '@/types';

export type ExecutionTarget = 'local' | 'server';

export interface TensorInfo {
  name: string;
  role: string;
  shape: number[];
  dtype: string;
  layout: string | null;
  device: string;
  bytes: number;
  elements: number;
  min: number | null;
  max: number | null;
  mean: number | null;
  std: number | null;
}

export interface PipelineSubstep {
  name: string;
  detail: string | null;
  duration_ms: number | null;
  note: string | null;
}

export interface PipelineStage {
  id: string;
  name: string;
  duration_ms: number | null;
  detail: string | null;
  tensors: TensorInfo[];
  substeps: PipelineSubstep[];
  device: string | null;
  runtime: string | null;
  note: string | null;
}

export interface PlaygroundTimings {
  model_load_ms: number | null;
  decode_ms: number | null;
  preprocess_ms: number;
  inference_ms: number;
  postprocess_ms: number;
  server_total_ms: number;
}

export interface ClassificationResult {
  class_id: number;
  label: string;
  probability: number;
}

export interface EmbeddingResult {
  dimension: number | null;
  preview: number[];
  norm: number | null;
  tokens: number | null;
  token_preview: string[] | null;
}

export interface PlaygroundTrace {
  request_id: string;
  execution: ExecutionTarget;
  task: string;
  modality: string;
  model: {
    model_id: string;
    display_name: string;
    family: string;
    parameters_millions: number | null;
    input_format: string;
    output_format: string;
  };
  runtime: {
    runtime_id: string;
    runtime_version: string | null;
    execution_provider: string | null;
    device: string;
    precision: string;
    input_size: number | null;
  };
  timings: PlaygroundTimings;
  memory: {
    process_rss_mb: number | null;
    input_tensor_bytes: number;
    output_tensor_bytes: number;
  };
  stages: PipelineStage[];
  result: {
    detections?: Detection[];
    count?: number;
    classifications?: ClassificationResult[];
    embedding?: EmbeddingResult;
    extra?: Record<string, string>;
  };
  /** Measured by the client: request start → response parsed. Server runs never
   *  report this themselves, and it is the only place network time can be seen. */
  client_round_trip_ms?: number;
}
