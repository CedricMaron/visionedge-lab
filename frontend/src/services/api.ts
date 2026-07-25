// Real API service for the detection / capabilities / models slice.

import { http } from './http';
import type {
  BenchmarkComparisonResponse,
  BenchmarkResult,
  BenchmarksResponse,
  Capabilities,
  ClassesResponse,
  DetectionStatus,
  HealthResponse,
  InferResponse,
  InferenceConfig,
  JobsResponse,
  ModelsResponse,
  RuntimeStatus,
  SessionsResponse,
  SwitchResponse,
} from '@/types';

export const api = {
  health: (signal?: AbortSignal) => http.get<HealthResponse>('/health', undefined, signal),

  capabilities: (signal?: AbortSignal) =>
    http.get<Capabilities>('/api/capabilities', undefined, signal),

  models: (signal?: AbortSignal) => http.get<ModelsResponse>('/api/models', undefined, signal),

  modelRegistry: (signal?: AbortSignal) =>
    http.get<Record<string, unknown>>('/api/model-registry', undefined, signal),

  classes: (signal?: AbortSignal) => http.get<ClassesResponse>('/api/classes', undefined, signal),

  runtimeStatus: (signal?: AbortSignal) =>
    http.get<RuntimeStatus>('/api/runtime-status', undefined, signal),

  detectionStatus: (signal?: AbortSignal) =>
    http.get<DetectionStatus>('/api/detection/status', undefined, signal),

  sessions: (signal?: AbortSignal) =>
    http.get<SessionsResponse>('/api/sessions', undefined, signal),

  benchmarks: (signal?: AbortSignal) =>
    http.get<BenchmarksResponse>('/api/benchmarks', undefined, signal),

  benchmarkComparison: (signal?: AbortSignal) =>
    http.get<BenchmarkComparisonResponse>('/api/benchmarks/comparison', undefined, signal),

  jobs: (signal?: AbortSignal) => http.get<JobsResponse>('/api/jobs', undefined, signal),

  switchDetection: (config: InferenceConfig) =>
    http.postJson<SwitchResponse>('/api/detection/switch', config),

  runBenchmark: (runs: number) =>
    http.postJson<BenchmarkResult>('/api/detection/benchmark', {}, { runs }),

  // Single-shot inference over a JPEG blob.
  infer: (
    file: Blob,
    params: { confidence?: number; iou?: number; classes?: string },
  ): Promise<InferResponse> => {
    const form = new FormData();
    form.append('file', file, 'frame.jpg');
    return http.postForm<InferResponse>('/api/infer', form, {
      confidence: params.confidence,
      iou: params.iou,
      classes: params.classes,
    });
  },
};
