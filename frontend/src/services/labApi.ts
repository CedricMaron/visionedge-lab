// InferenceLab benchmarking API client.
// Mirrors backend/app/api/lab.py. Every metric arrives already wrapped in a
// Measurement, so this layer never invents, defaults or rounds a value.

import { http } from './http';
import type {
  BenchmarkRun,
  CapabilityCell,
  CompareResult,
  IterationSample,
  LabModel,
  LabScenario,
  Overview,
  RunSummary,
  RuntimeCapability,
  UtilizationSample,
} from '@/types/lab';

export interface CreateRunBody {
  scenario_id: string;
  model_id: string;
  runtime_id?: string;
  device?: string;
  precision?: string;
  mode?: string;
  measured_iterations?: number;
  warmup_iterations?: number;
  seed?: number;
  label?: string;
  enable_sampler?: boolean;
}

export const labApi = {
  overview: (signal?: AbortSignal) => http.get<Overview>('/api/lab/overview', undefined, signal),

  runtimes: (signal?: AbortSignal) =>
    http.get<{ runtimes: RuntimeCapability[] }>('/api/lab/runtimes', undefined, signal),

  capabilityMatrix: (signal?: AbortSignal) =>
    http.get<{ cells: CapabilityCell[] }>('/api/lab/capability-matrix', undefined, signal),

  models: (signal?: AbortSignal) =>
    http.get<{ models: LabModel[] }>('/api/lab/models', undefined, signal),

  scenarios: (signal?: AbortSignal) =>
    http.get<{ scenarios: LabScenario[] }>('/api/lab/scenarios', undefined, signal),

  system: (signal?: AbortSignal) =>
    http.get<{
      hardware: BenchmarkRun['hardware'];
      software: BenchmarkRun['software'];
      runtimes: RuntimeCapability[];
    }>('/api/lab/system', undefined, signal),

  runs: (params?: { limit?: number; task?: string; model_id?: string }, signal?: AbortSignal) =>
    http.get<{ runs: RunSummary[] }>('/api/lab/runs', params, signal),

  run: (runId: string, signal?: AbortSignal) =>
    http.get<BenchmarkRun>(`/api/lab/runs/${runId}`, undefined, signal),

  iterations: (runId: string, signal?: AbortSignal) =>
    http.get<{ iterations: IterationSample[] }>(
      `/api/lab/runs/${runId}/iterations`,
      undefined,
      signal,
    ),

  utilization: (runId: string, signal?: AbortSignal) =>
    http.get<{ samples: UtilizationSample[] }>(
      `/api/lab/runs/${runId}/utilization`,
      undefined,
      signal,
    ),

  createRun: (body: CreateRunBody) =>
    http.postJson<{ run_token: string; run: BenchmarkRun }>('/api/lab/runs', body),

  cancelRun: (token: string) =>
    http.postJson<{ cancelled: boolean }>(`/api/lab/runs/${token}/cancel`, {}),

  compare: (runIds: string[]) =>
    http.postJson<CompareResult>('/api/lab/compare', { run_ids: runIds }),

  /** Export URL, used as an href so the browser handles the download. */
  exportUrl: (runId: string, format: 'json' | 'csv' | 'markdown') =>
    `/api/lab/runs/${runId}/export?format=${format}`,
};
