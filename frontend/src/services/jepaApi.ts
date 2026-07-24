// JEPA (Joint-Embedding Predictive Architecture) API — PLANNED, Phase 4.
// Structured client stub. These endpoints are not implemented in the current
// backend build; the interfaces below document the intended contract so the UI
// can be wired up when the slice lands. No fake data is returned.

export interface JepaTrainingRun {
  run_id: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  epoch: number;
  loss: number;
}

export interface JepaTrainConfig {
  dataset: string;
  epochs: number;
  batch_size: number;
  mask_ratio: number;
}

const NOT_IMPLEMENTED = 'JEPA training slice not implemented (Planned — Phase 4).';

export const jepaApi = {
  listRuns(): Promise<JepaTrainingRun[]> {
    return Promise.reject(new Error(NOT_IMPLEMENTED));
  },
  startRun(_config: JepaTrainConfig): Promise<JepaTrainingRun> {
    return Promise.reject(new Error(NOT_IMPLEMENTED));
  },
  stopRun(_runId: string): Promise<void> {
    return Promise.reject(new Error(NOT_IMPLEMENTED));
  },
};
