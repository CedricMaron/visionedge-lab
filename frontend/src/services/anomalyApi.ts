// Anomaly detection API — PLANNED, Phase 5.
// Structured client stub documenting the intended contract. Not implemented in
// the current backend build; returns rejected promises rather than fake scores.

export interface AnomalyEvent {
  event_id: string;
  timestamp: string;
  score: number;
  description: string;
}

const NOT_IMPLEMENTED = 'Anomaly detection slice not implemented (Planned — Phase 5).';

export const anomalyApi = {
  listEvents(): Promise<AnomalyEvent[]> {
    return Promise.reject(new Error(NOT_IMPLEMENTED));
  },
  score(_frame: Blob): Promise<AnomalyEvent> {
    return Promise.reject(new Error(NOT_IMPLEMENTED));
  },
};
