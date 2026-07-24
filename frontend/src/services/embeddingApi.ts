// Embedding explorer API — PLANNED, Phase 4.
// Structured client stub documenting the intended contract. Not implemented in
// the current backend build; returns rejected promises rather than fake vectors.

export interface EmbeddingPoint {
  id: string;
  label: string;
  x: number;
  y: number;
}

export interface EmbeddingQuery {
  text?: string;
  image_id?: string;
  top_k: number;
}

const NOT_IMPLEMENTED = 'Embedding explorer slice not implemented (Planned — Phase 4).';

export const embeddingApi = {
  projection(): Promise<EmbeddingPoint[]> {
    return Promise.reject(new Error(NOT_IMPLEMENTED));
  },
  nearest(_query: EmbeddingQuery): Promise<EmbeddingPoint[]> {
    return Promise.reject(new Error(NOT_IMPLEMENTED));
  },
};
