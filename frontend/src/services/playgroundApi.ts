// Playground inference client — one call per run, whatever the modality.

import { http } from './http';
import type { PlaygroundTrace } from '@/types/playground';

export interface PlaygroundRunParams {
  model_id: string;
  runtime_id: string;
  device: string;
  precision: string;
  input_size?: number | null;
  confidence?: number;
  iou?: number;
  classes?: number[];
  top_k?: number;
  file?: Blob;
  text?: string;
}

function toForm(params: PlaygroundRunParams): FormData {
  const form = new FormData();
  form.append('model_id', params.model_id);
  form.append('runtime_id', params.runtime_id);
  form.append('device', params.device);
  form.append('precision', params.precision);
  if (params.input_size) form.append('input_size', String(params.input_size));
  if (params.confidence !== undefined) form.append('confidence', String(params.confidence));
  if (params.iou !== undefined) form.append('iou', String(params.iou));
  if (params.classes && params.classes.length) form.append('classes', params.classes.join(','));
  if (params.top_k !== undefined) form.append('top_k', String(params.top_k));
  if (params.text) form.append('text', params.text);
  if (params.file) form.append('file', params.file, 'input.jpg');
  return form;
}

export const playgroundApi = {
  /**
   * Run one inference on the server.
   *
   * The round trip is timed here because only the client can see it: the server's
   * own total excludes upload, download and queueing, and calling the difference
   * "network" without measuring it would be a guess.
   */
  run: async (params: PlaygroundRunParams): Promise<PlaygroundTrace> => {
    const t0 = performance.now();
    const trace = await http.postForm<PlaygroundTrace>('/api/playground/infer', toForm(params));
    return { ...trace, client_round_trip_ms: performance.now() - t0 };
  },

  unload: () => http.postJson<{ unloaded: number }>('/api/playground/unload', {}),
};
