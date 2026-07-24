// VLM (vision-language model) API service.
// The VLM slice may or may not be enabled in a given backend build, so every
// call surfaces availability honestly rather than faking a response.

import { ApiError, http } from './http';
import type { VLMResponse, VlmModelsResponse } from '@/types';

export class VlmUnavailableError extends Error {
  constructor(message = 'VLM slice not enabled in this build') {
    super(message);
    this.name = 'VlmUnavailableError';
  }
}

function mapUnavailable(err: unknown): never {
  if (err instanceof ApiError && (err.status === 404 || err.status === 501)) {
    throw new VlmUnavailableError();
  }
  throw err;
}

export const vlmApi = {
  models: async (signal?: AbortSignal): Promise<VlmModelsResponse> => {
    try {
      return await http.get<VlmModelsResponse>('/api/vlm/models', undefined, signal);
    } catch (err) {
      return mapUnavailable(err);
    }
  },

  runtimeStatus: async (signal?: AbortSignal): Promise<Record<string, unknown>> => {
    try {
      return await http.get<Record<string, unknown>>('/api/vlm/runtime-status', undefined, signal);
    } catch (err) {
      return mapUnavailable(err);
    }
  },

  analyzeImage: async (
    file: Blob,
    opts: { question?: string; ground?: boolean } = {},
  ): Promise<VLMResponse> => {
    const form = new FormData();
    form.append('file', file, 'frame.jpg');
    if (opts.question) form.append('question', opts.question);
    if (opts.ground !== undefined) form.append('ground', String(opts.ground));
    try {
      return await http.postForm<VLMResponse>('/api/vlm/analyze-image', form);
    } catch (err) {
      return mapUnavailable(err);
    }
  },

  ask: async (file: Blob, question: string, ground: boolean): Promise<VLMResponse> => {
    const form = new FormData();
    form.append('file', file, 'frame.jpg');
    form.append('question', question);
    form.append('ground', String(ground));
    try {
      return await http.postForm<VLMResponse>('/api/vlm/ask', form);
    } catch (err) {
      return mapUnavailable(err);
    }
  },
};
