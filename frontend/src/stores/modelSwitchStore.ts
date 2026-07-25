// Model-switch store. Builds an InferenceConfig from the user's selectors and
// drives POST /api/detection/switch, tracking load progress, errors, and the
// backend's rollback signal.

import { create } from 'zustand';
import { api } from '@/services/api';
import { ApiError } from '@/services/http';
import type { ExecutionLocation, InferenceConfig, SwitchResponse } from '@/types';

export type SwitchStatus = 'idle' | 'loading' | 'success' | 'error' | 'rolled_back';

export interface DraftConfig {
  model_id: string;
  runtime: string;
  input_size: number;
  confidence: number;
  iou: number;
  execution_location: ExecutionLocation;
}

export const DEFAULT_DRAFT: DraftConfig = {
  model_id: '',
  runtime: '',
  input_size: 640,
  confidence: 0.25,
  iou: 0.45,
  execution_location: 'pc_local',
};

interface ModelSwitchState {
  draft: DraftConfig;
  status: SwitchStatus;
  message: string;
  activeConfig: InferenceConfig | null;
  rolledBack: boolean;

  setDraft: (patch: Partial<DraftConfig>) => void;
  buildConfig: (allowedClassIds: number[]) => InferenceConfig;
  applySwitch: (allowedClassIds: number[]) => Promise<SwitchResponse>;
  reset: () => void;
}

export function buildInferenceConfig(
  draft: DraftConfig,
  allowedClassIds: number[],
): InferenceConfig {
  return {
    model_id: draft.model_id,
    runtime: draft.runtime,
    input_size: draft.input_size,
    confidence: draft.confidence,
    iou: draft.iou,
    execution_location: draft.execution_location,
    allowed_class_ids: [...allowedClassIds].sort((a, b) => a - b),
  };
}

export const useModelSwitchStore = create<ModelSwitchState>((set, get) => ({
  draft: { ...DEFAULT_DRAFT },
  status: 'idle',
  message: '',
  activeConfig: null,
  rolledBack: false,

  setDraft: (patch) => set((state) => ({ draft: { ...state.draft, ...patch } })),

  buildConfig: (allowedClassIds) => buildInferenceConfig(get().draft, allowedClassIds),

  applySwitch: async (allowedClassIds) => {
    const config = buildInferenceConfig(get().draft, allowedClassIds);
    set({ status: 'loading', message: 'Applying configuration…', rolledBack: false });
    try {
      const res = await api.switchDetection(config);
      if (res.ok && !res.rolled_back) {
        set({
          status: 'success',
          message: res.message || 'Configuration applied.',
          activeConfig: res.config,
          rolledBack: false,
        });
      } else {
        set({
          status: 'rolled_back',
          message: res.message || 'Switch rolled back to the previous configuration.',
          activeConfig: res.config,
          rolledBack: true,
        });
      }
      return res;
    } catch (err) {
      const message =
        err instanceof ApiError ? err.message : err instanceof Error ? err.message : 'Switch failed';
      set({ status: 'error', message, rolledBack: false });
      throw err;
    }
  },

  reset: () => set({ status: 'idle', message: '', rolledBack: false }),
}));
