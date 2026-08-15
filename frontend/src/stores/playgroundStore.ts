// The Playground's configuration and its most recent inference trace.
//
// The trace lives here rather than in the page so the Pipeline and Performance
// pages can inspect the run the visitor just executed without re-running it. It is
// deliberately not persisted: a trace describes one execution on one machine, and
// restoring it from localStorage after a reload would present a stale measurement
// as a current one.

import { create } from 'zustand';
import type { Modality } from '@/lab/catalog';
import type { ExecutionTarget, PlaygroundTrace } from '@/types/playground';

export interface PlaygroundConfig {
  modality: Modality;
  task: string;
  modelId: string;
  execution: ExecutionTarget;
  runtimeId: string;
  device: string;
  precision: string;
  inputSize: number | null;
  confidence: number;
  iou: number;
  topK: number;
}

export const DEFAULT_CONFIG: PlaygroundConfig = {
  modality: 'image',
  task: '',
  modelId: '',
  execution: 'server',
  runtimeId: '',
  device: 'cpu',
  precision: 'fp32',
  inputSize: null,
  confidence: 0.25,
  iou: 0.45,
  topK: 5,
};

/** Streaming (video) statistics, which have no single-shot trace to live in. */
export interface StreamStats {
  execution: ExecutionTarget;
  fps: number;
  inferenceMs: number;
  processedFrames: number;
  droppedFrames: number;
  backend: string;
}

interface PlaygroundState {
  config: PlaygroundConfig;
  trace: PlaygroundTrace | null;
  stream: StreamStats | null;
  setConfig: (patch: Partial<PlaygroundConfig>) => void;
  setTrace: (trace: PlaygroundTrace | null) => void;
  setStream: (stats: StreamStats | null) => void;
}

export const usePlaygroundStore = create<PlaygroundState>((set) => ({
  config: { ...DEFAULT_CONFIG },
  trace: null,
  stream: null,
  setConfig: (patch) => set((state) => ({ config: { ...state.config, ...patch } })),
  setTrace: (trace) => set({ trace }),
  setStream: (stream) => set({ stream }),
}));
