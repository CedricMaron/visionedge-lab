// Shared interfaces for pluggable inference backends (Phase 3 groundwork).

import type { Detection } from '@/types';

export interface InferenceRequest {
  // Encoded JPEG frame or raw image bitmap source.
  frame: Blob | ImageBitmap;
  confidence: number;
  iou: number;
  allowedClassIds: number[];
}

export interface InferenceOutput {
  detections: Detection[];
  inferenceMs: number;
  backend: string;
}

export interface InferenceBackend {
  readonly name: string;
  readonly executionLocation: 'browser' | 'server';
  isAvailable(): Promise<boolean>;
  init(): Promise<void>;
  infer(request: InferenceRequest): Promise<InferenceOutput>;
  dispose(): Promise<void>;
}
