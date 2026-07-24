// Browser-side inference backend — PLANNED (Phase 3).
//
// This will run detection models fully client-side via onnxruntime-web / WebGPU
// so frames never leave the device. It is deliberately NOT implemented yet:
// methods throw explicit errors rather than returning fabricated detections.

import type { InferenceBackend, InferenceOutput, InferenceRequest } from './types';

const NOT_IMPLEMENTED =
  'Browser-side inference is not implemented in this build (Planned — Phase 3).';

export class BrowserInferenceBackend implements InferenceBackend {
  readonly name = 'browser-onnxruntime-web';
  readonly executionLocation = 'browser' as const;

  async isAvailable(): Promise<boolean> {
    // Real availability will probe WebGPU / WASM SIMD + model assets.
    return false;
  }

  async init(): Promise<void> {
    throw new Error(NOT_IMPLEMENTED);
  }

  async infer(_request: InferenceRequest): Promise<InferenceOutput> {
    throw new Error(NOT_IMPLEMENTED);
  }

  async dispose(): Promise<void> {
    // No-op until implemented.
  }
}
