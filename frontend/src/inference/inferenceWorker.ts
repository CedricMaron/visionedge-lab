// Web Worker entrypoint for browser-side inference — PLANNED (Phase 3).
//
// When implemented, the main thread will post frames here and the worker will
// run the ONNX model off the UI thread, posting back detections. For now it is
// a documented stub that replies with an explicit "not implemented" error so no
// caller ever receives fabricated results.

export interface WorkerInferMessage {
  type: 'infer';
  frameId: number;
  bitmap: ImageBitmap;
  confidence: number;
  iou: number;
  allowedClassIds: number[];
}

export interface WorkerReadyMessage {
  type: 'ready' | 'error';
  message?: string;
}

// Guard so this module is import-safe from the main thread (tests, bundling).
if (typeof self !== 'undefined' && 'onmessage' in self) {
  self.onmessage = (event: MessageEvent<WorkerInferMessage>) => {
    if (event.data?.type === 'infer') {
      const reply: WorkerReadyMessage = {
        type: 'error',
        message: 'Browser-side worker inference is not implemented (Planned — Phase 3).',
      };
      (self as unknown as Worker).postMessage(reply);
    }
  };
}
