// Server-side inference backend adapter.
//
// This wraps the real /api/infer endpoint behind the shared InferenceBackend
// interface so the Live page (Phase 3+) can swap browser vs server execution
// transparently. The single-shot path here is real; the low-latency streaming
// path used today lives in the WebSocket hook (useDetectionSocket).

import type { InferenceBackend, InferenceOutput, InferenceRequest } from './types';
import { api } from '@/services/api';

async function toJpegBlob(frame: Blob | ImageBitmap): Promise<Blob> {
  if (frame instanceof Blob) return frame;
  const canvas = document.createElement('canvas');
  canvas.width = frame.width;
  canvas.height = frame.height;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('2D canvas context unavailable');
  ctx.drawImage(frame, 0, 0);
  return new Promise<Blob>((resolve, reject) => {
    canvas.toBlob(
      (b) => (b ? resolve(b) : reject(new Error('toBlob failed'))),
      'image/jpeg',
      0.8,
    );
  });
}

export class ServerInferenceBackend implements InferenceBackend {
  readonly name = 'server-fastapi';
  readonly executionLocation = 'server' as const;

  async isAvailable(): Promise<boolean> {
    try {
      await api.health();
      return true;
    } catch {
      return false;
    }
  }

  async init(): Promise<void> {
    // Nothing to warm up for the HTTP path.
  }

  async infer(request: InferenceRequest): Promise<InferenceOutput> {
    const blob = await toJpegBlob(request.frame);
    const res = await api.infer(blob, {
      confidence: request.confidence,
      iou: request.iou,
      classes: request.allowedClassIds.length ? request.allowedClassIds.join(',') : undefined,
    });
    return {
      detections: res.detections,
      inferenceMs: res.timings.inference_ms,
      backend: res.backend,
      timings: res.timings,
    };
  }

  async dispose(): Promise<void> {
    // No-op.
  }
}
