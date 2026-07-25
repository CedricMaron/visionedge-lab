import { describe, it, expect, afterEach, vi } from 'vitest';
import { ServerInferenceBackend } from './serverBackend';

const BODY = {
  detections: [],
  timings: { preprocess_ms: 3, inference_ms: 140, postprocess_ms: 2, end_to_end_ms: 146 },
  backend: 'onnxruntime-cpu',
  count: 0,
};

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('ServerInferenceBackend', () => {
  it('passes the full timing breakdown through, not just inference_ms', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
      new Response(JSON.stringify(BODY), {
        status: 200, headers: { 'content-type': 'application/json' },
      }),
    ));

    const out = await new ServerInferenceBackend().infer({
      frame: new Blob(['x']), confidence: 0.25, iou: 0.45, allowedClassIds: [],
    });

    expect(out.inferenceMs).toBe(140);
    expect(out.timings?.preprocess_ms).toBe(3);
    expect(out.timings?.postprocess_ms).toBe(2);
    expect(out.timings?.end_to_end_ms).toBe(146);
  });
});
