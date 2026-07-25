import { describe, it, expect, afterEach, vi } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import { useDetectionSocket } from './useDetectionSocket';

// Minimal WebSocket stand-in: captures the instance so tests can push messages.
class FakeWebSocket {
  static instances: FakeWebSocket[] = [];
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  readonly url: string;
  readyState = 1;
  binaryType = 'blob';
  onopen: (() => void) | null = null;
  onmessage: ((e: MessageEvent) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;

  constructor(url: string) {
    this.url = url;
    FakeWebSocket.instances.push(this);
  }

  send() {}
  close() {
    this.readyState = 3;
    this.onclose?.();
  }

  emit(payload: unknown) {
    this.onmessage?.({ data: JSON.stringify(payload) } as MessageEvent);
  }
}

function frame(dropped: number) {
  return {
    frame_id: 1,
    detections: [],
    timings: { inference_ms: 12 },
    dropped,
    backend: 'onnxruntime-cpu',
  };
}

afterEach(() => {
  FakeWebSocket.instances = [];
  vi.unstubAllGlobals();
});

describe('useDetectionSocket dropped-frame accounting', () => {
  it('reports the running total the server sends, not one per message', () => {
    vi.stubGlobal('WebSocket', FakeWebSocket);
    const { result } = renderHook(() => useDetectionSocket());

    act(() => result.current.connect());
    const ws = FakeWebSocket.instances[0];
    act(() => ws.onopen?.());

    // The server's `dropped` field is a cumulative session total, so a client
    // that increments per message inflates it on every subsequent frame.
    act(() => ws.emit(frame(4)));
    expect(result.current.droppedCount).toBe(4);

    act(() => ws.emit(frame(4)));
    act(() => ws.emit(frame(4)));
    expect(result.current.droppedCount).toBe(4);

    act(() => ws.emit(frame(7)));
    expect(result.current.droppedCount).toBe(7);
  });

  it('exposes backend and inference latency from the message', () => {
    vi.stubGlobal('WebSocket', FakeWebSocket);
    const { result } = renderHook(() => useDetectionSocket());
    act(() => result.current.connect());
    const ws = FakeWebSocket.instances[0];
    act(() => ws.onopen?.());
    act(() => ws.emit(frame(0)));

    expect(result.current.backend).toBe('onnxruntime-cpu');
    expect(result.current.inferenceMs).toBe(12);
    expect(result.current.droppedCount).toBe(0);
  });
});
