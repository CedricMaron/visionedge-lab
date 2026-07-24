// WebSocket hook for live detection streaming.
// Client sends BINARY jpeg frames; server replies JSON WsDetectResult.
// Handles bounded auto-reconnect and exposes latest detections + live FPS.

import { useCallback, useEffect, useRef, useState } from 'react';
import { getWsUrl } from '@/config';
import type { Detection, WsDetectResult } from '@/types';

export type SocketStatus = 'idle' | 'connecting' | 'open' | 'closed' | 'error';

export interface DetectionSocket {
  status: SocketStatus;
  detections: Detection[];
  backend: string;
  inferenceMs: number;
  fps: number;
  droppedCount: number;
  reconnectAttempt: number;
  sendFrame: (blob: Blob) => void;
  connect: () => void;
  disconnect: () => void;
}

const MAX_RETRIES = 5;
const BASE_RETRY_MS = 1000;

export function useDetectionSocket(autoStart = false): DetectionSocket {
  const [status, setStatus] = useState<SocketStatus>('idle');
  const [detections, setDetections] = useState<Detection[]>([]);
  const [backend, setBackend] = useState('');
  const [inferenceMs, setInferenceMs] = useState(0);
  const [fps, setFps] = useState(0);
  const [droppedCount, setDroppedCount] = useState(0);
  const [reconnectAttempt, setReconnectAttempt] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);
  const retryRef = useRef(0);
  const wantOpenRef = useRef(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const frameTimes = useRef<number[]>([]);

  const clearTimer = () => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  };

  const open = useCallback(() => {
    if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
      return;
    }
    setStatus('connecting');
    let ws: WebSocket;
    try {
      ws = new WebSocket(getWsUrl('/api/ws/detect'));
    } catch {
      setStatus('error');
      return;
    }
    ws.binaryType = 'arraybuffer';
    wsRef.current = ws;

    ws.onopen = () => {
      retryRef.current = 0;
      setReconnectAttempt(0);
      setStatus('open');
    };

    ws.onmessage = (event: MessageEvent) => {
      if (typeof event.data !== 'string') return;
      try {
        const msg = JSON.parse(event.data) as WsDetectResult;
        setDetections(msg.detections ?? []);
        setBackend(msg.backend ?? '');
        setInferenceMs(msg.timings?.inference_ms ?? 0);
        if (msg.dropped) setDroppedCount((c) => c + 1);

        const now = performance.now();
        const times = frameTimes.current;
        times.push(now);
        while (times.length > 0 && now - times[0] > 1000) times.shift();
        setFps(times.length);
      } catch {
        // Ignore malformed frames.
      }
    };

    ws.onerror = () => {
      setStatus('error');
    };

    ws.onclose = () => {
      wsRef.current = null;
      if (!wantOpenRef.current) {
        setStatus('closed');
        return;
      }
      if (retryRef.current >= MAX_RETRIES) {
        setStatus('error');
        return;
      }
      retryRef.current += 1;
      setReconnectAttempt(retryRef.current);
      setStatus('connecting');
      const delay = BASE_RETRY_MS * 2 ** (retryRef.current - 1);
      clearTimer();
      timerRef.current = setTimeout(() => {
        if (wantOpenRef.current) open();
      }, delay);
    };
  }, []);

  const connect = useCallback(() => {
    wantOpenRef.current = true;
    retryRef.current = 0;
    setReconnectAttempt(0);
    open();
  }, [open]);

  const disconnect = useCallback(() => {
    wantOpenRef.current = false;
    clearTimer();
    frameTimes.current = [];
    setFps(0);
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setStatus('idle');
  }, []);

  const sendFrame = useCallback((blob: Blob) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      blob.arrayBuffer().then((buf) => {
        if (ws.readyState === WebSocket.OPEN) ws.send(buf);
      });
    }
  }, []);

  useEffect(() => {
    if (autoStart) connect();
    return () => {
      wantOpenRef.current = false;
      clearTimer();
      if (wsRef.current) wsRef.current.close();
      wsRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return {
    status,
    detections,
    backend,
    inferenceMs,
    fps,
    droppedCount,
    reconnectAttempt,
    sendFrame,
    connect,
    disconnect,
  };
}
