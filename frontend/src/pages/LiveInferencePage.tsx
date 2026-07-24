import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '@/services/api';
import { ApiError } from '@/services/http';
import { useDetectionSocket } from '@/hooks/useDetectionSocket';
import { useClassStore } from '@/stores/classStore';
import { drawDetections } from '@/components/DetectionOverlay';
import { PageHeader, Badge, Field } from '@/components/ui';
import { Icon } from '@/components/Icon';
import { StatCard } from '@/components/StatCard';
import { PRESETS, DEFAULT_PRESET, type OptimizationPreset } from '@/utils/presets';
import type { InferenceConfig } from '@/types';

type SourceMode = 'camera' | 'image' | 'video';

const RESOLUTIONS = [
  { label: '640 × 480', w: 640, h: 480 },
  { label: '1280 × 720', w: 1280, h: 720 },
  { label: '1920 × 1080', w: 1920, h: 1080 },
];

export default function LiveInferencePage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const captureRef = useRef<HTMLCanvasElement>(document.createElement('canvas'));
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef<number | null>(null);
  const lastSendRef = useRef(0);
  const imageUrlRef = useRef<string | null>(null);

  const socket = useDetectionSocket();
  const selectedIds = useClassStore((s) => s.selectedIds);
  const classes = useClassStore((s) => s.classes);

  const [sourceMode, setSourceMode] = useState<SourceMode>('camera');
  const [cameras, setCameras] = useState<MediaDeviceInfo[]>([]);
  const [deviceId, setDeviceId] = useState('');
  const [resolution, setResolution] = useState(RESOLUTIONS[1]);
  const [running, setRunning] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [preset, setPreset] = useState<OptimizationPreset>(DEFAULT_PRESET);
  const [presetMsg, setPresetMsg] = useState<string | null>(null);
  const [config, setConfig] = useState<InferenceConfig | null>(null);
  const [health, setHealth] = useState<string>('unknown');

  // Keep latest values available to the render loop without re-binding it.
  const socketRef = useRef(socket);
  socketRef.current = socket;
  const presetRef = useRef(preset);
  presetRef.current = preset;

  /* --------------------------- status ---------------------------- */
  useEffect(() => {
    const controller = new AbortController();
    api
      .detectionStatus(controller.signal)
      .then((s) => {
        setConfig(s.config);
        setHealth(s.health);
      })
      .catch(() => setHealth('unreachable'));
    return () => controller.abort();
  }, []);

  /* --------------------------- cameras --------------------------- */
  const refreshCameras = useCallback(async () => {
    if (!navigator.mediaDevices?.enumerateDevices) return;
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cams = devices.filter((d) => d.kind === 'videoinput');
    setCameras(cams);
    if (cams.length && !deviceId) setDeviceId(cams[0].deviceId);
  }, [deviceId]);

  useEffect(() => {
    refreshCameras();
  }, [refreshCameras]);

  /* --------------------------- capture loop ---------------------- */
  const drawOverlay = useCallback(() => {
    const overlay = overlayRef.current;
    if (!overlay) return;
    let sw = 0;
    let sh = 0;
    if (sourceMode === 'image' && imageRef.current) {
      sw = imageRef.current.naturalWidth;
      sh = imageRef.current.naturalHeight;
    } else if (videoRef.current) {
      sw = videoRef.current.videoWidth;
      sh = videoRef.current.videoHeight;
    }
    if (sw && sh && (overlay.width !== sw || overlay.height !== sh)) {
      overlay.width = sw;
      overlay.height = sh;
    }
    drawDetections(overlay, socketRef.current.detections, sw, sh);
  }, [sourceMode]);

  const loop = useCallback(() => {
    const now = performance.now();
    const interval = 1000 / presetRef.current.targetFps;
    const source =
      sourceMode === 'image' ? imageRef.current : videoRef.current;

    if (source && now - lastSendRef.current >= interval) {
      lastSendRef.current = now;
      const sw =
        sourceMode === 'image'
          ? (imageRef.current?.naturalWidth ?? 0)
          : (videoRef.current?.videoWidth ?? 0);
      const sh =
        sourceMode === 'image'
          ? (imageRef.current?.naturalHeight ?? 0)
          : (videoRef.current?.videoHeight ?? 0);
      if (sw && sh) {
        const canvas = captureRef.current;
        canvas.width = sw;
        canvas.height = sh;
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.drawImage(source as CanvasImageSource, 0, 0, sw, sh);
          canvas.toBlob(
            (blob) => {
              if (blob) socketRef.current.sendFrame(blob);
            },
            'image/jpeg',
            presetRef.current.jpegQuality,
          );
        }
      }
    }

    drawOverlay();
    rafRef.current = requestAnimationFrame(loop);
  }, [sourceMode, drawOverlay]);

  /* --------------------------- start/stop ------------------------ */
  const stopStream = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
  }, []);

  const startCamera = useCallback(async () => {
    setCameraError(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError('Camera API (getUserMedia) is not available in this browser/context.');
      return false;
    }
    try {
      stopStream();
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          deviceId: deviceId ? { exact: deviceId } : undefined,
          width: { ideal: resolution.w },
          height: { ideal: resolution.h },
        },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play().catch(() => undefined);
      }
      await refreshCameras();
      return true;
    } catch (err) {
      const name = err instanceof DOMException ? err.name : '';
      if (name === 'NotAllowedError' || name === 'SecurityError') {
        setCameraError('Camera permission was denied. Grant access in your browser settings and try again.');
      } else if (name === 'NotFoundError') {
        setCameraError('No camera device was found.');
      } else {
        setCameraError(err instanceof Error ? err.message : 'Failed to start camera.');
      }
      return false;
    }
  }, [deviceId, resolution, refreshCameras, stopStream]);

  const start = useCallback(async () => {
    if (sourceMode === 'camera') {
      const ok = await startCamera();
      if (!ok) return;
    }
    socket.connect();
    setRunning(true);
    lastSendRef.current = 0;
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(loop);
  }, [sourceMode, startCamera, socket, loop]);

  const stop = useCallback(() => {
    setRunning(false);
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    socket.disconnect();
    stopStream();
  }, [socket, stopStream]);

  const switchCamera = useCallback(async () => {
    if (cameras.length < 2) return;
    const idx = cameras.findIndex((c) => c.deviceId === deviceId);
    const next = cameras[(idx + 1) % cameras.length];
    setDeviceId(next.deviceId);
  }, [cameras, deviceId]);

  // Restart camera when device/resolution changes mid-session.
  useEffect(() => {
    if (running && sourceMode === 'camera') {
      startCamera();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [deviceId, resolution]);

  useEffect(() => {
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      stopStream();
      if (imageUrlRef.current) URL.revokeObjectURL(imageUrlRef.current);
    };
  }, [stopStream]);

  /* --------------------------- uploads --------------------------- */
  function onImageUpload(file: File) {
    stop();
    setSourceMode('image');
    if (imageUrlRef.current) URL.revokeObjectURL(imageUrlRef.current);
    const url = URL.createObjectURL(file);
    imageUrlRef.current = url;
    if (imageRef.current) imageRef.current.src = url;
  }

  function onVideoUpload(file: File) {
    stop();
    setSourceMode('video');
    stopStream();
    if (imageUrlRef.current) URL.revokeObjectURL(imageUrlRef.current);
    const url = URL.createObjectURL(file);
    imageUrlRef.current = url;
    if (videoRef.current) {
      videoRef.current.srcObject = null;
      videoRef.current.src = url;
      videoRef.current.loop = true;
      videoRef.current.play().catch(() => undefined);
    }
  }

  /* --------------------------- presets --------------------------- */
  async function applyPreset(p: OptimizationPreset) {
    setPreset(p);
    setPresetMsg(null);
    if (!config) {
      setPresetMsg('Applied client-side (no active detector config to switch).');
      return;
    }
    const next: InferenceConfig = {
      ...config,
      input_size: p.inputSize,
      confidence: p.confidence,
      allowed_class_ids: selectedIds,
    };
    try {
      const res = await api.switchDetection(next);
      setConfig(res.config);
      setPresetMsg(
        res.rolled_back ? `Rolled back: ${res.message}` : res.message || 'Preset applied.',
      );
    } catch (err) {
      setPresetMsg(
        `Switch failed: ${err instanceof ApiError ? err.message : err instanceof Error ? err.message : 'error'}`,
      );
    }
  }

  const statusTone =
    socket.status === 'open' ? 'good' : socket.status === 'error' ? 'bad' : 'warn';
  const activeClassNames = classes.filter((c) => selectedIds.includes(c.id)).map((c) => c.name);

  return (
    <div>
      <PageHeader
        title="Live Inference"
        subtitle="Stream frames from your camera, an uploaded video, or an image to the detector over WebSocket and see detections overlaid in real time."
        actions={
          <div className="flex items-center gap-2">
            <Badge tone={statusTone}>ws: {socket.status}</Badge>
            {socket.reconnectAttempt > 0 && <Badge tone="warn">retry {socket.reconnectAttempt}/5</Badge>}
          </div>
        }
      />

      <div className="grid gap-4 lg:grid-cols-[1fr_320px]">
        {/* Viewport */}
        <div className="space-y-4">
          <div className="card relative aspect-video overflow-hidden bg-black">
            <video
              ref={videoRef}
              className={`absolute inset-0 h-full w-full object-contain ${sourceMode === 'image' ? 'hidden' : ''}`}
              playsInline
              muted
            />
            <img
              ref={imageRef}
              alt="uploaded source"
              className={`absolute inset-0 h-full w-full object-contain ${sourceMode === 'image' ? '' : 'hidden'}`}
              onLoad={drawOverlay}
            />
            <canvas
              ref={overlayRef}
              className="absolute inset-0 h-full w-full object-contain"
            />
            {!running && sourceMode === 'camera' && (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-slate-500">
                <Icon name="camera" className="h-8 w-8" />
                <span className="text-sm">Press Start to begin streaming</span>
              </div>
            )}
            {cameraError && (
              <div className="absolute inset-x-4 bottom-4 rounded-lg border border-bad/40 bg-bad/15 px-3 py-2 text-sm text-bad">
                {cameraError}
              </div>
            )}
            {socket.status === 'error' && running && (
              <div className="absolute inset-x-4 top-4 rounded-lg border border-warn/40 bg-warn/15 px-3 py-2 text-sm text-warn">
                Lost connection to the detector. Auto-reconnect exhausted after 5 attempts — press Stop then Start to retry.
              </div>
            )}
          </div>

          {/* Controls */}
          <div className="flex flex-wrap items-center gap-2">
            {!running ? (
              <button className="btn-primary" onClick={start}>
                <Icon name="live" className="h-4 w-4" /> Start
              </button>
            ) : (
              <button className="btn-danger" onClick={stop}>
                Stop
              </button>
            )}
            <button
              className="btn-ghost"
              onClick={switchCamera}
              disabled={sourceMode !== 'camera' || cameras.length < 2}
            >
              <Icon name="refresh" className="h-4 w-4" /> Switch camera
            </button>

            <label className="btn-ghost cursor-pointer">
              Upload image
              <input
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => e.target.files?.[0] && onImageUpload(e.target.files[0])}
              />
            </label>
            <label className="btn-ghost cursor-pointer">
              Upload video
              <input
                type="file"
                accept="video/*"
                className="hidden"
                onChange={(e) => e.target.files?.[0] && onVideoUpload(e.target.files[0])}
              />
            </label>
            {sourceMode !== 'camera' && (
              <button
                className="btn-ghost"
                onClick={() => {
                  stop();
                  setSourceMode('camera');
                }}
              >
                Use camera
              </button>
            )}
          </div>

          {/* Live stats */}
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <StatCard label="FPS" value={socket.fps} icon="gauge" tone="accent" />
            <StatCard label="Inference" value={socket.inferenceMs ? socket.inferenceMs.toFixed(1) : '—'} unit="ms" icon="clock" />
            <StatCard label="Objects" value={socket.detections.length} icon="tag" />
            <StatCard label="Dropped" value={socket.droppedCount} icon="alert" tone={socket.droppedCount > 0 ? 'warn' : 'neutral'} />
          </div>
        </div>

        {/* Side panel */}
        <aside className="space-y-4">
          <div className="card card-pad space-y-3">
            <h2 className="label">Source</h2>
            <div className="flex gap-1 rounded-lg bg-surface-900 p-1 text-xs">
              {(['camera', 'video', 'image'] as SourceMode[]).map((m) => (
                <button
                  key={m}
                  onClick={() => {
                    stop();
                    setSourceMode(m);
                  }}
                  className={`flex-1 rounded-md px-2 py-1.5 capitalize transition ${
                    sourceMode === m ? 'bg-accent text-surface-950' : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {m}
                </button>
              ))}
            </div>

            {sourceMode === 'camera' && (
              <>
                <Field label="Camera">
                  <select className="input" value={deviceId} onChange={(e) => setDeviceId(e.target.value)}>
                    {cameras.map((c, i) => (
                      <option key={c.deviceId} value={c.deviceId}>
                        {c.label || `Camera ${i + 1}`}
                      </option>
                    ))}
                    {cameras.length === 0 && <option value="">(start to grant access)</option>}
                  </select>
                </Field>
                <Field label="Resolution">
                  <select
                    className="input"
                    value={resolution.label}
                    onChange={(e) =>
                      setResolution(RESOLUTIONS.find((r) => r.label === e.target.value) ?? RESOLUTIONS[1])
                    }
                  >
                    {RESOLUTIONS.map((r) => (
                      <option key={r.label} value={r.label}>
                        {r.label}
                      </option>
                    ))}
                  </select>
                </Field>
              </>
            )}
          </div>

          <div className="card card-pad space-y-2">
            <h2 className="label">Optimization preset</h2>
            <div className="grid grid-cols-1 gap-1.5">
              {PRESETS.map((p) => (
                <button
                  key={p.key}
                  onClick={() => applyPreset(p)}
                  className={`rounded-lg border px-3 py-2 text-left text-sm transition ${
                    preset.key === p.key
                      ? 'border-accent/50 bg-accent/10 text-slate-100'
                      : 'border-surface-700 bg-surface-900 text-slate-300 hover:border-surface-600'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-medium">{p.label}</span>
                    <span className="font-mono text-[11px] text-slate-500">
                      {p.inputSize}px · {p.targetFps}fps
                    </span>
                  </div>
                  <div className="text-[11px] text-slate-500">{p.description}</div>
                </button>
              ))}
            </div>
            {presetMsg && <p className="text-xs text-slate-400">{presetMsg}</p>}
          </div>

          <div className="card card-pad space-y-2 text-sm">
            <h2 className="label">Active configuration</h2>
            <Row label="Model" value={config?.model_id ?? '—'} />
            <Row label="Runtime" value={config?.runtime ?? '—'} />
            <Row label="Input size" value={config ? String(config.input_size) : '—'} />
            <Row label="Confidence" value={config ? config.confidence.toFixed(2) : '—'} />
            <Row label="Execution" value={config?.execution_location ?? '—'} />
            <Row label="Backend (live)" value={socket.backend || '—'} />
            <Row
              label="Health"
              value={health}
              tone={health === 'healthy' || health === 'ok' ? 'good' : health === 'unreachable' ? 'bad' : 'warn'}
            />
            <div className="flex items-center justify-between">
              <span className="text-slate-500">Classes</span>
              <span className="text-slate-300">
                {selectedIds.length === classes.length || classes.length === 0
                  ? 'all'
                  : `${selectedIds.length} selected`}
              </span>
            </div>
            {activeClassNames.length > 0 && activeClassNames.length < classes.length && (
              <p className="text-[11px] text-slate-500">
                {activeClassNames.slice(0, 8).join(', ')}
                {activeClassNames.length > 8 ? ` +${activeClassNames.length - 8} more` : ''}
              </p>
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}

function Row({ label, value, tone }: { label: string; value: string; tone?: string }) {
  const toneCls =
    tone === 'good' ? 'text-good' : tone === 'bad' ? 'text-bad' : tone === 'warn' ? 'text-warn' : 'text-slate-300';
  return (
    <div className="flex items-center justify-between">
      <span className="text-slate-500">{label}</span>
      <span className={`font-mono text-xs ${toneCls}`}>{value}</span>
    </div>
  );
}
