/**
 * Playground — the one place inference happens.
 *
 * Input → Task → Model → Execution → Pipeline → Output, in that order and on one
 * screen. Nothing here is model-specific: the tasks come from the models the server
 * has adapters for, the runtimes come from its probes, and the result panel renders
 * whichever output kind the task produces.
 *
 * Impossible combinations are not offered. Where something a visitor would expect
 * is missing — local execution, CUDA, a second model — the reason is shown instead
 * of a disabled control with no explanation.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { useAsync } from '@/hooks/useAsync';
import { labApi } from '@/services/labApi';
import { api } from '@/services/api';
import { playgroundApi } from '@/services/playgroundApi';
import { useDetectionSocket } from '@/hooks/useDetectionSocket';
import { useClassStore } from '@/stores/classStore';
import { usePlaygroundStore } from '@/stores/playgroundStore';
import { detectBrowserCaps, type BrowserCaps } from '@/utils/browserCaps';
import { drawDetections } from '@/utils/drawDetections';
import { ClassPicker } from '@/components/ClassPicker';
import { ExecutionBadge } from '@/components/ExecutionBadge';
import { Icon } from '@/components/Icon';
import { Badge, ErrorState, Field, PageHeader, Spinner } from '@/components/ui';
import { formatMb } from '@/utils/format';
import {
  INPUT_SIZES,
  MODALITIES,
  localAvailability,
  offeredPrecisions,
  modelsForTask,
  resolveConfig,
  serverAvailability,
  taskLabel,
  tasksForModality,
  type Modality,
} from '@/lab/catalog';
import type { Capabilities } from '@/types';
import type { LabModel, RuntimeCapability } from '@/types/lab';
import type { ExecutionTarget, PlaygroundTrace } from '@/types/playground';

const CAMERA_RESOLUTIONS = [
  { label: '640 × 480', w: 640, h: 480 },
  { label: '1280 × 720', w: 1280, h: 720 },
  { label: '1920 × 1080', w: 1920, h: 1080 },
];

const STREAM_TARGET_FPS = 12;

export default function PlaygroundPage() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();

  const models = useAsync<{ models: LabModel[] }>((s) => labApi.models(s), []);
  const runtimes = useAsync<{ runtimes: RuntimeCapability[] }>((s) => labApi.runtimes(s), []);
  // Used only to rule out accelerators the server does not have: ONNX Runtime lists
  // a CUDA provider whenever the GPU build is installed, GPU or no GPU.
  const hostCaps = useAsync<Capabilities>((s) => api.capabilities(s), []);

  const config = usePlaygroundStore((s) => s.config);
  const setConfig = usePlaygroundStore((s) => s.setConfig);
  const trace = usePlaygroundStore((s) => s.trace);
  const setTrace = usePlaygroundStore((s) => s.setTrace);
  const setStream = usePlaygroundStore((s) => s.setStream);
  const selectedClassIds = useClassStore((s) => s.selectedIds);
  const allClasses = useClassStore((s) => s.classes);

  const [caps, setCaps] = useState<BrowserCaps | null>(null);
  useEffect(() => setCaps(detectBrowserCaps()), []);

  /* ------------------------------ inputs ------------------------------ */
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [imageSize, setImageSize] = useState<{ w: number; h: number } | null>(null);
  const [text, setText] = useState('A man is riding a bicycle next to a red car.');
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);

  const overlayRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const videoOverlayRef = useRef<HTMLCanvasElement>(null);
  const captureRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef<number | null>(null);
  const lastSendRef = useRef(0);
  const mediaUrlRef = useRef<string | null>(null);

  /* --------------------------- derived options ------------------------ */
  const allModels = useMemo(() => models.data?.models ?? [], [models.data]);
  const allRuntimes = useMemo(() => runtimes.data?.runtimes ?? [], [runtimes.data]);

  const tasks = useMemo(
    () => tasksForModality(allModels, config.modality),
    [allModels, config.modality],
  );
  const taskModels = useMemo(
    () => modelsForTask(allModels, config.modality, config.task),
    [allModels, config.modality, config.task],
  );
  const model = taskModels.find((m) => m.model_id === config.modelId);

  const serverGpuCount = hostCaps.data ? hostCaps.data.gpus.length : null;
  const server = useMemo(
    () => serverAvailability(model, allRuntimes, serverGpuCount),
    [model, allRuntimes, serverGpuCount],
  );
  const local = useMemo(() => localAvailability(model, caps), [model, caps]);
  const availability = config.execution === 'server' ? server : local;

  const runtimeOption = availability.runtimes.find((r) => r.runtime_id === config.runtimeId);
  const precision = useMemo(
    () => offeredPrecisions(model, availability, config.runtimeId, config.device),
    [model, availability, config.runtimeId, config.device],
  );

  /* --------------------- keep the selection valid --------------------- */
  // A model chosen on the Models page arrives as ?model=…; adopt it once.
  useEffect(() => {
    const wanted = searchParams.get('model');
    if (!wanted || allModels.length === 0) return;
    const found = allModels.find((m) => m.model_id === wanted);
    if (found) {
      const modality: Modality = found.modality === 'text' ? 'text' : 'image';
      setConfig({ modality, task: found.task, modelId: found.model_id });
    }
    searchParams.delete('model');
    setSearchParams(searchParams, { replace: true });
  }, [searchParams, setSearchParams, allModels, setConfig]);

  useEffect(() => {
    if (tasks.length > 0 && !tasks.includes(config.task)) setConfig({ task: tasks[0] });
  }, [tasks, config.task, setConfig]);

  useEffect(() => {
    if (taskModels.length > 0 && !taskModels.some((m) => m.model_id === config.modelId)) {
      setConfig({ modelId: taskModels[0].model_id, inputSize: taskModels[0].input_size });
    }
  }, [taskModels, config.modelId, setConfig]);

  useEffect(() => {
    // Fall back to whichever target can actually run this model.
    if (config.execution === 'local' && !local.available && server.available) {
      setConfig({ execution: 'server' });
    }
  }, [config.execution, local.available, server.available, setConfig]);

  useEffect(() => {
    const resolved = resolveConfig(availability, config);
    if (!resolved) return;
    if (
      resolved.runtime_id !== config.runtimeId ||
      resolved.device !== config.device ||
      resolved.precision !== config.precision
    ) {
      setConfig(resolved);
    }
  }, [availability, config, setConfig]);

  useEffect(() => {
    // resolveConfig clamps to what the runtime reports; this clamps further to what
    // the installed artefact can honestly be run at.
    if (precision.values.length > 0 && !precision.values.includes(config.precision)) {
      setConfig({ precision: precision.values[0] });
    }
  }, [precision, config.precision, setConfig]);

  /* ------------------------------ image run --------------------------- */
  function onPickImage(file: File) {
    if (mediaUrlRef.current) URL.revokeObjectURL(mediaUrlRef.current);
    const url = URL.createObjectURL(file);
    mediaUrlRef.current = url;
    setImageFile(file);
    setImageUrl(url);
    setTrace(null);
    setRunError(null);
  }

  const redrawImageOverlay = useCallback(() => {
    const canvas = overlayRef.current;
    if (!canvas || !imageSize) return;
    if (canvas.width !== imageSize.w || canvas.height !== imageSize.h) {
      canvas.width = imageSize.w;
      canvas.height = imageSize.h;
    }
    drawDetections(canvas, trace?.result.detections ?? [], imageSize.w, imageSize.h);
  }, [imageSize, trace]);

  useEffect(() => {
    redrawImageOverlay();
  }, [redrawImageOverlay]);

  async function runSingleShot() {
    if (!model) return;
    setRunning(true);
    setRunError(null);
    try {
      const result = await playgroundApi.run({
        model_id: model.model_id,
        runtime_id: config.runtimeId,
        device: config.device,
        precision: config.precision,
        input_size: config.inputSize ?? undefined,
        confidence: config.task === 'object_detection' ? config.confidence : undefined,
        iou: config.task === 'object_detection' ? config.iou : undefined,
        classes:
          config.task === 'object_detection' && selectedClassIds.length < allClasses.length
            ? selectedClassIds
            : undefined,
        top_k: config.topK,
        file: config.modality === 'image' ? (imageFile ?? undefined) : undefined,
        text: config.modality === 'text' ? text : undefined,
      });
      setTrace(result);
    } catch (err) {
      setTrace(null);
      setRunError(err instanceof Error ? err.message : 'inference failed');
    } finally {
      setRunning(false);
    }
  }

  /* ------------------------------ streaming --------------------------- */
  const socket = useDetectionSocket();
  const socketRef = useRef(socket);
  socketRef.current = socket;

  const [videoSource, setVideoSource] = useState<'camera' | 'file'>('camera');
  const [cameras, setCameras] = useState<MediaDeviceInfo[]>([]);
  const [cameraId, setCameraId] = useState('');
  const [resolution, setResolution] = useState(CAMERA_RESOLUTIONS[1]);
  const [streaming, setStreaming] = useState(false);
  const [streamError, setStreamError] = useState<string | null>(null);
  const [switchMessage, setSwitchMessage] = useState<string | null>(null);
  const processedRef = useRef(0);
  const [processed, setProcessed] = useState(0);

  useEffect(() => {
    setStream(
      streaming
        ? {
            execution: 'server',
            fps: socket.fps,
            inferenceMs: socket.inferenceMs,
            processedFrames: processed,
            droppedFrames: socket.droppedCount,
            backend: socket.backend,
          }
        : null,
    );
  }, [streaming, socket.fps, socket.inferenceMs, socket.droppedCount, socket.backend, processed, setStream]);

  const streamLoop = useCallback(() => {
    const video = videoRef.current;
    const overlay = videoOverlayRef.current;
    if (video && overlay) {
      const sw = video.videoWidth;
      const sh = video.videoHeight;
      if (sw && sh) {
        if (overlay.width !== sw || overlay.height !== sh) {
          overlay.width = sw;
          overlay.height = sh;
        }
        drawDetections(overlay, socketRef.current.detections, sw, sh);

        const now = performance.now();
        if (now - lastSendRef.current >= 1000 / STREAM_TARGET_FPS) {
          lastSendRef.current = now;
          if (!captureRef.current) captureRef.current = document.createElement('canvas');
          const canvas = captureRef.current;
          canvas.width = sw;
          canvas.height = sh;
          const ctx = canvas.getContext('2d');
          if (ctx) {
            ctx.drawImage(video, 0, 0, sw, sh);
            canvas.toBlob(
              (blob) => {
                if (blob) {
                  socketRef.current.sendFrame(blob);
                  processedRef.current += 1;
                  setProcessed(processedRef.current);
                }
              },
              'image/jpeg',
              0.7,
            );
          }
        }
      }
    }
    rafRef.current = requestAnimationFrame(streamLoop);
  }, []);

  const stopStreaming = useCallback(() => {
    setStreaming(false);
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    socketRef.current.disconnect();
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
  }, []);

  const startStreaming = useCallback(async () => {
    setStreamError(null);
    setSwitchMessage(null);
    processedRef.current = 0;
    setProcessed(0);

    // Align the server's streaming detector with the Playground selection — the
    // WebSocket path uses the server's active configuration, so leaving them out of
    // sync would report one model's numbers under another model's name.
    if (model) {
      try {
        const res = await api.switchDetection({
          model_id: model.model_id,
          runtime: `onnxruntime-${config.device}`,
          input_size: config.inputSize ?? 640,
          confidence: config.confidence,
          iou: config.iou,
          execution_location: 'local_server',
          allowed_class_ids: selectedClassIds,
        });
        if (res.rolled_back) setSwitchMessage(`Rolled back: ${res.message}`);
      } catch (err) {
        setSwitchMessage(
          `Could not apply this configuration to the streaming detector: ${
            err instanceof Error ? err.message : 'switch failed'
          }. Streaming will use the server's current configuration.`,
        );
      }
    }

    if (videoSource === 'camera') {
      if (!navigator.mediaDevices?.getUserMedia) {
        setStreamError('getUserMedia is not available in this browser or context.');
        return;
      }
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            deviceId: cameraId ? { exact: cameraId } : undefined,
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
        const devices = await navigator.mediaDevices.enumerateDevices();
        setCameras(devices.filter((d) => d.kind === 'videoinput'));
      } catch (err) {
        const name = err instanceof DOMException ? err.name : '';
        setStreamError(
          name === 'NotAllowedError'
            ? 'Camera permission was denied.'
            : name === 'NotFoundError'
              ? 'No camera device was found.'
              : err instanceof Error
                ? err.message
                : 'The camera could not be started.',
        );
        return;
      }
    } else if (!videoRef.current?.src) {
      setStreamError('Upload a video file first.');
      return;
    } else {
      await videoRef.current.play().catch(() => undefined);
    }

    socketRef.current.connect();
    setStreaming(true);
    lastSendRef.current = 0;
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(streamLoop);
  }, [
    model,
    config.device,
    config.inputSize,
    config.confidence,
    config.iou,
    selectedClassIds,
    videoSource,
    cameraId,
    resolution,
    streamLoop,
  ]);

  function onPickVideo(file: File) {
    stopStreaming();
    if (mediaUrlRef.current) URL.revokeObjectURL(mediaUrlRef.current);
    const url = URL.createObjectURL(file);
    mediaUrlRef.current = url;
    setVideoSource('file');
    if (videoRef.current) {
      videoRef.current.srcObject = null;
      videoRef.current.src = url;
      videoRef.current.loop = true;
    }
  }

  useEffect(
    () => () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      if (streamRef.current) streamRef.current.getTracks().forEach((t) => t.stop());
      if (mediaUrlRef.current) URL.revokeObjectURL(mediaUrlRef.current);
    },
    [],
  );

  // Leaving the video modality must not leave the camera on.
  useEffect(() => {
    if (config.modality !== 'video') stopStreaming();
  }, [config.modality, stopStreaming]);

  /* ------------------------------- render ----------------------------- */
  const loading = models.loading || runtimes.loading;
  const loadError = models.error || runtimes.error;
  const canRun =
    Boolean(model) &&
    availability.available &&
    (config.modality === 'image' ? Boolean(imageFile) : text.trim().length > 0);

  return (
    <div>
      <PageHeader
        title="Playground"
        subtitle="Run a model on your own input and see exactly what happened: the output, the measured latency, and the pipeline that produced it."
        actions={
          model && (
            <div className="flex flex-wrap items-center gap-2">
              <ExecutionBadge target={config.execution} />
              <Badge tone={availability.available ? 'good' : 'warn'}>
                {availability.available ? 'ready' : 'unavailable'}
              </Badge>
            </div>
          )
        }
      />

      {loading && <Spinner label="Loading models and runtime probes…" />}
      {loadError && <ErrorState message={loadError} onRetry={models.reload} />}

      {!loading && !loadError && (
        <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_320px]">
          {/* ------------------------- input + output ------------------------ */}
          <div className="min-w-0 space-y-4">
            <div className="card card-pad">
              <div className="label mb-2">Input</div>
              <div className="flex gap-1 rounded-lg bg-elevated p-1 text-sm">
                {MODALITIES.map((m) => (
                  <button
                    key={m.key}
                    onClick={() => setConfig({ modality: m.key })}
                    className={`flex-1 rounded-md px-3 py-2 transition ${
                      config.modality === m.key
                        ? 'bg-accent text-accent-contrast'
                        : 'text-secondary hover:text-primary'
                    }`}
                  >
                    {m.label}
                  </button>
                ))}
              </div>

              {tasks.length === 0 && (
                <p className="mt-3 text-sm text-muted">
                  No installed model in this build consumes {config.modality} input.
                </p>
              )}

              {/* Image */}
              {config.modality === 'image' && (
                <div className="mt-3 space-y-3">
                  <div className="relative overflow-hidden rounded-lg border border-subtle bg-elevated">
                    {imageUrl ? (
                      <div className="relative">
                        <img
                          src={imageUrl}
                          alt="input"
                          className="mx-auto block max-h-[60vh] w-full object-contain"
                          onLoad={(e) => {
                            const img = e.currentTarget;
                            setImageSize({ w: img.naturalWidth, h: img.naturalHeight });
                          }}
                        />
                        <canvas
                          ref={overlayRef}
                          className="pointer-events-none absolute inset-0 h-full w-full object-contain"
                        />
                      </div>
                    ) : (
                      <label className="flex h-48 cursor-pointer flex-col items-center justify-center gap-2 text-muted sm:h-64">
                        <Icon name="grid" className="h-7 w-7" />
                        <span className="text-sm">Choose an image</span>
                        <input
                          type="file"
                          accept="image/*"
                          className="hidden"
                          onChange={(e) => e.target.files?.[0] && onPickImage(e.target.files[0])}
                        />
                      </label>
                    )}
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    <label className="btn-ghost cursor-pointer">
                      {imageUrl ? 'Replace image' : 'Upload image'}
                      <input
                        type="file"
                        accept="image/*"
                        className="hidden"
                        onChange={(e) => e.target.files?.[0] && onPickImage(e.target.files[0])}
                      />
                    </label>
                    <button className="btn-primary" disabled={!canRun || running} onClick={runSingleShot}>
                      {running ? 'Running…' : 'Run inference'}
                    </button>
                    {imageSize && (
                      <span className="font-mono text-2xs text-muted">
                        {imageSize.w} × {imageSize.h}
                      </span>
                    )}
                  </div>
                </div>
              )}

              {/* Text */}
              {config.modality === 'text' && (
                <div className="mt-3 space-y-3">
                  <textarea
                    className="input min-h-[8rem] resize-y font-normal"
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    placeholder="A man is riding a bicycle next to a red car."
                  />
                  <div className="flex flex-wrap items-center gap-2">
                    <button className="btn-primary" disabled={!canRun || running} onClick={runSingleShot}>
                      {running ? 'Running…' : 'Run inference'}
                    </button>
                    <span className="text-2xs text-muted">{text.trim().length} characters</span>
                  </div>
                </div>
              )}

              {/* Video */}
              {config.modality === 'video' && (
                <div className="mt-3 space-y-3">
                  <div className="relative aspect-video overflow-hidden rounded-lg border border-subtle bg-black">
                    <video
                      ref={videoRef}
                      className="absolute inset-0 h-full w-full object-contain"
                      playsInline
                      muted
                    />
                    <canvas
                      ref={videoOverlayRef}
                      className="pointer-events-none absolute inset-0 h-full w-full object-contain"
                    />
                    {!streaming && (
                      <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-muted">
                        <Icon name="camera" className="h-7 w-7" />
                        <span className="text-sm">Start to stream frames to the server</span>
                      </div>
                    )}
                  </div>

                  <div className="flex flex-wrap items-center gap-2">
                    {!streaming ? (
                      <button className="btn-primary" onClick={startStreaming} disabled={!server.available}>
                        <Icon name="live" className="h-4 w-4" /> Start
                      </button>
                    ) : (
                      <button className="btn-danger" onClick={stopStreaming}>
                        Stop
                      </button>
                    )}
                    <div className="flex gap-1 rounded-lg bg-elevated p-1 text-xs">
                      {(['camera', 'file'] as const).map((s) => (
                        <button
                          key={s}
                          onClick={() => {
                            stopStreaming();
                            setVideoSource(s);
                          }}
                          className={`rounded-md px-2.5 py-1.5 capitalize ${
                            videoSource === s
                              ? 'bg-accent text-accent-contrast'
                              : 'text-secondary hover:text-primary'
                          }`}
                        >
                          {s}
                        </button>
                      ))}
                    </div>
                    <label className="btn-ghost cursor-pointer">
                      Upload video
                      <input
                        type="file"
                        accept="video/*"
                        className="hidden"
                        onChange={(e) => e.target.files?.[0] && onPickVideo(e.target.files[0])}
                      />
                    </label>
                    {videoSource === 'camera' && cameras.length > 1 && (
                      <select
                        className="input max-w-[12rem]"
                        value={cameraId}
                        onChange={(e) => setCameraId(e.target.value)}
                      >
                        {cameras.map((c, i) => (
                          <option key={c.deviceId} value={c.deviceId}>
                            {c.label || `Camera ${i + 1}`}
                          </option>
                        ))}
                      </select>
                    )}
                    {videoSource === 'camera' && (
                      <select
                        className="input max-w-[10rem]"
                        value={resolution.label}
                        onChange={(e) =>
                          setResolution(
                            CAMERA_RESOLUTIONS.find((r) => r.label === e.target.value) ??
                              CAMERA_RESOLUTIONS[1],
                          )
                        }
                      >
                        {CAMERA_RESOLUTIONS.map((r) => (
                          <option key={r.label} value={r.label}>
                            {r.label}
                          </option>
                        ))}
                      </select>
                    )}
                  </div>

                  {streamError && (
                    <p className="rounded border border-bad/40 bg-bad-soft px-3 py-2 text-sm text-bad">
                      {streamError}
                    </p>
                  )}
                  {switchMessage && <p className="text-xs text-warn">{switchMessage}</p>}

                  {streaming && (
                    <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                      <Metric label="FPS" value={socket.fps.toFixed(0)} target="server" />
                      <Metric
                        label="Inference"
                        value={socket.inferenceMs ? `${socket.inferenceMs.toFixed(1)} ms` : '—'}
                        target="server"
                      />
                      <Metric label="Frames sent" value={String(processed)} target="server" />
                      <Metric
                        label="Dropped"
                        value={String(socket.droppedCount)}
                        target="server"
                      />
                    </div>
                  )}
                  <p className="text-2xs text-muted">
                    Streaming runs over the WebSocket path on the server. Frames are JPEG-encoded
                    in the browser at {STREAM_TARGET_FPS} fps; the server drops any frame
                    superseded before inference and reports the count.
                  </p>
                </div>
              )}
            </div>

            {runError && (
              <div className="card card-pad border-bad/30 bg-bad/5 text-sm text-bad">{runError}</div>
            )}

            {/* Output */}
            {trace && config.modality !== 'video' && (
              <ResultPanel trace={trace} onInspect={() => navigate('/pipeline')} />
            )}
          </div>

          {/* --------------------------- configuration ----------------------- */}
          <aside className="min-w-0 space-y-4">
            <div className="card card-pad space-y-3">
              <Field label="Task">
                <select
                  className="input"
                  value={config.task}
                  onChange={(e) => setConfig({ task: e.target.value })}
                >
                  {tasks.map((t) => (
                    <option key={t} value={t}>
                      {taskLabel(t)}
                    </option>
                  ))}
                </select>
              </Field>

              <Field label="Model" hint={model?.notes || undefined}>
                <select
                  className="input"
                  value={config.modelId}
                  onChange={(e) => {
                    const next = taskModels.find((m) => m.model_id === e.target.value);
                    setConfig({ modelId: e.target.value, inputSize: next?.input_size ?? null });
                  }}
                >
                  {taskModels.map((m) => (
                    <option key={m.model_id} value={m.model_id}>
                      {m.display_name}
                      {m.deployment_status === 'installed' ? '' : ' (not installed)'}
                    </option>
                  ))}
                </select>
              </Field>

              <div>
                <span className="label">Execution</span>
                <div className="mt-1.5 flex gap-1 rounded-lg bg-elevated p-1 text-sm">
                  {(['local', 'server'] as ExecutionTarget[]).map((target) => {
                    const info = target === 'server' ? server : local;
                    return (
                      <button
                        key={target}
                        onClick={() => setConfig({ execution: target })}
                        disabled={!info.available}
                        className={`flex-1 rounded-md px-3 py-1.5 capitalize transition disabled:cursor-not-allowed disabled:opacity-45 ${
                          config.execution === target
                            ? 'bg-accent text-accent-contrast'
                            : 'text-secondary hover:text-primary'
                        }`}
                      >
                        {target === 'local' ? 'Local device' : 'Server'}
                      </button>
                    );
                  })}
                </div>
                {!availability.available && availability.reason && (
                  <p className="mt-1.5 text-2xs leading-snug text-muted">{availability.reason}</p>
                )}
                {config.execution === 'server' && !local.available && local.reason && (
                  <p className="mt-1.5 text-2xs leading-snug text-muted">
                    Local unavailable — {local.reason}
                  </p>
                )}
              </div>

              {availability.available && (
                <>
                  <Field label="Runtime">
                    <select
                      className="input"
                      value={config.runtimeId}
                      onChange={(e) => setConfig({ runtimeId: e.target.value })}
                    >
                      {availability.runtimes.map((r) => (
                        <option key={r.runtime_id} value={r.runtime_id}>
                          {r.runtime_id}
                          {r.version ? ` ${r.version}` : ''}
                        </option>
                      ))}
                    </select>
                  </Field>

                  <div className="grid grid-cols-2 gap-3">
                    <Field label="Device">
                      <select
                        className="input"
                        value={config.device}
                        onChange={(e) => setConfig({ device: e.target.value })}
                      >
                        {(runtimeOption?.devices ?? []).map((d) => (
                          <option key={d} value={d}>
                            {d}
                          </option>
                        ))}
                      </select>
                    </Field>
                    <Field label="Precision">
                      <select
                        className="input"
                        value={config.precision}
                        onChange={(e) => setConfig({ precision: e.target.value })}
                      >
                        {precision.values.map((p) => (
                          <option key={p} value={p}>
                            {p}
                          </option>
                        ))}
                      </select>
                    </Field>
                  </div>

                  {precision.note && (
                    <p className="-mt-1 text-2xs leading-snug text-muted">{precision.note}</p>
                  )}

                  {config.task === 'object_detection' && (
                    <Field label="Input resolution">
                      <select
                        className="input"
                        value={String(config.inputSize ?? '')}
                        onChange={(e) => setConfig({ inputSize: Number(e.target.value) })}
                      >
                        {INPUT_SIZES.map((size) => (
                          <option key={size} value={size}>
                            {size} × {size}
                          </option>
                        ))}
                      </select>
                    </Field>
                  )}
                </>
              )}
            </div>

            <details className="card card-pad">
              <summary className="cursor-pointer text-sm font-medium text-primary">
                Advanced settings
              </summary>
              <div className="mt-3 space-y-3">
                {config.task === 'object_detection' && (
                  <>
                    <Field label={`Confidence — ${config.confidence.toFixed(2)}`}>
                      <input
                        type="range"
                        min={0.05}
                        max={0.95}
                        step={0.05}
                        className="w-full accent-accent"
                        value={config.confidence}
                        onChange={(e) => setConfig({ confidence: Number(e.target.value) })}
                      />
                    </Field>
                    <Field label={`IoU — ${config.iou.toFixed(2)}`}>
                      <input
                        type="range"
                        min={0.1}
                        max={0.9}
                        step={0.05}
                        className="w-full accent-accent"
                        value={config.iou}
                        onChange={(e) => setConfig({ iou: Number(e.target.value) })}
                      />
                    </Field>
                    <div>
                      <div className="label mb-2">
                        Classes — {selectedClassIds.length}/{allClasses.length}
                      </div>
                      <ClassPicker compact />
                    </div>
                  </>
                )}
                {config.task === 'image_classification' && (
                  <Field label={`Top-k — ${config.topK}`}>
                    <input
                      type="range"
                      min={1}
                      max={10}
                      step={1}
                      className="w-full accent-accent"
                      value={config.topK}
                      onChange={(e) => setConfig({ topK: Number(e.target.value) })}
                    />
                  </Field>
                )}
                {config.task === 'text_embedding' && (
                  <p className="text-xs text-muted">
                    This task has no tunable decoding parameters. Sequence length is decided by the
                    tokenizer and shown on the Pipeline page.
                  </p>
                )}
              </div>
            </details>
          </aside>
        </div>
      )}
    </div>
  );
}

/* --------------------------------------------------------------------- */

function Metric({
  label,
  value,
  target,
  sub,
}: {
  label: string;
  value: string;
  target: ExecutionTarget;
  sub?: string;
}) {
  return (
    <div className="rounded border border-subtle bg-panel px-3 py-2">
      <div className="flex items-center justify-between gap-2">
        <span className="label truncate">{label}</span>
        <ExecutionBadge target={target} />
      </div>
      <div className="mt-1 font-mono text-lg text-primary">{value}</div>
      {sub && <div className="text-2xs text-muted">{sub}</div>}
    </div>
  );
}

function ResultPanel({ trace, onInspect }: { trace: PlaygroundTrace; onInspect: () => void }) {
  const t = trace.timings;
  const detections = trace.result.detections ?? [];
  const classifications = trace.result.classifications ?? [];
  const embedding = trace.result.embedding;
  const network =
    trace.client_round_trip_ms !== undefined
      ? trace.client_round_trip_ms - t.server_total_ms
      : null;

  return (
    <div className="space-y-4">
      <div className="card card-pad">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <h2 className="text-sm font-semibold text-primary">Output</h2>
          <span className="font-mono text-2xs text-muted">
            {trace.model.display_name} · {trace.runtime.runtime_id}/{trace.runtime.device}/
            {trace.runtime.precision}
          </span>
        </div>

        {detections.length > 0 && (
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th scope="col">Class</th>
                  <th scope="col" className="text-right">Confidence</th>
                  <th scope="col" className="text-right">Box (x1, y1, x2, y2)</th>
                </tr>
              </thead>
              <tbody>
                {detections.map((d, i) => (
                  <tr key={`${d.className}-${i}`}>
                    <td className="text-primary">{d.className}</td>
                    <td className="num">{(d.confidence * 100).toFixed(1)}%</td>
                    <td className="num text-2xs">
                      {d.x1.toFixed(0)}, {d.y1.toFixed(0)}, {d.x2.toFixed(0)}, {d.y2.toFixed(0)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {trace.task === 'object_detection' && detections.length === 0 && (
          <p className="text-sm text-muted">
            No object passed the confidence threshold. Lower it under Advanced settings.
          </p>
        )}

        {classifications.length > 0 && (
          <ul className="space-y-2">
            {classifications.map((c) => (
              <li key={c.class_id}>
                <div className="flex items-baseline justify-between gap-3 text-sm">
                  <span className="truncate text-primary">{c.label}</span>
                  <span className="font-mono text-xs text-secondary">
                    {(c.probability * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="mt-1 h-1.5 rounded-sm bg-elevated">
                  <div
                    className="h-1.5 rounded-sm bg-accent"
                    style={{ width: `${Math.max(1, c.probability * 100)}%` }}
                  />
                </div>
              </li>
            ))}
          </ul>
        )}

        {embedding && (
          <div className="space-y-2 text-sm">
            <div className="font-mono text-primary">
              [1, {embedding.dimension}] float32 · L2 norm {embedding.norm?.toFixed(4) ?? '—'}
            </div>
            <div className="overflow-x-auto rounded border border-subtle bg-elevated p-2 font-mono text-2xs text-secondary">
              [{embedding.preview.map((v) => v.toFixed(4)).join(', ')}
              {embedding.dimension && embedding.dimension > embedding.preview.length ? ', …' : ''}]
            </div>
            {embedding.token_preview && (
              <div>
                <div className="label mb-1">Tokens — {embedding.tokens}</div>
                <div className="flex flex-wrap gap-1">
                  {embedding.token_preview.map((token, i) => (
                    <span
                      key={`${token}-${i}`}
                      className="rounded bg-elevated px-1.5 py-0.5 font-mono text-2xs text-secondary"
                    >
                      {token}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      <div className="card card-pad">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <h2 className="text-sm font-semibold text-primary">Performance</h2>
          <ExecutionBadge target={trace.execution} />
        </div>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Metric label="Inference" value={`${t.inference_ms.toFixed(2)} ms`} target={trace.execution} />
          <Metric
            label="Server total"
            value={`${t.server_total_ms.toFixed(2)} ms`}
            target={trace.execution}
            sub="decode → response"
          />
          <Metric
            label="End-to-end"
            value={
              trace.client_round_trip_ms !== undefined
                ? `${trace.client_round_trip_ms.toFixed(0)} ms`
                : '—'
            }
            target={trace.execution}
            sub="measured in the browser"
          />
          <Metric
            label="Throughput"
            value={`${(1000 / t.inference_ms).toFixed(1)} /s`}
            target={trace.execution}
            sub="1 / inference latency"
          />
        </div>
        <dl className="mt-3 divide-y divide-subtle text-sm">
          {[
            ['Decode', t.decode_ms === null ? 'not applicable' : `${t.decode_ms.toFixed(2)} ms`],
            ['Preprocess', `${t.preprocess_ms.toFixed(2)} ms`],
            ['Post-process', `${t.postprocess_ms.toFixed(2)} ms`],
            [
              'Network + client overhead',
              network === null ? 'unavailable' : `${network.toFixed(0)} ms`,
            ],
            [
              'Model load (this call)',
              t.model_load_ms === null ? 'session already warm' : `${t.model_load_ms.toFixed(0)} ms`,
            ],
            [
              'Process RSS',
              trace.memory.process_rss_mb === null
                ? 'unavailable'
                : formatMb(trace.memory.process_rss_mb),
            ],
          ].map(([term, value]) => (
            <div key={term} className="flex justify-between gap-3 py-1.5">
              <dt className="text-secondary">{term}</dt>
              <dd className="text-right font-mono text-xs text-primary">{value}</dd>
            </div>
          ))}
        </dl>
        <p className="mt-2 text-2xs text-muted">
          Network + client overhead is the browser round trip minus the server's own total. It
          includes upload, download and client parsing, and is not attributed to any single one.
        </p>
        <button className="btn-ghost mt-3 w-full" onClick={onInspect}>
          <Icon name="blueprint" className="h-4 w-4" /> Inspect pipeline
        </button>
      </div>
    </div>
  );
}
