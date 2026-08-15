/**
 * Modality → Task → Model → Execution target → Runtime.
 *
 * The Playground, the Model library and the Environment page all ask the same
 * question in different words: what can this build actually run, and where? This
 * module answers it once, from the server's model registry and runtime probes plus
 * the browser's own capability probes, so no page hardcodes a model name or a
 * runtime list.
 *
 * Nothing here declares a capability it did not read from a probe. A combination
 * that cannot run is not silently hidden either — it carries the reason, so the UI
 * can explain rather than grey out.
 */
import type { LabModel, RuntimeCapability } from '@/types/lab';
import type { BrowserCaps } from '@/utils/browserCaps';

export type Modality = 'image' | 'video' | 'text';

export const MODALITIES: { key: Modality; label: string }[] = [
  { key: 'image', label: 'Image' },
  { key: 'video', label: 'Video' },
  { key: 'text', label: 'Text' },
];

/** One item of input. Multiple items are allowed so image+text tasks can be added
 *  without reshaping every caller around a single `inputType`. */
export interface InputItem {
  modality: Modality;
  /** Encoded bytes for image/video frames; the string itself for text. */
  data: Blob | string | null;
  label?: string;
}

export const TASK_LABELS: Record<string, string> = {
  object_detection: 'Object detection',
  image_classification: 'Image classification',
  image_captioning: 'Image captioning',
  vision_language: 'Visual question answering',
  text_embedding: 'Text embedding',
  image_segmentation: 'Image segmentation',
  video_understanding: 'Video understanding',
};

export function taskLabel(task: string): string {
  return TASK_LABELS[task] ?? task.replace(/_/g, ' ');
}

/** Which input modalities a model's task can be driven from in this build.
 *  Video is image-model territory only for tasks we can run frame by frame. */
export function modalitiesForModel(model: LabModel): Modality[] {
  if (model.modality === 'text') return ['text'];
  if (model.modality === 'image') {
    return model.task === 'object_detection' ? ['image', 'video'] : ['image'];
  }
  return [];
}

export function modelsForModality(models: LabModel[], modality: Modality): LabModel[] {
  return models.filter((m) => modalitiesForModel(m).includes(modality));
}

export function tasksForModality(models: LabModel[], modality: Modality): string[] {
  return [...new Set(modelsForModality(models, modality).map((m) => m.task))];
}

export function modelsForTask(models: LabModel[], modality: Modality, task: string): LabModel[] {
  return modelsForModality(models, modality).filter((m) => m.task === task);
}

/* ------------------------------------------------------------------ */
/* Execution targets                                                   */
/* ------------------------------------------------------------------ */

export interface RuntimeOption {
  runtime_id: string;
  version: string | null;
  devices: string[];
  precisions_by_device: Record<string, string[]>;
  execution_providers: string[];
}

export interface ExecutionAvailability {
  available: boolean;
  reason: string | null;
  runtimes: RuntimeOption[];
}

/**
 * Server execution for one model: the intersection of what the model declares it
 * supports and what the server's runtime probes actually reported.
 *
 * `serverGpuCount` refines that intersection. ONNX Runtime lists
 * CUDAExecutionProvider whenever the GPU build is installed, whether or not a GPU
 * exists — so a machine with zero detected GPUs must not be offered a CUDA device,
 * which would fail only after the session was created.
 */
export function serverAvailability(
  model: LabModel | undefined,
  runtimes: RuntimeCapability[],
  serverGpuCount?: number | null,
): ExecutionAvailability {
  if (!model) return { available: false, reason: 'no model selected', runtimes: [] };
  if (model.deployment_status !== 'installed') {
    return {
      available: false,
      reason: model.not_installed_reason ?? 'the model weights are not installed on the server',
      runtimes: [],
    };
  }
  const gpuAbsent = serverGpuCount !== undefined && serverGpuCount !== null && serverGpuCount === 0;
  const usable = runtimes
    .filter((r) => r.available && model.supported_runtimes.includes(r.runtime_id))
    .map<RuntimeOption>((r) => ({
      runtime_id: r.runtime_id,
      version: r.version,
      devices: r.devices.filter(
        (d) => model.supported_devices.includes(d) && !(gpuAbsent && d !== 'cpu'),
      ),
      precisions_by_device: r.precisions_by_device,
      execution_providers: r.execution_providers,
    }))
    .filter((r) => r.devices.length > 0);

  if (usable.length === 0) {
    return {
      available: false,
      reason:
        'no runtime this model supports is available on the server — see Environment for each runtime’s probe result',
      runtimes: [],
    };
  }
  return { available: true, reason: null, runtimes: usable };
}

/**
 * Local (in-browser) execution.
 *
 * The browser capabilities are probed for real, but this build ships no in-browser
 * model weights and no ONNX Runtime Web bundle, so no model can execute locally
 * yet. That is reported as the reason rather than hidden: a visitor on a WebGPU
 * machine should see that their device is capable and that the *build* is the
 * limitation.
 */
export function localAvailability(
  model: LabModel | undefined,
  caps: BrowserCaps | null,
): ExecutionAvailability {
  if (!caps) {
    return { available: false, reason: 'browser capabilities have not been probed yet', runtimes: [] };
  }
  const engines: string[] = [];
  if (caps.webGpu) engines.push('WebGPU');
  if (caps.wasmSimd) engines.push('WASM SIMD');
  const detected = engines.length ? engines.join(' and ') : 'no accelerated compute API';
  return {
    available: false,
    reason:
      `this build ships no in-browser model weights or ONNX Runtime Web bundle, so ${
        model ? model.display_name : 'this model'
      } cannot execute on your device. Your browser exposes ${detected}.`,
    runtimes: [],
  };
}

/** Local device runtimes the browser actually exposes, for the Environment table. */
export interface LocalRuntimeStatus {
  runtime_id: string;
  label: string;
  status: 'available' | 'unavailable' | 'unknown';
  detail: string;
}

export function localRuntimeStatuses(caps: BrowserCaps | null): LocalRuntimeStatus[] {
  if (!caps) return [];
  const webnn = typeof navigator !== 'undefined' && 'ml' in navigator;
  return [
    {
      runtime_id: 'webgpu',
      label: 'WebGPU',
      status: caps.webGpu ? 'available' : 'unavailable',
      detail: caps.webGpu
        ? 'navigator.gpu is present. The adapter name is not exposed by the browser.'
        : 'navigator.gpu is not present in this browser/context.',
    },
    {
      runtime_id: 'webnn',
      label: 'WebNN',
      status: webnn ? 'available' : 'unavailable',
      detail: webnn ? 'navigator.ml is present.' : 'navigator.ml is not present in this browser.',
    },
    {
      runtime_id: 'wasm',
      label: 'WebAssembly (SIMD)',
      status: caps.wasmSimd ? 'available' : caps.webAssembly ? 'unavailable' : 'unavailable',
      detail: caps.wasmSimd
        ? 'A v128 module validated, so SIMD kernels would run.'
        : 'SIMD probe failed; only scalar WASM would be available.',
    },
    {
      runtime_id: 'onnxruntime-web',
      label: 'ONNX Runtime Web',
      status: 'unavailable',
      detail: 'not bundled in this frontend build, so no model can execute in the browser.',
    },
    {
      runtime_id: 'webgl2',
      label: 'WebGL 2',
      status: caps.webGl2 ? 'available' : 'unavailable',
      detail: caps.webGl2 ? 'A webgl2 context was created.' : 'No webgl2 context could be created.',
    },
  ];
}

/* ------------------------------------------------------------------ */
/* Configuration resolution                                            */
/* ------------------------------------------------------------------ */

export interface ResolvedConfig {
  runtime_id: string;
  device: string;
  precision: string;
}

/** Clamp a desired configuration to one the chosen execution target can run. */
export function resolveConfig(
  availability: ExecutionAvailability,
  desired: Partial<ResolvedConfig>,
): ResolvedConfig | null {
  const runtime =
    availability.runtimes.find((r) => r.runtime_id === desired.runtime_id) ??
    availability.runtimes[0];
  if (!runtime) return null;
  const device = runtime.devices.includes(desired.device ?? '')
    ? (desired.device as string)
    : runtime.devices[0];
  const precisions = runtime.precisions_by_device[device] ?? [];
  const precision = precisions.includes(desired.precision ?? '')
    ? (desired.precision as string)
    : (precisions[0] ?? 'fp32');
  return { runtime_id: runtime.runtime_id, device, precision };
}

export function precisionsFor(
  availability: ExecutionAvailability,
  runtimeId: string,
  device: string,
): string[] {
  const runtime = availability.runtimes.find((r) => r.runtime_id === runtimeId);
  return runtime?.precisions_by_device[device] ?? [];
}

/**
 * Precisions worth offering for an actual run.
 *
 * A runtime that supports int8 does not make an fp32 export an int8 model: ONNX
 * Runtime records the precision it was asked for without verifying it, so offering
 * int8 against an fp32 artefact would relabel the same graph rather than quantize
 * it. Only precisions the installed artefact can genuinely be run at are offered,
 * and the note says why the list is shorter than the runtime's.
 */
export function offeredPrecisions(
  model: LabModel | undefined,
  availability: ExecutionAvailability,
  runtimeId: string,
  device: string,
): { values: string[]; note: string | null } {
  const supported = precisionsFor(availability, runtimeId, device).filter(
    (p) => !model || model.supported_precisions.includes(p),
  );
  const quantizedArtefact = /int8|int4|fp16|quant/i.test(model?.local_path ?? model?.model_id ?? '');
  if (quantizedArtefact || supported.length <= 1) {
    return { values: supported.length ? supported : ['fp32'], note: null };
  }
  return {
    values: supported.filter((p) => p === 'fp32'),
    note: 'The installed artefact is an FP32 export. Running it at another precision would relabel the same graph, not quantize it — install a quantized export to measure one.',
  };
}

/** Input resolutions offered for models with a fixed square input. */
export const INPUT_SIZES = [320, 416, 512, 640];
