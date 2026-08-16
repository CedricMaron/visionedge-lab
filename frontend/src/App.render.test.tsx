/**
 * The app must render — at the entry route and at a route reached by refresh.
 *
 * This exists because it did not. `resolveConfig` returns the API's snake_case
 * (`runtime_id`) and the store holds camelCase (`runtimeId`); spreading one into
 * the other wrote a stray key, left `runtimeId` empty, and the effect that clamps
 * the configuration re-fired forever until React tore the tree down. The page went
 * white with no failing test anywhere, because TypeScript only excess-checks object
 * literals and nothing ever mounted this page in a test.
 *
 * So the assertion is deliberately blunt: mount the real App against realistic
 * payloads and require that React logged nothing. "Maximum update depth exceeded"
 * arrives as a console error, which is exactly what a white page looks like from
 * here.
 */
import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from './App';

const model = (over: Record<string, unknown>) => ({
  model_id: 'yolov8n-onnx',
  display_name: 'YOLOv8 Nano (ONNX)',
  family: 'yolov8',
  task: 'object_detection',
  modality: 'image',
  adapter: 'yolov8',
  source_repository: null,
  paper_url: null,
  model_license: 'AGPL-3.0',
  weights_license: 'AGPL-3.0',
  commercial_use_permitted: false,
  parameters_millions: 3.2,
  file_size_bytes: 12851139,
  local_path: 'models/yolov8n.onnx',
  companion_files: [],
  input_size: 640,
  supported_runtimes: ['onnxruntime'],
  supported_devices: ['cpu', 'cuda'],
  supported_precisions: ['fp32', 'int8'],
  deployment_status: 'installed',
  not_installed_reason: null,
  install_hint: null,
  notes: '',
  ...over,
});

// Mirrors the production probe: ONNX Runtime lists a CUDA provider even on a
// machine with no GPU, which is why the UI cross-checks the host's GPU count.
const PAYLOADS: Record<string, unknown> = {
  '/health': { status: 'ok', detection_health: 'ready', warnings: [] },
  '/api/lab/models': {
    models: [
      model({}),
      model({
        model_id: 'all-minilm-l6-v2-onnx',
        display_name: 'all-MiniLM-L6-v2 (ONNX)',
        family: 'minilm',
        task: 'text_embedding',
        modality: 'text',
        adapter: 'minilm',
        input_size: null,
        local_path: 'models/embedding/all-MiniLM-L6-v2.onnx',
      }),
    ],
  },
  '/api/lab/runtimes': {
    runtimes: [
      {
        runtime_id: 'onnxruntime',
        available: true,
        unavailable_reason: null,
        version: '1.19.2',
        execution_providers: ['CPUExecutionProvider'],
        devices: ['cpu', 'cuda'],
        precisions_by_device: { cpu: ['fp32', 'int8'], cuda: ['fp32', 'fp16', 'int8'] },
        supports_device_synchronization: true,
        supports_profiling: true,
        notes: [],
      },
    ],
  },
  '/api/capabilities': {
    os: 'Windows',
    os_version: '10',
    python_version: '3.12.0',
    cpu_model: 'CPU',
    cpu_cores_physical: 2,
    cpu_cores_logical: 4,
    ram_total_mb: 8192,
    ram_available_mb: 4096,
    gpus: [],
    nvidia_gpu_present: false,
    runtimes: {
      onnxruntime: true,
      onnxruntime_providers: ['CPUExecutionProvider'],
      onnxruntime_cuda: false,
      pytorch: false,
      pytorch_cuda: false,
      cuda_version: null,
      openvino: false,
      tensorrt: false,
    },
    supported_precisions: ['fp32'],
  },
  '/api/classes': { classes: [{ id: 0, name: 'person' }], groups: { people: ['person'] } },
  '/api/lab/scenarios': { scenarios: [] },
  '/api/lab/runs': { runs: [] },
  '/api/detection/status': {
    config: null,
    health: 'ready',
    events: [],
    metrics: {
      processed_fps: 0,
      inference_latency_p50_ms: 0,
      inference_latency_p95_ms: 0,
      inference_latency_p99_ms: 0,
      end_to_end_p50_ms: 0,
      processed_frames: 0,
      dropped_frames: 0,
    },
  },
  '/api/lab/system': {
    hardware: {
      cpu_model: 'CPU',
      cpu_cores_physical: 2,
      cpu_cores_logical: 4,
      cpu_instruction_sets: [],
      cpu_max_freq_mhz: null,
      ram_total_mb: 8192,
      gpus: [],
      gpu_count: 0,
      cuda_version: null,
      cudnn_version: null,
      nvml_available: false,
    },
    software: {
      os: 'Windows',
      os_version: '10',
      kernel_version: null,
      python_version: '3.12.0',
      package_versions: {},
      relevant_env_vars: {},
    },
    runtimes: [],
  },
};

let consoleErrors: string[];

beforeEach(() => {
  consoleErrors = [];
  vi.spyOn(console, 'error').mockImplementation((...args) => {
    const message = args.map(String).join(' ');
    // `act(...)` warnings are an artefact of driving async effects from a test,
    // not a defect in the page. Everything else counts — "Maximum update depth
    // exceeded", the error that blanked the site, arrives through here.
    if (message.includes('not wrapped in act(')) return;
    // jsdom ships no canvas implementation. The overlay already treats a null
    // 2D context as "nothing to draw", so this is the environment talking, not
    // the page.
    if (message.includes('HTMLCanvasElement.prototype.getContext')) return;
    consoleErrors.push(message);
  });
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const { pathname } = new URL(String(input), 'http://localhost');
      const body = PAYLOADS[pathname];
      return new Response(JSON.stringify(body ?? { detail: 'not found' }), {
        status: body ? 200 : 404,
        headers: { 'content-type': 'application/json' },
      });
    }),
  );
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

async function mountAt(path: string) {
  render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  );
  // Long enough for the three fetches to resolve and every clamping effect to
  // settle. A runaway effect exceeds React's nesting limit well inside this.
  await new Promise((resolve) => setTimeout(resolve, 1200));
}

describe('the application renders', () => {
  it('mounts the Playground at the entry route without a runaway effect', async () => {
    await mountAt('/');

    await waitFor(() => expect(screen.getAllByText('Playground').length).toBeGreaterThan(0));
    expect(screen.getByText('Run inference')).toBeTruthy();
    expect(consoleErrors).toEqual([]);
  });

  it('mounts a nested route directly, as a refresh does', async () => {
    await mountAt('/pipeline');

    await waitFor(() => expect(screen.getAllByText('Pipeline').length).toBeGreaterThan(0));
    expect(consoleErrors).toEqual([]);
  });

  it('still renders when every API call fails', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('{}', { status: 500 })));

    await mountAt('/');

    // A dead backend is an error state, not a white page.
    await waitFor(() => expect(screen.getAllByText('Playground').length).toBeGreaterThan(0));
    expect(consoleErrors).toEqual([]);
  });
});
