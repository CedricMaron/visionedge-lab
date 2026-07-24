// Client-side capability probes. We do NOT rely on user-agent parsing.

export interface BrowserCaps {
  hardwareConcurrency: number | null;
  deviceMemoryGb: number | null;
  webAssembly: boolean;
  wasmSimd: boolean;
  webWorkers: boolean;
  webGpu: boolean;
  webGl2: boolean;
  offscreenCanvas: boolean;
  mediaDevices: boolean;
  secureContext: boolean;
  network: { effectiveType?: string; downlinkMbps?: number; rtt?: number } | null;
}

// Probe WASM SIMD support by compiling a tiny module that uses v128.
function detectWasmSimd(): boolean {
  try {
    // Minimal wasm module header + a v128.const-using body.
    const bytes = new Uint8Array([
      0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253,
      15, 253, 98, 11,
    ]);
    return WebAssembly.validate(bytes);
  } catch {
    return false;
  }
}

function detectWebGl2(): boolean {
  try {
    const canvas = document.createElement('canvas');
    return !!canvas.getContext('webgl2');
  } catch {
    return false;
  }
}

export function detectBrowserCaps(): BrowserCaps {
  const nav = navigator as Navigator & {
    deviceMemory?: number;
    gpu?: unknown;
    connection?: { effectiveType?: string; downlink?: number; rtt?: number };
  };

  return {
    hardwareConcurrency: typeof nav.hardwareConcurrency === 'number' ? nav.hardwareConcurrency : null,
    deviceMemoryGb: typeof nav.deviceMemory === 'number' ? nav.deviceMemory : null,
    webAssembly: typeof WebAssembly === 'object',
    wasmSimd: detectWasmSimd(),
    webWorkers: typeof Worker !== 'undefined',
    webGpu: 'gpu' in navigator && nav.gpu != null,
    webGl2: detectWebGl2(),
    offscreenCanvas: typeof OffscreenCanvas !== 'undefined',
    mediaDevices: !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia),
    secureContext: typeof isSecureContext !== 'undefined' ? isSecureContext : false,
    network: nav.connection
      ? {
          effectiveType: nav.connection.effectiveType,
          downlinkMbps: nav.connection.downlink,
          rtt: nav.connection.rtt,
        }
      : null,
  };
}

export interface CameraProbe {
  cameras: { deviceId: string; label: string }[];
  error: string | null;
}

// Enumerate cameras (labels require a prior getUserMedia grant).
export async function probeCameras(): Promise<CameraProbe> {
  if (!navigator.mediaDevices?.enumerateDevices) {
    return { cameras: [], error: 'enumerateDevices not supported' };
  }
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cameras = devices
      .filter((d) => d.kind === 'videoinput')
      .map((d, i) => ({ deviceId: d.deviceId, label: d.label || `Camera ${i + 1}` }));
    return { cameras, error: null };
  } catch (err) {
    return { cameras: [], error: err instanceof Error ? err.message : 'enumeration failed' };
  }
}
