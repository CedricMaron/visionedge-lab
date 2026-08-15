/**
 * Environment — this device, the server, and which runtimes exist on each.
 *
 * The split is the point. "GPU" means something different in a browser tab than it
 * does in the server process, and merging the two into one hardware panel is how
 * capability pages end up claiming things neither side can prove. Browser facts come
 * from feature probes (never from a user-agent string); server facts come from the
 * same probes the benchmark engine records into every run.
 */
import { useEffect, useState } from 'react';
import { useAsync } from '@/hooks/useAsync';
import { labApi } from '@/services/labApi';
import { ErrorState, PageHeader, Spinner } from '@/components/ui';
import { Icon } from '@/components/Icon';
import { localRuntimeStatuses } from '@/lab/catalog';
import { detectBrowserCaps, probeCameras, type BrowserCaps } from '@/utils/browserCaps';
import {
  detectDeviceClass,
  deviceClassLabel,
  type DeviceClassification,
} from '@/utils/deviceClass';
import type { BenchmarkRun, RuntimeCapability } from '@/types/lab';

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-4 border-b border-subtle py-2 last:border-0">
      <span className="text-sm text-secondary">{label}</span>
      <span className="text-right font-mono text-sm text-primary">{value}</span>
    </div>
  );
}

function StatePill({ state }: { state: 'available' | 'unavailable' | 'unknown' }) {
  const tone =
    state === 'available'
      ? 'bg-good-soft text-good'
      : state === 'unknown'
        ? 'bg-elevated text-muted'
        : 'bg-elevated text-muted';
  return <span className={`pill ${tone}`}>{state}</span>;
}

export default function EnvironmentPage() {
  const system = useAsync<{
    hardware: BenchmarkRun['hardware'];
    software: BenchmarkRun['software'];
    runtimes: RuntimeCapability[];
  }>((s) => labApi.system(s), []);

  const [caps, setCaps] = useState<BrowserCaps | null>(null);
  const [deviceClass, setDeviceClass] = useState<DeviceClassification | null>(null);
  const [cameras, setCameras] = useState<number | null>(null);

  useEffect(() => {
    setCaps(detectBrowserCaps());
    setDeviceClass(detectDeviceClass());
    probeCameras().then((r) => setCameras(r.error ? null : r.cameras.length));
  }, []);

  const localRuntimes = localRuntimeStatuses(caps);
  const serverRuntimes = system.data?.runtimes ?? [];

  return (
    <div>
      <PageHeader
        title="Environment"
        subtitle="What your device can do, what the server can do, and which runtimes exist on each. Every row is a probe result."
        actions={
          <button className="btn-ghost" onClick={system.reload}>
            <Icon name="refresh" className="h-4 w-4" /> Refresh
          </button>
        }
      />

      <div className="grid gap-4 lg:grid-cols-2">
        {/* ------------------------------ this device ------------------------ */}
        <section className="card card-pad">
          <div className="mb-3 flex items-baseline justify-between gap-3">
            <h2 className="text-sm font-semibold text-primary">This device</h2>
            {deviceClass && (
              <span className="pill bg-accent-soft text-accent">
                {deviceClassLabel(deviceClass.deviceClass)}
                {deviceClass.deviceClass !== 'unknown' &&
                  ` · ${(deviceClass.confidence * 100).toFixed(0)}%`}
              </span>
            )}
          </div>

          {caps ? (
            <>
              <Row
                label="Logical CPU cores"
                value={caps.hardwareConcurrency ?? 'not exposed by browser'}
              />
              <Row
                label="Device memory"
                value={
                  caps.deviceMemoryGb !== null
                    ? `≈ ${caps.deviceMemoryGb} GB (coarse)`
                    : 'not exposed by browser'
                }
              />
              <Row label="GPU model" value="not exposed by browser" />
              <Row label="WebGPU" value={caps.webGpu ? 'available' : 'unavailable'} />
              <Row
                label="WebNN"
                value={
                  typeof navigator !== 'undefined' && 'ml' in navigator
                    ? 'available'
                    : 'unavailable'
                }
              />
              <Row label="WebAssembly SIMD" value={caps.wasmSimd ? 'available' : 'unavailable'} />
              <Row label="WebGL 2" value={caps.webGl2 ? 'available' : 'unavailable'} />
              <Row label="Web Workers" value={caps.webWorkers ? 'available' : 'unavailable'} />
              <Row
                label="OffscreenCanvas"
                value={caps.offscreenCanvas ? 'available' : 'unavailable'}
              />
              <Row label="Secure context" value={caps.secureContext ? 'yes' : 'no'} />
              <Row label="Camera API" value={caps.mediaDevices ? 'available' : 'unavailable'} />
              <Row
                label="Cameras enumerated"
                value={cameras === null ? 'unknown' : String(cameras)}
              />
              <Row
                label="Network (effective type)"
                value={caps.network?.effectiveType ?? 'not exposed by browser'}
              />
              <p className="mt-3 text-xs text-muted">
                Device class is voted on probed APIs — pointer type, touch points, screen size,
                cores. No user-agent string is parsed, and a signal the browser does not expose
                casts no vote.
              </p>
            </>
          ) : (
            <Spinner label="Probing browser capabilities…" />
          )}
        </section>

        {/* -------------------------------- server --------------------------- */}
        <section className="card card-pad">
          <h2 className="mb-3 text-sm font-semibold text-primary">Server</h2>
          {system.loading && <Spinner label="Probing the server…" />}
          {system.error && <ErrorState message={system.error} onRetry={system.reload} />}
          {system.data && (
            <>
              <Row
                label="OS"
                value={`${system.data.software.os} ${system.data.software.os_version}`}
              />
              <Row label="Kernel" value={system.data.software.kernel_version ?? 'unknown'} />
              <Row label="Python" value={system.data.software.python_version} />
              <Row label="CPU" value={system.data.hardware.cpu_model} />
              <Row
                label="Cores (physical / logical)"
                value={`${system.data.hardware.cpu_cores_physical ?? '?'} / ${system.data.hardware.cpu_cores_logical}`}
              />
              <Row
                label="Instruction sets"
                value={system.data.hardware.cpu_instruction_sets.join(' ') || 'undetected'}
              />
              <Row
                label="RAM"
                value={`${(system.data.hardware.ram_total_mb / 1024).toFixed(1)} GB`}
              />
              <Row
                label="GPU"
                value={
                  system.data.hardware.gpus.length === 0
                    ? 'none detected'
                    : system.data.hardware.gpus.map((g) => g.name).join(', ')
                }
              />
              <Row
                label="VRAM"
                value={
                  system.data.hardware.gpus.length === 0
                    ? 'unavailable — no GPU'
                    : system.data.hardware.gpus
                        .map((g) => (g.memory_total_mb ? `${g.memory_total_mb} MB` : 'unknown'))
                        .join(', ')
                }
              />
              <Row label="CUDA" value={system.data.hardware.cuda_version ?? 'not present'} />
              <Row label="cuDNN" value={system.data.hardware.cudnn_version ?? 'not present'} />
              <Row
                label="NVML (power / thermal)"
                value={system.data.hardware.nvml_available ? 'available' : 'unavailable'}
              />
              <div className="mt-3">
                <div className="label mb-1">Execution-relevant packages</div>
                <div className="max-h-40 overflow-y-auto rounded border border-subtle bg-elevated p-2">
                  {Object.entries(system.data.software.package_versions).map(([pkg, version]) => (
                    <div key={pkg} className="flex justify-between gap-3 font-mono text-2xs">
                      <span className="text-secondary">{pkg}</span>
                      <span className="text-primary">{version}</span>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}
        </section>
      </div>

      {/* -------------------------------- runtimes --------------------------- */}
      <section className="card card-pad mt-4">
        <h2 className="mb-1 text-sm font-semibold text-primary">Runtimes</h2>
        <p className="mb-3 text-xs text-muted">
          A runtime is listed as available only where its probe succeeded. "—" means the runtime
          does not exist on that side at all, not that it failed.
        </p>
        <div className="overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th scope="col">Runtime</th>
                <th scope="col" className="text-center">Local</th>
                <th scope="col" className="text-center">Server</th>
                <th scope="col">Status</th>
              </tr>
            </thead>
            <tbody>
              {localRuntimes.map((runtime) => (
                <tr key={`local-${runtime.runtime_id}`}>
                  <th scope="row" className="px-3 py-2 text-left font-mono text-xs text-primary">
                    {runtime.label}
                  </th>
                  <td className="text-center">
                    <StatePill state={runtime.status} />
                  </td>
                  <td className="text-center text-muted">—</td>
                  <td className="text-xs text-muted">{runtime.detail}</td>
                </tr>
              ))}
              {serverRuntimes.map((runtime) => (
                <tr key={`server-${runtime.runtime_id}`}>
                  <th scope="row" className="px-3 py-2 text-left font-mono text-xs text-primary">
                    {runtime.runtime_id}
                  </th>
                  <td className="text-center text-muted">—</td>
                  <td className="text-center">
                    <StatePill state={runtime.available ? 'available' : 'unavailable'} />
                  </td>
                  <td className="text-xs text-muted">
                    {runtime.available
                      ? `${runtime.version ? `v${runtime.version} · ` : ''}${runtime.devices.join(', ')}${
                          runtime.execution_providers.length
                            ? ` · ${runtime.execution_providers.join(', ')}`
                            : ''
                        }`
                      : (runtime.unavailable_reason ?? 'unavailable')}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {deviceClass && (
        <section className="card card-pad mt-4">
          <h2 className="mb-3 text-sm font-semibold text-primary">Device classification evidence</h2>
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th scope="col">Signal</th>
                  <th scope="col">Value</th>
                  <th scope="col" className="text-right">Weight</th>
                  <th scope="col">Vote</th>
                </tr>
              </thead>
              <tbody>
                {deviceClass.evidence.map((item) => (
                  <tr key={item.signal} className={item.available ? undefined : 'opacity-60'}>
                    <td className="text-xs">{item.signal}</td>
                    <td className="font-mono text-xs">
                      {item.available ? item.value : <span className="italic text-muted">unavailable</span>}
                    </td>
                    <td className="num text-xs">{item.weight}</td>
                    <td className="text-xs">
                      {item.vote === 'none' ? (
                        <span className="text-muted">—</span>
                      ) : (
                        <span className={item.vote === 'phone' ? 'text-accent' : 'text-secondary'}>
                          {item.vote === 'phone' ? 'phone' : 'PC'}
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
}
