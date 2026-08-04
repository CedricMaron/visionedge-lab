import { useEffect, useMemo, useState } from 'react';
import { api } from '@/services/api';
import { useAsync } from '@/hooks/useAsync';
import { PageHeader, Spinner, ErrorState, Badge } from '@/components/ui';
import { Icon } from '@/components/Icon';
import { formatMb } from '@/utils/format';
import { detectBrowserCaps, probeCameras, type BrowserCaps } from '@/utils/browserCaps';
import {
  detectDeviceClass,
  deviceClassLabel,
  type DeviceClassification,
} from '@/utils/deviceClass';
import type { Capabilities } from '@/types';

function CapRow({ label, ok, value }: { label: string; ok?: boolean; value?: string }) {
  return (
    <div className="flex items-center justify-between gap-3 border-b border-subtle py-2 last:border-0">
      <span className="text-sm text-secondary">{label}</span>
      <span className="flex items-center gap-2 text-sm">
        {value !== undefined && <span className="font-mono text-primary">{value}</span>}
        {ok !== undefined && (
          <span className={`pill ${ok ? 'bg-good/15 text-good' : 'bg-bad/15 text-bad'}`}>
            {ok ? 'available' : 'unavailable'}
          </span>
        )}
      </span>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="card card-pad">
      <h2 className="mb-2 text-sm font-semibold text-primary">{title}</h2>
      {children}
    </div>
  );
}

function DeviceClassCard({ classification }: { classification: DeviceClassification }) {
  const { deviceClass, confidence, evidence } = classification;
  const tone =
    deviceClass === 'unknown' ? 'bg-elevated text-muted' : 'bg-accent-soft text-accent';

  return (
    <div className="card card-pad">
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold text-primary">This device</h2>
          <p className="mt-0.5 text-xs text-muted">
            Classified from probed browser APIs. No user-agent string is parsed.
          </p>
        </div>
        <div className="text-right">
          <span className={`pill text-base ${tone}`}>{deviceClassLabel(deviceClass)}</span>
          <div className="mt-1 text-xs text-muted">
            {deviceClass === 'unknown'
              ? 'signals disagree or are unavailable'
              : `confidence ${(confidence * 100).toFixed(0)}%`}
          </div>
        </div>
      </div>

      <table className="data-table mt-3">
        <thead>
          <tr>
            <th scope="col">Signal</th>
            <th scope="col">Value</th>
            <th scope="col">Weight</th>
            <th scope="col">Vote</th>
          </tr>
        </thead>
        <tbody>
          {evidence.map((item) => (
            <tr key={item.signal} className={item.available ? undefined : 'opacity-60'}>
              <td className="text-xs">{item.signal}</td>
              <td className="font-mono text-xs">
                {item.available ? item.value : <span className="italic text-muted">unavailable</span>}
              </td>
              <td className="text-xs">{item.weight}</td>
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

      <p className="mt-2 text-xs text-muted">
        Signals that the browser does not expose cast no vote and are shown as
        unavailable rather than assumed.
      </p>
    </div>
  );
}

export default function DeviceCapabilitiesPage() {
  const { data, error, loading, reload } = useAsync<Capabilities>((s) => api.capabilities(s), []);
  const [browser, setBrowser] = useState<BrowserCaps | null>(null);
  const [deviceClass, setDeviceClass] = useState<DeviceClassification | null>(null);
  const [cameras, setCameras] = useState<{ deviceId: string; label: string }[]>([]);
  const [cameraErr, setCameraErr] = useState<string | null>(null);

  useEffect(() => {
    setBrowser(detectBrowserCaps());
    setDeviceClass(detectDeviceClass());
    probeCameras().then((r) => {
      setCameras(r.cameras);
      setCameraErr(r.error);
    });
  }, []);

  const rt = data?.runtimes;
  const providers = useMemo(() => rt?.onnxruntime_providers?.join(', ') || '—', [rt]);

  return (
    <div>
      <PageHeader
        title="Device Capabilities"
        subtitle="Backend hardware/runtime capabilities plus live client-side browser probes. No user-agent guessing — features are actually tested."
        actions={
          <button className="btn-ghost" onClick={reload}>
            <Icon name="refresh" className="h-4 w-4" /> Refresh
          </button>
        }
      />

      {deviceClass && (
        <div className="mb-4">
          <DeviceClassCard classification={deviceClass} />
        </div>
      )}

      <div className="grid gap-4 lg:grid-cols-2">
        {/* Backend */}
        <div className="space-y-4">
          <h2 className="label flex items-center gap-2">
            <Icon name="server" className="h-4 w-4" /> Backend (server)
          </h2>
          {loading && <Spinner label="Fetching /api/capabilities…" />}
          {error && <ErrorState message={error} onRetry={reload} />}
          {data && (
            <>
              <Section title="System">
                <CapRow label="OS" value={`${data.os} ${data.os_version}`} />
                <CapRow label="Python" value={data.python_version} />
                <CapRow label="CPU" value={data.cpu_model} />
                <CapRow
                  label="Cores (phys / logical)"
                  value={`${data.cpu_cores_physical} / ${data.cpu_cores_logical}`}
                />
                <CapRow label="RAM total" value={formatMb(data.ram_total_mb)} />
                <CapRow label="RAM available" value={formatMb(data.ram_available_mb)} />
              </Section>

              <Section title="GPU">
                <CapRow label="NVIDIA GPU present" ok={data.nvidia_gpu_present} />
                {data.gpus.length === 0 && <p className="py-1 text-sm text-muted">No GPUs reported.</p>}
                {data.gpus.map((g, i) => (
                  <div key={i} className="border-b border-subtle py-2 last:border-0">
                    <div className="text-sm text-primary">{g.name}</div>
                    <div className="mt-1 flex flex-wrap gap-2 text-xs text-muted">
                      <span>mem {formatMb(g.memory_used_mb)} / {formatMb(g.memory_total_mb)}</span>
                      <span>driver {g.driver_version}</span>
                    </div>
                  </div>
                ))}
              </Section>

              <Section title="Runtimes">
                {rt && (
                  <>
                    <CapRow label="ONNX Runtime" ok={rt.onnxruntime} />
                    <CapRow label="ORT CUDA" ok={rt.onnxruntime_cuda} />
                    <CapRow label="ORT providers" value={providers} />
                    <CapRow label="PyTorch" ok={rt.pytorch} />
                    <CapRow label="PyTorch CUDA" ok={rt.pytorch_cuda} />
                    <CapRow label="CUDA version" value={rt.cuda_version ?? '—'} />
                    <CapRow label="OpenVINO" ok={rt.openvino} />
                    <CapRow label="TensorRT" ok={rt.tensorrt} />
                  </>
                )}
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {data.supported_precisions.map((p) => (
                    <Badge key={p} tone="accent">
                      {p}
                    </Badge>
                  ))}
                </div>
              </Section>
            </>
          )}
        </div>

        {/* Browser */}
        <div className="space-y-4">
          <h2 className="label flex items-center gap-2">
            <Icon name="chip" className="h-4 w-4" /> Browser (this device)
          </h2>
          {browser && (
            <>
              <Section title="Compute">
                <CapRow
                  label="Logical CPU cores"
                  value={browser.hardwareConcurrency?.toString() ?? 'unknown'}
                />
                <CapRow
                  label="Device memory"
                  value={browser.deviceMemoryGb ? `${browser.deviceMemoryGb} GB` : 'unknown'}
                />
                <CapRow label="WebAssembly" ok={browser.webAssembly} />
                <CapRow label="WASM SIMD" ok={browser.wasmSimd} />
                <CapRow label="Web Workers" ok={browser.webWorkers} />
                <CapRow label="OffscreenCanvas" ok={browser.offscreenCanvas} />
              </Section>

              <Section title="Graphics & acceleration">
                <CapRow label="WebGPU (navigator.gpu)" ok={browser.webGpu} />
                <CapRow label="WebGL2" ok={browser.webGl2} />
              </Section>

              <Section title="Media & network">
                <CapRow label="Camera (getUserMedia)" ok={browser.mediaDevices} />
                <CapRow label="Secure context" ok={browser.secureContext} />
                <CapRow
                  label="Cameras detected"
                  value={cameras.length > 0 ? String(cameras.length) : cameraErr ? 'blocked' : '0 (grant to see labels)'}
                />
                {browser.network ? (
                  <>
                    <CapRow label="Network type" value={browser.network.effectiveType ?? '—'} />
                    <CapRow
                      label="Downlink"
                      value={browser.network.downlinkMbps ? `${browser.network.downlinkMbps} Mbps` : '—'}
                    />
                    <CapRow label="RTT" value={browser.network.rtt ? `${browser.network.rtt} ms` : '—'} />
                  </>
                ) : (
                  <CapRow label="Network Information API" ok={false} />
                )}
              </Section>

              {cameras.length > 0 && (
                <Section title="Cameras">
                  {cameras.map((c) => (
                    <div key={c.deviceId} className="py-1 text-sm text-secondary">
                      {c.label}
                    </div>
                  ))}
                </Section>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
