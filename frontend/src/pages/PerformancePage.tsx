import { useEffect, useRef, useState } from 'react';
import { api } from '@/services/api';
import { PageHeader, Badge } from '@/components/ui';
import { StatCard } from '@/components/StatCard';
import { MetricChart, type ChartPoint } from '@/components/MetricChart';
import { formatMb } from '@/utils/format';
import type { Capabilities, DetectionStatus } from '@/types';

const MAX_POINTS = 40;
const POLL_MS = 2000;

export default function PerformancePage() {
  const [points, setPoints] = useState<ChartPoint[]>([]);
  const [latest, setLatest] = useState<DetectionStatus | null>(null);
  const [caps, setCaps] = useState<Capabilities | null>(null);
  const [connected, setConnected] = useState<boolean | null>(null);
  const startRef = useRef<number>(Date.now());

  useEffect(() => {
    let active = true;
    const controller = new AbortController();

    async function tick() {
      try {
        const [status, capabilities] = await Promise.all([
          api.detectionStatus(controller.signal),
          api.capabilities(controller.signal).catch(() => null),
        ]);
        if (!active) return;
        setLatest(status);
        if (capabilities) setCaps(capabilities);
        setConnected(true);

        const t = Math.round((Date.now() - startRef.current) / 1000);
        const m = status.metrics;
        const ramUsedPct =
          capabilities && capabilities.ram_total_mb
            ? ((capabilities.ram_total_mb - capabilities.ram_available_mb) /
                capabilities.ram_total_mb) *
              100
            : 0;
        setPoints((prev) => {
          const next: ChartPoint = {
            t,
            fps: Number((m.processed_fps ?? 0).toFixed(2)),
            inf_p50: Number((m.inference_latency_p50_ms ?? 0).toFixed(2)),
            inf_p95: Number((m.inference_latency_p95_ms ?? 0).toFixed(2)),
            e2e_p50: Number((m.end_to_end_p50_ms ?? 0).toFixed(2)),
            dropped: m.dropped_frames ?? 0,
            ram_pct: Number(ramUsedPct.toFixed(1)),
          };
          return [...prev, next].slice(-MAX_POINTS);
        });
      } catch {
        if (active) setConnected(false);
      }
    }

    tick();
    const id = setInterval(tick, POLL_MS);
    return () => {
      active = false;
      controller.abort();
      clearInterval(id);
    };
  }, []);

  const m = latest?.metrics;
  const ramUsed = caps ? caps.ram_total_mb - caps.ram_available_mb : null;

  return (
    <div>
      <PageHeader
        title="Performance"
        subtitle="Live detection telemetry polled from /api/detection/status and /api/runtime-status, with host CPU/RAM from /api/capabilities."
        actions={
          connected === null ? (
            <Badge tone="warn">connecting…</Badge>
          ) : connected ? (
            <Badge tone="good">streaming</Badge>
          ) : (
            <Badge tone="bad">backend unreachable</Badge>
          )
        }
      />

      <div className="mb-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <StatCard label="Processed FPS" value={m ? m.processed_fps.toFixed(1) : '—'} icon="gauge" tone="accent" />
        <StatCard label="Inference p50" value={m ? m.inference_latency_p50_ms.toFixed(1) : '—'} unit="ms" icon="clock" />
        <StatCard label="End-to-end p50" value={m ? m.end_to_end_p50_ms.toFixed(1) : '—'} unit="ms" icon="clock" />
        <StatCard
          label="Dropped frames"
          value={m ? m.dropped_frames : '—'}
          icon="alert"
          tone={m && m.dropped_frames > 0 ? 'warn' : 'neutral'}
        />
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <MetricChart
          title="Throughput (FPS)"
          data={points}
          unit="frames/s"
          series={[{ key: 'fps', label: 'processed FPS', color: '#38bdf8' }]}
        />
        <MetricChart
          title="Inference latency"
          data={points}
          unit="ms"
          series={[
            { key: 'inf_p50', label: 'p50', color: '#34d399' },
            { key: 'inf_p95', label: 'p95', color: '#fbbf24' },
          ]}
        />
        <MetricChart
          title="End-to-end latency"
          data={points}
          unit="ms"
          series={[{ key: 'e2e_p50', label: 'e2e p50', color: '#a78bfa' }]}
        />
        <MetricChart
          title="Dropped frames & host RAM"
          data={points}
          series={[
            { key: 'dropped', label: 'dropped (count)', color: '#f87171' },
            { key: 'ram_pct', label: 'RAM used (%)', color: '#22d3ee' },
          ]}
        />
      </div>

      {caps && (
        <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
          <StatCard label="Host CPU cores" value={caps.cpu_cores_logical} sub={`${caps.cpu_cores_physical} physical`} icon="chip" />
          <StatCard label="RAM used" value={formatMb(ramUsed)} icon="chip" />
          <StatCard label="RAM available" value={formatMb(caps.ram_available_mb)} icon="chip" />
          <StatCard label="GPUs" value={caps.gpus.length} sub={caps.nvidia_gpu_present ? 'NVIDIA present' : 'no NVIDIA GPU'} icon="server" />
        </div>
      )}

      {latest && latest.metrics && (
        <div className="card card-pad mt-4">
          <h3 className="label mb-3">Detailed metrics snapshot</h3>
          <div className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-3">
            {Object.entries(latest.metrics).map(([k, v]) => (
              <div key={k} className="flex justify-between border-b border-surface-800 py-1">
                <span className="text-slate-500">{k}</span>
                <span className="font-mono text-slate-200">{typeof v === 'number' ? v.toFixed(2) : String(v)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
