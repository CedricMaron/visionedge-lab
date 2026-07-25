import { useState } from 'react';
import { api } from '@/services/api';
import { useAsync } from '@/hooks/useAsync';
import { ApiError } from '@/services/http';
import { PageHeader, Spinner, ErrorState, Badge, Field } from '@/components/ui';
import { Icon } from '@/components/Icon';
import { StatCard } from '@/components/StatCard';
import { formatMb, formatMs } from '@/utils/format';
import type {
  BenchmarkComparisonResponse,
  BenchmarkComparisonRow,
  BenchmarkResult,
  BenchmarksResponse,
} from '@/types';

function BenchTable({ rows, highlightFirst }: { rows: BenchmarkResult[]; highlightFirst?: boolean }) {
  return (
    <div className="card overflow-x-auto">
      <table className="w-full min-w-[760px] text-sm">
        <thead>
          <tr className="border-b border-surface-700 text-left text-xs uppercase tracking-wide text-slate-500">
            <th className="px-3 py-2.5">Model</th>
            <th className="px-3 py-2.5">Backend</th>
            <th className="px-3 py-2.5">Device / provider</th>
            <th className="px-3 py-2.5">Input</th>
            <th className="px-3 py-2.5">Prec</th>
            <th className="px-3 py-2.5 text-right">FPS</th>
            <th className="px-3 py-2.5 text-right">Mean</th>
            <th className="px-3 py-2.5 text-right">p50</th>
            <th className="px-3 py-2.5 text-right">p95</th>
            <th className="px-3 py-2.5 text-right">p99</th>
            <th className="px-3 py-2.5 text-right">RSS</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((b, i) => (
            <tr
              key={i}
              className={`border-b border-surface-800 last:border-0 ${
                highlightFirst && i === 0 ? 'bg-accent/5' : ''
              }`}
            >
              <td className="px-3 py-2.5 font-mono text-xs text-slate-200">{b.model_id}</td>
              <td className="px-3 py-2.5 text-slate-300">{b.backend}</td>
              <td className="px-3 py-2.5 text-slate-400">
                {b.device} / {b.provider}
              </td>
              <td className="px-3 py-2.5 font-mono text-slate-300">{b.input_size}</td>
              <td className="px-3 py-2.5 text-slate-300">{b.precision}</td>
              <td className="px-3 py-2.5 text-right font-mono text-accent">{b.fps.toFixed(1)}</td>
              <td className="px-3 py-2.5 text-right font-mono text-slate-300">{formatMs(b.latency_mean_ms)}</td>
              <td className="px-3 py-2.5 text-right font-mono text-slate-300">{formatMs(b.latency_p50_ms)}</td>
              <td className="px-3 py-2.5 text-right font-mono text-slate-300">{formatMs(b.latency_p95_ms)}</td>
              <td className="px-3 py-2.5 text-right font-mono text-slate-300">{formatMs(b.latency_p99_ms)}</td>
              <td className="px-3 py-2.5 text-right font-mono text-slate-300">{formatMb(b.memory_rss_mb)}</td>
            </tr>
          ))}
          {rows.length === 0 && (
            <tr>
              <td colSpan={11} className="px-3 py-6 text-center text-slate-500">
                No benchmarks recorded yet.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

export function ComparisonTable({ rows }: { rows: BenchmarkComparisonRow[] }) {
  return (
    <div className="card overflow-x-auto">
      <table className="w-full min-w-[680px] text-sm">
        <thead>
          <tr className="border-b border-surface-700 text-left text-xs uppercase tracking-wide text-slate-500">
            <th className="px-3 py-2.5">Model</th>
            <th className="px-3 py-2.5">Device / provider</th>
            <th className="px-3 py-2.5">Input</th>
            <th className="px-3 py-2.5 text-right">Median FPS</th>
            <th className="px-3 py-2.5 text-right">Median p50</th>
            <th className="px-3 py-2.5 text-right">Runs</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((g) => (
            <tr
              key={`${g.model_id}-${g.provider}-${g.input_size}-${g.precision}`}
              className="border-b border-surface-800 last:border-0"
            >
              <td className="px-3 py-2.5 font-mono text-xs text-slate-200">
                {g.model_id}
                {g.any_concurrent_traffic && (
                  <span
                    className="pill ml-2 bg-warn/15 text-warn"
                    title="Measured while live inference was running — the machine was loaded"
                  >
                    loaded
                  </span>
                )}
              </td>
              <td className="px-3 py-2.5 text-slate-400">
                {g.device} / {g.provider ?? '—'}
              </td>
              <td className="px-3 py-2.5 font-mono text-slate-300">{g.input_size}</td>
              <td className="px-3 py-2.5 text-right font-mono text-accent">
                {g.median_fps?.toFixed(1) ?? '—'}
              </td>
              <td className="px-3 py-2.5 text-right font-mono text-slate-300">
                {g.median_p50_ms !== null ? formatMs(g.median_p50_ms) : '—'}
              </td>
              <td className="px-3 py-2.5 text-right font-mono text-slate-400">n={g.n}</td>
            </tr>
          ))}
          {rows.length === 0 && (
            <tr>
              <td colSpan={6} className="px-3 py-6 text-center text-slate-500">
                No benchmarks recorded yet. Switch a model to record one automatically.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

export default function BenchmarksPage() {
  const history = useAsync<BenchmarksResponse>((s) => api.benchmarks(s), []);
  const comparison = useAsync<BenchmarkComparisonResponse>((s) => api.benchmarkComparison(s), []);
  const [runs, setRuns] = useState(50);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<BenchmarkResult | null>(null);
  const [runError, setRunError] = useState<string | null>(null);

  async function runBench() {
    setRunning(true);
    setRunError(null);
    try {
      const res = await api.runBenchmark(runs);
      setResult(res);
      history.reload();
    } catch (err) {
      setRunError(err instanceof ApiError ? err.message : err instanceof Error ? err.message : 'Benchmark failed');
    } finally {
      setRunning(false);
    }
  }

  return (
    <div>
      <PageHeader
        title="Benchmarks"
        subtitle="Run a real benchmark against the active configuration and compare with history. All results are measured on this device's backend."
      />

      <div className="card card-pad mb-4 flex flex-col gap-4 sm:flex-row sm:items-end">
        <Field label="Runs">
          <input
            type="number"
            min={1}
            max={1000}
            className="input w-32"
            value={runs}
            onChange={(e) => setRuns(Math.max(1, Number(e.target.value)))}
          />
        </Field>
        <button className="btn-primary" disabled={running} onClick={runBench}>
          {running ? (
            <>
              <span className="h-4 w-4 animate-spin rounded-full border-2 border-surface-950/40 border-t-surface-950" />
              Running {runs} runs…
            </>
          ) : (
            <>
              <Icon name="gauge" className="h-4 w-4" /> Run benchmark
            </>
          )}
        </button>
        <Badge tone="accent">measured on this device</Badge>
      </div>

      {runError && <ErrorState message={runError} onRetry={runBench} />}

      {result && (
        <div className="mb-6">
          <h2 className="label mb-2">Latest measured result</h2>
          <div className="mb-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
            <StatCard label="FPS" value={result.fps.toFixed(1)} tone="accent" icon="gauge" />
            <StatCard label="Mean latency" value={formatMs(result.latency_mean_ms)} icon="clock" />
            <StatCard label="p95 latency" value={formatMs(result.latency_p95_ms)} icon="clock" />
            <StatCard label="Memory RSS" value={formatMb(result.memory_rss_mb)} icon="chip" />
          </div>
          <BenchTable rows={[result]} highlightFirst />
        </div>
      )}

      <h2 className="label mb-2 mt-6">Model comparison (median across runs)</h2>
      <p className="mb-2 text-xs text-slate-500">
        Median rather than best, so one lucky run can&apos;t flatter a model. Only comparable
        within this host. Rows marked <span className="text-warn">loaded</span> were measured
        while live inference was running.
      </p>
      {comparison.error && <ErrorState message={comparison.error} onRetry={comparison.reload} />}
      <ComparisonTable rows={comparison.data?.groups ?? []} />

      <h2 className="label mb-2 mt-6">History</h2>
      {history.loading && <Spinner label="Loading benchmark history…" />}
      {history.error && <ErrorState message={history.error} onRetry={history.reload} />}
      {history.data && <BenchTable rows={history.data.benchmarks} />}
    </div>
  );
}
