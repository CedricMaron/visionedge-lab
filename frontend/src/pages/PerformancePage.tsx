/**
 * Performance — live telemetry, benchmarks and history in one place.
 *
 * Three things used to live on three pages: what the server is doing right now,
 * what a controlled benchmark measured, and what past runs recorded. They belong
 * together, but they are not the same kind of number, so each section says which it
 * is and where it executed. Nothing is averaged across configurations: runs carry a
 * fingerprint precisely so incomparable ones are never pooled.
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { useAsync } from '@/hooks/useAsync';
import { api } from '@/services/api';
import { labApi } from '@/services/labApi';
import { ExecutionBadge } from '@/components/ExecutionBadge';
import { MetricChart, type ChartPoint } from '@/components/MetricChart';
import { Badge, EmptyState, ErrorState, Field, PageHeader, Spinner } from '@/components/ui';
import { Icon } from '@/components/Icon';
import { usePlaygroundStore } from '@/stores/playgroundStore';
import { formatMb } from '@/utils/format';
import { taskLabel } from '@/lab/catalog';
import type { Capabilities, DetectionStatus } from '@/types';
import type { CompareResult, LabScenario, RunSummary } from '@/types/lab';

const MAX_POINTS = 40;
const POLL_MS = 2000;

export default function PerformancePage() {
  const config = usePlaygroundStore((s) => s.config);
  const trace = usePlaygroundStore((s) => s.trace);
  const stream = usePlaygroundStore((s) => s.stream);

  /* ----------------------------- live telemetry ---------------------------- */
  const [points, setPoints] = useState<ChartPoint[]>([]);
  const [status, setStatus] = useState<DetectionStatus | null>(null);
  const [caps, setCaps] = useState<Capabilities | null>(null);
  const [connected, setConnected] = useState<boolean | null>(null);
  const startRef = useRef<number>(Date.now());

  useEffect(() => {
    let active = true;
    const controller = new AbortController();

    async function tick() {
      try {
        const [detection, capabilities] = await Promise.all([
          api.detectionStatus(controller.signal),
          api.capabilities(controller.signal).catch(() => null),
        ]);
        if (!active) return;
        setStatus(detection);
        if (capabilities) setCaps(capabilities);
        setConnected(true);

        const t = Math.round((Date.now() - startRef.current) / 1000);
        const m = detection.metrics;
        const ramUsedPct =
          capabilities && capabilities.ram_total_mb
            ? ((capabilities.ram_total_mb - capabilities.ram_available_mb) /
                capabilities.ram_total_mb) *
              100
            : 0;
        setPoints((prev) =>
          [
            ...prev,
            {
              t,
              fps: Number((m.processed_fps ?? 0).toFixed(2)),
              inf_p50: Number((m.inference_latency_p50_ms ?? 0).toFixed(2)),
              inf_p95: Number((m.inference_latency_p95_ms ?? 0).toFixed(2)),
              e2e_p50: Number((m.end_to_end_p50_ms ?? 0).toFixed(2)),
              dropped: m.dropped_frames ?? 0,
              ram_pct: Number(ramUsedPct.toFixed(1)),
            },
          ].slice(-MAX_POINTS),
        );
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

  const metrics = status?.metrics;
  const hasGpu = Boolean(caps && caps.gpus.length > 0);

  /* -------------------------------- benchmark ------------------------------ */
  const scenarios = useAsync<{ scenarios: LabScenario[] }>((s) => labApi.scenarios(s), []);
  const runs = useAsync<{ runs: RunSummary[] }>((s) => labApi.runs({ limit: 50 }, s), []);

  const taskScenarios = useMemo(
    () => (scenarios.data?.scenarios ?? []).filter((s) => !config.task || s.task === config.task),
    [scenarios.data, config.task],
  );
  const [scenarioId, setScenarioId] = useState('');
  const [warmup, setWarmup] = useState<number | ''>('');
  const [iterations, setIterations] = useState<number | ''>('');
  const [benchmarking, setBenchmarking] = useState(false);
  const [benchError, setBenchError] = useState<string | null>(null);
  const [lastRunId, setLastRunId] = useState<string | null>(null);

  useEffect(() => {
    if (taskScenarios.length > 0 && !taskScenarios.some((s) => s.id === scenarioId)) {
      setScenarioId(taskScenarios[0].id);
    }
  }, [taskScenarios, scenarioId]);

  async function runBenchmark() {
    if (!scenarioId || !config.modelId) return;
    setBenchmarking(true);
    setBenchError(null);
    try {
      const result = await labApi.createRun({
        scenario_id: scenarioId,
        model_id: config.modelId,
        runtime_id: config.runtimeId || 'onnxruntime',
        device: config.device,
        precision: config.precision,
        warmup_iterations: warmup === '' ? undefined : warmup,
        measured_iterations: iterations === '' ? undefined : iterations,
      });
      setLastRunId(result.run.identity.run_id);
      runs.reload();
    } catch (err) {
      setBenchError(err instanceof Error ? err.message : 'the benchmark could not be run');
    } finally {
      setBenchmarking(false);
    }
  }

  /* ------------------------------- comparison ------------------------------ */
  const [selected, setSelected] = useState<string[]>([]);
  const [comparison, setComparison] = useState<CompareResult | null>(null);
  const [comparing, setComparing] = useState(false);

  const toggleRun = (runId: string) =>
    setSelected((current) =>
      current.includes(runId) ? current.filter((id) => id !== runId) : [...current, runId].slice(-8),
    );

  async function compare() {
    setComparing(true);
    try {
      setComparison(await labApi.compare(selected));
    } catch {
      /* the panel simply does not open; the run list is unchanged */
    } finally {
      setComparing(false);
    }
  }

  return (
    <div>
      <PageHeader
        title="Performance"
        subtitle="Live telemetry from the server, controlled benchmarks, and every stored run. Local and server measurements are never mixed."
        actions={
          connected === null ? (
            <Badge tone="warn">connecting…</Badge>
          ) : connected ? (
            <Badge tone="good">telemetry live</Badge>
          ) : (
            <Badge tone="bad">backend unreachable</Badge>
          )
        }
      />

      {/* Active configuration ------------------------------------------------ */}
      <div className="card card-pad mb-4">
        <div className="mb-2 flex items-center justify-between gap-2">
          <h2 className="label">Active configuration</h2>
          <ExecutionBadge target={config.execution} />
        </div>
        <div className="flex flex-wrap gap-x-6 gap-y-1 font-mono text-sm text-primary">
          <span>{config.modelId || '—'}</span>
          <span className="text-secondary">{taskLabel(config.task) || '—'}</span>
          <span className="text-secondary">{config.runtimeId || '—'}</span>
          <span className="text-secondary">{config.device}</span>
          <span className="text-secondary">{config.precision}</span>
          {config.inputSize && (
            <span className="text-secondary">
              {config.inputSize} × {config.inputSize}
            </span>
          )}
        </div>
      </div>

      {/* Last run ------------------------------------------------------------ */}
      {(trace || stream) && (
        <div className="mb-4 grid gap-4 md:grid-cols-2">
          {trace && (
            <section className="card card-pad">
              <div className="mb-3 flex items-center justify-between gap-2">
                <h2 className="text-sm font-semibold text-primary">Last single-shot run</h2>
                <ExecutionBadge target={trace.execution} />
              </div>
              <dl className="divide-y divide-subtle text-sm">
                {[
                  ['Inference', `${trace.timings.inference_ms.toFixed(2)} ms`],
                  ['Preprocess', `${trace.timings.preprocess_ms.toFixed(2)} ms`],
                  ['Post-process', `${trace.timings.postprocess_ms.toFixed(2)} ms`],
                  ['Server total', `${trace.timings.server_total_ms.toFixed(2)} ms`],
                  [
                    'End-to-end (browser)',
                    trace.client_round_trip_ms !== undefined
                      ? `${trace.client_round_trip_ms.toFixed(0)} ms`
                      : 'unavailable',
                  ],
                  [
                    'Network + client overhead',
                    trace.client_round_trip_ms !== undefined
                      ? `${(trace.client_round_trip_ms - trace.timings.server_total_ms).toFixed(0)} ms`
                      : 'unavailable',
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
              <Link className="mt-3 inline-block text-xs text-accent hover:underline" to="/pipeline">
                Inspect its pipeline →
              </Link>
            </section>
          )}
          {stream && (
            <section className="card card-pad">
              <div className="mb-3 flex items-center justify-between gap-2">
                <h2 className="text-sm font-semibold text-primary">Live stream</h2>
                <ExecutionBadge target={stream.execution} />
              </div>
              <dl className="divide-y divide-subtle text-sm">
                {[
                  ['Throughput', `${stream.fps.toFixed(0)} fps`],
                  ['Inference', `${stream.inferenceMs.toFixed(1)} ms`],
                  ['Frames sent', String(stream.processedFrames)],
                  ['Dropped by server', String(stream.droppedFrames)],
                  ['Backend', stream.backend || '—'],
                ].map(([term, value]) => (
                  <div key={term} className="flex justify-between gap-3 py-1.5">
                    <dt className="text-secondary">{term}</dt>
                    <dd className="text-right font-mono text-xs text-primary">{value}</dd>
                  </div>
                ))}
              </dl>
            </section>
          )}
        </div>
      )}

      {/* Live telemetry ------------------------------------------------------ */}
      <section className="mb-6">
        <div className="mb-2 flex items-center justify-between gap-2">
          <h2 className="label">Live server telemetry</h2>
          <ExecutionBadge target="server" />
        </div>
        <div className="mb-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Stat label="Processed FPS" value={metrics ? metrics.processed_fps.toFixed(1) : '—'} />
          <Stat
            label="Inference p50"
            value={metrics ? `${metrics.inference_latency_p50_ms.toFixed(1)} ms` : '—'}
          />
          <Stat
            label="Inference p95"
            value={metrics ? `${metrics.inference_latency_p95_ms.toFixed(1)} ms` : '—'}
          />
          <Stat
            label="End-to-end p50"
            value={metrics ? `${metrics.end_to_end_p50_ms.toFixed(1)} ms` : '—'}
          />
        </div>

        <div className="grid gap-4 lg:grid-cols-2">
          <MetricChart
            title="Throughput"
            data={points}
            unit="frames/s"
            series={[{ key: 'fps', label: 'processed FPS', color: 'rgb(var(--series-1))' }]}
          />
          <MetricChart
            title="Inference latency"
            data={points}
            unit="ms"
            series={[
              { key: 'inf_p50', label: 'p50', color: 'rgb(var(--series-3))' },
              { key: 'inf_p95', label: 'p95', color: 'rgb(var(--series-2))' },
            ]}
          />
          <MetricChart
            title="End-to-end latency"
            data={points}
            unit="ms"
            series={[{ key: 'e2e_p50', label: 'e2e p50', color: 'rgb(var(--series-4))' }]}
          />
          <MetricChart
            title="Dropped frames & host RAM"
            data={points}
            series={[
              { key: 'dropped', label: 'dropped (count)', color: 'rgb(var(--series-5))' },
              { key: 'ram_pct', label: 'RAM used (%)', color: 'rgb(var(--series-6))' },
            ]}
          />
        </div>

        <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {caps && (
            <>
              <Stat label="CPU cores" value={`${caps.cpu_cores_logical}`} sub={`${caps.cpu_cores_physical} physical`} />
              <Stat label="RAM used" value={formatMb(caps.ram_total_mb - caps.ram_available_mb)} />
              <Stat label="RAM total" value={formatMb(caps.ram_total_mb)} />
              <Stat
                label="GPU"
                value={hasGpu ? caps.gpus[0].name : 'none detected'}
                sub={hasGpu ? formatMb(caps.gpus[0].memory_total_mb) : undefined}
              />
            </>
          )}
        </div>
        {!hasGpu && (
          <p className="mt-2 text-xs text-muted">
            GPU utilization, VRAM, power and temperature are unavailable — no compatible GPU was
            detected on the server.
          </p>
        )}
      </section>

      {/* Benchmark ----------------------------------------------------------- */}
      <section className="mb-6">
        <div className="mb-2 flex items-center justify-between gap-2">
          <h2 className="label">Benchmark</h2>
          <ExecutionBadge target="server" />
        </div>
        <div className="card card-pad">
          {scenarios.loading && <Spinner label="Loading scenarios…" />}
          {scenarios.error && <ErrorState message={scenarios.error} onRetry={scenarios.reload} />}
          {scenarios.data && taskScenarios.length === 0 && (
            <p className="text-sm text-muted">
              No benchmark scenario is defined for {taskLabel(config.task) || 'this task'}.
            </p>
          )}
          {taskScenarios.length > 0 && (
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <Field label="Scenario">
                <select
                  className="input"
                  value={scenarioId}
                  onChange={(e) => setScenarioId(e.target.value)}
                >
                  {taskScenarios.map((s) => (
                    <option key={s.id} value={s.id}>
                      {s.id}
                    </option>
                  ))}
                </select>
              </Field>
              <Field label="Warm-up iterations">
                <input
                  type="number"
                  min={0}
                  max={100}
                  className="input"
                  placeholder={String(
                    taskScenarios.find((s) => s.id === scenarioId)?.warmup_iterations ?? 5,
                  )}
                  value={warmup}
                  onChange={(e) => setWarmup(e.target.value === '' ? '' : Number(e.target.value))}
                />
              </Field>
              <Field label="Measured iterations">
                <input
                  type="number"
                  min={1}
                  max={1000}
                  className="input"
                  placeholder={String(
                    taskScenarios.find((s) => s.id === scenarioId)?.measured_iterations ?? 20,
                  )}
                  value={iterations}
                  onChange={(e) =>
                    setIterations(e.target.value === '' ? '' : Number(e.target.value))
                  }
                />
              </Field>
              <div className="flex items-end">
                <button
                  className="btn-primary w-full"
                  disabled={benchmarking || !config.modelId}
                  onClick={runBenchmark}
                >
                  {benchmarking ? 'Running…' : 'Run benchmark'}
                </button>
              </div>
            </div>
          )}
          {benchError && (
            <p className="mt-3 rounded border border-bad/40 bg-bad-soft px-3 py-2 text-sm text-bad">
              {benchError}
            </p>
          )}
          {lastRunId && (
            <p className="mt-3 text-sm text-secondary">
              Stored as{' '}
              <Link className="font-mono text-accent hover:underline" to={`/runs/${lastRunId}`}>
                {lastRunId}
              </Link>
              .
            </p>
          )}
          <p className="mt-3 text-xs text-muted">
            The benchmark runs the active model on the server with the configuration above, using
            the scenario's synthetic input so the measurement reflects the runtime path rather than
            the contents of one image.
          </p>
        </div>
      </section>

      {/* History / comparison ------------------------------------------------ */}
      <section>
        <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
          <h2 className="label">Run history & comparison</h2>
          <div className="flex items-center gap-2">
            {selected.length >= 2 && (
              <button className="btn-primary" disabled={comparing} onClick={compare}>
                {comparing ? 'Comparing…' : `Compare ${selected.length}`}
              </button>
            )}
            <button className="btn-ghost" onClick={runs.reload}>
              <Icon name="refresh" className="h-4 w-4" /> Refresh
            </button>
          </div>
        </div>

        {runs.loading && <Spinner label="Loading runs…" />}
        {runs.error && <ErrorState message={runs.error} onRetry={runs.reload} />}

        {comparison && (
          <div className="card card-pad mb-3">
            <div className="flex items-start justify-between gap-3">
              <h3 className="text-sm font-semibold text-primary">Comparison</h3>
              <button className="btn-ghost" onClick={() => setComparison(null)}>
                close
              </button>
            </div>
            {comparison.warning && (
              <p className="mt-2 rounded border border-warn/40 bg-warn-soft px-3 py-2 text-sm text-warn">
                {comparison.warning}
              </p>
            )}
            <ul className="mt-2 space-y-1.5">
              {comparison.comparisons.map((c) => (
                <li key={c.run_id} className="text-sm">
                  <span className="font-mono text-xs text-primary">{c.run_id}</span>{' '}
                  <span className={c.comparable ? 'text-good' : 'text-bad'}>
                    {c.comparable ? 'comparable' : 'not comparable'}
                  </span>
                  {c.blocking_differences.length > 0 && (
                    <ul className="list-inside list-disc text-xs text-muted">
                      {c.blocking_differences.map((d) => (
                        <li key={d}>{d}</li>
                      ))}
                    </ul>
                  )}
                </li>
              ))}
            </ul>
          </div>
        )}

        {runs.data && runs.data.runs.length === 0 && (
          <EmptyState
            title="No benchmark runs stored yet"
            hint="Run one above, or from the CLI with `inference-lab benchmark run`."
          />
        )}

        {runs.data && runs.data.runs.length > 0 && (
          <div className="card overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th scope="col" className="w-8" />
                  <th scope="col">Model</th>
                  <th scope="col">Execution</th>
                  <th scope="col">Runtime</th>
                  <th scope="col">Device</th>
                  <th scope="col">Precision</th>
                  <th scope="col">Scenario</th>
                  <th scope="col" className="text-right">req/s</th>
                  <th scope="col" className="text-right">p50 ms</th>
                  <th scope="col" className="text-right">p95 ms</th>
                  <th scope="col" className="text-right">Peak RSS</th>
                  <th scope="col">Fingerprint</th>
                </tr>
              </thead>
              <tbody>
                {runs.data.runs.map((run) => (
                  <tr key={run.run_id}>
                    <td>
                      <input
                        type="checkbox"
                        checked={selected.includes(run.run_id)}
                        onChange={() => toggleRun(run.run_id)}
                        aria-label={`select run ${run.run_id}`}
                      />
                    </td>
                    <td className="text-xs">
                      <Link className="text-accent hover:underline" to={`/runs/${run.run_id}`}>
                        {run.model_id}
                      </Link>
                    </td>
                    <td>
                      <ExecutionBadge target="server" />
                    </td>
                    <td className="text-xs">{run.runtime_id}</td>
                    <td className="text-xs">{run.device}</td>
                    <td className="text-xs">{run.precision}</td>
                    <td className="text-2xs">{run.scenario_id}</td>
                    <td className="num">{run.throughput_per_s?.toFixed(2) ?? '—'}</td>
                    <td className="num">{run.latency_p50_ms?.toFixed(2) ?? '—'}</td>
                    <td className="num">{run.latency_p95_ms?.toFixed(2) ?? '—'}</td>
                    <td className="num">{run.peak_rss_mb ? formatMb(run.peak_rss_mb) : '—'}</td>
                    <td className="font-mono text-2xs text-muted">{run.fingerprint}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        <p className="mt-2 text-xs text-muted">
          Every stored run executed on the server. Runs sharing a fingerprint measured equivalent
          configurations and may be pooled; different fingerprints may be compared side by side but
          never averaged together.
        </p>
      </section>
    </div>
  );
}

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="card px-3 py-2.5">
      <div className="label truncate">{label}</div>
      <div className="mt-1 truncate font-mono text-lg text-primary" title={value}>
        {value}
      </div>
      {sub && <div className="text-2xs text-muted">{sub}</div>}
    </div>
  );
}
