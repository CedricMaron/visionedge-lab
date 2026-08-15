/**
 * One benchmark run in full: the decomposition, the raw evidence behind it, and
 * every caveat that qualifies it.
 *
 * Warnings render above the numbers, not below. A reader who stops after the
 * headline latency should already have seen that it was measured on a throttled
 * GPU, or from three samples, or with iterations that failed.
 */
import { useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { useAsync } from '@/hooks/useAsync';
import { labApi } from '@/services/labApi';
import { getApiBase } from '@/config';
import { ErrorState, PageHeader, Spinner } from '@/components/ui';
import { LatencyDecomposition } from '@/components/LatencyDecomposition';
import { MeasurementList } from '@/components/MeasurementValue';
import { UtilizationChart } from '@/components/UtilizationChart';
import { IterationTable } from '@/components/IterationTable';
import { TraceWaterfall } from '@/components/TraceWaterfall';
import type { BenchmarkRun } from '@/types/lab';

const STATUS_TONE: Record<string, string> = {
  completed: 'bg-good-soft text-good',
  partial: 'bg-warn-soft text-warn',
  failed: 'bg-bad-soft text-bad',
  cancelled: 'bg-elevated text-muted',
  timed_out: 'bg-warn-soft text-warn',
};

function Panel({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
}) {
  return (
    <section className="card card-pad">
      <h2 className="text-sm font-semibold text-primary">{title}</h2>
      {subtitle && <p className="mt-0.5 text-xs text-muted">{subtitle}</p>}
      <div className="mt-3">{children}</div>
    </section>
  );
}

function summaryValue(value: number | null, unit = 'ms'): string {
  return value === null ? '—' : `${value.toFixed(2)} ${unit}`;
}

function unsynchronizedExecution(run: BenchmarkRun): boolean {
  return run.iterations.some(
    (it) =>
      it.group === 'measured' &&
      it.succeeded &&
      it.spans.some((s) => s.phase === 'model_execution' && !s.device_synchronized),
  );
}

export default function RunDetailPage() {
  const { runId = '' } = useParams();
  const { data: run, error, loading, reload } = useAsync<BenchmarkRun>(
    (s) => labApi.run(runId, s),
    [runId],
  );
  const [copied, setCopied] = useState(false);

  if (loading) return <Spinner label="Loading run…" />;
  if (error) return <ErrorState message={error} onRetry={reload} />;
  if (!run) return null;

  const cold = run.cold_warm;
  const base = getApiBase();

  return (
    <div>
      <PageHeader
        title={run.model.display_name || run.model.model_id}
        subtitle={`${run.task.replace(/_/g, ' ')} · scenario ${run.scenario.id} · ${run.runtime.runtime_id}/${run.runtime.device}/${run.runtime.precision}`}
        actions={
          <div className="flex flex-wrap items-center gap-2">
            <span className={`pill ${STATUS_TONE[run.status] ?? 'bg-elevated text-muted'}`}>
              {run.status}
            </span>
            {(['json', 'csv', 'markdown'] as const).map((format) => (
              <a
                key={format}
                className="btn-ghost"
                href={`${base}${labApi.exportUrl(run.identity.run_id, format)}`}
                download
              >
                {format.toUpperCase()}
              </a>
            ))}
          </div>
        }
      />

      {/* Caveats first, deliberately. */}
      {(run.warnings.length > 0 || run.errors.failures.length > 0) && (
        <section className="mb-4 space-y-2" aria-label="Result caveats">
          {run.errors.failures.length > 0 && (
            <div className="rounded border border-bad/40 bg-bad-soft px-3 py-2 text-sm text-bad">
              <strong>
                {run.errors.failures.length} of {run.scenario.measured_iterations} measured
                iterations failed.
              </strong>{' '}
              {run.errors.statistics_exclude_failures
                ? 'Statistics below exclude them, so every figure describes successful work only.'
                : 'Statistics below include them.'}
              <ul className="mt-1 list-inside list-disc text-xs">
                {run.errors.failures.slice(0, 5).map((f) => (
                  <li key={f.index}>
                    iteration {f.index}: <code>{f.error_type}</code> — {f.error_message}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {run.warnings.map((warning) => (
            <div
              key={warning}
              className="rounded border border-warn/40 bg-warn-soft px-3 py-2 text-sm text-warn"
            >
              {warning}
            </div>
          ))}
        </section>
      )}

      {/* Headline figures. */}
      <div className="mb-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
        {[
          { label: 'p50 latency', value: summaryValue(run.timings.total.p50_ms) },
          { label: 'p95 latency', value: summaryValue(run.timings.total.p95_ms) },
          {
            label: 'Throughput',
            value:
              run.throughput.requests_per_second?.value != null
                ? `${run.throughput.requests_per_second.value.toFixed(2)} req/s`
                : '—',
          },
          {
            label: 'Measured iterations',
            value: `${run.successful_iterations}${
              run.failed_iterations ? ` (+${run.failed_iterations} failed)` : ''
            }`,
          },
        ].map((stat) => (
          <div key={stat.label} className="card card-pad">
            <div className="label">{stat.label}</div>
            <div className="mt-1 font-mono text-lg text-primary">{stat.value}</div>
          </div>
        ))}
      </div>

      <div className="space-y-4">
        <Panel
          title="Latency decomposition"
          subtitle="Where every millisecond went. Unattributed time is shown as residual overhead rather than folded into a neighbouring phase."
        >
          <LatencyDecomposition
            breakdown={run.timings}
            unsynchronized={unsynchronizedExecution(run)}
          />
        </Panel>

        <div className="grid gap-4 lg:grid-cols-2">
          <Panel
            title="Cold start vs. steady state"
            subtitle="What a first request costs after a deploy, separated from the warm path."
          >
            <dl className="divide-y divide-subtle">
              {[
                ['Model load', cold.model_load_ms],
                ['Graph compilation', cold.graph_compilation_ms],
                ['Kernel warm-up', cold.kernel_warmup_ms],
                ['First inference', cold.first_inference_ms],
                ['Cold start total', cold.cold_start_total_ms],
              ].map(([label, value]) => (
                <div key={label as string} className="flex justify-between py-2">
                  <dt className="text-sm text-secondary">{label as string}</dt>
                  <dd className="font-mono text-sm text-primary">
                    {value === null || value === undefined
                      ? 'not measured for this run'
                      : `${(value as number).toFixed(2)} ms`}
                  </dd>
                </div>
              ))}
              <div className="flex justify-between py-2">
                <dt className="text-sm font-medium text-secondary">Warm p50</dt>
                <dd className="font-mono text-sm text-primary">
                  {summaryValue(cold.warm_inference.p50_ms)}{' '}
                  <span className="text-xs text-muted">over {cold.warm_inference.n} iters</span>
                </dd>
              </div>
            </dl>
          </Panel>

          <Panel
            title="Energy"
            subtitle="Derived by integrating measured GPU power. GPU-only — CPU package and RAM draw are not readable here."
          >
            <MeasurementList measurements={run.energy} showProvenance />
          </Panel>
        </div>

        <div className="grid gap-4 lg:grid-cols-2">
          <Panel title="Throughput">
            <MeasurementList measurements={run.throughput} />
          </Panel>
          <Panel
            title="Memory"
            subtitle="Allocated, reserved, process and device totals are four different quantities and are kept apart."
          >
            <MeasurementList measurements={run.memory} skip={['snapshots']} showProvenance />
          </Panel>
        </div>

        <Panel
          title="Hardware utilization"
          subtitle={`Sampled every ${run.utilization.sample_interval_ms} ms from ${
            run.utilization.sources.join(', ') || 'no probe'
          }.`}
        >
          <UtilizationChart series={run.utilization} />
        </Panel>

        <Panel
          title="Trace waterfall"
          subtitle="Each iteration on a shared time axis — which run was slow, and which phase made it slow."
        >
          <TraceWaterfall iterations={run.iterations} />
        </Panel>

        {run.artifacts.length > 0 && (
          <Panel
            title="Profiler artifacts"
            subtitle="Large traces are stored separately and referenced here, not embedded in the result."
          >
            <ul className="space-y-2">
              {run.artifacts.map((artifact) => (
                <li
                  key={artifact.path}
                  className="flex flex-wrap items-center justify-between gap-2 rounded border border-subtle px-3 py-2"
                >
                  <span>
                    <span className="font-mono text-xs text-primary">{artifact.kind}</span>
                    <span className="ml-2 text-xs text-muted">
                      {(artifact.size_bytes / 1024).toFixed(0)} KB
                    </span>
                    {artifact.note && (
                      <span className="mt-0.5 block text-2xs text-muted">{artifact.note}</span>
                    )}
                  </span>
                  <a
                    className="btn-ghost shrink-0"
                    href={`${base}/api/lab/runs/${run.identity.run_id}/artifacts/${artifact.path.split('/').pop()}`}
                    download
                  >
                    Download
                  </a>
                </li>
              ))}
            </ul>
          </Panel>
        )}

        <Panel
          title="Per-iteration samples"
          subtitle="The raw evidence behind every aggregate above. Warm-up and failed iterations are retained and marked, not discarded."
        >
          <IterationTable iterations={run.iterations} />
        </Panel>

        <div className="grid gap-4 lg:grid-cols-2">
          <Panel title="Environment">
            <dl className="divide-y divide-subtle text-sm">
              {[
                ['CPU', run.hardware.cpu_model],
                ['Cores', `${run.hardware.cpu_cores_logical} logical`],
                ['RAM', `${(run.hardware.ram_total_mb / 1024).toFixed(1)} GB`],
                ['GPU', run.hardware.gpus.map((g) => g.name).join(', ') || 'none'],
                ['CUDA', run.hardware.cuda_version ?? 'n/a'],
                ['OS', `${run.software.os} ${run.software.kernel_version ?? ''}`],
                ['Python', run.software.python_version],
                ['Execution provider', run.runtime.execution_provider ?? '—'],
                ['Fingerprint', run.fingerprint.digest],
              ].map(([label, value]) => (
                <div key={label} className="flex justify-between gap-4 py-2">
                  <dt className="text-secondary">{label}</dt>
                  <dd className="text-right font-mono text-xs text-primary">{value}</dd>
                </div>
              ))}
            </dl>
          </Panel>

          <Panel
            title="Reproducibility"
            subtitle="Everything needed to run this again and expect the same answer."
          >
            <dl className="divide-y divide-subtle text-sm">
              <div className="flex justify-between py-2">
                <dt className="text-secondary">Git commit</dt>
                <dd className="font-mono text-xs text-primary">
                  {run.reproducibility.git_commit?.slice(0, 12) ?? 'unknown'}
                  {run.reproducibility.git_dirty && (
                    <span className="ml-2 pill bg-warn-soft text-warn">dirty tree</span>
                  )}
                </dd>
              </div>
              <div className="flex justify-between py-2">
                <dt className="text-secondary">Seed</dt>
                <dd className="font-mono text-xs text-primary">
                  {run.reproducibility.random_seed ?? 'none'}
                </dd>
              </div>
              <div className="flex justify-between py-2">
                <dt className="text-secondary">Instrumentation overhead</dt>
                <dd className="font-mono text-xs text-primary">
                  {run.instrumentation_overhead_ms === null
                    ? 'not measured'
                    : `${run.instrumentation_overhead_ms.toFixed(2)} ms/sample`}
                </dd>
              </div>
            </dl>

            {run.reproducibility.reproduction_command && (
              <div className="mt-3">
                <div className="label mb-1">Reproduce</div>
                <div className="flex items-start gap-2">
                  <code className="flex-1 overflow-x-auto rounded border border-subtle bg-elevated p-2 text-xs text-primary">
                    {run.reproducibility.reproduction_command}
                  </code>
                  <button
                    className="btn-ghost shrink-0"
                    onClick={() => {
                      navigator.clipboard
                        ?.writeText(run.reproducibility.reproduction_command ?? '')
                        .then(() => {
                          setCopied(true);
                          setTimeout(() => setCopied(false), 1500);
                        })
                        .catch(() => undefined);
                    }}
                  >
                    {copied ? 'copied' : 'copy'}
                  </button>
                </div>
              </div>
            )}
          </Panel>
        </div>

        <p className="text-xs text-muted">
          <Link to="/performance" className="text-accent hover:underline">
            ← all results
          </Link>
        </p>
      </div>
    </div>
  );
}
