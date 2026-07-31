/**
 * Stored benchmark runs, with selection for comparison.
 *
 * Comparison is gated by the backend's compatibility check: runs that measured
 * materially different things are refused with the specific differences listed,
 * rather than being plotted on a shared axis where the difference would be invisible.
 */
import { useState } from 'react';
import { Link } from 'react-router-dom';
import { useAsync } from '@/hooks/useAsync';
import { labApi } from '@/services/labApi';
import { getApiBase } from '@/config';
import { EmptyState, ErrorState, PageHeader, Spinner } from '@/components/ui';
import { Icon } from '@/components/Icon';
import type { CompareResult, RunSummary } from '@/types/lab';

const STATUS_TONE: Record<string, string> = {
  completed: 'bg-good-soft text-good',
  partial: 'bg-warn-soft text-warn',
  failed: 'bg-bad-soft text-bad',
  cancelled: 'bg-elevated text-muted',
  timed_out: 'bg-warn-soft text-warn',
};

function ComparePanel({
  result,
  onClose,
}: {
  result: CompareResult;
  onClose: () => void;
}) {
  return (
    <div className="card card-pad mb-4">
      <div className="flex items-start justify-between gap-4">
        <h2 className="text-sm font-semibold text-primary">Comparison</h2>
        <button className="btn-ghost" onClick={onClose}>
          close
        </button>
      </div>

      {result.warning && (
        <p className="mt-2 rounded border border-warn/40 bg-warn-soft px-3 py-2 text-sm text-warn">
          {result.warning}
        </p>
      )}

      <div className="mt-3 space-y-2">
        {result.comparisons.map((comparison) => (
          <div
            key={comparison.run_id}
            className="rounded border border-subtle px-3 py-2 text-sm"
          >
            <div className="flex items-center justify-between gap-2">
              <span className="font-mono text-xs text-primary">{comparison.run_id}</span>
              <span
                className={`pill ${
                  comparison.comparable ? 'bg-good-soft text-good' : 'bg-bad-soft text-bad'
                }`}
              >
                {comparison.comparable ? 'comparable' : 'not comparable'}
              </span>
            </div>
            {comparison.blocking_differences.length > 0 && (
              <ul className="mt-1 list-inside list-disc text-xs text-muted">
                {comparison.blocking_differences.map((difference) => (
                  <li key={difference}>{difference}</li>
                ))}
              </ul>
            )}
          </div>
        ))}
      </div>

      {result.all_comparable && (
        <div className="mt-4 overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th scope="col">Run</th>
                <th scope="col">Model</th>
                <th scope="col" className="text-right">p50 ms</th>
                <th scope="col" className="text-right">p95 ms</th>
                <th scope="col" className="text-right">req/s</th>
              </tr>
            </thead>
            <tbody>
              {result.runs.map((run) => (
                <tr key={run.identity.run_id}>
                  <td className="font-mono text-xs">{run.identity.run_id}</td>
                  <td className="text-xs">{run.model.model_id}</td>
                  <td className="num">{run.timings.total.p50_ms?.toFixed(2) ?? '—'}</td>
                  <td className="num">{run.timings.total.p95_ms?.toFixed(2) ?? '—'}</td>
                  <td className="num">
                    {run.throughput.requests_per_second?.value?.toFixed(2) ?? '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default function ResultsPage() {
  const { data, error, loading, reload } = useAsync<{ runs: RunSummary[] }>(
    (s) => labApi.runs({ limit: 100 }, s),
    [],
  );
  const [selected, setSelected] = useState<string[]>([]);
  const [comparison, setComparison] = useState<CompareResult | null>(null);
  const [comparing, setComparing] = useState(false);

  const toggle = (runId: string) =>
    setSelected((current) =>
      current.includes(runId)
        ? current.filter((id) => id !== runId)
        : [...current, runId].slice(-8),
    );

  async function compare() {
    setComparing(true);
    try {
      setComparison(await labApi.compare(selected));
    } catch {
      /* surfaced by the disabled state; nothing to recover */
    } finally {
      setComparing(false);
    }
  }

  return (
    <div>
      <PageHeader
        title="Results"
        subtitle="Every stored benchmark run. Select two or more to compare — incompatible runs are refused with the reason."
        actions={
          <div className="flex items-center gap-2">
            <a className="btn-ghost" href={`${getApiBase()}/api/lab/export/summary`} download>
              Export CSV
            </a>
            <button className="btn-ghost" onClick={reload}>
              <Icon name="refresh" className="h-4 w-4" /> Refresh
            </button>
          </div>
        }
      />

      {loading && <Spinner label="Loading runs…" />}
      {error && <ErrorState message={error} onRetry={reload} />}

      {comparison && (
        <ComparePanel result={comparison} onClose={() => setComparison(null)} />
      )}

      {selected.length >= 2 && !comparison && (
        <div className="mb-4 flex items-center gap-3">
          <button className="btn-primary" disabled={comparing} onClick={compare}>
            {comparing ? 'Comparing…' : `Compare ${selected.length} runs`}
          </button>
          <button className="btn-ghost" onClick={() => setSelected([])}>
            Clear selection
          </button>
        </div>
      )}

      {data && data.runs.length === 0 && (
        <EmptyState
          title="No benchmark runs stored yet"
          hint="Run one from the Run Benchmark page, or from the CLI with `inference-lab benchmark run`."
        />
      )}

      {data && data.runs.length > 0 && (
        <div className="card overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th scope="col" className="w-8" />
                <th scope="col">Run</th>
                <th scope="col">Status</th>
                <th scope="col">Task</th>
                <th scope="col">Model</th>
                <th scope="col">Runtime</th>
                <th scope="col" className="text-right">p50 ms</th>
                <th scope="col" className="text-right">p95 ms</th>
                <th scope="col" className="text-right">n</th>
                <th scope="col">Fingerprint</th>
              </tr>
            </thead>
            <tbody>
              {data.runs.map((run) => (
                <tr key={run.run_id}>
                  <td>
                    <input
                      type="checkbox"
                      checked={selected.includes(run.run_id)}
                      onChange={() => toggle(run.run_id)}
                      aria-label={`select run ${run.run_id}`}
                    />
                  </td>
                  <td>
                    <Link
                      to={`/lab/results/${run.run_id}`}
                      className="font-mono text-xs text-accent hover:underline"
                    >
                      {run.run_id}
                    </Link>
                  </td>
                  <td>
                    <span className={`pill ${STATUS_TONE[run.status] ?? 'bg-elevated text-muted'}`}>
                      {run.status}
                    </span>
                  </td>
                  <td className="text-xs">{run.task.replace(/_/g, ' ')}</td>
                  <td className="text-xs">{run.model_id}</td>
                  <td className="text-xs">
                    {run.runtime_id}/{run.device}/{run.precision}
                  </td>
                  <td className="num">{run.latency_p50_ms?.toFixed(2) ?? '—'}</td>
                  <td className="num">{run.latency_p95_ms?.toFixed(2) ?? '—'}</td>
                  <td className="num">
                    {run.measured_iterations}
                    {run.failed_iterations > 0 && (
                      <span className="ml-1 text-bad">+{run.failed_iterations}✗</span>
                    )}
                  </td>
                  <td className="font-mono text-2xs text-muted">{run.fingerprint}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {data && data.runs.length > 0 && (
        <p className="mt-2 text-xs text-muted">
          Runs sharing a fingerprint were produced on equivalent configurations and may be
          pooled. Different fingerprints may be compared, but never averaged together.
        </p>
      )}
    </div>
  );
}
