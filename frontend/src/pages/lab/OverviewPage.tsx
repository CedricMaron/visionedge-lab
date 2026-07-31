/**
 * Landing page.
 *
 * Leaders are reported per task and never pooled: a combined ranking over detection
 * latency and embedding latency would be arithmetically valid and completely
 * meaningless.
 */
import { Link } from 'react-router-dom';
import { useAsync } from '@/hooks/useAsync';
import { labApi } from '@/services/labApi';
import { EmptyState, ErrorState, PageHeader, Spinner } from '@/components/ui';
import type { Overview, RunSummary } from '@/types/lab';

function StatCard({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="card card-pad">
      <div className="label">{label}</div>
      <div className="mt-1 font-mono text-xl text-primary">{value}</div>
      {hint && <div className="mt-0.5 text-xs text-muted">{hint}</div>}
    </div>
  );
}

function LeaderRow({ title, run, metric }: { title: string; run: RunSummary | null; metric: string }) {
  if (!run) {
    return (
      <div className="flex justify-between gap-3 py-1.5 text-sm">
        <span className="text-secondary">{title}</span>
        <span className="text-xs text-muted">no run with this metric yet</span>
      </div>
    );
  }
  return (
    <div className="flex items-baseline justify-between gap-3 py-1.5 text-sm">
      <span className="text-secondary">{title}</span>
      <span className="text-right">
        <Link
          to={`/lab/results/${run.run_id}`}
          className="font-mono text-xs text-accent hover:underline"
        >
          {run.model_id}
        </Link>
        <span className="ml-2 font-mono text-xs text-primary">{metric}</span>
      </span>
    </div>
  );
}

export default function OverviewPage() {
  const { data, error, loading, reload } = useAsync<Overview>((s) => labApi.overview(s), []);

  return (
    <div>
      <PageHeader
        title="Overview"
        subtitle="Multimodal AI inference, profiling and benchmarking — decomposed from input decoding to device execution."
      />

      {loading && <Spinner label="Loading…" />}
      {error && <ErrorState message={error} onRetry={reload} />}

      {data && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <StatCard label="Stored runs" value={String(data.total_runs)} />
            <StatCard
              label="Tasks measured"
              value={String(Object.keys(data.leaders_by_task).length)}
            />
            <StatCard
              label="Runtimes available"
              value={String(data.runtimes_available.length)}
              hint={`${data.runtimes_unavailable} unavailable, each with a reason`}
            />
            <StatCard
              label="Recent failures"
              value={String(data.recent_failures.length)}
              hint={data.recent_failures.length > 0 ? 'see Results' : 'none'}
            />
          </div>

          {data.total_runs === 0 ? (
            <EmptyState
              title="No benchmark runs yet"
              hint="Start one from Run Benchmark, or from the CLI with `inference-lab benchmark run`."
            />
          ) : (
            <>
              <div className="grid gap-4 lg:grid-cols-2">
                <section className="card card-pad">
                  <h2 className="text-sm font-semibold text-primary">Leaders by task</h2>
                  <p className="mt-0.5 text-xs text-muted">
                    Ranked within a task only. Comparing latency across different tasks would be
                    meaningless.
                  </p>
                  <div className="mt-3 space-y-4">
                    {Object.entries(data.leaders_by_task).map(([task, leaders]) => (
                      <div key={task}>
                        <h3 className="label">
                          {task.replace(/_/g, ' ')} · {leaders.run_count} run(s)
                        </h3>
                        <div className="mt-1 divide-y divide-subtle">
                          <LeaderRow
                            title="Lowest p50 latency"
                            run={leaders.fastest}
                            metric={
                              leaders.fastest?.latency_p50_ms != null
                                ? `${leaders.fastest.latency_p50_ms.toFixed(2)} ms`
                                : '—'
                            }
                          />
                          <LeaderRow
                            title="Lowest peak memory"
                            run={leaders.most_memory_efficient}
                            metric={
                              leaders.most_memory_efficient?.peak_rss_mb != null
                                ? `${leaders.most_memory_efficient.peak_rss_mb.toFixed(0)} MB`
                                : '—'
                            }
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </section>

                <section className="card card-pad">
                  <h2 className="text-sm font-semibold text-primary">Recent runs</h2>
                  <div className="mt-3 overflow-x-auto">
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th scope="col">Run</th>
                          <th scope="col">Model</th>
                          <th scope="col" className="text-right">p50 ms</th>
                          <th scope="col">Status</th>
                        </tr>
                      </thead>
                      <tbody>
                        {data.recent_runs.map((run) => (
                          <tr key={run.run_id}>
                            <td>
                              <Link
                                to={`/lab/results/${run.run_id}`}
                                className="font-mono text-xs text-accent hover:underline"
                              >
                                {run.run_id.slice(0, 12)}
                              </Link>
                            </td>
                            <td className="text-xs">{run.model_id}</td>
                            <td className="num">{run.latency_p50_ms?.toFixed(2) ?? '—'}</td>
                            <td className="text-xs">{run.status}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <Link
                    to="/lab/results"
                    className="mt-3 inline-block text-xs text-accent hover:underline"
                  >
                    all results →
                  </Link>
                </section>
              </div>

              {data.recent_failures.length > 0 && (
                <section className="card card-pad">
                  <h2 className="text-sm font-semibold text-primary">Recent failures</h2>
                  <p className="mt-0.5 text-xs text-muted">
                    Failed and partial runs are retained rather than discarded — a run that
                    disappeared would hide a real problem.
                  </p>
                  <ul className="mt-2 space-y-1 text-sm">
                    {data.recent_failures.map((run) => (
                      <li key={run.run_id} className="flex justify-between gap-3">
                        <Link
                          to={`/lab/results/${run.run_id}`}
                          className="font-mono text-xs text-accent hover:underline"
                        >
                          {run.run_id}
                        </Link>
                        <span className="text-xs text-muted">
                          {run.model_id} · {run.status} · {run.failed_iterations} failed iteration(s)
                        </span>
                      </li>
                    ))}
                  </ul>
                </section>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
