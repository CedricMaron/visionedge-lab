import { useEffect, useMemo, useState } from 'react';
import { api } from '@/services/api';
import { useAsync } from '@/hooks/useAsync';
import { useModelSwitchStore } from '@/stores/modelSwitchStore';
import { useClassStore } from '@/stores/classStore';
import { ModelCard } from '@/components/ModelCard';
import { PageHeader, Spinner, ErrorState, Field } from '@/components/ui';
import { Icon } from '@/components/Icon';
import type { ExecutionLocation, JobRecord, ModelEntry, ModelsResponse } from '@/types';

const INPUT_SIZES = [320, 416, 512, 640, 960, 1280];
// Must match the backend ExecutionLocation enum exactly — anything else is a 422.
const EXEC_LOCATIONS: { value: ExecutionLocation; label: string }[] = [
  { value: 'pc_local', label: 'PC (local)' },
  { value: 'phone_local', label: 'Phone (local)' },
  { value: 'local_server', label: 'Local server' },
  { value: 'remote_server', label: 'Remote server' },
];

const TERMINAL_JOB_STATES = ['completed', 'failed', 'stopped'];

export function BenchmarkJobStatus({ job }: { job: JobRecord | null }) {
  if (!job) return null;
  if (job.state === 'failed') {
    return (
      <p className="mt-2 text-sm text-bad">Benchmark failed: {job.error ?? 'unknown error'}</p>
    );
  }
  if (job.state === 'running' || job.state === 'queued') {
    return (
      <p className="mt-2 text-sm text-secondary">
        Benchmarking this model in the background… {job.current_step} / {job.total_steps} runs
      </p>
    );
  }
  if (job.state === 'completed') {
    return <p className="mt-2 text-sm text-good">Benchmark recorded — see the Benchmarks page.</p>;
  }
  return null;
}

/** Poll the newest benchmark job after a successful switch until it finishes. */
function useBenchmarkJob(switchStatus: string): JobRecord | null {
  const [job, setJob] = useState<JobRecord | null>(null);

  useEffect(() => {
    if (switchStatus !== 'success') return;
    let cancelled = false;

    const tick = async () => {
      try {
        const { jobs } = await api.jobs();
        const latest =
          jobs
            .filter((j) => j.kind === 'benchmark')
            .sort((a, b) => b.created_at - a.created_at)[0] ?? null;
        if (cancelled) return;
        setJob(latest);
        if (latest && TERMINAL_JOB_STATES.includes(latest.state)) clearInterval(timer);
      } catch {
        // Job status is advisory — a failed poll must not break the page.
      }
    };

    const timer = setInterval(tick, 1000);
    void tick();
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [switchStatus]);

  return job;
}

export default function ModelSelectorPage() {
  const { data, error, loading, reload } = useAsync<ModelsResponse>((s) => api.models(s), []);
  const draft = useModelSwitchStore((s) => s.draft);
  const setDraft = useModelSwitchStore((s) => s.setDraft);
  const applySwitch = useModelSwitchStore((s) => s.applySwitch);
  const status = useModelSwitchStore((s) => s.status);
  const benchmarkJob = useBenchmarkJob(status);
  const message = useModelSwitchStore((s) => s.message);
  const selectedIds = useClassStore((s) => s.selectedIds);

  const [selected, setSelected] = useState<ModelEntry | null>(null);

  // Memoized so the effect below doesn't re-run on every render.
  const models = useMemo(() => data?.detection_models ?? [], [data]);

  useEffect(() => {
    if (!selected && models.length > 0) setSelected(models[0]);
  }, [models, selected]);

  function chooseModel(m: ModelEntry) {
    setSelected(m);
    const size = Array.isArray(m.input_size) ? m.input_size[0] : m.input_size;
    setDraft({
      model_id: m.model_id,
      runtime: m.supported_runtimes[0] ?? '',
      input_size: size,
    });
  }

  const runtimeOptions = useMemo(() => selected?.supported_runtimes ?? [], [selected]);

  async function onApply() {
    try {
      await applySwitch(selectedIds);
    } catch {
      /* status already reflects the error */
    }
  }

  return (
    <div>
      <PageHeader
        title="Model Selector"
        subtitle="Compare detection models and switch the active inference configuration."
        actions={
          <button className="btn-ghost" onClick={reload}>
            <Icon name="refresh" className="h-4 w-4" /> Refresh
          </button>
        }
      />

      {loading && <Spinner label="Loading models…" />}
      {error && <ErrorState message={error} onRetry={reload} />}

      {data && (
        <div className="grid gap-4 lg:grid-cols-[1fr_320px]">
          <div className="grid gap-4 sm:grid-cols-2">
            {models.map((m) => (
              <ModelCard
                key={m.model_id}
                model={m}
                active={selected?.model_id === m.model_id}
                onSelect={chooseModel}
              />
            ))}
            {models.length === 0 && (
              <p className="text-sm text-muted">No detection models reported.</p>
            )}
          </div>

          {/* Config panel */}
          <aside className="lg:sticky lg:top-4 lg:self-start">
            <div className="card card-pad space-y-4">
              <h2 className="text-sm font-semibold text-primary">Switch configuration</h2>
              {selected ? (
                <>
                  <div className="rounded-lg bg-elevated px-3 py-2 text-xs">
                    <div className="text-secondary">Model family / size / version</div>
                    <div className="mt-0.5 font-mono text-primary">
                      {selected.family} · {selected.size} · {selected.version}
                    </div>
                  </div>

                  <Field label="Execution provider / runtime">
                    <select
                      className="input"
                      value={draft.runtime}
                      onChange={(e) => setDraft({ runtime: e.target.value })}
                    >
                      {runtimeOptions.map((r) => (
                        <option key={r} value={r}>
                          {r}
                        </option>
                      ))}
                    </select>
                  </Field>

                  <Field label="Format / precision" hint={`format: ${selected.format}`}>
                    <select
                      className="input"
                      value={draft.model_id}
                      onChange={(e) => setDraft({ model_id: e.target.value })}
                    >
                      <option value={selected.model_id}>
                        {selected.precision} ({selected.model_id})
                      </option>
                    </select>
                  </Field>

                  <Field label="Input resolution">
                    <select
                      className="input"
                      value={draft.input_size}
                      onChange={(e) => setDraft({ input_size: Number(e.target.value) })}
                    >
                      {INPUT_SIZES.map((s) => (
                        <option key={s} value={s}>
                          {s} × {s}
                        </option>
                      ))}
                    </select>
                  </Field>

                  <Field label="Execution location">
                    <select
                      className="input"
                      value={draft.execution_location}
                      onChange={(e) =>
                        setDraft({ execution_location: e.target.value as ExecutionLocation })
                      }
                    >
                      {EXEC_LOCATIONS.map((l) => (
                        <option key={l.value} value={l.value}>
                          {l.label}
                        </option>
                      ))}
                    </select>
                  </Field>

                  <div className="grid grid-cols-2 gap-2">
                    <Field label="Confidence">
                      <input
                        type="number"
                        step="0.05"
                        min="0"
                        max="1"
                        className="input"
                        value={draft.confidence}
                        onChange={(e) => setDraft({ confidence: Number(e.target.value) })}
                      />
                    </Field>
                    <Field label="IoU">
                      <input
                        type="number"
                        step="0.05"
                        min="0"
                        max="1"
                        className="input"
                        value={draft.iou}
                        onChange={(e) => setDraft({ iou: Number(e.target.value) })}
                      />
                    </Field>
                  </div>

                  <p className="text-xs text-muted">
                    Applies to {selectedIds.length} allowed class{selectedIds.length === 1 ? '' : 'es'}{' '}
                    (from Class Selector).
                  </p>

                  <button className="btn-primary w-full" disabled={status === 'loading'} onClick={onApply}>
                    {status === 'loading' ? (
                      <>
                        <span className="h-4 w-4 animate-spin rounded-full border-2 border-accent-contrast border-t-accent-contrast" />
                        Applying…
                      </>
                    ) : (
                      'Switch to this config'
                    )}
                  </button>

                  {status !== 'idle' && status !== 'loading' && (
                    <div
                      className={`rounded-lg border px-3 py-2 text-sm ${
                        status === 'success'
                          ? 'border-good/30 bg-good/10 text-good'
                          : status === 'rolled_back'
                            ? 'border-warn/30 bg-warn/10 text-warn'
                            : 'border-bad/30 bg-bad/10 text-bad'
                      }`}
                    >
                      {status === 'rolled_back' && <strong className="mr-1">Rolled back:</strong>}
                      {status === 'error' && <strong className="mr-1">Error:</strong>}
                      {message}
                    </div>
                  )}

                  <BenchmarkJobStatus job={benchmarkJob} />
                </>
              ) : (
                <p className="text-sm text-muted">Select a model to configure.</p>
              )}
            </div>
          </aside>
        </div>
      )}
    </div>
  );
}
