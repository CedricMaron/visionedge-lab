import { useEffect, useMemo, useState } from 'react';
import { api } from '@/services/api';
import { useAsync } from '@/hooks/useAsync';
import { useModelSwitchStore } from '@/stores/modelSwitchStore';
import { useClassStore } from '@/stores/classStore';
import { ModelCard } from '@/components/ModelCard';
import { PageHeader, Spinner, ErrorState, Field } from '@/components/ui';
import { Icon } from '@/components/Icon';
import type { ModelEntry, ModelsResponse } from '@/types';

const INPUT_SIZES = [320, 416, 512, 640, 960, 1280];
const EXEC_LOCATIONS = ['server', 'browser', 'edge'] as const;

export default function ModelSelectorPage() {
  const { data, error, loading, reload } = useAsync<ModelsResponse>((s) => api.models(s), []);
  const draft = useModelSwitchStore((s) => s.draft);
  const setDraft = useModelSwitchStore((s) => s.setDraft);
  const applySwitch = useModelSwitchStore((s) => s.applySwitch);
  const status = useModelSwitchStore((s) => s.status);
  const message = useModelSwitchStore((s) => s.message);
  const selectedIds = useClassStore((s) => s.selectedIds);

  const [selected, setSelected] = useState<ModelEntry | null>(null);

  const models = data?.detection_models ?? [];

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
              <p className="text-sm text-slate-500">No detection models reported.</p>
            )}
          </div>

          {/* Config panel */}
          <aside className="lg:sticky lg:top-4 lg:self-start">
            <div className="card card-pad space-y-4">
              <h2 className="text-sm font-semibold text-slate-200">Switch configuration</h2>
              {selected ? (
                <>
                  <div className="rounded-lg bg-surface-900 px-3 py-2 text-xs">
                    <div className="text-slate-400">Model family / size / version</div>
                    <div className="mt-0.5 font-mono text-slate-200">
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
                        setDraft({ execution_location: e.target.value as (typeof EXEC_LOCATIONS)[number] })
                      }
                    >
                      {EXEC_LOCATIONS.map((l) => (
                        <option key={l} value={l}>
                          {l}
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

                  <p className="text-xs text-slate-500">
                    Applies to {selectedIds.length} allowed class{selectedIds.length === 1 ? '' : 'es'}{' '}
                    (from Class Selector).
                  </p>

                  <button className="btn-primary w-full" disabled={status === 'loading'} onClick={onApply}>
                    {status === 'loading' ? (
                      <>
                        <span className="h-4 w-4 animate-spin rounded-full border-2 border-surface-950/40 border-t-surface-950" />
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
                </>
              ) : (
                <p className="text-sm text-slate-500">Select a model to configure.</p>
              )}
            </div>
          </aside>
        </div>
      )}
    </div>
  );
}
