/**
 * Model catalogue.
 *
 * Licensing is shown as two fields, because code and weights genuinely differ and
 * only reporting one would mislead someone deciding whether they may ship a model.
 * Install status is derived from disk on the server, so a model listed as installed
 * is one that will actually load.
 */
import { useMemo, useState } from 'react';
import { useAsync } from '@/hooks/useAsync';
import { labApi } from '@/services/labApi';
import { EmptyState, ErrorState, PageHeader, Spinner } from '@/components/ui';
import type { LabModel } from '@/types/lab';

const STATUS_TONE: Record<LabModel['deployment_status'], string> = {
  installed: 'bg-good-soft text-good',
  not_installed: 'bg-elevated text-muted',
  incomplete: 'bg-warn-soft text-warn',
  missing: 'bg-bad-soft text-bad',
};

function formatBytes(bytes: number | null): string {
  if (bytes === null) return '—';
  const mb = bytes / (1024 * 1024);
  return mb >= 1024 ? `${(mb / 1024).toFixed(2)} GB` : `${mb.toFixed(1)} MB`;
}

function ModelCard({ model }: { model: LabModel }) {
  return (
    <article className="card card-pad">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-primary">{model.display_name}</h3>
          <p className="mt-0.5 font-mono text-xs text-muted">{model.model_id}</p>
        </div>
        <span className={`pill shrink-0 ${STATUS_TONE[model.deployment_status]}`}>
          {model.deployment_status.replace(/_/g, ' ')}
        </span>
      </div>

      <div className="mt-3 flex flex-wrap gap-1.5">
        <span className="pill bg-accent-soft text-accent">{model.task.replace(/_/g, ' ')}</span>
        <span className="pill bg-elevated text-secondary">{model.modality}</span>
        <span className="pill bg-elevated text-secondary">{model.family}</span>
        {model.commercial_use_permitted === false && (
          <span className="pill bg-warn-soft text-warn">non-commercial weights</span>
        )}
        {model.commercial_use_permitted === null && (
          <span className="pill bg-elevated text-muted">licence unreviewed</span>
        )}
      </div>

      <dl className="mt-3 divide-y divide-subtle text-sm">
        {[
          ['Parameters', model.parameters_millions ? `${model.parameters_millions}M` : '—'],
          ['Size on disk', formatBytes(model.file_size_bytes)],
          ['Code licence', model.model_license],
          ['Weights licence', model.weights_license],
          ['Runtimes', model.supported_runtimes.join(', ') || '—'],
          ['Devices', model.supported_devices.join(', ') || '—'],
          ['Precisions', model.supported_precisions.join(', ') || '—'],
        ].map(([term, value]) => (
          <div key={term} className="flex justify-between gap-3 py-1.5">
            <dt className="text-secondary">{term}</dt>
            <dd className="text-right font-mono text-xs text-primary">{value}</dd>
          </div>
        ))}
      </dl>

      {model.companion_files.length > 0 && (
        <div className="mt-3">
          <div className="label mb-1">Required companion files</div>
          <ul className="space-y-1 text-xs text-muted">
            {model.companion_files.map((file) => (
              <li key={file.file_name}>
                <span className="font-mono text-secondary">{file.file_name}</span> — {file.purpose}
              </li>
            ))}
          </ul>
        </div>
      )}

      {model.deployment_status !== 'installed' && (
        <div className="mt-3 rounded border border-subtle bg-elevated px-3 py-2 text-xs">
          {model.not_installed_reason && (
            <p className="text-muted">{model.not_installed_reason}</p>
          )}
          {model.install_hint && (
            <p className="mt-1">
              <span className="text-muted">install: </span>
              <code className="text-primary">{model.install_hint}</code>
            </p>
          )}
        </div>
      )}

      {model.notes && <p className="mt-3 text-xs text-muted">{model.notes}</p>}

      <div className="mt-3 flex gap-3 text-xs">
        {model.source_repository && (
          <a
            className="text-accent hover:underline"
            href={model.source_repository}
            target="_blank"
            rel="noreferrer noopener"
          >
            repository
          </a>
        )}
        {model.paper_url && (
          <a
            className="text-accent hover:underline"
            href={model.paper_url}
            target="_blank"
            rel="noreferrer noopener"
          >
            paper
          </a>
        )}
      </div>
    </article>
  );
}

export default function LabModelsPage() {
  const { data, error, loading, reload } = useAsync<{ models: LabModel[] }>(
    (s) => labApi.models(s),
    [],
  );
  const [task, setTask] = useState('all');
  const [installedOnly, setInstalledOnly] = useState(false);

  const tasks = useMemo(
    () => ['all', ...new Set((data?.models ?? []).map((m) => m.task))],
    [data],
  );
  const visible = useMemo(
    () =>
      (data?.models ?? []).filter(
        (m) =>
          (task === 'all' || m.task === task) &&
          (!installedOnly || m.deployment_status === 'installed'),
      ),
    [data, task, installedOnly],
  );

  return (
    <div>
      <PageHeader
        title="Models"
        subtitle="Models with an adapter in this build. Status is derived from what is on disk, not declared."
      />

      {loading && <Spinner label="Loading models…" />}
      {error && <ErrorState message={error} onRetry={reload} />}

      {data && (
        <>
          <div className="mb-4 flex flex-wrap items-center gap-3">
            <div className="flex flex-wrap gap-1.5">
              {tasks.map((option) => (
                <button
                  key={option}
                  onClick={() => setTask(option)}
                  className={`pill border ${
                    task === option
                      ? 'border-accent bg-accent-soft text-accent'
                      : 'border-subtle bg-panel text-secondary hover:border-strong'
                  }`}
                >
                  {option.replace(/_/g, ' ')}
                </button>
              ))}
            </div>
            <label className="flex items-center gap-2 text-xs text-secondary">
              <input
                type="checkbox"
                checked={installedOnly}
                onChange={(e) => setInstalledOnly(e.target.checked)}
              />
              installed only
            </label>
          </div>

          {visible.length === 0 ? (
            <EmptyState title="No models match this filter" />
          ) : (
            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
              {visible.map((model) => (
                <ModelCard key={model.model_id} model={model} />
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
