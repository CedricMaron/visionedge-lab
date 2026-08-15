/**
 * Model library — every model this build has an adapter for.
 *
 * One card per model, showing what it does, what it costs, where it can run, and
 * under what licence. Availability is not declared: install status comes from the
 * server's own disk check, server runtimes from its probes, and local execution
 * from what the browser exposes.
 */
import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAsync } from '@/hooks/useAsync';
import { labApi } from '@/services/labApi';
import { ExecutionBadge } from '@/components/ExecutionBadge';
import { EmptyState, ErrorState, PageHeader, Spinner } from '@/components/ui';
import { localAvailability, serverAvailability, taskLabel } from '@/lab/catalog';
import { detectBrowserCaps, type BrowserCaps } from '@/utils/browserCaps';
import { formatBytes } from '@/utils/format';
import { api } from '@/services/api';
import type { Capabilities } from '@/types';
import type { LabModel, RuntimeCapability } from '@/types/lab';

type Filter = 'all' | 'vision' | 'text' | 'multimodal';

const FILTERS: { key: Filter; label: string }[] = [
  { key: 'all', label: 'All' },
  { key: 'vision', label: 'Vision' },
  { key: 'text', label: 'Text' },
  { key: 'multimodal', label: 'Multimodal' },
];

/** Structural outline of a model family. Architecture, not measurement. */
const ARCHITECTURE: Record<string, string[]> = {
  object_detection: ['Input', 'CSP backbone', 'Feature pyramid (PAN)', 'Detection head', 'Predictions'],
  image_classification: ['Input', 'Stem', 'Inverted-residual blocks', 'Global pooling', 'Classifier'],
  text_embedding: [
    'Tokens',
    'Embedding layer',
    'Transformer encoder × 6',
    'Attention-masked mean pooling',
    'L2 normalization',
  ],
};

function modalityGroup(model: LabModel): Filter {
  if (model.modality === 'text') return 'text';
  if (model.modality === 'multimodal') return 'multimodal';
  return 'vision';
}

function ModelCard({
  model,
  runtimes,
  caps,
  serverGpuCount,
  onUse,
}: {
  model: LabModel;
  runtimes: RuntimeCapability[];
  caps: BrowserCaps | null;
  serverGpuCount: number | null;
  onUse: () => void;
}) {
  const [details, setDetails] = useState(false);
  const server = serverAvailability(model, runtimes, serverGpuCount);
  const local = localAvailability(model, caps);
  const installed = model.deployment_status === 'installed';

  return (
    <article className="card card-pad flex flex-col">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="truncate text-sm font-semibold text-primary">{model.display_name}</h3>
          <p className="mt-0.5 truncate font-mono text-2xs text-muted">{model.model_id}</p>
        </div>
        <span
          className={`pill shrink-0 ${
            installed ? 'bg-good-soft text-good' : 'bg-elevated text-muted'
          }`}
        >
          {model.deployment_status.replace(/_/g, ' ')}
        </span>
      </div>

      <div className="mt-2 flex flex-wrap gap-1.5">
        <span className="pill bg-accent-soft text-accent">{taskLabel(model.task)}</span>
        <span className="pill bg-elevated text-secondary">{model.modality}</span>
        {model.commercial_use_permitted === false && (
          <span className="pill bg-warn-soft text-warn">non-commercial weights</span>
        )}
      </div>

      <dl className="mt-3 divide-y divide-subtle text-sm">
        {[
          ['Parameters', model.parameters_millions ? `${model.parameters_millions}M` : 'unknown'],
          ['Size on disk', formatBytes(model.file_size_bytes)],
          ['Input', model.input_size ? `${model.input_size} × ${model.input_size}` : model.modality],
          ['Precisions', model.supported_precisions.join(', ') || 'unknown'],
          ['Weights licence', model.weights_license],
        ].map(([term, value]) => (
          <div key={term} className="flex justify-between gap-3 py-1.5">
            <dt className="text-secondary">{term}</dt>
            <dd className="text-right font-mono text-xs text-primary">{value}</dd>
          </div>
        ))}
      </dl>

      <div className="mt-3 space-y-2">
        <div>
          <div className="mb-1 flex items-center gap-2">
            <ExecutionBadge target="local" />
            <span className="text-xs text-muted">
              {local.available ? 'available' : 'unavailable'}
            </span>
          </div>
          {!local.available && local.reason && (
            <p className="text-2xs leading-snug text-muted">{local.reason}</p>
          )}
        </div>
        <div>
          <div className="mb-1 flex items-center gap-2">
            <ExecutionBadge target="server" />
            <span className="text-xs text-muted">
              {server.available ? 'available' : 'unavailable'}
            </span>
          </div>
          {server.available ? (
            <ul className="space-y-0.5 text-2xs text-secondary">
              {server.runtimes.map((r) => (
                <li key={r.runtime_id} className="font-mono">
                  {r.runtime_id} · {r.devices.join(', ')}
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-2xs leading-snug text-muted">{server.reason}</p>
          )}
        </div>
      </div>

      {details && (
        <div className="mt-3 space-y-3 border-t border-subtle pt-3 text-xs">
          <div>
            <div className="label mb-1">Family structure</div>
            <ol className="space-y-0.5 font-mono text-2xs text-secondary">
              {(ARCHITECTURE[model.task] ?? ['Input', 'Model', 'Output']).map((step, i, arr) => (
                <li key={step}>
                  {step}
                  {i < arr.length - 1 && <span className="text-muted"> ↓</span>}
                </li>
              ))}
            </ol>
            <p className="mt-1 text-2xs text-muted">
              Typical structure of the {model.family} family, from its published architecture — not
              read from the graph.
            </p>
          </div>
          <div>
            <div className="label mb-1">Adapter</div>
            <p className="font-mono text-2xs text-secondary">{model.adapter}</p>
          </div>
          {model.companion_files.length > 0 && (
            <div>
              <div className="label mb-1">Companion files</div>
              <ul className="space-y-0.5 text-2xs text-muted">
                {model.companion_files.map((file) => (
                  <li key={file.file_name}>
                    <span className="font-mono text-secondary">{file.file_name}</span> —{' '}
                    {file.purpose}
                  </li>
                ))}
              </ul>
            </div>
          )}
          <div>
            <div className="label mb-1">Licences</div>
            <p className="text-2xs text-muted">
              code {model.model_license} · weights {model.weights_license}
            </p>
          </div>
          {model.notes && <p className="text-2xs text-muted">{model.notes}</p>}
          {!installed && model.install_hint && (
            <p className="text-2xs">
              <span className="text-muted">install: </span>
              <code className="text-primary">{model.install_hint}</code>
            </p>
          )}
          <div className="flex gap-3">
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
        </div>
      )}

      <div className="mt-4 flex gap-2">
        <button className="btn-primary flex-1" disabled={!server.available} onClick={onUse}>
          Use model
        </button>
        <button className="btn-ghost" onClick={() => setDetails((d) => !d)}>
          {details ? 'Less' : 'Details'}
        </button>
      </div>
    </article>
  );
}

export default function ModelsPage() {
  const navigate = useNavigate();
  const models = useAsync<{ models: LabModel[] }>((s) => labApi.models(s), []);
  const runtimes = useAsync<{ runtimes: RuntimeCapability[] }>((s) => labApi.runtimes(s), []);
  const hostCaps = useAsync<Capabilities>((s) => api.capabilities(s), []);
  const [filter, setFilter] = useState<Filter>('all');
  const [caps, setCaps] = useState<BrowserCaps | null>(null);
  useEffect(() => setCaps(detectBrowserCaps()), []);

  const visible = useMemo(
    () =>
      (models.data?.models ?? []).filter((m) => filter === 'all' || modalityGroup(m) === filter),
    [models.data, filter],
  );

  return (
    <div>
      <PageHeader
        title="Models"
        subtitle="Every model with an adapter in this build, with the licence, the install status read from disk, and where it can actually execute."
      />

      {(models.loading || runtimes.loading) && <Spinner label="Loading model library…" />}
      {models.error && <ErrorState message={models.error} onRetry={models.reload} />}

      {models.data && (
        <>
          <div className="mb-4 flex flex-wrap gap-1.5">
            {FILTERS.map((f) => (
              <button
                key={f.key}
                onClick={() => setFilter(f.key)}
                className={`pill border ${
                  filter === f.key
                    ? 'border-accent bg-accent-soft text-accent'
                    : 'border-subtle bg-panel text-secondary hover:border-strong'
                }`}
              >
                {f.label}
              </button>
            ))}
          </div>

          {visible.length === 0 ? (
            <EmptyState title="No model matches this filter" />
          ) : (
            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
              {visible.map((model) => (
                <ModelCard
                  key={model.model_id}
                  model={model}
                  runtimes={runtimes.data?.runtimes ?? []}
                  caps={caps}
                  serverGpuCount={hostCaps.data ? hostCaps.data.gpus.length : null}
                  onUse={() => navigate(`/?model=${encodeURIComponent(model.model_id)}`)}
                />
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
