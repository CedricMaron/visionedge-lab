/**
 * The pipeline as it was actually executed.
 *
 * Stages come from the backend trace, not from a diagram hardcoded for one model:
 * a detection run and a text-embedding run produce different stages because their
 * adapters ran different phases. A step the adapter performs internally without its
 * own clock is listed with its shapes and an explicit "not separately instrumented"
 * note rather than an invented duration.
 */
import { useState } from 'react';
import type { PipelineStage, TensorInfo } from '@/types/playground';
import { formatBytes } from '@/utils/format';

function num(value: number | null, digits = 3): string {
  if (value === null || Number.isNaN(value)) return '—';
  const abs = Math.abs(value);
  if (abs !== 0 && abs < 0.001) return value.toExponential(2);
  return value.toFixed(digits);
}

export function TensorCard({ tensor }: { tensor: TensorInfo }) {
  return (
    <div className="rounded border border-subtle bg-elevated px-3 py-2">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <span className="font-mono text-xs text-primary">{tensor.name}</span>
        <span className="font-mono text-2xs text-muted">
          {tensor.dtype}
          {tensor.layout ? ` · ${tensor.layout}` : ''} · {tensor.device}
        </span>
      </div>
      <div className="mt-1 font-mono text-sm text-primary">[{tensor.shape.join(', ')}]</div>
      <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-2xs sm:grid-cols-3">
        {[
          ['memory', formatBytes(tensor.bytes)],
          ['elements', tensor.elements.toLocaleString()],
          ['min', num(tensor.min)],
          ['max', num(tensor.max)],
          ['mean', num(tensor.mean)],
          ['std', num(tensor.std)],
        ].map(([term, value]) => (
          <div key={term} className="flex justify-between gap-2">
            <dt className="text-muted">{term}</dt>
            <dd className="font-mono text-secondary">{value}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function StageCard({ stage, expanded, onToggle }: {
  stage: PipelineStage;
  expanded: boolean;
  onToggle: () => void;
}) {
  const inspectable = stage.tensors.length > 0 || stage.substeps.length > 0;
  return (
    <div className="card overflow-hidden">
      <button
        type="button"
        onClick={onToggle}
        disabled={!inspectable}
        className="flex w-full items-start justify-between gap-3 px-3 py-2.5 text-left disabled:cursor-default sm:px-4"
      >
        <span className="min-w-0">
          <span className="block text-sm font-medium text-primary">{stage.name}</span>
          {stage.detail && (
            <span className="mt-0.5 block truncate font-mono text-2xs text-muted">
              {stage.detail}
            </span>
          )}
          {(stage.device || stage.runtime) && (
            <span className="mt-0.5 block font-mono text-2xs text-muted">
              {[stage.device, stage.runtime].filter(Boolean).join(' · ')}
            </span>
          )}
        </span>
        <span className="shrink-0 text-right">
          <span className="block font-mono text-sm text-primary">
            {stage.duration_ms === null ? (
              <span className="text-2xs italic text-muted">not timed</span>
            ) : (
              `${stage.duration_ms.toFixed(2)} ms`
            )}
          </span>
          {inspectable && (
            <span className="mt-0.5 block text-2xs text-accent">
              {expanded ? 'hide' : 'inspect'}
            </span>
          )}
        </span>
      </button>

      {expanded && (
        <div className="space-y-3 border-t border-subtle px-3 py-3 sm:px-4">
          {stage.substeps.length > 0 && (
            <ol className="space-y-1.5">
              {stage.substeps.map((step) => (
                <li key={step.name} className="text-xs">
                  <span className="text-secondary">{step.name}</span>
                  {step.detail && <span className="text-muted"> — {step.detail}</span>}
                  <span className="ml-1 text-2xs italic text-muted">({step.note})</span>
                </li>
              ))}
            </ol>
          )}
          {stage.tensors.length > 0 && (
            <div className="space-y-2">
              <div className="label">Tensors</div>
              {stage.tensors.map((tensor) => (
                <TensorCard key={`${stage.id}-${tensor.name}`} tensor={tensor} />
              ))}
            </div>
          )}
          {stage.note && <p className="text-2xs text-muted">{stage.note}</p>}
        </div>
      )}
    </div>
  );
}

export function PipelineFlow({ stages }: { stages: PipelineStage[] }) {
  const [open, setOpen] = useState<string | null>(null);
  return (
    <div className="space-y-0">
      {stages.map((stage, index) => (
        <div key={stage.id}>
          <StageCard
            stage={stage}
            expanded={open === stage.id}
            onToggle={() => setOpen(open === stage.id ? null : stage.id)}
          />
          {index < stages.length - 1 && (
            <div className="flex justify-center py-1" aria-hidden="true">
              <span className="text-muted">↓</span>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

/** Measured stages only, as proportional CSS bars. Untimed stages are listed
 *  beneath rather than drawn, because a bar implies a measured width. */
export function PipelineTimeline({ stages }: { stages: PipelineStage[] }) {
  const timed = stages.filter((s) => s.duration_ms !== null && s.duration_ms > 0);
  const untimed = stages.filter((s) => s.duration_ms === null);
  const total = timed.reduce((sum, s) => sum + (s.duration_ms ?? 0), 0);

  if (timed.length === 0) {
    return <p className="text-sm text-muted">No stage of this run was separately timed.</p>;
  }

  return (
    <div>
      <div className="space-y-1.5">
        {timed.map((stage) => {
          const ms = stage.duration_ms ?? 0;
          const width = total > 0 ? Math.max(1, (ms / total) * 100) : 0;
          return (
            <div key={stage.id} className="grid grid-cols-[7rem_1fr_4.5rem] items-center gap-2 sm:grid-cols-[10rem_1fr_5.5rem]">
              <span className="truncate text-xs text-secondary">{stage.name}</span>
              <span className="h-3 rounded-sm bg-elevated">
                <span
                  className="block h-3 rounded-sm bg-accent"
                  style={{ width: `${width}%` }}
                  title={`${ms.toFixed(2)} ms · ${width.toFixed(1)}% of the timed total`}
                />
              </span>
              <span className="text-right font-mono text-xs text-primary">{ms.toFixed(2)} ms</span>
            </div>
          );
        })}
      </div>
      <p className="mt-2 text-2xs text-muted">
        {total.toFixed(2)} ms across {timed.length} measured stages.
        {untimed.length > 0 && (
          <> Not separately timed: {untimed.map((s) => s.name).join(', ')}.</>
        )}
      </p>
    </div>
  );
}
