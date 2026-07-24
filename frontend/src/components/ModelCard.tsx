import type { ModelEntry } from '@/types';
import { formatBytes, formatMb } from '@/utils/format';
import { Badge } from './ui';

function speedTone(cat: string): string {
  const c = cat.toLowerCase();
  if (c.includes('fast') || c.includes('real')) return 'good';
  if (c.includes('slow')) return 'warn';
  return 'accent';
}

function qualityTone(cat: string): string {
  const c = cat.toLowerCase();
  if (c.includes('high') || c.includes('best')) return 'good';
  if (c.includes('low')) return 'warn';
  return 'accent';
}

function statusTone(status: string): string {
  const s = status.toLowerCase();
  if (s.includes('installed') || s.includes('ready') || s.includes('deployed')) return 'good';
  if (s.includes('download') || s.includes('pending')) return 'warn';
  if (s.includes('unavailable') || s.includes('error')) return 'bad';
  return 'neutral';
}

export function ModelCard({
  model,
  active,
  onSelect,
}: {
  model: ModelEntry;
  active?: boolean;
  onSelect?: (m: ModelEntry) => void;
}) {
  const inputSize = Array.isArray(model.input_size)
    ? model.input_size.join('×')
    : model.input_size;

  return (
    <div
      className={`card card-pad flex flex-col gap-3 transition ${
        active ? 'border-accent/60 ring-1 ring-accent/40' : 'hover:border-surface-600'
      }`}
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="font-semibold text-slate-100">{model.display_name}</h3>
          <p className="mt-0.5 font-mono text-xs text-slate-500">{model.model_id}</p>
        </div>
        <Badge tone={statusTone(model.deployment_status)}>{model.deployment_status}</Badge>
      </div>

      <div className="flex flex-wrap gap-1.5">
        <Badge tone="neutral">{model.family}</Badge>
        <Badge tone="neutral">{model.size}</Badge>
        <Badge tone="neutral">{model.format}</Badge>
        <Badge tone="neutral">{model.precision}</Badge>
        <Badge tone={speedTone(model.speed_category)}>{model.speed_category}</Badge>
        <Badge tone={qualityTone(model.quality_category)}>{model.quality_category}</Badge>
      </div>

      <dl className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs">
        <div className="flex justify-between">
          <dt className="text-slate-500">Input</dt>
          <dd className="font-mono text-slate-300">{inputSize}</dd>
        </div>
        <div className="flex justify-between">
          <dt className="text-slate-500">Arch</dt>
          <dd className="text-slate-300">{model.architecture}</dd>
        </div>
        <div className="flex justify-between">
          <dt className="text-slate-500">File</dt>
          <dd className="font-mono text-slate-300">{formatBytes(model.file_size_bytes)}</dd>
        </div>
        <div className="flex justify-between">
          <dt className="text-slate-500">Est. RAM</dt>
          <dd className="font-mono text-slate-300">{formatMb(model.expected_memory_mb)}</dd>
        </div>
        <div className="col-span-2 flex justify-between">
          <dt className="text-slate-500">Devices</dt>
          <dd className="text-right text-slate-300">{model.supported_devices.join(', ')}</dd>
        </div>
        <div className="col-span-2 flex justify-between">
          <dt className="text-slate-500">Runtimes</dt>
          <dd className="text-right text-slate-300">{model.supported_runtimes.join(', ')}</dd>
        </div>
      </dl>

      {model.notes && <p className="text-xs text-slate-500">{model.notes}</p>}

      <div className="mt-auto flex items-center justify-between pt-1">
        <span className="text-[11px] text-slate-600">{model.license}</span>
        {onSelect && (
          <button className={active ? 'btn-ghost' : 'btn-primary'} onClick={() => onSelect(model)}>
            {active ? 'Configure' : 'Switch to this config'}
          </button>
        )}
      </div>
    </div>
  );
}
