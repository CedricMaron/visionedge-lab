import type { VLMResponse } from '@/types';
import { formatMb, formatMs } from '@/utils/format';
import { Icon } from './Icon';
import { StatCard } from './StatCard';

export function VLMResponsePanel({ response }: { response: VLMResponse }) {
  const local =
    response.execution_location.toLowerCase().includes('local') ||
    response.execution_location.toLowerCase().includes('browser') ||
    response.execution_location.toLowerCase().includes('edge');

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <span className="pill bg-accent/15 text-accent">{response.model_id}</span>
        <span className="pill bg-surface-700 text-slate-300">{response.runtime}</span>
        <span
          className={`pill ${local ? 'bg-good/15 text-good' : 'bg-warn/15 text-warn'}`}
          title="Where the model executed"
        >
          <Icon name={local ? 'chip' : 'server'} className="h-3.5 w-3.5" />
          {local ? 'On-device (private)' : `Server (${response.execution_location})`}
        </span>
      </div>

      <div className="card card-pad">
        <h3 className="label mb-2">Response</h3>
        <p className="whitespace-pre-wrap text-sm leading-relaxed text-slate-200">
          {response.text || '(no text returned)'}
        </p>
      </div>

      {response.structured_output && (
        <div className="card card-pad">
          <h3 className="label mb-2">Structured output</h3>
          <pre className="overflow-x-auto rounded-lg bg-surface-950 p-3 text-xs text-slate-300">
            {JSON.stringify(response.structured_output, null, 2)}
          </pre>
        </div>
      )}

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        <StatCard label="Time to first token" value={formatMs(response.time_to_first_token_ms)} icon="clock" />
        <StatCard label="Generation" value={formatMs(response.generation_latency_ms)} icon="clock" />
        <StatCard label="Total latency" value={formatMs(response.total_latency_ms)} icon="gauge" tone="accent" />
        <StatCard label="Prompt tokens" value={response.prompt_tokens} />
        <StatCard label="Generated tokens" value={response.generated_tokens} />
        <StatCard label="Memory" value={formatMb(response.memory_usage_mb)} icon="chip" />
      </div>

      {response.warnings.length > 0 && (
        <div className="card card-pad border-warn/30 bg-warn/5">
          <h3 className="label mb-1 text-warn">Warnings</h3>
          <ul className="list-inside list-disc text-sm text-slate-300">
            {response.warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
