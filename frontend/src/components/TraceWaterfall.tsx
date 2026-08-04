/**
 * Per-iteration span waterfall.
 *
 * A flame-chart-inspired view of where each individual iteration spent its time.
 * The aggregate decomposition answers "where does latency come from on average";
 * this answers "which iteration was the slow one, and which phase made it slow" —
 * the question a p99 outlier actually raises.
 *
 * Bars are laid out on a shared time axis so a long iteration is visibly long, not
 * normalized away. Warm-up and failed iterations are drawn and marked rather than
 * hidden, because an outlier that vanished from the chart would be the one worth
 * seeing.
 */
import { useMemo, useState } from 'react';
import { phaseTone } from './latencyRows';
import type { IterationSample } from '@/types/lab';

interface Props {
  iterations: IterationSample[];
}

const GROUP_MARK: Record<string, string> = {
  warmup: 'W',
  measured: '',
  cold_start: 'C',
};

export function TraceWaterfall({ iterations }: Props) {
  const [showWarmup, setShowWarmup] = useState(true);

  const { rows, scaleMs } = useMemo(() => {
    const visible = showWarmup
      ? iterations
      : iterations.filter((it) => it.group !== 'warmup');
    // Scale to the slowest visible iteration so relative cost stays readable; using
    // a per-row scale would make every iteration look equally expensive.
    const max = Math.max(1, ...visible.map((it) => it.total_ms ?? 0));
    return { rows: visible, scaleMs: max };
  }, [iterations, showWarmup]);

  if (iterations.length === 0) {
    return <p className="text-sm text-muted">No iterations were recorded.</p>;
  }

  const warmupCount = iterations.filter((it) => it.group === 'warmup').length;

  return (
    <div>
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <p className="text-xs text-muted">
          Each row is one iteration, drawn on a shared {scaleMs.toFixed(1)} ms axis.
        </p>
        {warmupCount > 0 && (
          <label className="flex items-center gap-2 text-xs text-secondary">
            <input
              type="checkbox"
              checked={showWarmup}
              onChange={(e) => setShowWarmup(e.target.checked)}
            />
            show {warmupCount} warm-up iteration(s)
          </label>
        )}
      </div>

      <div className="max-h-80 overflow-y-auto rounded border border-subtle p-2">
        {rows.map((iteration) => {
          const topLevel = iteration.spans.filter((s) => s.parent === null);
          const total = iteration.total_ms ?? 0;
          const widthPercent = (total / scaleMs) * 100;

          return (
            <div key={iteration.index} className="flex items-center gap-2 py-0.5">
              <span className="w-10 shrink-0 text-right font-mono text-2xs text-muted">
                {iteration.index}
                {GROUP_MARK[iteration.group] && (
                  <span className="ml-0.5 text-accent">{GROUP_MARK[iteration.group]}</span>
                )}
              </span>

              <div className="relative h-4 flex-1 rounded bg-elevated">
                {iteration.succeeded ? (
                  <div className="flex h-full" style={{ width: `${widthPercent}%` }}>
                    {topLevel.map((span, index) => (
                      <div
                        key={`${span.phase}-${index}`}
                        className={`${phaseTone(span.phase)} h-full first:rounded-l last:rounded-r`}
                        style={{
                          width: `${total > 0 ? (span.duration_ms / total) * 100 : 0}%`,
                          // Unsynchronized GPU spans measure dispatch, not execution.
                          // Marked visually so an implausibly short bar is explicable.
                          opacity:
                            span.phase === 'model_execution' && !span.device_synchronized
                              ? 0.55
                              : 1,
                        }}
                        title={`${span.phase.replace(/_/g, ' ')}: ${span.duration_ms.toFixed(2)} ms${
                          span.phase === 'model_execution' && !span.device_synchronized
                            ? ' (device not synchronized — dispatch time)'
                            : ''
                        }`}
                      />
                    ))}
                  </div>
                ) : (
                  <div
                    className="flex h-full items-center rounded bg-bad-soft px-2 text-2xs text-bad"
                    title={iteration.error_message ?? undefined}
                  >
                    {iteration.error_type ?? 'failed'}
                  </div>
                )}
              </div>

              <span className="w-16 shrink-0 text-right font-mono text-2xs text-secondary">
                {iteration.total_ms === null ? '—' : `${iteration.total_ms.toFixed(1)} ms`}
              </span>
            </div>
          );
        })}
      </div>

      <p className="mt-2 text-xs text-muted">
        <span className="font-mono text-accent">W</span> marks warm-up iterations, which are
        recorded but excluded from statistics. Faded bars were timed without device
        synchronization.
      </p>
    </div>
  );
}
