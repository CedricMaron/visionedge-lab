/**
 * Raw per-iteration samples — the evidence behind every aggregate.
 *
 * Warm-up and failed iterations are shown, marked, and excluded from the summary
 * strip. Hiding them would make the aggregates unauditable; silently mixing them in
 * would make the aggregates wrong.
 */
import { useMemo, useState } from 'react';
import type { IterationSample } from '@/types/lab';

const GROUP_TONE: Record<string, string> = {
  warmup: 'bg-elevated text-muted',
  measured: 'bg-accent-soft text-accent',
  cold_start: 'bg-warn-soft text-warn',
};

function phaseMs(iteration: IterationSample, phase: string): string {
  const span = iteration.spans.find((s) => s.phase === phase && s.parent === null);
  return span ? span.duration_ms.toFixed(2) : '—';
}

export function IterationTable({ iterations }: { iterations: IterationSample[] }) {
  const [showAll, setShowAll] = useState(false);
  const [onlyCounted, setOnlyCounted] = useState(false);

  const { visible, counted, phases } = useMemo(() => {
    const countedRows = iterations.filter((it) => it.succeeded && it.group === 'measured');
    const filtered = onlyCounted ? countedRows : iterations;
    // Column set is derived from the data, so a text pipeline shows `tokenization`
    // rather than an empty `preprocessing` column.
    const phaseNames = [
      ...new Set(
        iterations.flatMap((it) => it.spans.filter((s) => s.parent === null).map((s) => s.phase)),
      ),
    ];
    return {
      visible: showAll ? filtered : filtered.slice(0, 25),
      counted: countedRows,
      phases: phaseNames,
    };
  }, [iterations, showAll, onlyCounted]);

  if (iterations.length === 0) {
    return <p className="text-sm text-muted">No iterations were recorded.</p>;
  }

  const total = onlyCounted ? counted.length : iterations.length;

  return (
    <div>
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <p className="text-xs text-muted">
          {iterations.length} recorded ·{' '}
          <strong className="text-secondary">{counted.length}</strong> counted toward statistics ·{' '}
          {iterations.filter((i) => i.group === 'warmup').length} warm-up ·{' '}
          {iterations.filter((i) => !i.succeeded).length} failed
        </p>
        <label className="flex items-center gap-2 text-xs text-secondary">
          <input
            type="checkbox"
            checked={onlyCounted}
            onChange={(e) => setOnlyCounted(e.target.checked)}
          />
          show only iterations counted in statistics
        </label>
      </div>

      <div className="max-h-96 overflow-auto rounded border border-subtle">
        <table className="data-table">
          <thead className="sticky top-0 bg-panel">
            <tr>
              <th scope="col">#</th>
              <th scope="col">Group</th>
              {phases.map((phase) => (
                <th key={phase} scope="col" className="text-right">
                  {phase.replace(/_/g, ' ')}
                </th>
              ))}
              <th scope="col" className="text-right">Total ms</th>
              <th scope="col">Result</th>
            </tr>
          </thead>
          <tbody>
            {visible.map((iteration) => (
              <tr
                key={iteration.index}
                className={!iteration.succeeded ? 'bg-bad-soft/40' : undefined}
              >
                <td className="font-mono text-xs">{iteration.index}</td>
                <td>
                  <span className={`pill ${GROUP_TONE[iteration.group] ?? ''}`}>
                    {iteration.group}
                  </span>
                </td>
                {phases.map((phase) => (
                  <td key={phase} className="num text-xs">
                    {phaseMs(iteration, phase)}
                  </td>
                ))}
                <td className="num text-xs">
                  {iteration.total_ms === null ? '—' : iteration.total_ms.toFixed(2)}
                </td>
                <td className="text-xs">
                  {iteration.succeeded ? (
                    <span className="text-good">ok</span>
                  ) : (
                    <span className="text-bad" title={iteration.error_message ?? undefined}>
                      {iteration.error_type}
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {total > 25 && (
        <button className="btn-ghost mt-2" onClick={() => setShowAll((s) => !s)}>
          {showAll ? 'show first 25' : `show all ${total}`}
        </button>
      )}
    </div>
  );
}
