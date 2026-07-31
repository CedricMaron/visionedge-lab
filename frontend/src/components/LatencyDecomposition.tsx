/**
 * End-to-end latency decomposition.
 *
 * The point of the platform in one component: not "32 ms of inference" but where
 * every millisecond went, from input decoding through device execution to
 * serialization — and, explicitly, the part that no phase accounted for.
 *
 * Two presentational rules follow from the measurement rules:
 *
 * 1. Unattributed time is shown as **residual overhead**, in a visually distinct
 *    hatched bar. It is never folded into a neighbouring phase and never labelled
 *    "network", because it is not known to be either.
 * 2. A phase timed without device synchronization is flagged, since on an
 *    asynchronous device it measures dispatch rather than execution.
 */
import { useMemo } from 'react';
import type { PhaseBreakdown } from '@/types/lab';
import { buildLatencyRows, phaseTone } from './latencyRows';

function ms(value: number | null | undefined): string {
  if (value === null || value === undefined) return '—';
  if (Math.abs(value) < 0.01) return '<0.01';
  return value.toFixed(2);
}

interface Props {
  breakdown: PhaseBreakdown;
  /** Set when any model_execution span was timed without device synchronization. */
  unsynchronized?: boolean;
  className?: string;
}

export function LatencyDecomposition({ breakdown, unsynchronized, className }: Props) {
  const { rows, totalMs } = useMemo(() => buildLatencyRows(breakdown), [breakdown]);

  if (breakdown.total.n === 0) {
    return (
      <div className="card card-pad text-sm text-muted">
        No measured iterations completed, so there is no latency to decompose.
      </div>
    );
  }

  return (
    <section className={className} aria-label="End-to-end latency decomposition">
      {/* Proportional bar: the whole pipeline at a glance. */}
      <div
        className="flex h-7 w-full overflow-hidden rounded border border-subtle"
        role="img"
        aria-label={rows
          .map((r) => `${r.label} ${ms(r.valueMs)} milliseconds`)
          .join(', ')}
      >
        {rows.map((row) => (
          <div
            key={row.phase}
            className={`${row.isResidual ? 'bg-elevated' : phaseTone(row.phase)} relative`}
            style={{
              width: `${Math.max(0, row.share) * 100}%`,
              // Hatching marks the residual as "not a measured phase" without
              // relying on colour alone.
              backgroundImage: row.isResidual
                ? 'repeating-linear-gradient(45deg, rgb(var(--border-strong)) 0 3px, transparent 3px 6px)'
                : undefined,
            }}
            title={`${row.label}: ${ms(row.valueMs)} ms (${(row.share * 100).toFixed(1)}%)`}
          />
        ))}
      </div>

      <table className="data-table mt-4">
        <thead>
          <tr>
            <th scope="col">Phase</th>
            <th scope="col" className="text-right">Mean ms</th>
            <th scope="col" className="text-right">p50</th>
            <th scope="col" className="text-right">p95</th>
            <th scope="col" className="text-right">p99</th>
            <th scope="col" className="text-right">Share</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.phase}>
              <th scope="row" className="px-3 py-2 text-left text-sm font-normal">
                <span className="flex items-center gap-2">
                  <span
                    aria-hidden
                    className={`inline-block h-2.5 w-2.5 shrink-0 rounded-sm ${
                      row.isResidual ? 'border border-strong bg-elevated' : phaseTone(row.phase)
                    }`}
                  />
                  <span className={row.isResidual ? 'italic text-muted' : 'text-primary'}>
                    {row.label}
                  </span>
                </span>
                {row.isResidual && (
                  <span className="mt-0.5 block pl-4.5 text-2xs text-muted">
                    measured total minus the sum of measured phases — not attributed to any
                    stage
                  </span>
                )}
              </th>
              <td className="num">{ms(row.valueMs)}</td>
              <td className="num">{ms(row.stats?.p50_ms)}</td>
              <td className="num">{ms(row.stats?.p95_ms)}</td>
              <td className="num">{ms(row.stats?.p99_ms)}</td>
              <td className="num">{(row.share * 100).toFixed(1)}%</td>
            </tr>
          ))}
        </tbody>
        <tfoot>
          <tr className="font-semibold">
            <th scope="row" className="px-3 py-2 text-left text-sm text-primary">
              End-to-end
            </th>
            <td className="num">{ms(breakdown.total.mean_ms)}</td>
            <td className="num">{ms(breakdown.total.p50_ms)}</td>
            <td className="num">{ms(breakdown.total.p95_ms)}</td>
            <td className="num">{ms(breakdown.total.p99_ms)}</td>
            <td className="num">100%</td>
          </tr>
        </tfoot>
      </table>

      <p className="mt-2 text-xs text-muted">
        Aggregated over <strong className="text-secondary">{breakdown.total.n}</strong> measured
        iterations · min {ms(breakdown.total.min_ms)} ms · max {ms(breakdown.total.max_ms)} ms
        {breakdown.total.stddev_ms !== null && <> · σ {ms(breakdown.total.stddev_ms)} ms</>}
        {breakdown.total.coefficient_of_variation !== null && (
          <> · CV {breakdown.total.coefficient_of_variation.toFixed(3)}</>
        )}
        {totalMs > 0 && <> · warm-up and failed iterations excluded</>}
      </p>

      {unsynchronized && (
        <p className="mt-2 rounded border border-warn/40 bg-warn-soft px-3 py-2 text-xs text-warn">
          Model execution was timed without device synchronization. On an asynchronous device
          this measures kernel dispatch rather than completion, and would understate the true
          execution time.
        </p>
      )}
    </section>
  );
}
