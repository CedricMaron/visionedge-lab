import { describe, expect, it } from 'vitest';
import { buildLatencyRows } from './latencyRows';
import type { DurationStats, PhaseBreakdown } from '@/types/lab';

function stats(meanMs: number, n = 10): DurationStats {
  return {
    n,
    min_ms: meanMs * 0.9,
    max_ms: meanMs * 1.1,
    mean_ms: meanMs,
    median_ms: meanMs,
    stddev_ms: 1,
    p50_ms: meanMs,
    p90_ms: meanMs * 1.05,
    p95_ms: meanMs * 1.08,
    p99_ms: meanMs * 1.1,
    coefficient_of_variation: 0.1,
  };
}

function breakdown(
  phases: Record<string, number>,
  totalMs: number,
  residual: number | null = null,
): PhaseBreakdown {
  return {
    phases: Object.fromEntries(Object.entries(phases).map(([k, v]) => [k, stats(v)])),
    total: stats(totalMs),
    residual_ms: residual,
  };
}

describe('buildLatencyRows', () => {
  it('orders phases by pipeline position, not by magnitude', () => {
    // Postprocessing is tiny and model execution is huge; pipeline order must win,
    // otherwise the chart stops reading as a pipeline.
    const { rows } = buildLatencyRows(
      breakdown({ postprocessing: 1, model_execution: 50, preprocessing: 8 }, 59),
    );
    expect(rows.map((r) => r.phase)).toEqual([
      'preprocessing',
      'model_execution',
      'postprocessing',
    ]);
  });

  it('places tokenization where preprocessing would sit for text pipelines', () => {
    const { rows } = buildLatencyRows(
      breakdown({ model_execution: 20, tokenization: 2, postprocessing: 0.3 }, 22.3),
    );
    expect(rows.map((r) => r.phase)).toEqual([
      'tokenization',
      'model_execution',
      'postprocessing',
    ]);
  });

  it('surfaces residual overhead as its own row', () => {
    // The whole point: unattributed time must remain visible rather than being
    // folded into a neighbouring phase.
    const { rows } = buildLatencyRows(
      breakdown({ preprocessing: 5, model_execution: 10 }, 20, 5),
    );
    const residual = rows.find((r) => r.isResidual);
    expect(residual).toBeDefined();
    expect(residual!.valueMs).toBe(5);
    expect(residual!.label).toMatch(/residual/i);
  });

  it('omits a negligible residual', () => {
    const { rows } = buildLatencyRows(
      breakdown({ preprocessing: 5, model_execution: 10 }, 15, 0.001),
    );
    expect(rows.some((r) => r.isResidual)).toBe(false);
  });

  it('omits the residual when the backend reports none', () => {
    const { rows } = buildLatencyRows(breakdown({ model_execution: 10 }, 10, null));
    expect(rows.some((r) => r.isResidual)).toBe(false);
  });

  it('computes shares against the measured total', () => {
    const { rows, totalMs } = buildLatencyRows(
      breakdown({ preprocessing: 25, model_execution: 75 }, 100),
    );
    expect(totalMs).toBe(100);
    expect(rows.find((r) => r.phase === 'model_execution')!.share).toBeCloseTo(0.75);
    expect(rows.find((r) => r.phase === 'preprocessing')!.share).toBeCloseTo(0.25);
  });

  it('does not divide by zero when nothing was measured', () => {
    const empty = breakdown({}, 0);
    const { rows } = buildLatencyRows(empty);
    expect(rows).toEqual([]);
  });

  it('sorts unrecognized phases last rather than first', () => {
    const { rows } = buildLatencyRows(
      breakdown({ some_future_phase: 1, model_execution: 10 }, 11),
    );
    expect(rows[rows.length - 1].phase).toBe('some_future_phase');
  });

  it('keeps server-side remote phases in transport order', () => {
    const { rows } = buildLatencyRows(
      breakdown(
        { download: 3, upload: 2, server_model_execution: 20, server_queue: 1 },
        26,
      ),
    );
    expect(rows.map((r) => r.phase)).toEqual([
      'upload',
      'server_queue',
      'server_model_execution',
      'download',
    ]);
  });
});
