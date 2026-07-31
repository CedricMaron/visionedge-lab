/**
 * Hardware utilization over the life of a run.
 *
 * Renders only the series that were actually sampled. A probe that was unavailable
 * is named with its reason rather than drawn as a flat zero line, which would look
 * like an idle GPU rather than a missing sensor.
 */
import { useMemo } from 'react';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { UtilizationSeries } from '@/types/lab';

interface SeriesDef {
  key: string;
  label: string;
  unit: string;
  colour: string;
  axis: 'percent' | 'absolute';
}

const CANDIDATES: SeriesDef[] = [
  { key: 'cpu_percent', label: 'CPU', unit: '%', colour: 'var(--series-1)', axis: 'percent' },
  {
    key: 'process_cpu_percent',
    label: 'Process CPU',
    unit: '%',
    colour: 'var(--series-2)',
    axis: 'percent',
  },
  { key: 'gpu_percent', label: 'GPU', unit: '%', colour: 'var(--series-3)', axis: 'percent' },
  {
    key: 'gpu_power_w',
    label: 'GPU power',
    unit: 'W',
    colour: 'var(--series-5)',
    axis: 'absolute',
  },
  {
    key: 'gpu_temperature_c',
    label: 'GPU temp',
    unit: '°C',
    colour: 'var(--series-4)',
    axis: 'absolute',
  },
];

export function UtilizationChart({ series }: { series: UtilizationSeries }) {
  const { data, present } = useMemo(() => {
    const rows = series.samples.map((sample) => ({
      t: Number((sample.t_offset_ms / 1000).toFixed(2)),
      ...sample,
    }));
    // A series counts as present only if at least one sample carried a value —
    // otherwise it is a missing sensor, not a zero reading.
    const available = CANDIDATES.filter((candidate) =>
      series.samples.some(
        (sample) => (sample as unknown as Record<string, number | null>)[candidate.key] != null,
      ),
    );
    return { data: rows, present: available };
  }, [series]);

  if (series.samples.length === 0) {
    const reasons = Object.entries(series.unavailable);
    return (
      <div className="text-sm text-muted">
        <p>No utilization samples were collected for this run.</p>
        {reasons.length > 0 && (
          <ul className="mt-1 list-inside list-disc text-xs">
            {reasons.map(([probe, reason]) => (
              <li key={probe}>
                <span className="font-mono">{probe}</span>: {reason}
              </li>
            ))}
          </ul>
        )}
      </div>
    );
  }

  return (
    <div>
      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: -12 }}>
            <CartesianGrid stroke="rgb(var(--border-subtle))" strokeDasharray="2 4" />
            <XAxis
              dataKey="t"
              tick={{ fontSize: 11, fill: 'rgb(var(--text-muted))' }}
              stroke="rgb(var(--border-strong))"
              label={{
                value: 'seconds from run start',
                position: 'insideBottom',
                offset: -2,
                style: { fontSize: 11, fill: 'rgb(var(--text-muted))' },
              }}
            />
            <YAxis
              tick={{ fontSize: 11, fill: 'rgb(var(--text-muted))' }}
              stroke="rgb(var(--border-strong))"
            />
            <Tooltip
              contentStyle={{
                background: 'rgb(var(--overlay))',
                border: '1px solid rgb(var(--border-subtle))',
                borderRadius: 6,
                fontSize: 12,
                color: 'rgb(var(--text-primary))',
              }}
              formatter={(value: number, name: string) => {
                const def = present.find((s) => s.label === name);
                return [`${value?.toFixed?.(1) ?? value} ${def?.unit ?? ''}`, name];
              }}
              labelFormatter={(t) => `t = ${t}s`}
            />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            {present.map((def) => (
              <Line
                key={def.key}
                type="monotone"
                dataKey={def.key}
                name={def.label}
                stroke={`rgb(${def.colour})`}
                strokeWidth={1.5}
                dot={false}
                isAnimationActive={false}
                connectNulls
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      <p className="mt-2 text-xs text-muted">
        {series.samples.length} samples at {series.sample_interval_ms} ms
        {series.sampler_overhead_ms.value !== null && (
          <>
            {' '}
            · sampler cost {series.sampler_overhead_ms.value.toFixed(2)} ms per tick
            {series.sample_interval_ms > 0 && (
              <>
                {' '}
                (
                {(
                  (series.sampler_overhead_ms.value / series.sample_interval_ms) *
                  100
                ).toFixed(1)}
                % of the interval)
              </>
            )}
          </>
        )}
      </p>

      {Object.entries(series.unavailable).length > 0 && (
        <ul className="mt-1 list-inside list-disc text-xs text-muted">
          {Object.entries(series.unavailable).map(([probe, reason]) => (
            <li key={probe}>
              <span className="font-mono">{probe}</span> unavailable: {reason}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
