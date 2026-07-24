// Thin recharts wrapper for time-series line charts used on Performance.

import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

export interface SeriesDef {
  key: string;
  label: string;
  color: string;
}

export interface ChartPoint {
  t: number;
  [key: string]: number;
}

export function MetricChart({
  title,
  data,
  series,
  unit,
  height = 220,
}: {
  title: string;
  data: ChartPoint[];
  series: SeriesDef[];
  unit?: string;
  height?: number;
}) {
  return (
    <div className="card card-pad">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-medium text-slate-200">{title}</h3>
        {unit && <span className="text-xs text-slate-500">{unit}</span>}
      </div>
      <div style={{ width: '100%', height }}>
        <ResponsiveContainer>
          <LineChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -12 }}>
            <CartesianGrid stroke="#1c2333" strokeDasharray="3 3" />
            <XAxis
              dataKey="t"
              tick={{ fill: '#64748b', fontSize: 11 }}
              tickFormatter={(v: number) => `${v}s`}
              stroke="#28324a"
            />
            <YAxis tick={{ fill: '#64748b', fontSize: 11 }} stroke="#28324a" width={44} />
            <Tooltip
              contentStyle={{
                background: '#0f1420',
                border: '1px solid #28324a',
                borderRadius: 8,
                fontSize: 12,
              }}
              labelStyle={{ color: '#94a3b8' }}
              labelFormatter={(v) => `t = ${v}s`}
            />
            {series.map((s) => (
              <Line
                key={s.key}
                type="monotone"
                dataKey={s.key}
                name={s.label}
                stroke={s.color}
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-2 flex flex-wrap gap-3">
        {series.map((s) => (
          <span key={s.key} className="flex items-center gap-1.5 text-xs text-slate-400">
            <span className="h-2 w-2 rounded-full" style={{ background: s.color }} />
            {s.label}
          </span>
        ))}
      </div>
    </div>
  );
}
