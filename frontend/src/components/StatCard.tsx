import type { ReactNode } from 'react';
import { Icon, type IconName } from './Icon';

export function StatCard({
  label,
  value,
  unit,
  icon,
  tone = 'neutral',
  sub,
}: {
  label: string;
  value: ReactNode;
  unit?: string;
  icon?: IconName;
  tone?: 'neutral' | 'good' | 'warn' | 'bad' | 'accent';
  sub?: string;
}) {
  const toneText: Record<string, string> = {
    neutral: 'text-slate-100',
    good: 'text-good',
    warn: 'text-warn',
    bad: 'text-bad',
    accent: 'text-accent',
  };
  return (
    <div className="card card-pad">
      <div className="flex items-center justify-between">
        <span className="label">{label}</span>
        {icon && <Icon name={icon} className="h-4 w-4 text-slate-500" />}
      </div>
      <div className="mt-2 flex items-baseline gap-1.5">
        <span className={`text-2xl font-semibold tabular-nums ${toneText[tone]}`}>{value}</span>
        {unit && <span className="text-sm text-slate-500">{unit}</span>}
      </div>
      {sub && <p className="mt-1 text-xs text-slate-500">{sub}</p>}
    </div>
  );
}
