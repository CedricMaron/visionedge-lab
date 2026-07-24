// Small shared UI primitives.

import type { ReactNode } from 'react';
import { Icon } from './Icon';

export function PageHeader({
  title,
  subtitle,
  actions,
}: {
  title: string;
  subtitle?: string;
  actions?: ReactNode;
}) {
  return (
    <header className="mb-5 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
      <div>
        <h1 className="text-xl font-semibold tracking-tight text-slate-100 sm:text-2xl">
          {title}
        </h1>
        {subtitle && <p className="mt-1 max-w-2xl text-sm text-slate-400">{subtitle}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </header>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="flex items-center gap-3 text-sm text-slate-400">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-surface-600 border-t-accent" />
      {label ?? 'Loading…'}
    </div>
  );
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="card card-pad flex flex-col items-start gap-3 border-bad/30 bg-bad/5">
      <div className="flex items-center gap-2 text-bad">
        <Icon name="alert" className="h-5 w-5" />
        <span className="font-medium">Request failed</span>
      </div>
      <p className="text-sm text-slate-300">{message}</p>
      {onRetry && (
        <button className="btn-ghost" onClick={onRetry}>
          <Icon name="refresh" className="h-4 w-4" /> Retry
        </button>
      )}
    </div>
  );
}

export function EmptyState({ title, hint }: { title: string; hint?: string }) {
  return (
    <div className="card card-pad text-center text-slate-400">
      <p className="font-medium text-slate-300">{title}</p>
      {hint && <p className="mt-1 text-sm">{hint}</p>}
    </div>
  );
}

export function Field({
  label,
  children,
  hint,
}: {
  label: string;
  children: ReactNode;
  hint?: string;
}) {
  return (
    <label className="block">
      <span className="label">{label}</span>
      <div className="mt-1.5">{children}</div>
      {hint && <span className="mt-1 block text-xs text-slate-500">{hint}</span>}
    </label>
  );
}

const TONE: Record<string, string> = {
  good: 'bg-good/15 text-good',
  warn: 'bg-warn/15 text-warn',
  bad: 'bg-bad/15 text-bad',
  neutral: 'bg-surface-700 text-slate-300',
  accent: 'bg-accent/15 text-accent',
};

export function Badge({
  children,
  tone = 'neutral',
}: {
  children: ReactNode;
  tone?: keyof typeof TONE | string;
}) {
  return <span className={`pill ${TONE[tone] ?? TONE.neutral}`}>{children}</span>;
}
