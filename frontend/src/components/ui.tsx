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
        <h1 className="text-xl font-semibold tracking-tight text-primary sm:text-2xl">
          {title}
        </h1>
        {subtitle && <p className="mt-1 max-w-2xl text-sm text-secondary">{subtitle}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </header>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="flex items-center gap-3 text-sm text-secondary">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-strong border-t-accent" />
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
      <p className="text-sm text-secondary">{message}</p>
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
    <div className="card card-pad text-center text-secondary">
      <p className="font-medium text-secondary">{title}</p>
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
      {hint && <span className="mt-1 block text-xs text-muted">{hint}</span>}
    </label>
  );
}

const TONE: Record<string, string> = {
  good: 'bg-good/15 text-good',
  warn: 'bg-warn/15 text-warn',
  bad: 'bg-bad/15 text-bad',
  neutral: 'bg-elevated text-secondary',
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
