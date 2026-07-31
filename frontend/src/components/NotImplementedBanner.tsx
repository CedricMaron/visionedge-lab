// Honest placeholder banner for planned pages. Renders a real component shell
// with an explicit "not yet implemented" notice — never fake data or charts.

import type { ReactNode } from 'react';
import { Icon } from './Icon';
import { PageHeader } from './ui';

export function NotImplementedBanner({
  phase,
  description,
}: {
  phase: string;
  description: string;
}) {
  return (
    <div className="card card-pad border-warn/30 bg-warn/5">
      <div className="flex items-start gap-3">
        <div className="mt-0.5 rounded-lg bg-warn/15 p-2 text-warn">
          <Icon name="flask" className="h-5 w-5" />
        </div>
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <span className="pill bg-warn/15 text-warn">Planned — {phase}</span>
            <span className="text-sm font-medium text-primary">Not yet implemented in this build</span>
          </div>
          <p className="mt-2 max-w-2xl text-sm text-secondary">{description}</p>
          <p className="mt-2 text-xs text-muted">
            This screen is an honest shell: no simulated data, metrics, or charts are shown. The UI
            will light up when the backend slice ships.
          </p>
        </div>
      </div>
    </div>
  );
}

// Full planned-page scaffold: header + banner + optional planned-feature list.
export function PlannedPage({
  title,
  subtitle,
  phase,
  description,
  plannedFeatures,
  children,
}: {
  title: string;
  subtitle?: string;
  phase: string;
  description: string;
  plannedFeatures?: string[];
  children?: ReactNode;
}) {
  return (
    <div>
      <PageHeader title={title} subtitle={subtitle} />
      <NotImplementedBanner phase={phase} description={description} />
      {plannedFeatures && plannedFeatures.length > 0 && (
        <div className="card card-pad mt-4">
          <h2 className="label mb-3">Planned capabilities</h2>
          <ul className="space-y-2">
            {plannedFeatures.map((f) => (
              <li key={f} className="flex items-start gap-2 text-sm text-secondary">
                <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-accent/70" />
                {f}
              </li>
            ))}
          </ul>
        </div>
      )}
      {children}
    </div>
  );
}
