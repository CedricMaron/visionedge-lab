// LOCAL / SERVER provenance marker.
//
// Every metric in this app belongs to exactly one execution location, and mixing
// the two would make a comparison meaningless. This badge is the visual contract:
// wherever numbers appear, the location appears with them.

import type { ExecutionTarget } from '@/types/playground';

export function ExecutionBadge({
  target,
  className = '',
}: {
  target: ExecutionTarget;
  className?: string;
}) {
  const tone =
    target === 'local' ? 'bg-accent-soft text-accent' : 'bg-elevated text-secondary';
  return (
    <span className={`pill font-mono text-2xs uppercase tracking-wider ${tone} ${className}`}>
      {target}
    </span>
  );
}
