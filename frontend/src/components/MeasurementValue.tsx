/**
 * Renders a Measurement, including the case where there isn't one.
 *
 * This component is the UI half of the platform's central promise. A metric the
 * environment could not provide renders as an explicit "unavailable" with the
 * backend's reason attached — never as a dash, a zero, or a blank cell, each of
 * which reads as a value.
 *
 * Derived and estimated values are badged, because "18.3 ms measured" and
 * "18.3 ms inferred from other numbers" deserve different amounts of trust.
 */
import type { Measurement } from '@/types/lab';

const KIND_BADGE: Record<Measurement['kind'], { text: string; className: string } | null> = {
  // Measured is the default expectation and needs no badge; badging everything
  // would make the badges invisible.
  measured: null,
  derived: { text: 'derived', className: 'bg-elevated text-muted' },
  estimated: { text: 'estimated', className: 'bg-warn-soft text-warn' },
};

function formatValue(value: number | string | boolean | null, unit: string): string {
  if (value === null) return '—';
  if (typeof value === 'boolean') return value ? 'yes' : 'no';
  if (typeof value === 'string') return value;

  const abs = Math.abs(value);
  let text: string;
  if (abs === 0) text = '0';
  else if (abs < 0.01) text = value.toExponential(2);
  else if (abs < 1) text = value.toFixed(3);
  else if (abs < 100) text = value.toFixed(2);
  else if (abs < 10000) text = value.toFixed(1);
  else text = value.toLocaleString(undefined, { maximumFractionDigits: 0 });

  return unit ? `${text} ${unit}` : text;
}

interface Props {
  measurement: Measurement<number | string | boolean> | undefined;
  /** Show the instrumentation source and any caveat note beneath the value. */
  showProvenance?: boolean;
  className?: string;
}

export function MeasurementValue({ measurement, showProvenance, className }: Props) {
  if (!measurement) {
    return <span className="text-muted">—</span>;
  }

  if (measurement.value === null) {
    return (
      <span className={`inline-flex flex-col gap-0.5 ${className ?? ''}`}>
        <span className="text-sm italic text-muted">unavailable</span>
        {measurement.unavailable_reason && (
          <span className="text-2xs leading-snug text-muted">
            {measurement.unavailable_reason}
          </span>
        )}
      </span>
    );
  }

  const badge = KIND_BADGE[measurement.kind];

  return (
    <span className={`inline-flex flex-col gap-0.5 ${className ?? ''}`}>
      <span className="flex items-baseline gap-1.5">
        <span className="font-mono text-sm text-primary">
          {formatValue(measurement.value, measurement.unit)}
        </span>
        {badge && (
          <span className={`pill ${badge.className}`} title={measurement.note ?? undefined}>
            {badge.text}
          </span>
        )}
      </span>
      {showProvenance && (measurement.source || measurement.note) && (
        <span className="text-2xs leading-snug text-muted">
          {measurement.source && <>via {measurement.source}</>}
          {measurement.source && measurement.note && ' · '}
          {measurement.note}
        </span>
      )}
    </span>
  );
}

/** A labelled row of measurements, the standard layout for a metrics block. */
export function MeasurementList({
  measurements,
  showProvenance,
  skip = [],
}: {
  measurements: Record<string, Measurement>;
  showProvenance?: boolean;
  skip?: string[];
}) {
  const entries = Object.entries(measurements).filter(
    ([key, value]) =>
      !skip.includes(key) && value && typeof value === 'object' && 'kind' in value,
  );

  if (entries.length === 0) {
    return <p className="text-sm text-muted">No metrics in this group.</p>;
  }

  // Available metrics first: a block that leads with ten "unavailable" rows buries
  // the numbers that do exist.
  const ordered = [...entries].sort(([, a], [, b]) => {
    const aAvailable = a.value !== null ? 0 : 1;
    const bAvailable = b.value !== null ? 0 : 1;
    return aAvailable - bAvailable;
  });

  return (
    <dl className="divide-y divide-subtle">
      {ordered.map(([key, measurement]) => (
        <div key={key} className="flex items-start justify-between gap-4 py-2">
          <dt className="text-sm capitalize text-secondary">{key.replace(/_/g, ' ')}</dt>
          <dd className="text-right">
            <MeasurementValue measurement={measurement} showProvenance={showProvenance} />
          </dd>
        </div>
      ))}
    </dl>
  );
}
