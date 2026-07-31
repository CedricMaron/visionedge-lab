/**
 * System inspection: what this machine actually is, and what it can actually run.
 *
 * The capability matrix is the load-bearing part. Every unsupported cell carries the
 * reason it is unsupported, because a greyed-out checkbox with no explanation is
 * indistinguishable from a bug.
 */
import { useMemo, useState } from 'react';
import { useAsync } from '@/hooks/useAsync';
import { labApi } from '@/services/labApi';
import { ErrorState, PageHeader, Spinner } from '@/components/ui';
import { Icon } from '@/components/Icon';
import type { CapabilityCell, RuntimeCapability } from '@/types/lab';

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-4 border-b border-subtle py-2 last:border-0">
      <span className="text-sm text-secondary">{label}</span>
      <span className="text-right font-mono text-sm text-primary">{value}</span>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="card card-pad">
      <h2 className="mb-3 text-sm font-semibold text-primary">{title}</h2>
      {children}
    </div>
  );
}

function RuntimeRow({ runtime }: { runtime: RuntimeCapability }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <tr className="cursor-pointer" onClick={() => setOpen((o) => !o)}>
        <td className="font-mono text-xs text-primary">{runtime.runtime_id}</td>
        <td>
          <span
            className={`pill ${
              runtime.available ? 'bg-good-soft text-good' : 'bg-elevated text-muted'
            }`}
          >
            {runtime.available ? 'available' : 'unavailable'}
          </span>
        </td>
        <td className="font-mono text-xs">{runtime.version ?? '—'}</td>
        <td className="text-xs">
          {runtime.available
            ? runtime.devices.join(', ') || '—'
            : (runtime.unavailable_reason ?? '')}
        </td>
      </tr>
      {open && (runtime.notes.length > 0 || runtime.execution_providers.length > 0) && (
        <tr>
          <td colSpan={4} className="bg-elevated text-xs">
            {runtime.execution_providers.length > 0 && (
              <p className="mb-1">
                <span className="text-muted">providers: </span>
                <span className="font-mono">{runtime.execution_providers.join(', ')}</span>
              </p>
            )}
            {runtime.notes.map((note) => (
              <p key={note} className="text-muted">
                {note}
              </p>
            ))}
          </td>
        </tr>
      )}
    </>
  );
}

function CapabilityMatrix({ cells }: { cells: CapabilityCell[] }) {
  const [showUnsupported, setShowUnsupported] = useState(false);

  const { runtimes, columns, lookup } = useMemo(() => {
    const runtimeIds = [...new Set(cells.map((c) => c.runtime_id))];
    const cols = [...new Set(cells.map((c) => `${c.device}/${c.precision}`))];
    const map = new Map(cells.map((c) => [`${c.runtime_id}|${c.device}/${c.precision}`, c]));
    return { runtimes: runtimeIds, columns: cols, lookup: map };
  }, [cells]);

  const visibleRuntimes = showUnsupported
    ? runtimes
    : runtimes.filter((r) =>
        columns.some((col) => lookup.get(`${r}|${col}`)?.supported),
      );

  return (
    <div className="card card-pad">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div>
          <h2 className="text-sm font-semibold text-primary">
            Capability matrix — runtime × device × precision
          </h2>
          <p className="mt-0.5 text-xs text-muted">
            Hover any cell for the reason it is or is not supported here.
          </p>
        </div>
        <label className="flex items-center gap-2 text-xs text-secondary">
          <input
            type="checkbox"
            checked={showUnsupported}
            onChange={(e) => setShowUnsupported(e.target.checked)}
          />
          show runtimes with no supported combination
        </label>
      </div>

      <div className="overflow-x-auto">
        <table className="data-table">
          <thead>
            <tr>
              <th scope="col">Runtime</th>
              {columns.map((col) => (
                <th key={col} scope="col" className="text-center font-mono text-2xs">
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {visibleRuntimes.map((runtimeId) => (
              <tr key={runtimeId}>
                <th scope="row" className="px-3 py-2 text-left font-mono text-xs text-primary">
                  {runtimeId}
                </th>
                {columns.map((col) => {
                  const cell = lookup.get(`${runtimeId}|${col}`);
                  return (
                    <td
                      key={col}
                      className="text-center"
                      title={cell?.supported ? 'supported here' : (cell?.reason ?? 'unknown')}
                    >
                      {cell?.supported ? (
                        <span className="text-good" aria-label="supported">
                          ●
                        </span>
                      ) : (
                        <span className="text-muted" aria-label="not supported">
                          ○
                        </span>
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {visibleRuntimes.length < runtimes.length && (
        <p className="mt-2 text-xs text-muted">
          {runtimes.length - visibleRuntimes.length} runtime(s) hidden because no combination is
          supported on this machine.
        </p>
      )}
    </div>
  );
}

export default function SystemPage() {
  const system = useAsync((s) => labApi.system(s), []);
  const matrix = useAsync((s) => labApi.capabilityMatrix(s), []);

  return (
    <div>
      <PageHeader
        title="System"
        subtitle="Hardware, software and runtime availability on the machine serving this page. Every field is a live probe, not a configuration file."
        actions={
          <button className="btn-ghost" onClick={system.reload}>
            <Icon name="refresh" className="h-4 w-4" /> Refresh
          </button>
        }
      />

      {system.loading && <Spinner label="Probing system…" />}
      {system.error && <ErrorState message={system.error} onRetry={system.reload} />}

      {system.data && (
        <div className="space-y-4">
          <div className="grid gap-4 lg:grid-cols-2">
            <Section title="Compute">
              <Row label="CPU" value={system.data.hardware.cpu_model} />
              <Row
                label="Cores (physical / logical)"
                value={`${system.data.hardware.cpu_cores_physical ?? '?'} / ${system.data.hardware.cpu_cores_logical}`}
              />
              <Row
                label="Instruction sets"
                value={
                  system.data.hardware.cpu_instruction_sets.length > 0
                    ? system.data.hardware.cpu_instruction_sets.join(' ')
                    : 'undetected'
                }
              />
              <Row
                label="RAM"
                value={`${(system.data.hardware.ram_total_mb / 1024).toFixed(1)} GB`}
              />
            </Section>

            <Section title="Accelerators">
              {system.data.hardware.gpus.length === 0 ? (
                <p className="py-2 text-sm text-muted">
                  No NVIDIA GPU detected. GPU utilization, VRAM and power metrics will report
                  as unavailable with that reason.
                </p>
              ) : (
                system.data.hardware.gpus.map((gpu) => (
                  <div key={gpu.index} className="border-b border-subtle py-2 last:border-0">
                    <div className="text-sm text-primary">{gpu.name}</div>
                    <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted">
                      <span>{gpu.memory_total_mb} MB</span>
                      <span>driver {gpu.driver_version}</span>
                      <span>compute {gpu.compute_capability}</span>
                      {gpu.power_limit_w && <span>{gpu.power_limit_w} W limit</span>}
                    </div>
                  </div>
                ))
              )}
              <Row label="CUDA" value={system.data.hardware.cuda_version ?? 'not present'} />
              <Row label="cuDNN" value={system.data.hardware.cudnn_version ?? 'not present'} />
              <Row
                label="NVML (power / thermal probes)"
                value={
                  system.data.hardware.nvml_available ? (
                    <span className="text-good">available</span>
                  ) : (
                    <span className="text-muted">unavailable</span>
                  )
                }
              />
            </Section>
          </div>

          <div className="grid gap-4 lg:grid-cols-2">
            <Section title="Software">
              <Row
                label="OS"
                value={`${system.data.software.os} ${system.data.software.kernel_version ?? ''}`}
              />
              <Row label="Python" value={system.data.software.python_version} />
              <div className="mt-3">
                <div className="label mb-1">Execution-relevant packages</div>
                <div className="max-h-48 overflow-y-auto rounded border border-subtle bg-elevated p-2">
                  {Object.entries(system.data.software.package_versions).map(([pkg, version]) => (
                    <div key={pkg} className="flex justify-between font-mono text-xs">
                      <span className="text-secondary">{pkg}</span>
                      <span className="text-primary">{version}</span>
                    </div>
                  ))}
                </div>
              </div>
              {Object.keys(system.data.software.relevant_env_vars).length > 0 && (
                <div className="mt-3">
                  <div className="label mb-1">Environment (allow-listed)</div>
                  {Object.entries(system.data.software.relevant_env_vars).map(([k, v]) => (
                    <div key={k} className="flex justify-between font-mono text-xs">
                      <span className="text-secondary">{k}</span>
                      <span className="text-primary">{v}</span>
                    </div>
                  ))}
                </div>
              )}
            </Section>

            <Section title="Runtimes">
              <div className="overflow-x-auto">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th scope="col">Runtime</th>
                      <th scope="col">Status</th>
                      <th scope="col">Version</th>
                      <th scope="col">Detail</th>
                    </tr>
                  </thead>
                  <tbody>
                    {system.data.runtimes.map((runtime) => (
                      <RuntimeRow key={runtime.runtime_id} runtime={runtime} />
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="mt-2 text-xs text-muted">
                A runtime is listed as available only when its probe succeeded on this machine.
                Click a row for execution providers and caveats.
              </p>
            </Section>
          </div>

          {matrix.data && <CapabilityMatrix cells={matrix.data.cells} />}
        </div>
      )}
    </div>
  );
}
