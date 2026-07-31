/**
 * Guided benchmark configuration.
 *
 * Only compatible options are offered: scenarios are filtered to the selected
 * model's task, and device/precision choices come from the runtime's probe rather
 * than a hardcoded list. Choosing a combination the machine cannot run should be
 * impossible here, not an error afterwards.
 */
import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAsync } from '@/hooks/useAsync';
import { labApi } from '@/services/labApi';
import { ErrorState, Field, PageHeader, Spinner } from '@/components/ui';
import type { LabModel, LabScenario, RuntimeCapability } from '@/types/lab';

const MODES = [
  { value: 'standard', label: 'Standard', hint: 'Minimal overhead. The default for comparable results.' },
  { value: 'detailed', label: 'Detailed', hint: 'High-frequency hardware sampling. Not comparable with standard.' },
  { value: 'profiler', label: 'Profiler', hint: 'Framework profiling on. Materially perturbs timing.' },
];

export default function RunBenchmarkPage() {
  const navigate = useNavigate();
  const models = useAsync<{ models: LabModel[] }>((s) => labApi.models(s), []);
  const scenarios = useAsync<{ scenarios: LabScenario[] }>((s) => labApi.scenarios(s), []);
  const runtimes = useAsync<{ runtimes: RuntimeCapability[] }>((s) => labApi.runtimes(s), []);

  const [modelId, setModelId] = useState('');
  const [scenarioId, setScenarioId] = useState('');
  const [runtimeId, setRuntimeId] = useState('');
  const [device, setDevice] = useState('cpu');
  const [precision, setPrecision] = useState('fp32');
  const [mode, setMode] = useState('standard');
  const [iterations, setIterations] = useState<number | ''>('');
  const [label, setLabel] = useState('');
  const [running, setRunning] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);

  const installed = useMemo(
    () => (models.data?.models ?? []).filter((m) => m.deployment_status === 'installed'),
    [models.data],
  );
  const availableRuntimes = useMemo(
    () => (runtimes.data?.runtimes ?? []).filter((r) => r.available),
    [runtimes.data],
  );

  const selectedModel = installed.find((m) => m.model_id === modelId);
  const selectedRuntime = availableRuntimes.find((r) => r.runtime_id === runtimeId);

  // Only scenarios for this model's task: running a detection scenario against an
  // embedding model would fail after the model had already loaded.
  const compatibleScenarios = useMemo(
    () =>
      (scenarios.data?.scenarios ?? []).filter((s) => !selectedModel || s.task === selectedModel.task),
    [scenarios.data, selectedModel],
  );

  // Memoized: a fresh [] literal each render would retrigger the effects below
  // on every render rather than only when the selection actually changes.
  const devices = useMemo(() => selectedRuntime?.devices ?? [], [selectedRuntime]);
  const precisions = useMemo(
    () => selectedRuntime?.precisions_by_device?.[device] ?? [],
    [selectedRuntime, device],
  );

  // Keep dependent selections valid as the parent selection changes.
  useEffect(() => {
    if (installed.length > 0 && !modelId) setModelId(installed[0].model_id);
  }, [installed, modelId]);
  useEffect(() => {
    if (availableRuntimes.length > 0 && !runtimeId) setRuntimeId(availableRuntimes[0].runtime_id);
  }, [availableRuntimes, runtimeId]);
  useEffect(() => {
    if (compatibleScenarios.length > 0 && !compatibleScenarios.some((s) => s.id === scenarioId)) {
      setScenarioId(compatibleScenarios[0].id);
    }
  }, [compatibleScenarios, scenarioId]);
  useEffect(() => {
    if (devices.length > 0 && !devices.includes(device)) setDevice(devices[0]);
  }, [devices, device]);
  useEffect(() => {
    if (precisions.length > 0 && !precisions.includes(precision)) setPrecision(precisions[0]);
  }, [precisions, precision]);

  const scenario = compatibleScenarios.find((s) => s.id === scenarioId);
  const effectiveIterations = iterations === '' ? (scenario?.measured_iterations ?? 0) : iterations;
  const ready = Boolean(modelId && scenarioId && runtimeId && device && precision);

  async function start() {
    setRunning(true);
    setFailure(null);
    try {
      const result = await labApi.createRun({
        scenario_id: scenarioId,
        model_id: modelId,
        runtime_id: runtimeId,
        device,
        precision,
        mode,
        measured_iterations: iterations === '' ? undefined : iterations,
        label: label || undefined,
      });
      navigate(`/lab/results/${result.run.identity.run_id}`);
    } catch (err) {
      setFailure(err instanceof Error ? err.message : 'the benchmark could not be started');
    } finally {
      setRunning(false);
    }
  }

  const loading = models.loading || scenarios.loading || runtimes.loading;
  const loadError = models.error || scenarios.error || runtimes.error;

  return (
    <div>
      <PageHeader
        title="Run benchmark"
        subtitle="Only combinations this machine can actually execute are offered."
      />

      {loading && <Spinner label="Loading options…" />}
      {loadError && <ErrorState message={loadError} onRetry={models.reload} />}

      {!loading && !loadError && (
        <div className="grid gap-4 lg:grid-cols-[1fr_320px]">
          <div className="card card-pad space-y-4">
            <Field label="Model" hint={selectedModel?.notes}>
              <select className="input" value={modelId} onChange={(e) => setModelId(e.target.value)}>
                {installed.map((model) => (
                  <option key={model.model_id} value={model.model_id}>
                    {model.display_name} — {model.task.replace(/_/g, ' ')}
                  </option>
                ))}
              </select>
            </Field>
            {installed.length === 0 && (
              <p className="text-sm text-bad">
                No models are installed. Install one with{' '}
                <code>python scripts/download_models.py --list</code>.
              </p>
            )}

            <Field
              label="Scenario"
              hint={
                scenario
                  ? `${scenario.warmup_iterations} warm-up + ${scenario.measured_iterations} measured, batch ${scenario.batch_size}`
                  : undefined
              }
            >
              <select
                className="input"
                value={scenarioId}
                onChange={(e) => setScenarioId(e.target.value)}
              >
                {compatibleScenarios.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.id}
                  </option>
                ))}
              </select>
            </Field>
            {scenario?.description && (
              <p className="-mt-2 text-xs text-muted">{scenario.description}</p>
            )}

            <div className="grid gap-4 sm:grid-cols-3">
              <Field label="Runtime">
                <select
                  className="input"
                  value={runtimeId}
                  onChange={(e) => setRuntimeId(e.target.value)}
                >
                  {availableRuntimes.map((runtime) => (
                    <option key={runtime.runtime_id} value={runtime.runtime_id}>
                      {runtime.runtime_id} {runtime.version ?? ''}
                    </option>
                  ))}
                </select>
              </Field>
              <Field label="Device">
                <select className="input" value={device} onChange={(e) => setDevice(e.target.value)}>
                  {devices.map((d) => (
                    <option key={d} value={d}>
                      {d}
                    </option>
                  ))}
                </select>
              </Field>
              <Field label="Precision">
                <select
                  className="input"
                  value={precision}
                  onChange={(e) => setPrecision(e.target.value)}
                >
                  {precisions.map((p) => (
                    <option key={p} value={p}>
                      {p}
                    </option>
                  ))}
                </select>
              </Field>
            </div>

            <Field
              label="Measured iterations"
              hint={
                effectiveIterations < 10
                  ? 'Below ~10 samples, percentiles above p50 are not statistically meaningful — the result will be flagged.'
                  : undefined
              }
            >
              <input
                type="number"
                min={1}
                max={1000}
                className="input"
                placeholder={String(scenario?.measured_iterations ?? 20)}
                value={iterations}
                onChange={(e) => setIterations(e.target.value === '' ? '' : Number(e.target.value))}
              />
            </Field>

            <Field label="Instrumentation mode">
              <div className="space-y-1.5">
                {MODES.map((option) => (
                  <label
                    key={option.value}
                    className={`flex cursor-pointer gap-2 rounded border px-3 py-2 text-sm ${
                      mode === option.value
                        ? 'border-accent bg-accent-soft/40'
                        : 'border-subtle hover:border-strong'
                    }`}
                  >
                    <input
                      type="radio"
                      name="mode"
                      className="mt-0.5"
                      checked={mode === option.value}
                      onChange={() => setMode(option.value)}
                    />
                    <span>
                      <span className="font-medium text-primary">{option.label}</span>
                      <span className="block text-xs text-muted">{option.hint}</span>
                    </span>
                  </label>
                ))}
              </div>
            </Field>

            <Field label="Label (optional)">
              <input
                className="input"
                value={label}
                maxLength={120}
                placeholder="e.g. baseline before quantization"
                onChange={(e) => setLabel(e.target.value)}
              />
            </Field>

            {failure && (
              <p className="rounded border border-bad/40 bg-bad-soft px-3 py-2 text-sm text-bad">
                {failure}
              </p>
            )}
          </div>

          <aside className="lg:sticky lg:top-4 lg:self-start">
            <div className="card card-pad space-y-3">
              <h2 className="text-sm font-semibold text-primary">Review</h2>
              <dl className="divide-y divide-subtle text-sm">
                {[
                  ['Model', selectedModel?.display_name ?? '—'],
                  ['Task', selectedModel?.task.replace(/_/g, ' ') ?? '—'],
                  ['Licence (weights)', selectedModel?.weights_license ?? '—'],
                  ['Scenario', scenarioId || '—'],
                  ['Runtime', `${runtimeId}/${device}/${precision}`],
                  ['Warm-up', String(scenario?.warmup_iterations ?? '—')],
                  ['Measured', String(effectiveIterations || '—')],
                  ['Mode', mode],
                ].map(([term, value]) => (
                  <div key={term} className="flex justify-between gap-3 py-1.5">
                    <dt className="text-secondary">{term}</dt>
                    <dd className="text-right font-mono text-xs text-primary">{value}</dd>
                  </div>
                ))}
              </dl>

              {mode !== 'standard' && (
                <p className="rounded border border-warn/40 bg-warn-soft px-2.5 py-2 text-xs text-warn">
                  Results from {mode} mode are not comparable with standard-mode results.
                </p>
              )}

              <button className="btn-primary w-full" disabled={!ready || running} onClick={start}>
                {running ? 'Running…' : 'Run benchmark'}
              </button>
              <p className="text-xs text-muted">
                The run executes on the server and is stored when it finishes. Long sweeps are
                better run from the CLI.
              </p>
            </div>
          </aside>
        </div>
      )}
    </div>
  );
}
