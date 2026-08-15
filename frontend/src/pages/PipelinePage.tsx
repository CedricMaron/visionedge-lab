/**
 * Pipeline inspector — how the last input became the last output.
 *
 * Everything on this page comes from the trace of a real execution. There is no
 * diagram for a model that has not been run: without a trace the page says so and
 * sends you to the Playground, rather than drawing a plausible pipeline.
 */
import { Link } from 'react-router-dom';
import { ExecutionBadge } from '@/components/ExecutionBadge';
import { PipelineFlow, PipelineTimeline, TensorCard } from '@/components/PipelineFlow';
import { EmptyState, PageHeader } from '@/components/ui';
import { taskLabel } from '@/lab/catalog';
import { usePlaygroundStore } from '@/stores/playgroundStore';
import { formatBytes } from '@/utils/format';

export default function PipelinePage() {
  const trace = usePlaygroundStore((s) => s.trace);

  if (!trace) {
    return (
      <div>
        <PageHeader
          title="Pipeline"
          subtitle="Every stage of the most recent inference, with the tensors that crossed between them."
        />
        <EmptyState
          title="No inference has been run yet"
          hint="Run one in the Playground — the pipeline is generated from that execution, not from a template."
        />
        <div className="mt-3">
          <Link className="btn-primary" to="/">
            Open Playground
          </Link>
        </div>
      </div>
    );
  }

  const allTensors = trace.stages.flatMap((stage) =>
    stage.tensors.map((tensor) => ({ stage: stage.name, tensor })),
  );

  return (
    <div>
      <PageHeader
        title="Pipeline"
        subtitle="Generated from the last run's instrumentation. Stages the adapter does not time separately are marked, never estimated."
        actions={<ExecutionBadge target={trace.execution} />}
      />

      <div className="card card-pad mb-4">
        <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-3 lg:grid-cols-6">
          {[
            ['Task', taskLabel(trace.task)],
            ['Model', trace.model.display_name],
            ['Runtime', trace.runtime.runtime_id],
            ['Device', trace.runtime.device],
            ['Precision', trace.runtime.precision],
            [
              'Provider',
              trace.runtime.execution_provider ?? 'not reported',
            ],
          ].map(([term, value]) => (
            <div key={term}>
              <dt className="label">{term}</dt>
              <dd className="mt-0.5 truncate font-mono text-xs text-primary" title={String(value)}>
                {value}
              </dd>
            </div>
          ))}
        </dl>
      </div>

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <section className="min-w-0">
          <h2 className="label mb-2">Stages</h2>
          <PipelineFlow stages={trace.stages} />
        </section>

        <div className="min-w-0 space-y-4">
          <section className="card card-pad">
            <h2 className="label mb-3">Timeline</h2>
            <PipelineTimeline stages={trace.stages} />
          </section>

          <section className="card card-pad">
            <h2 className="label mb-3">Tensors</h2>
            {allTensors.length === 0 ? (
              <p className="text-sm text-muted">This run exposed no inspectable tensors.</p>
            ) : (
              <div className="space-y-3">
                {allTensors.map(({ stage, tensor }) => (
                  <div key={`${stage}-${tensor.name}`}>
                    <div className="mb-1 text-2xs uppercase tracking-wider text-muted">{stage}</div>
                    <TensorCard tensor={tensor} />
                  </div>
                ))}
              </div>
            )}
            <dl className="mt-3 divide-y divide-subtle text-sm">
              {[
                ['Input tensor memory', formatBytes(trace.memory.input_tensor_bytes)],
                ['Output tensor memory', formatBytes(trace.memory.output_tensor_bytes)],
                ['Model input format', trace.model.input_format],
                ['Model output format', trace.model.output_format],
              ].map(([term, value]) => (
                <div key={term} className="flex justify-between gap-3 py-1.5">
                  <dt className="text-secondary">{term}</dt>
                  <dd className="text-right font-mono text-2xs text-primary">{value}</dd>
                </div>
              ))}
            </dl>
          </section>
        </div>
      </div>
    </div>
  );
}
