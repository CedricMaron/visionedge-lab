import { PageHeader } from '@/components/ui';

function Node({ title, sub, tone = 'accent' }: { title: string; sub: string; tone?: string }) {
  const tones: Record<string, string> = {
    accent: 'border-accent/40 bg-accent/10',
    good: 'border-good/40 bg-good/10',
    warn: 'border-warn/40 bg-warn/10',
    neutral: 'border-surface-600 bg-surface-800',
  };
  return (
    <div className={`rounded-lg border px-3 py-2 text-center ${tones[tone]}`}>
      <div className="text-sm font-medium text-slate-100">{title}</div>
      <div className="text-[11px] text-slate-400">{sub}</div>
    </div>
  );
}

function Arrow() {
  return <div className="flex items-center justify-center text-slate-600">↓</div>;
}

export default function ArchitecturePage() {
  return (
    <div>
      <PageHeader
        title="Architecture"
        subtitle="How this frontend maps onto the VisionEdge Lab backend and where future phases plug in."
      />

      <div className="grid gap-4 lg:grid-cols-[320px_1fr]">
        <div className="card card-pad space-y-2">
          <h2 className="label mb-1">Live inference data path</h2>
          <Node title="Camera / video / image" sub="MediaDevices getUserMedia" tone="neutral" />
          <Arrow />
          <Node title="Frame grab" sub="video → canvas → toBlob(jpeg)" tone="neutral" />
          <Arrow />
          <Node title="WebSocket" sub="ws(s)://…/api/ws/detect (binary)" tone="accent" />
          <Arrow />
          <Node title="FastAPI detector" sub="ONNX / PyTorch runtime" tone="good" />
          <Arrow />
          <Node title="Detections JSON" sub="{ x1..y2, className, conf }" tone="accent" />
          <Arrow />
          <Node title="Canvas overlay" sub="bounding boxes + labels" tone="neutral" />
        </div>

        <div className="space-y-4">
          <div className="card card-pad">
            <h2 className="mb-2 text-sm font-semibold text-slate-200">Frontend layers</h2>
            <ul className="space-y-2 text-sm text-slate-300">
              <li>
                <span className="font-medium text-slate-100">Services</span> — a thin typed fetch
                wrapper (<code className="text-accent">services/http.ts</code>) plus one module per
                backend slice (<code className="text-accent">api</code>,{' '}
                <code className="text-accent">vlmApi</code>, and Phase-4/5 stubs{' '}
                <code className="text-accent">jepaApi / embeddingApi / anomalyApi</code>).
              </li>
              <li>
                <span className="font-medium text-slate-100">Stores</span> — zustand:{' '}
                <code className="text-accent">classStore</code> (persisted allowed classes),{' '}
                <code className="text-accent">modelSwitchStore</code> (config draft + switch),{' '}
                <code className="text-accent">settingsStore</code> (API base, defaults).
              </li>
              <li>
                <span className="font-medium text-slate-100">Hooks</span> —{' '}
                <code className="text-accent">useDetectionSocket</code> (streaming + bounded
                reconnect), <code className="text-accent">useHealth</code>,{' '}
                <code className="text-accent">useAsync</code> (fetch + polling).
              </li>
              <li>
                <span className="font-medium text-slate-100">Inference abstraction</span> — a shared{' '}
                <code className="text-accent">InferenceBackend</code> interface with a real{' '}
                <code className="text-accent">serverBackend</code> and honest{' '}
                <code className="text-accent">browserBackend</code> /{' '}
                <code className="text-accent">inferenceWorker</code> stubs for Phase-3 on-device
                inference.
              </li>
            </ul>
          </div>

          <div className="card card-pad">
            <h2 className="mb-2 text-sm font-semibold text-slate-200">Execution locations</h2>
            <div className="grid gap-3 sm:grid-cols-3">
              <Node title="Server" sub="FastAPI + GPU/CPU runtime (today)" tone="good" />
              <Node title="Browser" sub="onnxruntime-web / WebGPU (Phase 3)" tone="warn" />
              <Node title="Edge" sub="dedicated inference node (Phase 5)" tone="warn" />
            </div>
            <p className="mt-3 text-sm text-slate-400">
              The Live page and Model Selector already carry an{' '}
              <code className="text-accent">execution_location</code> field end-to-end, so browser
              and edge backends slot in without a data-model change.
            </p>
          </div>

          <div className="card card-pad">
            <h2 className="mb-2 text-sm font-semibold text-slate-200">Roadmap</h2>
            <ul className="space-y-1.5 text-sm text-slate-300">
              <li><span className="text-good">Phase 2 (this build)</span> — detection slice: live streaming, model/class/config control, capabilities, performance, benchmarks.</li>
              <li><span className="text-warn">Phase 3</span> — browser-side inference + streamed logs.</li>
              <li><span className="text-warn">Phase 4</span> — VLM depth, JEPA training, embeddings, model comparison, optimization advisor.</li>
              <li><span className="text-warn">Phase 5</span> — temporal analysis, anomaly detection, multi-server routing.</li>
              <li><span className="text-warn">Phase 6</span> — world models, cross-modal search.</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
