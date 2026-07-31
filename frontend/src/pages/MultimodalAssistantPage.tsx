import { useEffect, useRef, useState } from 'react';
import { vlmApi, VlmUnavailableError } from '@/services/vlmApi';
import { useSettingsStore } from '@/stores/settingsStore';
import { PageHeader, Field, Spinner, EmptyState } from '@/components/ui';
import { Icon } from '@/components/Icon';
import { NotImplementedBanner } from '@/components/NotImplementedBanner';
import { VLMResponsePanel } from '@/components/VLMResponsePanel';
import type { VLMResponse, VlmModelEntry } from '@/types';

export default function MultimodalAssistantPage() {
  const settings = useSettingsStore();
  const [available, setAvailable] = useState<'checking' | 'yes' | 'no'>('checking');
  const [models, setModels] = useState<VlmModelEntry[]>([]);
  const [modelId, setModelId] = useState('');
  const [file, setFile] = useState<Blob | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [question, setQuestion] = useState('Describe what you see in this image.');
  const [ground, setGround] = useState(settings.vlmGrounding);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<VLMResponse | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let active = true;
    const controller = new AbortController();
    vlmApi
      .models(controller.signal)
      .then((res) => {
        if (!active) return;
        setAvailable('yes');
        setModels(res.models ?? []);
        if (res.models?.length) setModelId(res.models[0].model_id);
      })
      .catch((err) => {
        if (!active) return;
        if (err instanceof VlmUnavailableError) setAvailable('no');
        else setAvailable('no');
      });
    return () => {
      active = false;
      controller.abort();
    };
  }, []);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  function onFile(f: File | null) {
    if (!f) return;
    setFile(f);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(URL.createObjectURL(f));
    setResponse(null);
  }

  async function ask() {
    if (!file) {
      setError('Attach an image first.');
      return;
    }
    setBusy(true);
    setError(null);
    setResponse(null);
    try {
      const res = await vlmApi.ask(file, question, ground);
      setResponse(res.response);
    } catch (err) {
      if (err instanceof VlmUnavailableError) {
        setAvailable('no');
      } else {
        setError(err instanceof Error ? err.message : 'Request failed');
      }
    } finally {
      setBusy(false);
    }
  }

  if (available === 'checking') {
    return (
      <div>
        <PageHeader title="Multimodal Assistant" />
        <Spinner label="Checking VLM availability…" />
      </div>
    );
  }

  if (available === 'no') {
    return (
      <div>
        <PageHeader
          title="Multimodal Assistant"
          subtitle="Ask a vision-language model questions about a captured or uploaded image."
        />
        <NotImplementedBanner
          phase="VLM slice"
          description="The VLM slice is not enabled in this backend build (the /api/vlm endpoints returned unavailable). When the backend exposes VLM models, this page will let you attach an image, ask questions, toggle detector-grounding and structured output, and inspect latency/token/memory metrics. No responses are fabricated here."
        />
      </div>
    );
  }

  const isLocal = (() => {
    const m = models.find((x) => x.model_id === modelId);
    const loc = String(m?.execution_location ?? '').toLowerCase();
    return loc.includes('local') || loc.includes('browser') || loc.includes('edge');
  })();

  return (
    <div>
      <PageHeader
        title="Multimodal Assistant"
        subtitle="Ask a vision-language model about an image. Grounding feeds detector output to the model for more precise, verifiable answers."
      />

      <div className="grid gap-4 lg:grid-cols-2">
        <div className="space-y-4">
          <div className="card card-pad space-y-3">
            <h2 className="label">Image</h2>
            <div
              className="flex aspect-video cursor-pointer items-center justify-center overflow-hidden rounded-lg border border-dashed border-strong bg-elevated"
              onClick={() => fileInput.current?.click()}
            >
              {previewUrl ? (
                <img src={previewUrl} alt="input" className="h-full w-full object-contain" />
              ) : (
                <div className="text-center text-sm text-muted">
                  <Icon name="camera" className="mx-auto mb-2 h-6 w-6" />
                  Click to upload an image
                </div>
              )}
            </div>
            <input
              ref={fileInput}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => onFile(e.target.files?.[0] ?? null)}
            />
          </div>

          <div className="card card-pad space-y-3">
            <Field label="Model">
              <select className="input" value={modelId} onChange={(e) => setModelId(e.target.value)}>
                {models.map((m) => (
                  <option key={m.model_id} value={m.model_id}>
                    {m.display_name ?? m.model_id}
                  </option>
                ))}
                {models.length === 0 && <option value="">(default model)</option>}
              </select>
            </Field>

            <Field label="Question">
              <textarea
                className="input min-h-[80px] resize-y"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
              />
            </Field>

            <label className="flex items-center gap-2 text-sm text-secondary">
              <input type="checkbox" className="accent-accent" checked={ground} onChange={(e) => setGround(e.target.checked)} />
              Detector-grounding (pass detected objects as context)
            </label>

            <div className="flex items-center gap-2">
              <span
                className={`pill ${isLocal ? 'bg-good/15 text-good' : 'bg-warn/15 text-warn'}`}
                title="Where inference runs"
              >
                <Icon name={isLocal ? 'chip' : 'server'} className="h-3.5 w-3.5" />
                {isLocal ? 'on-device (private)' : 'server-side'}
              </span>
            </div>

            <button className="btn-primary w-full" disabled={busy} onClick={ask}>
              {busy ? <Spinner label="Analysing…" /> : 'Ask'}
            </button>
            {error && <p className="text-sm text-bad">{error}</p>}
          </div>
        </div>

        <div>
          {response ? (
            <VLMResponsePanel response={response} />
          ) : (
            <EmptyState
              title="No response yet"
              hint="Attach an image and ask a question. Latency, token counts, memory usage, and a privacy indicator will appear here."
            />
          )}
        </div>
      </div>
    </div>
  );
}
