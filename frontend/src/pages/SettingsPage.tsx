import { useState } from 'react';
import { useSettingsStore } from '@/stores/settingsStore';
import { getDefaultApiBase } from '@/config';
import { api } from '@/services/api';
import { PageHeader, Field, Badge } from '@/components/ui';
import { Icon } from '@/components/Icon';

function Toggle({
  checked,
  onChange,
  label,
  hint,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
  hint?: string;
}) {
  return (
    <label className="flex cursor-pointer items-start justify-between gap-4">
      <span>
        <span className="text-sm font-medium text-primary">{label}</span>
        {hint && <span className="mt-0.5 block text-xs text-muted">{hint}</span>}
      </span>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={`relative mt-0.5 h-6 w-11 shrink-0 rounded-full transition ${
          checked ? 'bg-accent' : 'bg-elevated'
        }`}
      >
        <span
          className={`absolute top-0.5 h-5 w-5 rounded-full bg-white transition ${
            checked ? 'left-[22px]' : 'left-0.5'
          }`}
        />
      </button>
    </label>
  );
}

export default function SettingsPage() {
  const theme = useSettingsStore((st) => st.theme);
  const setTheme = useSettingsStore((st) => st.setTheme);
  const s = useSettingsStore();
  const [apiBaseInput, setApiBaseInput] = useState(s.apiBase);
  const [testState, setTestState] = useState<'idle' | 'testing' | 'ok' | 'fail'>('idle');
  const [testMsg, setTestMsg] = useState('');

  function saveApiBase() {
    s.setApiBase(apiBaseInput.trim() || getDefaultApiBase());
    setTestState('idle');
  }

  async function testConnection() {
    s.setApiBase(apiBaseInput.trim() || getDefaultApiBase());
    setTestState('testing');
    try {
      const h = await api.health();
      setTestState('ok');
      setTestMsg(`status: ${h.status}, detector: ${h.detection_health}`);
    } catch (err) {
      setTestState('fail');
      setTestMsg(err instanceof Error ? err.message : 'unreachable');
    }
  }

  return (
    <div>
      <PageHeader
        title="Settings"
        subtitle="Configuration is stored in your browser's localStorage. No account required."
      />

      <div className="mx-auto max-w-2xl space-y-4">
        <div className="card card-pad space-y-3">
          <h2 className="text-sm font-semibold text-primary">Backend connection</h2>
          <Field label="API base URL" hint={`Default: ${getDefaultApiBase()} · WebSocket URL is derived automatically.`}>
            <input
              className="input font-mono"
              value={apiBaseInput}
              onChange={(e) => setApiBaseInput(e.target.value)}
              placeholder="http://localhost:8000"
            />
          </Field>
          <div className="flex flex-wrap items-center gap-2">
            <button className="btn-primary" onClick={saveApiBase}>
              Save
            </button>
            <button className="btn-ghost" onClick={testConnection} disabled={testState === 'testing'}>
              <Icon name="refresh" className="h-4 w-4" /> Test connection
            </button>
            <button
              className="btn-ghost"
              onClick={() => {
                setApiBaseInput(getDefaultApiBase());
                s.setApiBase(getDefaultApiBase());
              }}
            >
              Reset to default
            </button>
            {testState === 'ok' && <Badge tone="good">connected — {testMsg}</Badge>}
            {testState === 'fail' && <Badge tone="bad">failed — {testMsg}</Badge>}
            {testState === 'testing' && <Badge tone="warn">testing…</Badge>}
          </div>
        </div>

        <div className="card card-pad space-y-4">
          <h2 className="text-sm font-semibold text-primary">Detector defaults</h2>
          <div className="grid grid-cols-2 gap-3">
            <Field label={`Default confidence (${s.defaultConfidence.toFixed(2)})`}>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                className="w-full accent-accent"
                value={s.defaultConfidence}
                onChange={(e) => s.setDefaultConfidence(Number(e.target.value))}
              />
            </Field>
            <Field label={`Default IoU (${s.defaultIou.toFixed(2)})`}>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                className="w-full accent-accent"
                value={s.defaultIou}
                onChange={(e) => s.setDefaultIou(Number(e.target.value))}
              />
            </Field>
          </div>
        </div>

        <div className="card card-pad space-y-4">
          <h2 className="text-sm font-semibold text-primary">Multimodal assistant (VLM)</h2>
          <Toggle
            label="Detector-grounding by default"
            hint="Pass detected objects to the VLM as grounding context when analysing images."
            checked={s.vlmGrounding}
            onChange={s.setVlmGrounding}
          />
          <Toggle
            label="Request structured JSON output"
            hint="Ask the VLM for machine-readable structured_output when supported."
            checked={s.structuredOutput}
            onChange={s.setStructuredOutput}
          />
        </div>

        <div className="card card-pad">
          <h2 className="text-sm font-semibold text-primary">Theme</h2>
          <p className="mt-1 text-sm text-secondary">
            InferenceLab defaults to a light, low-chrome theme. Both themes are built from the
            same semantic tokens and both are contrast-checked against WCAG AA.
          </p>
          <div className="mt-3 flex gap-2">
            {(['light', 'dark'] as const).map((option) => (
              <button
                key={option}
                onClick={() => setTheme(option)}
                aria-pressed={theme === option}
                className={`btn ${
                  theme === option
                    ? 'bg-accent text-accent-contrast'
                    : 'border border-subtle bg-panel text-secondary hover:border-strong'
                }`}
              >
                {option === 'light' ? 'Light' : 'Dark'}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
