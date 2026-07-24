import { useHealth } from '@/hooks/useHealth';
import { useSettingsStore } from '@/stores/settingsStore';
import { Icon } from './Icon';

function HealthPill() {
  const { state, health, warnings } = useHealth();

  const map = {
    connecting: { tone: 'bg-warn/15 text-warn', dot: 'bg-warn', label: 'Connecting' },
    online: { tone: 'bg-good/15 text-good', dot: 'bg-good', label: 'Online' },
    offline: { tone: 'bg-bad/15 text-bad', dot: 'bg-bad', label: 'Offline' },
  }[state];

  return (
    <div className="flex items-center gap-2">
      <span className={`pill ${map.tone}`} title={warnings.join('\n') || undefined}>
        <span className={`h-1.5 w-1.5 rounded-full ${map.dot} ${state !== 'offline' ? 'animate-pulse' : ''}`} />
        {map.label}
      </span>
      {state === 'online' && (
        <span className="hidden text-xs text-slate-500 sm:inline">
          detector: <span className="text-slate-300">{health}</span>
          {warnings.length > 0 && (
            <span className="ml-1 text-warn">· {warnings.length} warning{warnings.length > 1 ? 's' : ''}</span>
          )}
        </span>
      )}
    </div>
  );
}

export function TopBar({ onMenu }: { onMenu: () => void }) {
  const apiBase = useSettingsStore((s) => s.apiBase);
  return (
    <header className="sticky top-0 z-20 flex items-center gap-3 border-b border-surface-800 bg-surface-950/85 px-4 py-3 backdrop-blur">
      <button
        className="btn-ghost px-2 py-2 lg:hidden"
        onClick={onMenu}
        aria-label="Open navigation"
      >
        <Icon name="menu" className="h-5 w-5" />
      </button>
      <HealthPill />
      <div className="ml-auto flex items-center gap-3">
        <span className="hidden max-w-[220px] truncate font-mono text-xs text-slate-500 sm:inline">
          {apiBase}
        </span>
      </div>
    </header>
  );
}
