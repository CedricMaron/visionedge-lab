// Reusable class picker: search, grouped presets, and per-class checkboxes.
// Reads and mutates the persisted classStore.

import { useMemo, useState } from 'react';
import { useClassStore, type GroupKey } from '@/stores/classStore';
import { Icon } from './Icon';

const PRESETS: { key: GroupKey; label: string }[] = [
  { key: 'people', label: 'People only' },
  { key: 'vehicles', label: 'Vehicles only' },
  { key: 'animals', label: 'Animals only' },
  { key: 'indoor', label: 'Indoor objects' },
];

export function ClassPicker({ compact = false }: { compact?: boolean }) {
  const classes = useClassStore((s) => s.classes);
  const groups = useClassStore((s) => s.groups);
  const selectedIds = useClassStore((s) => s.selectedIds);
  const toggle = useClassStore((s) => s.toggle);
  const enableAll = useClassStore((s) => s.enableAll);
  const disableAll = useClassStore((s) => s.disableAll);
  const selectGroup = useClassStore((s) => s.selectGroup);

  const [query, setQuery] = useState('');
  const selectedSet = useMemo(() => new Set(selectedIds), [selectedIds]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return classes;
    return classes.filter((c) => c.name.toLowerCase().includes(q));
  }, [classes, query]);

  const groupNameSet = useMemo(() => {
    const map: Record<string, Set<string>> = {};
    for (const [g, names] of Object.entries(groups)) map[g] = new Set(names);
    return map;
  }, [groups]);

  function groupOf(name: string): string | null {
    for (const g of ['people', 'vehicles', 'animals', 'indoor']) {
      if (groupNameSet[g]?.has(name)) return g;
    }
    return null;
  }

  const groupTone: Record<string, string> = {
    people: 'text-accent',
    vehicles: 'text-good',
    animals: 'text-warn',
    indoor: 'text-[#a78bfa]',
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <button className="btn-ghost" onClick={enableAll}>
          Enable all
        </button>
        <button className="btn-ghost" onClick={disableAll}>
          Disable all
        </button>
        {PRESETS.map((p) => (
          <button key={p.key} className="btn-ghost" onClick={() => selectGroup(p.key)}>
            {p.label}
          </button>
        ))}
        <span className="ml-auto text-xs text-muted">
          {selectedIds.length} / {classes.length} selected
        </span>
      </div>

      <div className="relative">
        <Icon name="search" className="pointer-events-none absolute left-3 top-2.5 h-4 w-4 text-muted" />
        <input
          className="input pl-9"
          placeholder="Search 80 COCO classes…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
      </div>

      <div
        className={`grid gap-1.5 ${
          compact ? 'grid-cols-2 sm:grid-cols-3' : 'grid-cols-2 sm:grid-cols-3 lg:grid-cols-4'
        } max-h-[46vh] overflow-y-auto pr-1`}
      >
        {filtered.map((c) => {
          const on = selectedSet.has(c.id);
          const g = groupOf(c.name);
          return (
            <label
              key={c.id}
              className={`flex cursor-pointer items-center gap-2 rounded-lg border px-2.5 py-1.5 text-sm transition ${
                on
                  ? 'border-accent/40 bg-accent/10 text-primary'
                  : 'border-subtle bg-elevated text-secondary hover:border-strong'
              }`}
            >
              <input
                type="checkbox"
                className="accent-accent"
                checked={on}
                onChange={() => toggle(c.id)}
              />
              <span className="truncate">{c.name}</span>
              {g && <span className={`ml-auto text-[9px] uppercase ${groupTone[g]}`}>{g[0]}</span>}
            </label>
          );
        })}
        {filtered.length === 0 && (
          <p className="col-span-full py-6 text-center text-sm text-muted">
            No classes match "{query}".
          </p>
        )}
      </div>
    </div>
  );
}
