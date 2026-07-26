// Class-selection store. Holds the COCO class catalog + the user's allowed set.
// The selection is persisted to localStorage and feeds the Live page
// (allowed_class_ids). Group operations resolve class names via the catalog.

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { ClassEntry, ClassGroups } from '@/types';

export type GroupKey = 'people' | 'vehicles' | 'animals' | 'indoor';

interface ClassState {
  classes: ClassEntry[];
  groups: ClassGroups;
  // undefined selectedIds means "not yet initialised"; null-safe via helpers.
  selectedIds: number[];
  loaded: boolean;

  setCatalog: (classes: ClassEntry[], groups: ClassGroups) => void;
  isSelected: (id: number) => boolean;
  toggle: (id: number) => void;
  setSelected: (ids: number[]) => void;
  enableAll: () => void;
  disableAll: () => void;
  selectGroup: (group: GroupKey, additive?: boolean) => void;
  idsForGroup: (group: GroupKey) => number[];
}

function resolveGroupIds(classes: ClassEntry[], names: string[] | undefined): number[] {
  if (!names) return [];
  const byName = new Map(classes.map((c) => [c.name, c.id]));
  const ids: number[] = [];
  for (const n of names) {
    const id = byName.get(n);
    if (id !== undefined) ids.push(id);
  }
  return ids;
}

export const useClassStore = create<ClassState>()(
  persist(
    (set, get) => ({
      classes: [],
      groups: { people: [], vehicles: [], animals: [], indoor: [] },
      selectedIds: [],
      loaded: false,

      setCatalog: (classes, groups) =>
        set((state) => ({
          classes,
          groups,
          loaded: true,
          // On first load with an empty selection, default to all classes on.
          selectedIds:
            state.selectedIds.length === 0 && !state.loaded
              ? classes.map((c) => c.id)
              : state.selectedIds,
        })),

      isSelected: (id) => get().selectedIds.includes(id),

      toggle: (id) =>
        set((state) => ({
          selectedIds: state.selectedIds.includes(id)
            ? state.selectedIds.filter((x) => x !== id)
            : [...state.selectedIds, id].sort((a, b) => a - b),
        })),

      setSelected: (ids) => set({ selectedIds: [...new Set(ids)].sort((a, b) => a - b) }),

      enableAll: () => set((state) => ({ selectedIds: state.classes.map((c) => c.id) })),

      disableAll: () => set({ selectedIds: [] }),

      idsForGroup: (group) => resolveGroupIds(get().classes, get().groups[group]),

      selectGroup: (group, additive = false) =>
        set((state) => {
          const groupIds = resolveGroupIds(state.classes, state.groups[group]);
          const next = additive
            ? [...new Set([...state.selectedIds, ...groupIds])]
            : groupIds;
          return { selectedIds: next.sort((a, b) => a - b) };
        }),
    }),
    {
      name: 'inferencelab.classes',
      partialize: (state) => ({ selectedIds: state.selectedIds }),
    },
  ),
);
