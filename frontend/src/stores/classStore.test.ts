import { describe, it, expect, beforeEach } from 'vitest';
import { useClassStore } from './classStore';
import type { ClassEntry, ClassGroups } from '@/types';

const CLASSES: ClassEntry[] = [
  { id: 0, name: 'person' },
  { id: 1, name: 'bicycle' },
  { id: 2, name: 'car' },
  { id: 3, name: 'dog' },
  { id: 4, name: 'cat' },
  { id: 5, name: 'chair' },
];

const GROUPS: ClassGroups = {
  people: ['person'],
  vehicles: ['bicycle', 'car'],
  animals: ['dog', 'cat'],
  indoor: ['chair'],
};

function resetStore() {
  useClassStore.setState({
    classes: [],
    groups: { people: [], vehicles: [], animals: [], indoor: [] },
    selectedIds: [],
    loaded: false,
  });
}

describe('classStore', () => {
  beforeEach(() => {
    localStorage.clear();
    resetStore();
  });

  it('defaults to all classes selected on first catalog load', () => {
    useClassStore.getState().setCatalog(CLASSES, GROUPS);
    expect(useClassStore.getState().selectedIds).toEqual([0, 1, 2, 3, 4, 5]);
    expect(useClassStore.getState().loaded).toBe(true);
  });

  it('toggles a single class off and back on', () => {
    const s = useClassStore.getState();
    s.setCatalog(CLASSES, GROUPS);
    s.toggle(2);
    expect(useClassStore.getState().isSelected(2)).toBe(false);
    useClassStore.getState().toggle(2);
    expect(useClassStore.getState().isSelected(2)).toBe(true);
  });

  it('selectGroup replaces selection with the group ids (non-additive)', () => {
    const s = useClassStore.getState();
    s.setCatalog(CLASSES, GROUPS);
    s.selectGroup('vehicles');
    expect(useClassStore.getState().selectedIds).toEqual([1, 2]);
  });

  it('selectGroup additive merges without duplicates and stays sorted', () => {
    const s = useClassStore.getState();
    s.setCatalog(CLASSES, GROUPS);
    s.selectGroup('animals'); // [3,4]
    s.selectGroup('people', true); // + [0]
    s.selectGroup('animals', true); // no dup
    expect(useClassStore.getState().selectedIds).toEqual([0, 3, 4]);
  });

  it('enableAll / disableAll work', () => {
    const s = useClassStore.getState();
    s.setCatalog(CLASSES, GROUPS);
    s.disableAll();
    expect(useClassStore.getState().selectedIds).toEqual([]);
    useClassStore.getState().enableAll();
    expect(useClassStore.getState().selectedIds).toEqual([0, 1, 2, 3, 4, 5]);
  });

  it('persists selection to localStorage', () => {
    const s = useClassStore.getState();
    s.setCatalog(CLASSES, GROUPS);
    s.selectGroup('people'); // [0]
    const raw = localStorage.getItem('visionedge.classes');
    expect(raw).toBeTruthy();
    const parsed = JSON.parse(raw as string);
    expect(parsed.state.selectedIds).toEqual([0]);
  });
});
