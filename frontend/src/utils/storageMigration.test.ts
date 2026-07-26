import { beforeEach, describe, expect, it } from 'vitest';
import { migrateLegacyStorage } from './storageMigration';

/** Minimal in-memory Storage so tests never depend on jsdom's global localStorage. */
function makeStorage(seed: Record<string, string> = {}): Storage {
  const map = new Map(Object.entries(seed));
  return {
    get length() {
      return map.size;
    },
    clear: () => map.clear(),
    getItem: (k: string) => (map.has(k) ? map.get(k)! : null),
    key: (i: number) => Array.from(map.keys())[i] ?? null,
    removeItem: (k: string) => void map.delete(k),
    setItem: (k: string, v: string) => void map.set(k, v),
  } as Storage;
}

describe('migrateLegacyStorage', () => {
  let storage: Storage;

  beforeEach(() => {
    storage = makeStorage();
  });

  it('moves a legacy key to its current name', () => {
    storage.setItem('visionedge.apiBase', 'https://example.test');

    const result = migrateLegacyStorage(storage);

    expect(storage.getItem('inferencelab.apiBase')).toBe('https://example.test');
    expect(storage.getItem('visionedge.apiBase')).toBeNull();
    expect(result.migrated).toContain('visionedge.apiBase');
  });

  it('migrates every known key', () => {
    storage.setItem('visionedge.apiBase', 'a');
    storage.setItem('visionedge.settings', '{"x":1}');
    storage.setItem('visionedge.classes', '[1,2]');

    expect(migrateLegacyStorage(storage).migrated).toHaveLength(3);
    expect(storage.getItem('inferencelab.settings')).toBe('{"x":1}');
    expect(storage.getItem('inferencelab.classes')).toBe('[1,2]');
  });

  it('never overwrites newer state under the current key', () => {
    // A visitor who already used the renamed build must not be reset by a stale value.
    storage.setItem('visionedge.settings', '{"old":true}');
    storage.setItem('inferencelab.settings', '{"new":true}');

    const result = migrateLegacyStorage(storage);

    expect(storage.getItem('inferencelab.settings')).toBe('{"new":true}');
    expect(result.skipped).toContain('visionedge.settings');
    // The legacy entry is left alone rather than deleted, so nothing is destroyed.
    expect(storage.getItem('visionedge.settings')).toBe('{"old":true}');
  });

  it('is a no-op when there is nothing to migrate', () => {
    const result = migrateLegacyStorage(storage);
    expect(result.migrated).toHaveLength(0);
    expect(storage.length).toBe(0);
  });

  it('is idempotent across repeated boots', () => {
    storage.setItem('visionedge.apiBase', 'a');
    migrateLegacyStorage(storage);
    const second = migrateLegacyStorage(storage);

    expect(second.migrated).toHaveLength(0);
    expect(storage.getItem('inferencelab.apiBase')).toBe('a');
  });

  it('preserves the legacy value when writing fails', () => {
    // Quota exhaustion or a blocked store must not destroy the only copy.
    const failing = makeStorage({ 'visionedge.apiBase': 'a' });
    failing.setItem = () => {
      throw new DOMException('QuotaExceededError');
    };

    const result = migrateLegacyStorage(failing);

    expect(result.skipped).toContain('visionedge.apiBase');
    expect(failing.getItem('visionedge.apiBase')).toBe('a');
  });

  it('tolerates storage that throws on read', () => {
    const blocked = makeStorage();
    blocked.getItem = () => {
      throw new DOMException('SecurityError');
    };

    expect(() => migrateLegacyStorage(blocked)).not.toThrow();
  });
});
