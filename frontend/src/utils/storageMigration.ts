// Carries browser-persisted state across the VisionEdge Lab -> InferenceLab rename.
//
// Zustand's `persist` middleware and config.ts both key off localStorage names. Changing
// those names without migrating would silently reset every returning visitor's API base,
// class selection and settings — the state would still be on disk, just orphaned under a
// key nothing reads any more.
//
// This runs once, before any store is created (see main.tsx), and is a no-op after the
// first visit.

const KEY_RENAMES: ReadonlyArray<readonly [legacy: string, current: string]> = [
  ['visionedge.apiBase', 'inferencelab.apiBase'],
  ['visionedge.settings', 'inferencelab.settings'],
  ['visionedge.classes', 'inferencelab.classes'],
];

export interface StorageMigrationResult {
  migrated: string[];
  skipped: string[];
}

/**
 * Move legacy localStorage entries to their current keys.
 *
 * An existing value under the current key always wins and the legacy entry is left
 * alone — a returning visitor who has already used the renamed build must never have
 * their newer state overwritten by a stale one.
 */
export function migrateLegacyStorage(storage: Storage = localStorage): StorageMigrationResult {
  const migrated: string[] = [];
  const skipped: string[] = [];

  for (const [legacy, current] of KEY_RENAMES) {
    let legacyValue: string | null = null;
    try {
      legacyValue = storage.getItem(legacy);
    } catch {
      // Private-mode or blocked storage: nothing to migrate, and nothing to report.
      continue;
    }
    if (legacyValue === null) continue;

    if (storage.getItem(current) !== null) {
      skipped.push(legacy);
      continue;
    }

    try {
      storage.setItem(current, legacyValue);
      storage.removeItem(legacy);
      migrated.push(legacy);
    } catch {
      // Quota or permission failure. The legacy value stays put so a later attempt
      // can retry; losing it would be worse than leaving it orphaned.
      skipped.push(legacy);
    }
  }

  return { migrated, skipped };
}
