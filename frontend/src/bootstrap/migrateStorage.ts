// Side-effect module: runs the localStorage key migration at import time.
//
// This exists as its own module purely for ordering. ES module imports are fully
// evaluated before any statement in the importing module's body, and zustand's
// `persist` middleware rehydrates from localStorage during store construction —
// which happens at module-evaluation time. Calling the migration from main.tsx's
// body would therefore run it *after* every store had already read the old,
// now-unread keys and fallen back to defaults.
//
// Importing this first in main.tsx guarantees the rename happens before any store
// module is evaluated.

import { migrateLegacyStorage } from '@/utils/storageMigration';

migrateLegacyStorage();
