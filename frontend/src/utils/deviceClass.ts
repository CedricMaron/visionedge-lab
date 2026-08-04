// What kind of device is this?
//
// The Device Capabilities page promises "no user-agent guessing — features are
// actually tested", and this classifier keeps that promise: it votes on probed
// browser APIs and never parses a UA string.
//
// `navigator.userAgentData.mobile` is the one exception that isn't one. It is a
// structured boolean the browser reports through the User-Agent Client Hints API,
// not a string anyone parses. It is Chromium-only, and its absence simply
// contributes no vote.
//
// Every signal, its observed value and the way it voted are returned alongside the
// verdict, so the UI can show its reasoning rather than asserting an answer.

export type DeviceClass = 'phone' | 'tablet' | 'desktop' | 'unknown';

export interface DeviceEvidence {
  signal: string;
  value: string;
  available: boolean;
  vote: 'phone' | 'desktop' | 'none';
  weight: number;
}

export interface DeviceClassification {
  deviceClass: DeviceClass;
  /** 0..1 — margin between the two sides over the total weight that voted. */
  confidence: number;
  evidence: DeviceEvidence[];
}

/** Raw probe results, injected so the classifier is testable without a browser. */
export interface DeviceSignals {
  uaDataMobile: boolean | null;
  pointerCoarse: boolean | null;
  anyPointerFine: boolean | null;
  maxTouchPoints: number | null;
  screenMinSideCss: number | null;
  hardwareConcurrency: number | null;
  deviceMemoryGb: number | null;
}

/** Screen width below which a coarse-pointer device is a phone rather than a tablet. */
const TABLET_MIN_SIDE_CSS_PX = 500;

export function readDeviceSignals(): DeviceSignals {
  const nav = navigator as Navigator & {
    deviceMemory?: number;
    userAgentData?: { mobile?: boolean };
  };

  const matches = (query: string): boolean | null => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return null;
    try {
      return window.matchMedia(query).matches;
    } catch {
      return null;
    }
  };

  // CSS pixels, not physical: screen.width already accounts for devicePixelRatio, so
  // a high-DPR phone is not mistaken for a desktop.
  const screenMinSideCss =
    typeof screen !== 'undefined' && screen.width && screen.height
      ? Math.min(screen.width, screen.height)
      : null;

  return {
    uaDataMobile: typeof nav.userAgentData?.mobile === 'boolean' ? nav.userAgentData.mobile : null,
    pointerCoarse: matches('(pointer: coarse)'),
    anyPointerFine: matches('(any-pointer: fine)'),
    maxTouchPoints: typeof nav.maxTouchPoints === 'number' ? nav.maxTouchPoints : null,
    screenMinSideCss,
    hardwareConcurrency:
      typeof nav.hardwareConcurrency === 'number' ? nav.hardwareConcurrency : null,
    deviceMemoryGb: typeof nav.deviceMemory === 'number' ? nav.deviceMemory : null,
  };
}

export function classifyDevice(signals: DeviceSignals): DeviceClassification {
  const evidence: DeviceEvidence[] = [];

  const record = (
    signal: string,
    raw: unknown,
    weight: number,
    decide: () => 'phone' | 'desktop' | 'none',
  ) => {
    const available = raw !== null && raw !== undefined;
    evidence.push({
      signal,
      value: available ? String(raw) : 'unavailable',
      available,
      vote: available ? decide() : 'none',
      weight,
    });
  };

  record('Mobile hint (User-Agent Client Hints)', signals.uaDataMobile, 3, () =>
    signals.uaDataMobile ? 'phone' : 'desktop',
  );
  record('Coarse pointer', signals.pointerCoarse, 3, () =>
    signals.pointerCoarse ? 'phone' : 'desktop',
  );
  record('Fine pointer present', signals.anyPointerFine, 3, () =>
    signals.anyPointerFine ? 'desktop' : 'phone',
  );
  record('Max touch points', signals.maxTouchPoints, 2, () =>
    (signals.maxTouchPoints ?? 0) > 0 ? 'phone' : 'desktop',
  );
  record('Screen min side (CSS px)', signals.screenMinSideCss, 2, () =>
    (signals.screenMinSideCss ?? 0) < TABLET_MIN_SIDE_CSS_PX ? 'phone' : 'desktop',
  );
  record('Logical CPU cores', signals.hardwareConcurrency, 1, () =>
    (signals.hardwareConcurrency ?? 0) <= 8 ? 'phone' : 'desktop',
  );
  record('Device memory (GB)', signals.deviceMemoryGb, 1, () =>
    (signals.deviceMemoryGb ?? 0) <= 4 ? 'phone' : 'desktop',
  );

  let phoneScore = 0;
  let desktopScore = 0;
  let totalWeight = 0;
  for (const item of evidence) {
    if (!item.available || item.vote === 'none') continue;
    totalWeight += item.weight;
    if (item.vote === 'phone') phoneScore += item.weight;
    else desktopScore += item.weight;
  }

  if (totalWeight === 0) {
    return { deviceClass: 'unknown', confidence: 0, evidence };
  }

  const confidence = Math.abs(phoneScore - desktopScore) / totalWeight;

  // A touch device with a large screen is a tablet, not a phone. Checked before the
  // phone/desktop comparison because a tablet votes "phone" on most touch signals.
  if (
    signals.pointerCoarse === true &&
    signals.screenMinSideCss !== null &&
    signals.screenMinSideCss >= TABLET_MIN_SIDE_CSS_PX
  ) {
    return { deviceClass: 'tablet', confidence, evidence };
  }

  if (phoneScore === desktopScore) {
    return { deviceClass: 'unknown', confidence: 0, evidence };
  }

  return {
    deviceClass: phoneScore > desktopScore ? 'phone' : 'desktop',
    confidence,
    evidence,
  };
}

export function detectDeviceClass(): DeviceClassification {
  return classifyDevice(readDeviceSignals());
}

/** Human label. "PC" rather than "desktop" because that is what the user asked about. */
export function deviceClassLabel(deviceClass: DeviceClass): string {
  switch (deviceClass) {
    case 'phone':
      return 'Phone';
    case 'tablet':
      return 'Tablet';
    case 'desktop':
      return 'PC';
    default:
      return 'Unknown';
  }
}
