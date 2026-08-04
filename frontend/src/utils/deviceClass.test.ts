import { describe, expect, it } from 'vitest';
import {
  classifyDevice,
  deviceClassLabel,
  type DeviceSignals,
} from './deviceClass';

/** Signals are injected, so no test has to monkey-patch `navigator`. */
function signals(overrides: Partial<DeviceSignals> = {}): DeviceSignals {
  return {
    uaDataMobile: null,
    pointerCoarse: null,
    anyPointerFine: null,
    maxTouchPoints: null,
    screenMinSideCss: null,
    hardwareConcurrency: null,
    deviceMemoryGb: null,
    ...overrides,
  };
}

const PHONE = signals({
  uaDataMobile: true,
  pointerCoarse: true,
  anyPointerFine: false,
  maxTouchPoints: 5,
  screenMinSideCss: 390,
  hardwareConcurrency: 8,
  deviceMemoryGb: 4,
});

const DESKTOP = signals({
  uaDataMobile: false,
  pointerCoarse: false,
  anyPointerFine: true,
  maxTouchPoints: 0,
  screenMinSideCss: 1080,
  hardwareConcurrency: 12,
  deviceMemoryGb: 8,
});

const TABLET = signals({
  uaDataMobile: false,
  pointerCoarse: true,
  anyPointerFine: false,
  maxTouchPoints: 5,
  screenMinSideCss: 820,
  hardwareConcurrency: 8,
  deviceMemoryGb: 4,
});

describe('classifyDevice', () => {
  it('identifies a phone', () => {
    const result = classifyDevice(PHONE);
    expect(result.deviceClass).toBe('phone');
    expect(result.confidence).toBeGreaterThan(0.8);
  });

  it('identifies a PC', () => {
    const result = classifyDevice(DESKTOP);
    expect(result.deviceClass).toBe('desktop');
    expect(result.confidence).toBeGreaterThan(0.8);
  });

  it('identifies a tablet by screen size despite touch signals', () => {
    // A tablet votes "phone" on almost every touch signal; only the screen
    // distinguishes it, which is why that check runs before the tally.
    expect(classifyDevice(TABLET).deviceClass).toBe('tablet');
  });

  it('returns unknown when nothing can be probed', () => {
    const result = classifyDevice(signals());
    expect(result.deviceClass).toBe('unknown');
    expect(result.confidence).toBe(0);
  });

  it('returns unknown on an exact tie rather than picking a side', () => {
    // Coarse pointer (3, phone) against fine pointer present (3, desktop).
    const result = classifyDevice(
      signals({ pointerCoarse: true, anyPointerFine: true }),
    );
    expect(result.deviceClass).toBe('unknown');
  });

  it('classifies from partial signals', () => {
    // Chromium-only hints are absent in Firefox and Safari; the rest must still work.
    const result = classifyDevice(
      signals({ pointerCoarse: true, anyPointerFine: false, maxTouchPoints: 5 }),
    );
    expect(result.deviceClass).toBe('phone');
  });

  it('does not let weak signals outvote strong ones', () => {
    // A low-core, low-memory desktop must not be called a phone: cores and memory
    // carry weight 1 each against 3 for each pointer signal.
    const result = classifyDevice(
      signals({
        pointerCoarse: false,
        anyPointerFine: true,
        hardwareConcurrency: 4,
        deviceMemoryGb: 4,
      }),
    );
    expect(result.deviceClass).toBe('desktop');
  });

  it('treats a high-DPR phone screen as small', () => {
    // screen.width is in CSS pixels, so a 3x-DPR phone reports ~390, not ~1170.
    const result = classifyDevice(
      signals({ pointerCoarse: true, anyPointerFine: false, screenMinSideCss: 390 }),
    );
    expect(result.deviceClass).toBe('phone');
  });
});

describe('evidence', () => {
  it('reports every signal, including the unavailable ones', () => {
    const result = classifyDevice(signals({ pointerCoarse: true }));
    expect(result.evidence).toHaveLength(7);
    expect(result.evidence.filter((e) => !e.available).length).toBe(6);
  });

  it('shows how each available signal voted', () => {
    const result = classifyDevice(PHONE);
    const coarse = result.evidence.find((e) => e.signal === 'Coarse pointer');
    expect(coarse).toMatchObject({ available: true, vote: 'phone', value: 'true' });
  });

  it('marks unavailable signals as casting no vote', () => {
    const result = classifyDevice(signals({ pointerCoarse: true }));
    const missing = result.evidence.filter((e) => !e.available);
    expect(missing.every((e) => e.vote === 'none')).toBe(true);
    expect(missing.every((e) => e.value === 'unavailable')).toBe(true);
  });

  it('never silently drops a signal', () => {
    const names = classifyDevice(signals()).evidence.map((e) => e.signal);
    expect(new Set(names).size).toBe(names.length);
  });
});

describe('deviceClassLabel', () => {
  it('says PC rather than desktop', () => {
    expect(deviceClassLabel('desktop')).toBe('PC');
  });

  it.each([
    ['phone', 'Phone'],
    ['tablet', 'Tablet'],
    ['unknown', 'Unknown'],
  ] as const)('labels %s as %s', (input, expected) => {
    expect(deviceClassLabel(input)).toBe(expected);
  });
});
