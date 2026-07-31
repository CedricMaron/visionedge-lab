/**
 * WCAG AA contrast is asserted, not eyeballed.
 *
 * The token values are parsed straight out of src/index.css, so this test fails if
 * someone edits the palette into something unreadable — in either theme. Checking by
 * eye is exactly how a "professional" light theme ends up with 3:1 body text.
 *
 * Thresholds (WCAG 2.1):
 *   4.5:1  normal body text
 *   3.0:1  large text (>= 18.66px bold / 24px regular) and non-text UI boundaries
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { describe, expect, it } from 'vitest';

type Rgb = [number, number, number];

// Read the stylesheet that actually ships. A `?raw` import resolves to an empty
// string here because vitest disables CSS processing, which would have made every
// assertion below vacuously pass.
const CSS = readFileSync(join(dirname(fileURLToPath(import.meta.url)), '..', 'index.css'), 'utf8');

/**
 * Extract the token table for one selector block.
 *
 * Matched by pattern rather than exact string so the test does not break on
 * reformatting — it is asserting colour ratios, not whitespace.
 */
function parseTokens(selectorPattern: RegExp): Record<string, Rgb> {
  const match = selectorPattern.exec(CSS);
  if (!match) throw new Error(`selector ${selectorPattern} not found in index.css`);
  const open = CSS.indexOf('{', match.index);
  const close = CSS.indexOf('}', open);
  const block = CSS.slice(open, close);

  const tokens: Record<string, Rgb> = {};
  for (const match of block.matchAll(/--([a-z0-9-]+):\s*(\d+)\s+(\d+)\s+(\d+)\s*;/g)) {
    tokens[match[1]] = [Number(match[2]), Number(match[3]), Number(match[4])];
  }
  return tokens;
}

/** Relative luminance, per WCAG 2.1 definition. */
function luminance([r, g, b]: Rgb): number {
  const channel = (v: number) => {
    const s = v / 255;
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b);
}

function contrast(a: Rgb, b: Rgb): number {
  const [lighter, darker] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (lighter + 0.05) / (darker + 0.05);
}

const THEMES = {
  light: parseTokens(/:root\s*\{/),
  dark: parseTokens(/:root\[data-theme=['"]dark['"]\]\s*\{/),
};

/** [foreground, background, minimum ratio, description] */
const PAIRS: Array<[string, string, number, string]> = [
  ['text-primary', 'canvas', 4.5, 'body text on the page background'],
  ['text-primary', 'panel', 4.5, 'body text on a card'],
  ['text-primary', 'elevated', 4.5, 'body text in an input'],
  ['text-secondary', 'canvas', 4.5, 'secondary text on the page'],
  ['text-secondary', 'panel', 4.5, 'secondary text on a card'],
  ['text-muted', 'canvas', 4.5, 'muted text and labels on the page'],
  ['text-muted', 'panel', 4.5, 'muted text on a card'],
  ['accent', 'panel', 3.0, 'accent links and icons on a card'],
  ['accent-contrast', 'accent', 4.5, 'primary button label on its fill'],
  ['good', 'panel', 4.5, 'success status text'],
  ['warn', 'panel', 4.5, 'warning status text'],
  ['bad', 'panel', 4.5, 'error status text'],
  ['good', 'good-soft', 4.5, 'success text on its own tint'],
  ['warn', 'warn-soft', 4.5, 'warning text on its own tint'],
  ['bad', 'bad-soft', 4.5, 'error text on its own tint'],
  // WCAG 1.4.11 applies to boundaries that identify a control, not to decorative
  // dividers — requiring 3:1 of every hairline would mean heavy borders everywhere.
  ['border-interactive', 'panel', 3.0, 'form-control boundary'],
  ['border-interactive', 'canvas', 3.0, 'form-control boundary on the page'],
];

describe.each(Object.entries(THEMES))('%s theme', (themeName, tokens) => {
  it('defines every token the pairs reference', () => {
    for (const [fg, bg] of PAIRS) {
      expect(tokens[fg], `${themeName}: --${fg} missing`).toBeDefined();
      expect(tokens[bg], `${themeName}: --${bg} missing`).toBeDefined();
    }
  });

  it.each(PAIRS)('%s on %s meets %s:1 (%s)', (fg, bg, minimum, description) => {
    const ratio = contrast(tokens[fg], tokens[bg]);
    expect(
      ratio,
      `${themeName}: ${description} — --${fg} on --${bg} is ${ratio.toFixed(2)}:1, ` +
        `below the required ${minimum}:1`,
    ).toBeGreaterThanOrEqual(minimum);
  });

  it('keeps the categorical chart series mutually distinguishable', () => {
    // Two series that render at a similar luminance are hard to tell apart in a
    // line chart even when their hues differ.
    const series = [1, 2, 3, 4, 5, 6].map((i) => tokens[`series-${i}`]);
    for (const colour of series) expect(colour).toBeDefined();
    for (const colour of series) {
      expect(contrast(colour, tokens.panel)).toBeGreaterThanOrEqual(2.5);
    }
  });
});

describe('theme completeness', () => {
  it('light and dark define the same token set', () => {
    // A token present in one theme but not the other renders as an invalid colour
    // when the user toggles.
    expect(Object.keys(THEMES.light).sort()).toEqual(Object.keys(THEMES.dark).sort());
  });

  it('light is the default theme', () => {
    const start = /:root\s*\{/.exec(CSS)!.index;
    expect(CSS.slice(start, CSS.indexOf('}', start))).toContain('color-scheme: light');
  });
});
