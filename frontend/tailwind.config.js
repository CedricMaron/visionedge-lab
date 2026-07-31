/** @type {import('tailwindcss').Config} */

// Colours resolve through the CSS variables defined in src/index.css. The
// <alpha-value> placeholder keeps Tailwind's opacity modifiers (bg-accent/10)
// working against a variable-based palette.
//
// The old `surface-*` scale is deliberately absent: a stray pre-migration class now
// fails the build instead of quietly rendering a dark patch on a light page.
const token = (name) => `rgb(var(--${name}) / <alpha-value>)`;

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  darkMode: ['selector', '[data-theme="dark"]'],
  theme: {
    extend: {
      colors: {
        canvas: token('canvas'),
        panel: token('panel'),
        elevated: token('elevated'),
        overlay: token('overlay'),

        accent: {
          DEFAULT: token('accent'),
          hover: token('accent-hover'),
          soft: token('accent-soft'),
          contrast: token('accent-contrast'),
        },

        good: { DEFAULT: token('good'), soft: token('good-soft') },
        warn: { DEFAULT: token('warn'), soft: token('warn-soft') },
        bad: { DEFAULT: token('bad'), soft: token('bad-soft') },

        series: {
          1: token('series-1'),
          2: token('series-2'),
          3: token('series-3'),
          4: token('series-4'),
          5: token('series-5'),
          6: token('series-6'),
        },
      },
      textColor: {
        primary: token('text-primary'),
        secondary: token('text-secondary'),
        muted: token('text-muted'),
        inverse: token('text-inverse'),
      },
      borderColor: {
        subtle: token('border-subtle'),
        strong: token('border-strong'),
        interactive: token('border-interactive'),
      },
      fontFamily: {
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Consolas', 'monospace'],
      },
      fontSize: {
        '2xs': ['0.6875rem', { lineHeight: '1rem' }],
      },
    },
  },
  plugins: [],
};
