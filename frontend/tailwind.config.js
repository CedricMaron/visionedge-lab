/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        surface: {
          950: '#080b12',
          900: '#0b0f17',
          850: '#0f1420',
          800: '#141a29',
          700: '#1c2333',
          600: '#28324a',
        },
        accent: {
          DEFAULT: '#38bdf8',
          muted: '#0ea5e9',
          soft: '#7dd3fc',
        },
        good: '#34d399',
        warn: '#fbbf24',
        bad: '#f87171',
      },
      fontFamily: {
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Consolas', 'monospace'],
      },
    },
  },
  plugins: [],
};
