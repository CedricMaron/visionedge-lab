// Settings store, persisted to localStorage. The API base is also mirrored into
// the config module so the fetch wrapper picks it up.

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { getDefaultApiBase, setApiBase } from '@/config';

/** Stamp the theme on <html>; the CSS variable blocks key off this attribute. */
export function applyTheme(theme: Theme): void {
  if (typeof document === 'undefined') return;
  document.documentElement.setAttribute('data-theme', theme);
}

export type Theme = 'light' | 'dark';

interface SettingsState {
  apiBase: string;
  theme: Theme;
  defaultConfidence: number;
  defaultIou: number;
  vlmGrounding: boolean;
  structuredOutput: boolean;

  setApiBase: (base: string) => void;
  setTheme: (theme: Theme) => void;
  setDefaultConfidence: (v: number) => void;
  setDefaultIou: (v: number) => void;
  setVlmGrounding: (v: boolean) => void;
  setStructuredOutput: (v: boolean) => void;
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      apiBase: getDefaultApiBase(),
      // Light by default: the platform is meant to read as an engineering tool in
      // ordinary lighting. Dark is available but opt-in.
      theme: 'light',
      defaultConfidence: 0.25,
      defaultIou: 0.45,
      vlmGrounding: true,
      structuredOutput: false,

      setApiBase: (base) => {
        setApiBase(base);
        set({ apiBase: base.replace(/\/+$/, '') });
      },
      setTheme: (theme) => {
        applyTheme(theme);
        set({ theme });
      },
      setDefaultConfidence: (v) => set({ defaultConfidence: v }),
      setDefaultIou: (v) => set({ defaultIou: v }),
      setVlmGrounding: (v) => set({ vlmGrounding: v }),
      setStructuredOutput: (v) => set({ structuredOutput: v }),
    }),
    {
      name: 'inferencelab.settings',
      onRehydrateStorage: () => (state) => {
        if (state?.apiBase) setApiBase(state.apiBase);
        // Apply on rehydrate so a reload does not flash the default theme.
        applyTheme(state?.theme ?? 'light');
      },
    },
  ),
);
