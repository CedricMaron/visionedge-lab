// Settings store, persisted to localStorage. The API base is also mirrored into
// the config module so the fetch wrapper picks it up.

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { getDefaultApiBase, setApiBase } from '@/config';

interface SettingsState {
  apiBase: string;
  defaultConfidence: number;
  defaultIou: number;
  vlmGrounding: boolean;
  structuredOutput: boolean;

  setApiBase: (base: string) => void;
  setDefaultConfidence: (v: number) => void;
  setDefaultIou: (v: number) => void;
  setVlmGrounding: (v: boolean) => void;
  setStructuredOutput: (v: boolean) => void;
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      apiBase: getDefaultApiBase(),
      defaultConfidence: 0.25,
      defaultIou: 0.45,
      vlmGrounding: true,
      structuredOutput: false,

      setApiBase: (base) => {
        setApiBase(base);
        set({ apiBase: base.replace(/\/+$/, '') });
      },
      setDefaultConfidence: (v) => set({ defaultConfidence: v }),
      setDefaultIou: (v) => set({ defaultIou: v }),
      setVlmGrounding: (v) => set({ vlmGrounding: v }),
      setStructuredOutput: (v) => set({ structuredOutput: v }),
    }),
    {
      name: 'visionedge.settings',
      onRehydrateStorage: () => (state) => {
        if (state?.apiBase) setApiBase(state.apiBase);
      },
    },
  ),
);
