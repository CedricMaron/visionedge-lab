// Polls /health for the TopBar connection/health pill.

import { api } from '@/services/api';
import { useAsync } from './useAsync';
import type { HealthResponse } from '@/types';

export type ConnState = 'connecting' | 'online' | 'offline';

export interface HealthView {
  state: ConnState;
  health: string;
  warnings: string[];
  reload: () => void;
}

export function useHealth(pollMs = 5000): HealthView {
  const { data, error, loading, reload } = useAsync<HealthResponse>(
    (signal) => api.health(signal),
    [],
    pollMs,
  );

  let state: ConnState = 'connecting';
  if (!loading || data) state = error ? 'offline' : 'online';
  if (error && !data) state = 'offline';

  return {
    state,
    health: data?.detection_health ?? (error ? 'unreachable' : 'unknown'),
    warnings: data?.warnings ?? [],
    reload,
  };
}
