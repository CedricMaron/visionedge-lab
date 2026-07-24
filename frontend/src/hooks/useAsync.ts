import { useCallback, useEffect, useRef, useState } from 'react';

export interface AsyncState<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
  reload: () => void;
}

// Runs an async fetcher on mount and exposes reload(). Optionally re-polls.
export function useAsync<T>(
  fetcher: (signal: AbortSignal) => Promise<T>,
  deps: unknown[] = [],
  pollMs?: number,
): AsyncState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [tick, setTick] = useState(0);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  const reload = useCallback(() => setTick((t) => t + 1), []);

  useEffect(() => {
    const controller = new AbortController();
    let active = true;
    setLoading(true);
    fetcherRef
      .current(controller.signal)
      .then((res) => {
        if (!active) return;
        setData(res);
        setError(null);
      })
      .catch((err: unknown) => {
        if (!active || controller.signal.aborted) return;
        setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    let interval: ReturnType<typeof setInterval> | undefined;
    if (pollMs && pollMs > 0) {
      interval = setInterval(() => {
        fetcherRef
          .current(controller.signal)
          .then((res) => {
            if (!active) return;
            setData(res);
            setError(null);
          })
          .catch((err: unknown) => {
            if (!active || controller.signal.aborted) return;
            setError(err instanceof Error ? err.message : String(err));
          });
      }, pollMs);
    }

    return () => {
      active = false;
      controller.abort();
      if (interval) clearInterval(interval);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tick, pollMs, ...deps]);

  return { data, error, loading, reload };
}
