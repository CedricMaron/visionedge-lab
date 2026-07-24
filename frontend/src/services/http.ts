// Tiny fetch wrapper. Adds base URL, JSON handling, and a typed error.

import { getApiBase } from '@/config';

export class ApiError extends Error {
  readonly status: number;
  readonly body: unknown;
  constructor(message: string, status: number, body: unknown) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.body = body;
  }
}

interface RequestOptions {
  method?: string;
  query?: Record<string, string | number | boolean | undefined>;
  body?: BodyInit | null;
  headers?: Record<string, string>;
  signal?: AbortSignal;
  json?: unknown;
}

function buildUrl(path: string, query?: RequestOptions['query']): string {
  const base = getApiBase();
  const p = path.startsWith('/') ? path : `/${path}`;
  const url = new URL(`${base}${p}`);
  if (query) {
    for (const [k, v] of Object.entries(query)) {
      if (v !== undefined) url.searchParams.set(k, String(v));
    }
  }
  return url.toString();
}

export async function request<T>(path: string, opts: RequestOptions = {}): Promise<T> {
  const headers: Record<string, string> = { ...opts.headers };
  let body = opts.body;

  if (opts.json !== undefined) {
    headers['Content-Type'] = 'application/json';
    body = JSON.stringify(opts.json);
  }

  const res = await fetch(buildUrl(path, opts.query), {
    method: opts.method ?? 'GET',
    headers,
    body,
    signal: opts.signal,
  });

  const text = await res.text();
  let data: unknown = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = text;
    }
  }

  if (!res.ok) {
    const message =
      (data && typeof data === 'object' && 'detail' in data
        ? String((data as { detail: unknown }).detail)
        : res.statusText) || `HTTP ${res.status}`;
    throw new ApiError(message, res.status, data);
  }

  return data as T;
}

export const http = {
  get: <T>(path: string, query?: RequestOptions['query'], signal?: AbortSignal) =>
    request<T>(path, { method: 'GET', query, signal }),
  postJson: <T>(path: string, json: unknown, query?: RequestOptions['query']) =>
    request<T>(path, { method: 'POST', json, query }),
  postForm: <T>(path: string, form: FormData, query?: RequestOptions['query']) =>
    request<T>(path, { method: 'POST', body: form, query }),
};
