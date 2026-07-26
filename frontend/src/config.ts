// Central runtime configuration. The API base is env-configurable
// (VITE_API_BASE) but can be overridden at runtime via Settings (localStorage).
// The WebSocket URL is always DERIVED from the API base — never hardcoded.

const LS_KEY = 'inferencelab.apiBase';

// When VITE_API_BASE is not baked in at build time, talk to the origin the app was
// served from. That makes one build artifact work for a same-origin deployment
// behind a reverse proxy (visionedge.c-maron.space) without a rebuild. Local dev
// sets VITE_API_BASE=http://localhost:8000 in .env, and Settings can override both.
function defaultApiBase(): string {
  const configured = import.meta.env.VITE_API_BASE;
  if (configured && configured.trim().length > 0) return configured;
  if (typeof window !== 'undefined' && window.location?.origin) return window.location.origin;
  return 'http://localhost:8000';
}

const DEFAULT_API_BASE = defaultApiBase();

export function getApiBase(): string {
  if (typeof localStorage !== 'undefined') {
    const stored = localStorage.getItem(LS_KEY);
    if (stored && stored.trim().length > 0) return stored.replace(/\/+$/, '');
  }
  return DEFAULT_API_BASE.replace(/\/+$/, '');
}

export function setApiBase(base: string): void {
  if (typeof localStorage === 'undefined') return;
  const clean = base.trim().replace(/\/+$/, '');
  if (clean.length === 0) {
    localStorage.removeItem(LS_KEY);
  } else {
    localStorage.setItem(LS_KEY, clean);
  }
}

export function getDefaultApiBase(): string {
  return DEFAULT_API_BASE.replace(/\/+$/, '');
}

// Derive ws:// or wss:// from the http(s) API base.
export function getWsUrl(path: string): string {
  const base = getApiBase();
  const wsBase = base.replace(/^http:/, 'ws:').replace(/^https:/, 'wss:');
  const p = path.startsWith('/') ? path : `/${path}`;
  return `${wsBase}${p}`;
}
