// Central runtime configuration. The API base is env-configurable
// (VITE_API_BASE) but can be overridden at runtime via Settings (localStorage).
// The WebSocket URL is always DERIVED from the API base — never hardcoded.

const LS_KEY = 'visionedge.apiBase';
const DEFAULT_API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000';

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
