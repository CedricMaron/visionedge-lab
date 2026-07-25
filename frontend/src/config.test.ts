import { describe, it, expect, beforeEach } from 'vitest';
import { getApiBase, getDefaultApiBase, setApiBase, getWsUrl } from './config';

// With no VITE_API_BASE baked in (the same-origin deployment case), the app must
// talk to the origin it was served from rather than a developer's localhost.
describe('getApiBase', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it('falls back to the serving origin when no API base is configured', () => {
    expect(getDefaultApiBase()).toBe(window.location.origin);
  });

  it('prefers a runtime override from Settings over the default', () => {
    setApiBase('https://elsewhere.example');
    expect(getApiBase()).toBe('https://elsewhere.example');
  });

  it('strips trailing slashes from an override', () => {
    setApiBase('https://elsewhere.example///');
    expect(getApiBase()).toBe('https://elsewhere.example');
  });

  it('clearing the override returns to the default', () => {
    setApiBase('https://elsewhere.example');
    setApiBase('');
    expect(getApiBase()).toBe(getDefaultApiBase());
  });

  it('derives a websocket URL from the API base scheme', () => {
    setApiBase('https://visionedge.c-maron.space');
    expect(getWsUrl('/api/ws/detect')).toBe('wss://visionedge.c-maron.space/api/ws/detect');
    setApiBase('http://localhost:8000');
    expect(getWsUrl('/api/ws/detect')).toBe('ws://localhost:8000/api/ws/detect');
  });
});
