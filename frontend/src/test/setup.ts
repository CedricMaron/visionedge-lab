// Vitest setup. jsdom provides localStorage, but we reset it between tests.
import { afterEach, beforeEach } from 'vitest';

beforeEach(() => {
  localStorage.clear();
});

afterEach(() => {
  localStorage.clear();
});
