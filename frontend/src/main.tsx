// MUST be the first import: migrates legacy localStorage keys before any store
// module is evaluated. See the module docstring for why ordering matters here.
import '@/bootstrap/migrateStorage';

import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import App from './App';
import './index.css';

// No service worker is registered any more, and any surviving one is removed.
//
// The previous worker was cache-first over every GET, so after a deploy it kept
// serving the old index.html against asset hashes that no longer existed and the
// page rendered blank. Registering `/sw.js` is still what retires it: that file is
// now a kill switch that clears the caches and unregisters itself. Unregistering
// from here as well covers the browser that has a worker but never fetches the
// update — belt and braces, because the failure mode is an invisible one.
if ('serviceWorker' in navigator && import.meta.env.PROD) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/sw.js').catch(() => {
      navigator.serviceWorker
        .getRegistrations()
        .then((registrations) => registrations.forEach((r) => r.unregister()))
        .catch(() => {
          /* Nothing further this page can do; a hard reload still recovers. */
        });
    });
  });
}

const rootEl = document.getElementById('root');
if (!rootEl) throw new Error('Root element #root not found');

ReactDOM.createRoot(rootEl).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>,
);
