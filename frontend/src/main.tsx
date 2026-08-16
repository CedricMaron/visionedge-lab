// MUST be the first import: migrates legacy localStorage keys before any store
// module is evaluated. See the module docstring for why ordering matters here.
import '@/bootstrap/migrateStorage';

import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import App from './App';
import './index.css';

// This page NEVER registers a service worker. It only removes one.
//
// Registering here is what put the site in a reload loop: `/sw.js` is a kill
// switch that unregisters itself and navigates its clients, so every load created
// a worker that immediately retired itself and reloaded the page, which created
// the next one. Registration and self-removal fought each other forever.
//
// A browser that still carries the old cache-first worker does not need this line
// to be rescued: it re-fetches /sw.js as its own update check on navigation, gets
// the kill switch, and that worker cleans up. This is only for the case where a
// registration survives without ever activating.
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker
      .getRegistrations()
      .then((registrations) => registrations.forEach((registration) => registration.unregister()))
      .catch(() => {
        /* Nothing further this page can do; a hard reload still recovers. */
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
