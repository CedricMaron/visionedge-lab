// InferenceLab — service worker removal.
//
// This file used to be a cache-first app-shell worker. That is why the site went
// blank after a deploy: it answered every GET from its cache, including
// `/index.html`, and it only reinstalls when this file's own bytes change. So a
// returning visitor got the PREVIOUS index.html, which referenced asset hashes the
// rebuild had deleted — the scripts 404'd, nothing executed, and the page rendered
// empty. Meanwhile the server was serving the new build correctly to anyone
// without the worker, which is what made it look like a server problem.
//
// It is not replaced with a smarter caching strategy, because there was nothing to
// win: every useful thing this app does — inference, probes, telemetry, benchmarks
// — requires the server, so an offline shell could only ever show a broken tool.
// The one job left is to remove itself from the browsers that still have it.
//
// Deleting the file would NOT have worked: `/sw.js` would then fall through the
// SPA rewrite and return index.html as text/html, the update would fail on the MIME
// check, and the old worker would stay registered forever. A worker can only be
// retired by a worker.
//
// Note the deliberate absence of a `fetch` handler: with none registered, requests
// go straight to the network from the moment this version activates, so a client is
// no longer served stale content even before the unregistration completes.

self.addEventListener('install', () => {
  // Do not wait for existing tabs to close — they are the broken ones.
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    (async () => {
      // Drop every cache this origin ever created, not just the known name: a
      // cache left behind under an older key would keep serving a stale shell.
      const keys = await caches.keys();
      await Promise.all(keys.map((key) => caches.delete(key)));

      await self.clients.claim();
      await self.registration.unregister();

      // Reload open windows ONLY when there was a stale cache to clear.
      //
      // Reloading unconditionally is what turned this file into a reload loop:
      // paired with a page that re-registered on load, each activation reloaded
      // the page, which registered another worker, which reloaded again. Gating on
      // real work means a worker that finds nothing to clean exits quietly, so the
      // cycle cannot repeat even if something registers this file again.
      if (keys.length === 0) return;

      const clients = await self.clients.matchAll({ type: 'window' });
      for (const client of clients) {
        client.navigate(client.url).catch(() => {
          /* A client that refuses to navigate recovers on its next reload. */
        });
      }
    })(),
  );
});
