// VisionEdge Lab — minimal, honest app-shell service worker.
// This is intentionally simple: it caches the app shell for offline load of
// static assets only. It NEVER caches API or WebSocket responses — live
// inference data must always come from the network.
const CACHE = 'visionedge-shell-v1';
const SHELL = ['/', '/index.html', '/manifest.json', '/favicon.svg'];

self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(CACHE).then((cache) => cache.addAll(SHELL)));
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))),
    ),
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  // Never touch API traffic — always go to network.
  if (url.pathname.startsWith('/api') || event.request.method !== 'GET') {
    return;
  }
  event.respondWith(
    caches.match(event.request).then((cached) => cached || fetch(event.request)),
  );
});
