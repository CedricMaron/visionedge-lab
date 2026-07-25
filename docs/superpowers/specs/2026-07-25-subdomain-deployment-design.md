# Public deployment at visionedge.c-maron.space — design

**Date:** 2026-07-25
**Status:** approved, not yet implemented

## Goal

Serve VisionEdge Lab from `visionedge.c-maron.space`, a subdomain of the author's
resume/portfolio site, as a single origin.

**In scope for this repo:** origin configuration, reverse-proxy config, compose
wiring, rate limiting, and deployment documentation.

**Not in scope:** DNS records and host provisioning. Those are performed by the
author outside the repo. This design states what they must point at.

## What already exists

Nothing about the network is hardcoded, so this is configuration, not a rewrite:

- `VE_CORS_ORIGINS` — comma-separated origins, `*` for dev (`app/main.py:45`).
- `VITE_API_BASE` — browser's backend base URL, overridable at runtime through the
  Settings page (localStorage).
- `getWsUrl()` derives `ws://`/`wss://` from the API base; the socket URL is never
  hardcoded.

## Architecture decision: one origin behind a reverse proxy

`visionedge.c-maron.space` serves the built frontend and proxies `/api/*` —
including the `/api/ws/detect` WebSocket — to the backend container.

Consequences:

- **CORS disappears.** Same origin, so no preflight and no origin list to maintain.
  `VE_CORS_ORIGINS` is set to the subdomain anyway as defence in depth for any
  direct backend exposure.
- **`wss://` works with no extra configuration**, because it is derived from an
  `https://` base.
- **One TLS certificate.**

Rejected: a split `api.visionedge.c-maron.space` (needs explicit CORS and a second
certificate; only pays off if the two are hosted on different machines), and
frontend-only publication (the demo would be dead for every visitor but the author).

## Components

**`deploy/Caddyfile`** (new) — Caddy over nginx for automatic certificate issuance
and renewal, which removes a manual TLS step from a portfolio deployment:

```
visionedge.c-maron.space {
    encode zstd gzip
    handle /api/* {
        reverse_proxy backend:8000
    }
    handle {
        root * /srv
        try_files {path} /index.html
        file_server
    }
}
```

Caddy proxies WebSocket upgrades without extra directives.

**`docker-compose.prod.yml`** (new) — caddy + backend + a frontend build stage that
emits static files into the volume Caddy serves. The existing `docker-compose.yml`
stays as the local-development file.

**`frontend/src/config.ts`** — when `VITE_API_BASE` is unset, default to
`window.location.origin` instead of `http://localhost:8000`. This makes a
same-origin deployment work from the same build artifact, with localhost remaining
the dev default via `.env`. The Settings override is unchanged.

**Rate limiting** — `/api/infer`, `/api/vlm/*` and the WebSocket run real inference
on every request, so an unauthenticated public endpoint is a cost and abuse vector.
Add a per-IP limit at the Caddy layer, plus a cap on concurrent WebSocket sessions
in the backend. Exact limits are set during implementation against measured
single-request cost.

**`.env.example`** — a documented production block: `VE_CORS_ORIGINS`,
`VITE_API_BASE`, and a note that `VE_ALLOW_FRAME_TRANSMISSION` must stay `false`
unless a remote VLM is deliberately configured.

**`README.md`** — a deployment section, plus the honesty note below.

## Honesty note (required)

The capability scanner and every benchmark report the hardware they actually run
on. Deployed to a VPS, the Device Capabilities page and all live numbers will
describe that VPS, not the RTX 2060 the README's framing is built around.

The README must state that the hosted demo runs on different hardware from the
development machine, and that benchmark figures shown there are that host's. Without
it, the deployed site and the README contradict each other, and the project's
stated honesty rule fails at exactly the place most visitors will look.

## Browser requirements

Camera capture (`getUserMedia`) requires a secure context, so TLS is mandatory —
satisfied by Caddy's automatic HTTPS. Over plain HTTP the Live Inference page can
only work with uploaded files.

## Testing

- `docker compose -f docker-compose.prod.yml config` validates.
- Against a locally running proxy: `/` serves the SPA, `/api/health` reaches the
  backend, and `/api/ws/detect` completes a WebSocket upgrade and returns a
  detection message for a sent frame.
- With `VITE_API_BASE` unset, `getApiBase()` returns `window.location.origin`; with
  it set, the explicit value wins; a Settings override beats both.
- Rate limiting returns 429 past the configured threshold rather than queueing.

## Deployment steps for the author (outside the repo)

1. `A`/`AAAA` record for `visionedge.c-maron.space` → the host's IP.
2. Ports 80 and 443 open (Caddy needs 80 for the ACME challenge).
3. `docker compose -f docker-compose.prod.yml up -d`.
