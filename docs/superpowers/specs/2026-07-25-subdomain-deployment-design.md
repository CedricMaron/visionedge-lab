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

## Gaps found when checking the container build (added 2026-07-25)

The first draft was written from the configuration surface and assumed the
existing `docker-compose.yml` worked. Verifying it turned up three blockers that
have nothing to do with the subdomain but stop any deployment:

**`frontend/Dockerfile` does not exist.** `docker-compose.yml` already references
it, so `docker compose build` fails today. `docker compose config` validates YAML
only and never checks that referenced Dockerfiles exist, so it passed while the
build was broken. A multi-stage node-build → static-serve Dockerfile is required.

**A fresh clone has no model.** `models/*.onnx` is gitignored; only
`registry.json` and a sidecar are tracked, and the `yolov8n-onnx` entry has
`"download_url": null`. The compose file mounts `./models` read-only from the
host, so a new server starts with no detector.

*Decision:* publish `yolov8n.onnx` (12.85 MB) as a GitHub release asset, set its
`download_url` in the registry, and let `scripts/download_models.py` fetch and
checksum-verify it at deploy time. That script already downloads and verifies
against `checksum_sha256`, which the registry already carries for this entry —
the only missing piece is the URL. Keeps the image small and needs no
ultralytics/torch on the server.

*Licensing:* the weights are AGPL-3.0 (Ultralytics) while this repo is MIT.
Publishing them as a release asset is redistribution, and AGPL §13 covers network
use, so the README must state that the detection weights are AGPL-3.0, that the
MIT license covers this project's own code only, and point to the source. The
repo being public already satisfies the substance of the source-offer
requirement.

**The database is not persisted.** `db_path` resolves inside the container and no
volume mounts it, so every restart discards benchmark history and sessions —
exactly the data the comparison view accumulates. It would read as permanently
empty in production. The prod compose file must mount it.

## Components

**`deploy/Caddyfile`** (new) — Caddy over nginx for automatic certificate issuance
and renewal, which removes a manual TLS step from a portfolio deployment:

```
{$SITE_ADDRESS:visionedge.c-maron.space} {
    encode zstd gzip
    handle /health {          # sits outside /api but is part of the API surface
        reverse_proxy backend:8000
    }
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

`/health` needs its own route: the frontend calls it and it is not under `/api`, so
without this it would 404 against the static file server. Validated against the real
`caddy:2.8-alpine` image (`caddy validate`), which also confirmed automatic HTTPS
and the HTTP→HTTPS redirect.

Caddy proxies WebSocket upgrades without extra directives.

**`frontend/Dockerfile`** (new) — multi-stage: `npm ci && npm run build`, then the
built `dist/` copied into a static-serving stage. Referenced by the existing
compose file, which currently cannot build without it.

**`docker-compose.prod.yml`** (new) — caddy + backend + a frontend build stage that
emits static files into the volume Caddy serves. The existing `docker-compose.yml`
stays as the local-development file. Mounts a named volume for the SQLite database
so benchmark history and sessions survive restarts, and mounts `./models` so the
fetched weights are visible to the backend.

**`frontend/src/config.ts`** — when `VITE_API_BASE` is unset, default to
`window.location.origin` instead of `http://localhost:8000`. This makes a
same-origin deployment work from the same build artifact, with localhost remaining
the dev default via `.env`. The Settings override is unchanged.

**Rate limiting** — `/api/infer`, `/api/vlm/*` and `/api/detection/benchmark` run a
model forward pass per request, so an unauthenticated public endpoint is a cost and
abuse vector.

*Revised during implementation:* the limit lives in the backend
(`app/api/ratelimit.py`), not at the proxy. Caddy v2 has no built-in `rate_limit`
directive — it is a third-party plugin requiring a custom `xcaddy` build, so the
originally specified config would have failed to start. Enforcing it in the app also
means it holds however the project is deployed, and it is unit-testable. Default
60/min per IP, set in `docker-compose.prod.yml`; `VE_RATE_LIMIT_PER_MIN=0` disables
it for local development. Cheap endpoints (`/health`, metadata reads) are exempt so
uptime checks cannot trip it.

*Trust model:* the limiter keys on `X-Forwarded-For`, which is safe only because
Caddy's `trusted_proxies` is empty and it therefore discards a client-supplied
header and writes the real peer address — verified empirically against the running
stack. Adding another proxy in front (Cloudflare, a load balancer) without
configuring `trusted_proxies` would put every user in one shared bucket.

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
3. `python scripts/download_models.py --install yolov8n-onnx` to fetch and
   checksum-verify the weights from the release asset.
4. `docker compose -f docker-compose.prod.yml up -d`.
