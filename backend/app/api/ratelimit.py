"""Per-IP rate limiting for the endpoints that run real inference.

Enforced here rather than at the reverse proxy so the limit holds however the app
is deployed — Caddy v2 has no built-in rate limiter (it is a third-party plugin
needing a custom build), and a public endpoint doing YOLO inference on demand is a
cost and abuse vector without one.

A fixed window is deliberate: it is exact, allocation-free per request, and needs
no background sweeper. The cost is burstiness at a window boundary, which is
acceptable for protecting a demo.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable

from starlette.requests import Request
from starlette.responses import JSONResponse

# Paths whose cost is a model forward pass. Health, metrics and metadata reads are
# cheap and must stay unlimited so uptime checks never trip the limiter.
EXPENSIVE_PREFIXES = ("/api/infer", "/api/vlm/", "/api/detection/benchmark")


class FixedWindowLimiter:
    """Allow ``limit`` events per ``window_s`` per key. ``limit=0`` disables it."""

    def __init__(
        self,
        limit: int,
        window_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.limit = int(limit)
        self.window_s = float(window_s)
        self._clock = clock
        self._lock = threading.Lock()
        self._hits: dict[str, tuple[float, int]] = {}  # key -> (window_start, count)

    def allow(self, key: str) -> bool:
        if self.limit <= 0:
            return True
        now = self._clock()
        with self._lock:
            start, count = self._hits.get(key, (now, 0))
            if now - start >= self.window_s:
                start, count = now, 0
            if count >= self.limit:
                self._hits[key] = (start, count)
                return False
            self._hits[key] = (start, count + 1)
            return True

    def retry_after(self, key: str) -> float:
        """Seconds until this key's window rolls over."""
        if self.limit <= 0:
            return 0.0
        with self._lock:
            start, _ = self._hits.get(key, (self._clock(), 0))
        return max(0.0, self.window_s - (self._clock() - start))


def client_key(request: Request) -> str:
    """Identify the caller: the first X-Forwarded-For hop, else the peer address.

    Trust model: this header is only trustworthy because the reverse proxy in front
    replaces it. Caddy's ``trusted_proxies`` defaults to empty, so a client-supplied
    X-Forwarded-For is discarded and rewritten to the real peer — verified against
    the running stack, where rotating a spoofed header did not win extra quota.

    If another proxy is ever put in front (Cloudflare, a load balancer), configure
    Caddy's ``trusted_proxies`` for it. Otherwise every request arrives with that
    proxy's address and all users end up sharing one bucket.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def install_rate_limit(app, limit_per_min: int) -> None:
    """Attach the limiter to ``app`` for the expensive endpoints only."""
    limiter = FixedWindowLimiter(limit=limit_per_min, window_s=60.0)
    app.state.rate_limiter = limiter

    @app.middleware("http")
    async def _rate_limit(request: Request, call_next):
        if not request.url.path.startswith(EXPENSIVE_PREFIXES):
            return await call_next(request)
        key = client_key(request)
        if limiter.allow(key):
            return await call_next(request)
        retry = int(limiter.retry_after(key)) + 1
        return JSONResponse(
            status_code=429,
            headers={"Retry-After": str(retry)},
            content={
                "detail": (
                    f"Rate limit exceeded ({limiter.limit} inference requests per minute). "
                    f"Retry in {retry}s."
                )
            },
        )
