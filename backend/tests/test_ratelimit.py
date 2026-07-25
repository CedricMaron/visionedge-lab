"""Per-IP rate limiting on the endpoints that run real inference."""
from __future__ import annotations

import pytest

from app.api.ratelimit import FixedWindowLimiter


def test_allows_up_to_the_limit_then_rejects():
    clock = [1000.0]
    limiter = FixedWindowLimiter(limit=3, window_s=60.0, clock=lambda: clock[0])

    assert [limiter.allow("1.2.3.4") for _ in range(3)] == [True, True, True]
    assert limiter.allow("1.2.3.4") is False


def test_limits_are_per_client():
    clock = [1000.0]
    limiter = FixedWindowLimiter(limit=1, window_s=60.0, clock=lambda: clock[0])

    assert limiter.allow("1.1.1.1") is True
    assert limiter.allow("1.1.1.1") is False
    assert limiter.allow("2.2.2.2") is True  # a different client is unaffected


def test_window_resets():
    clock = [1000.0]
    limiter = FixedWindowLimiter(limit=1, window_s=60.0, clock=lambda: clock[0])

    assert limiter.allow("1.2.3.4") is True
    assert limiter.allow("1.2.3.4") is False
    clock[0] += 61.0
    assert limiter.allow("1.2.3.4") is True


def test_retry_after_reports_seconds_until_the_window_rolls():
    clock = [1000.0]
    limiter = FixedWindowLimiter(limit=1, window_s=60.0, clock=lambda: clock[0])
    limiter.allow("1.2.3.4")
    clock[0] += 10.0

    assert limiter.allow("1.2.3.4") is False
    assert 49.0 <= limiter.retry_after("1.2.3.4") <= 50.0


def test_zero_limit_disables_limiting():
    limiter = FixedWindowLimiter(limit=0, window_s=60.0)
    assert all(limiter.allow("1.2.3.4") for _ in range(100))


def test_expensive_endpoint_returns_429_past_the_limit(monkeypatch):
    from fastapi.testclient import TestClient

    from app.core.config import REPO_ROOT, get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("VE_RATE_LIMIT_PER_MIN", "2")

    from app.main import create_app

    sample = REPO_ROOT / "benchmark-data" / "sample_bus.jpg"
    if not sample.exists():
        pytest.skip("sample image not installed")
    img = sample.read_bytes()

    try:
        with TestClient(create_app()) as c:
            codes = [
                c.post("/api/infer", files={"file": ("f.jpg", img, "image/jpeg")}).status_code
                for _ in range(3)
            ]
            assert codes[:2] == [200, 200]
            assert codes[2] == 429

            # Cheap endpoints are never rate limited — health checks must not trip it.
            assert c.get("/health").status_code == 200
    finally:
        get_settings.cache_clear()
