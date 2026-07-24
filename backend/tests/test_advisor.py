"""Tests for the Optimization Advisor API."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


def test_recommend_returns_plan_with_reasons(client):
    d = client.get("/api/advisor/recommend?target_fps=30").json()
    assert "plan" in d and d["plan"]
    # every stage placement must carry a reason (never an unexplained recommendation)
    for placement in d["plan"].values():
        assert placement["location"]
        assert placement["reason"]
    assert d["disclaimer"]


def test_privacy_mode_keeps_vlm_local(client):
    d = client.get("/api/advisor/recommend?privacy_mode=true").json()
    # with privacy on, no stage should be placed on a remote server
    locations = {p["location"] for p in d["plan"].values()}
    assert "remote_server" not in locations


def test_presets_listed_and_fetchable(client):
    presets = client.get("/api/advisor/presets").json()["presets"]
    assert "balanced" in presets and "max_speed" in presets
    for name in presets:
        plan = client.get(f"/api/advisor/preset/{name}").json()
        assert plan["preset"] == name
        assert plan["plan"]


def test_unknown_preset_reports_available(client):
    d = client.get("/api/advisor/preset/nope").json()
    assert "error" in d and "balanced" in d["available"]
