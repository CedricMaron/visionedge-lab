"""Application version, in one place.

Imported by both the FastAPI app title and /health, so a deploy asserting on the
health payload is asserting on the same value the app was built with.
"""
from __future__ import annotations

APP_VERSION = "0.2.0"
