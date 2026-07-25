"""Application configuration via environment variables.

Every tunable is an environment variable with a safe default. Nothing about the
local network (IPs, ports of other machines) is hardcoded — see `.env.example`.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root = two levels up from this file's package (backend/app/core -> repo)
REPO_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = REPO_ROOT / "models"
BENCHMARK_DIR = REPO_ROOT / "benchmark-data"
CALIBRATION_DIR = REPO_ROOT / "calibration"


class Settings(BaseSettings):
    """Runtime settings. Prefix all env vars with ``VE_``."""

    model_config = SettingsConfigDict(env_prefix="VE_", env_file=".env", extra="ignore")

    # --- server ---
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "*"  # comma-separated; "*" for dev only

    # --- paths ---
    models_dir: Path = MODELS_DIR
    registry_path: Path = MODELS_DIR / "registry.json"
    db_path: Path = REPO_ROOT / "backend" / "visionedge.db"

    # --- detection defaults ---
    default_model_id: str = "yolov8n-onnx"
    default_confidence: float = Field(0.25, ge=0.0, le=1.0)
    default_iou: float = Field(0.45, ge=0.0, le=1.0)
    default_input_size: int = 640

    # --- transport / backpressure ---
    max_frame_queue: int = 2          # bounded queue: drop frames under load
    max_upload_bytes: int = 8 * 1024 * 1024

    # Public deployments run real inference per request. 0 disables the limit
    # (the local-dev default); set a value for anything internet-facing.
    rate_limit_per_min: int = 0
    ws_recv_timeout_s: float = 30.0

    # --- vlm ---
    default_vlm_id: str = "mock-vlm"          # mock is the always-on default
    vlm_remote_url: str = ""                  # opt-in; empty disables remote
    vlm_remote_api_key: str = ""              # from env only
    vlm_remote_timeout_s: float = 30.0
    vlm_remote_max_retries: int = 2
    allow_frame_transmission: bool = False    # opt-in privacy gate for remote

    # --- misc ---
    log_json: bool = True
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
