"""Model registry: validated loader for ``models/registry.json``.

The registry is the single source of truth for which models exist, their formats,
supported runtimes/devices, install status and checksums. UI never hardcodes models —
it reads them from here via the API.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, Field

from app.core.config import REPO_ROOT, get_settings


class DetectionModelEntry(BaseModel):
    model_id: str
    display_name: str
    family: str
    size: str
    architecture: str
    version: str
    format: str
    precision: str
    input_size: int
    supported_runtimes: list[str]
    supported_devices: list[str]
    labels: str
    file_name: str
    file_size_bytes: int | None = None
    checksum_sha256: str | None = None
    expected_memory_mb: int | None = None
    download_url: str | None = None
    local_path: str
    deployment_status: str
    reference_backend: bool = False
    speed_category: str = "unknown"
    quality_category: str = "unknown"
    license: str = "unknown"
    notes: str = ""


class VLMModelEntry(BaseModel):
    model_id: str
    display_name: str
    family: str
    version: str
    task_capabilities: list[str]
    image_support: bool
    multi_image_support: bool
    video_support: bool
    structured_output_support: bool
    context_length: int
    estimated_vram_mb: int
    supported_quantization: list[str]
    supported_runtimes: list[str]
    supported_devices: list[str]
    model_source: str
    license: str
    checksum_sha256: str | None = None
    deployment_status: str
    nature: str = ""


class ModelRegistry(BaseModel):
    schema_version: int = 1
    detection_models: list[DetectionModelEntry] = Field(default_factory=list)
    vlm_models: list[VLMModelEntry] = Field(default_factory=list)

    def detection(self, model_id: str) -> DetectionModelEntry | None:
        return next((m for m in self.detection_models if m.model_id == model_id), None)

    def vlm(self, model_id: str) -> VLMModelEntry | None:
        return next((m for m in self.vlm_models if m.model_id == model_id), None)

    def abs_path(self, entry: DetectionModelEntry) -> Path:
        return REPO_ROOT / entry.local_path


def load_registry(path: Path | None = None) -> ModelRegistry:
    path = path or get_settings().registry_path
    data = json.loads(Path(path).read_text())
    return ModelRegistry.model_validate(data)


def verify_checksum(file_path: Path, expected_sha256: str) -> bool:
    """Return True if the file's SHA-256 matches. Used for model integrity checks."""
    if not file_path.exists():
        return False
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest() == expected_sha256


def refresh_deployment_status(reg: ModelRegistry) -> ModelRegistry:
    """Update deployment_status based on whether the file actually exists on disk."""
    for m in reg.detection_models:
        exists = (REPO_ROOT / m.local_path).exists()
        if m.deployment_status == "installed" and not exists:
            m.deployment_status = "missing"
        elif m.deployment_status == "not_installed" and exists:
            m.deployment_status = "installed"
    return reg
