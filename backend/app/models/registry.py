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


class CompanionFile(BaseModel):
    """A file a model cannot run without, beyond its weights.

    Tokenizers and preprocessing configs are not optional extras: without the right
    tokenizer the embeddings are wrong, and without the classifier config the
    normalization constants and class labels would have to be guessed. They are
    declared here so an install is complete or fails, never half-done.
    """

    file_name: str
    download_url: str
    checksum_sha256: str | None = None
    purpose: str = ""


class AdapterModelEntry(BaseModel):
    """A model served through the task-agnostic adapter architecture.

    Supersedes ``DetectionModelEntry`` for anything added after the InferenceLab
    migration. The older lists are retained so existing consumers keep working.
    """

    model_id: str
    display_name: str
    family: str
    task: str
    modality: str
    adapter: str = Field(description="Adapter kind, e.g. 'yolov8', 'mobilenet', 'minilm'.")

    source_repository: str | None = None
    paper_url: str | None = None
    model_license: str
    weights_license: str
    commercial_use_permitted: bool | None = None
    auto_download_permitted: bool = False

    revision: str | None = None
    parameters_millions: float | None = None
    file_name: str
    file_size_bytes: int | None = None
    checksum_sha256: str | None = None
    download_url: str | None = None
    local_path: str
    companion_files: list[CompanionFile] = Field(default_factory=list)

    input_size: int | None = None
    supported_runtimes: list[str] = Field(default_factory=list)
    supported_devices: list[str] = Field(default_factory=list)
    supported_precisions: list[str] = Field(default_factory=list)

    deployment_status: str = "not_installed"
    not_installed_reason: str | None = None
    install_hint: str | None = None
    notes: str = ""
    is_test_adapter: bool = False


class ModelRegistry(BaseModel):
    schema_version: int = 1
    detection_models: list[DetectionModelEntry] = Field(default_factory=list)
    vlm_models: list[VLMModelEntry] = Field(default_factory=list)
    #: Task-agnostic entries for the adapter architecture.
    models: list[AdapterModelEntry] = Field(default_factory=list)

    def detection(self, model_id: str) -> DetectionModelEntry | None:
        return next((m for m in self.detection_models if m.model_id == model_id), None)

    def vlm(self, model_id: str) -> VLMModelEntry | None:
        return next((m for m in self.vlm_models if m.model_id == model_id), None)

    def adapter_model(self, model_id: str) -> AdapterModelEntry | None:
        return next((m for m in self.models if m.model_id == model_id), None)

    def production_models(self) -> list[AdapterModelEntry]:
        """Adapter models fit to show in a production listing.

        Test adapters are excluded here rather than at the presentation layer, so a
        fabricating adapter cannot reach a results page through a route that forgot
        to filter.
        """
        return [m for m in self.models if not m.is_test_adapter]

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
    """Derive deployment_status from what is actually on disk.

    Status is never a hand-maintained claim: a registry saying "installed" for a
    file that is absent would have the UI offer a model that cannot load.
    """
    for m in reg.detection_models:
        exists = (REPO_ROOT / m.local_path).exists()
        if m.deployment_status == "installed" and not exists:
            m.deployment_status = "missing"
        elif m.deployment_status == "not_installed" and exists:
            m.deployment_status = "installed"

    for entry in reg.models:
        weights = (REPO_ROOT / entry.local_path).exists()
        # A model with a missing companion file is not installed: it would load and
        # then produce wrong output, which is worse than failing.
        companions_present = all(
            (REPO_ROOT / entry.local_path).parent.joinpath(c.file_name).exists()
            for c in entry.companion_files
        )
        if weights and companions_present:
            entry.deployment_status = "installed"
            entry.not_installed_reason = None
        elif weights and not companions_present:
            missing = [
                c.file_name
                for c in entry.companion_files
                if not (REPO_ROOT / entry.local_path).parent.joinpath(c.file_name).exists()
            ]
            entry.deployment_status = "incomplete"
            entry.not_installed_reason = (
                f"weights are present but required companion file(s) are missing: "
                f"{', '.join(missing)}"
            )
        else:
            entry.deployment_status = "not_installed"
            if not entry.not_installed_reason:
                entry.not_installed_reason = f"weights not found at {entry.local_path}"
    return reg
