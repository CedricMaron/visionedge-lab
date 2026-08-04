"""Profiler artifact capture and storage.

Framework profilers emit large files — an ONNX Runtime trace for a few hundred
iterations is megabytes of JSON. §20 of the brief requires these to be stored
separately from the summary record, so a run listing stays cheap to query and a
trace is fetched only when someone actually opens it.

Artifacts live under ``benchmarks/results/traces/`` keyed by run id. The
:class:`~app.schemas.run.BenchmarkRun` holds only a reference: kind, path and size.

Two safety properties:

* Artifact paths are derived from a validated run id, never from request input, and
  :func:`resolve_artifact_path` refuses anything that escapes the artifact root — a
  path assembled from user input would be a directory-traversal read primitive.
* Capture never fails a benchmark. A profiler that could not be started or flushed
  produces a warning and no artifact, because losing a diagnostic is much less bad
  than losing the measurement it was diagnosing.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

from app.core.config import REPO_ROOT
from app.core.logging import get_logger
from app.schemas.run import ProfilerArtifact

log = get_logger("benchmark.artifacts")

ARTIFACT_ROOT = REPO_ROOT / "benchmarks" / "results" / "traces"

#: Run ids are generated as `run_<12 hex>`; anything else is not one of ours.
_RUN_ID_PATTERN = re.compile(r"^run_[0-9a-f]{6,32}$")


class ArtifactError(ValueError):
    """Raised when an artifact path cannot be safely resolved."""


def artifact_dir(run_id: str, *, create: bool = False) -> Path:
    """Directory holding one run's artifacts.

    Validates the run id rather than sanitizing it: a value that is not a run id we
    generated has no business addressing the filesystem at all.
    """
    if not _RUN_ID_PATTERN.match(run_id):
        raise ArtifactError(f"not a valid run id: {run_id!r}")
    directory = ARTIFACT_ROOT / run_id
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return directory


def resolve_artifact_path(run_id: str, file_name: str) -> Path:
    """Resolve one artifact, refusing anything outside the run's directory."""
    directory = artifact_dir(run_id)
    # Reject separators outright before resolving, so the error names the real
    # problem rather than reporting a confusing "outside the artifact root".
    if "/" in file_name or "\\" in file_name or file_name in ("", ".", ".."):
        raise ArtifactError(f"illegal artifact file name: {file_name!r}")

    candidate = (directory / file_name).resolve()
    root = ARTIFACT_ROOT.resolve()
    if not candidate.is_relative_to(root):
        raise ArtifactError(f"artifact path escapes the artifact root: {file_name!r}")
    return candidate


def capture_ort_profile(adapter, run_id: str) -> ProfilerArtifact | None:
    """Flush an ONNX Runtime profile and file it under the run's directory.

    ORT writes its trace to the process working directory and returns the path from
    ``end_profiling()``. It is moved rather than copied so a long profiling session
    does not leave two multi-megabyte files behind.

    Returns None — with a logged warning — when profiling was not enabled or the
    flush failed. A missing diagnostic must never fail the benchmark it was
    diagnosing.
    """
    runtime = getattr(adapter, "runtime", None)
    handle = getattr(adapter, "_handle", None)
    if runtime is None or handle is None:
        return None
    end_profiling = getattr(runtime, "end_profiling", None)
    if end_profiling is None:
        return None

    try:
        source_path = end_profiling(handle)
    except Exception as exc:  # noqa: BLE001
        log.warning("profile_flush_failed", run_id=run_id, error=str(exc))
        return None

    if not source_path:
        return None

    source = Path(source_path)
    if not source.exists():
        log.warning("profile_file_missing", run_id=run_id, path=str(source))
        return None

    try:
        destination = artifact_dir(run_id, create=True) / "onnxruntime_profile.json"
        shutil.move(str(source), str(destination))
        size = destination.stat().st_size
    except Exception as exc:  # noqa: BLE001
        log.warning("profile_store_failed", run_id=run_id, error=str(exc))
        return None

    log.info("profile_captured", run_id=run_id, bytes=size)
    return ProfilerArtifact(
        kind="ort_profile",
        path=str(destination.relative_to(REPO_ROOT)),
        size_bytes=size,
        note=(
            "ONNX Runtime operator-level profile in Chrome trace format. Open at "
            "chrome://tracing or https://ui.perfetto.dev. Captured in profiler mode, "
            "whose timings are not comparable with standard-mode results."
        ),
    )


def list_artifacts(run_id: str) -> list[ProfilerArtifact]:
    """Artifacts on disk for a run. Empty when the run produced none."""
    try:
        directory = artifact_dir(run_id)
    except ArtifactError:
        return []
    if not directory.exists():
        return []
    return [
        ProfilerArtifact(
            kind="ort_profile" if "onnxruntime" in path.name else "trace",
            path=str(path.relative_to(REPO_ROOT)),
            size_bytes=path.stat().st_size,
        )
        for path in sorted(directory.iterdir())
        if path.is_file()
    ]


def delete_artifacts(run_id: str) -> int:
    """Remove a run's artifacts. Returns how many files were deleted."""
    try:
        directory = artifact_dir(run_id)
    except ArtifactError:
        return 0
    if not directory.exists():
        return 0
    count = sum(1 for p in directory.iterdir() if p.is_file())
    shutil.rmtree(directory)
    return count
