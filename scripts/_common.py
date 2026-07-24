"""Shared helpers for the model-optimization CLI scripts.

Every script in ``scripts/`` imports from here for:
  * repo-root resolution and ``sys.path`` bootstrap (so ``import app.*`` works even
    when ``PYTHONPATH`` is not set),
  * checksums / file sizes,
  * sidecar-metadata writing,
  * ONNX model introspection (input/output names, shapes, opset),
  * consistent, non-tracebacky error reporting.

Design rule (see the project HONESTY RULES): scripts never *claim* a capability that
is not actually importable. Helpers here probe real imports and return booleans; the
calling script decides what to print and whether to exit non-zero.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# --- repo layout -----------------------------------------------------------------
# scripts/_common.py -> scripts/ -> repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
MODELS_DIR = REPO_ROOT / "models"
BENCHMARK_DIR = REPO_ROOT / "benchmark-data"
CALIBRATION_DIR = REPO_ROOT / "calibration"
FRONTEND_MODELS_DIR = REPO_ROOT / "frontend" / "public" / "models"


def bootstrap_path() -> None:
    """Ensure ``import app.*`` resolves, whether or not PYTHONPATH=backend was set."""
    b = str(BACKEND_DIR)
    if b not in sys.path:
        sys.path.insert(0, b)


# --- console ---------------------------------------------------------------------
def info(msg: str) -> None:
    print(f"[info] {msg}")


def warn(msg: str) -> None:
    print(f"[warn] {msg}")


def ok(msg: str) -> None:
    print(f"[ ok ] {msg}")


def die(msg: str, code: int = 1) -> None:
    """Print a clean, actionable error (no traceback) and exit non-zero."""
    print(f"[error] {msg}", file=sys.stderr)
    raise SystemExit(code)


# --- files -----------------------------------------------------------------------
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def file_size(path: Path) -> int:
    return path.stat().st_size


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    return f"{size:.1f} TB"


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def write_sidecar(model_path: Path, metadata: dict[str, Any]) -> Path:
    """Write ``<model>.json`` next to the produced model and return its path."""
    sidecar = model_path.with_suffix(model_path.suffix + ".json")
    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return sidecar


# --- onnx introspection ----------------------------------------------------------
def onnx_io_info(model_path: Path) -> dict[str, Any]:
    """Return input/output names + shapes + opset for an ONNX model, using onnx.

    Shapes may contain strings for symbolic (dynamic) dims. Never raises for a
    readable model; callers should treat a raised exception as "not a valid ONNX".
    """
    import onnx

    model = onnx.load(str(model_path))
    graph = model.graph

    def _shape(value_info) -> list[Any]:
        dims: list[Any] = []
        for d in value_info.type.tensor_type.shape.dim:
            if d.HasField("dim_value"):
                dims.append(int(d.dim_value))
            elif d.dim_param:
                dims.append(d.dim_param)
            else:
                dims.append("?")
        return dims

    inputs = [{"name": vi.name, "shape": _shape(vi)} for vi in graph.input]
    outputs = [{"name": vi.name, "shape": _shape(vi)} for vi in graph.output]
    opset = max((imp.version for imp in model.opset_import), default=None)
    ir_version = model.ir_version
    return {
        "inputs": inputs,
        "outputs": outputs,
        "opset": opset,
        "ir_version": ir_version,
    }


def ort_run_zero(model_path: Path, providers: list[str] | None = None) -> dict[str, Any]:
    """Load an ONNX model in ORT, feed a zero tensor of the input's static shape, and
    return {output_shapes, all_finite, provider}. Used for output verification.

    Any dynamic input dim is replaced with 1 (batch) or the model input_size fallback.
    """
    import numpy as np
    import onnxruntime as ort

    providers = providers or ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(model_path), providers=providers)
    feeds: dict[str, Any] = {}
    for inp in sess.get_inputs():
        shape = []
        for i, d in enumerate(inp.shape):
            if isinstance(d, int) and d > 0:
                shape.append(d)
            else:
                # symbolic/dynamic: batch->1, spatial->640 fallback
                shape.append(1 if i == 0 else 640)
        dtype = np.float16 if "float16" in inp.type else np.float32
        feeds[inp.name] = np.zeros(shape, dtype=dtype)
    outs = sess.run(None, feeds)
    return {
        "provider": sess.get_providers()[0],
        "output_shapes": [list(o.shape) for o in outs],
        "output_ranks": [o.ndim for o in outs],
        "all_finite": all(bool(np.isfinite(o).all()) for o in outs),
    }


def runtime_versions() -> dict[str, str | None]:
    """Best-effort version strings for the runtimes a report might care about."""
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for mod in ("numpy", "onnx", "onnxruntime", "onnxslim", "torch", "ultralytics"):
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            versions[mod] = None
    return versions
