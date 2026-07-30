#!/usr/bin/env python3
"""List and install models declared in models/registry.json.

--list    prints every registry model with its install status and size.
--install MODEL_ID fetches ONLY that model:
    * detection ONNX models (yolov8n/s/m-onnx) are produced by exporting via ultralytics
      (delegates to scripts/export_onnx.py),
    * detection PyTorch weights (*-pt) are downloaded by ultralytics,
    * VLM models are transformers/opt-in and are NOT downloaded here (a pointer is shown).

Safety: anything that would require downloading > 5 GB requires an explicit confirmation
(-y). None of the current models approach that, but the guard is implemented. After a
detection model is installed its checksum is verified against the registry when present.

Examples:
    python scripts/download_models.py --list
    python scripts/download_models.py --install yolov8s-onnx
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from urllib.request import urlopen

import _common as C

C.bootstrap_path()

REGISTRY_PATH = C.MODELS_DIR / "registry.json"
FIVE_GB = 5 * 1024 * 1024 * 1024

# registry model_id -> export_onnx --model key
ONNX_EXPORT_KEY = {
    "yolov8n-onnx": "nano",
    "yolov8s-onnx": "small",
    "yolov8m-onnx": "medium",
}
PT_WEIGHTS = {
    "yolov8n-pt": "yolov8n.pt",
}


def _load_registry():
    """Load the registry with deployment status derived from disk.

    Without refresh_deployment_status the status shown is whatever the JSON happens
    to say, so an installed model can report "not_installed" — which is exactly the
    kind of stale claim the registry is supposed to prevent.
    """
    from app.models.registry import load_registry, refresh_deployment_status

    return refresh_deployment_status(load_registry(REGISTRY_PATH))


def _installed(local_path: str) -> bool:
    return (C.REPO_ROOT / local_path).exists()


def cmd_list() -> int:
    reg = _load_registry()
    print(f"{'model_id':<30}{'kind':<22}{'status':<15}{'size':<11}local_path")
    print("-" * 100)
    for m in reg.detection_models:
        status = "installed" if _installed(m.local_path) else "not_installed"
        p = C.REPO_ROOT / m.local_path
        size = C.human_size(C.file_size(p)) if p.exists() else (
            C.human_size(m.file_size_bytes) if m.file_size_bytes else "?"
        )
        print(f"{m.model_id:<30}{'detection':<22}{status:<15}{size:<11}{m.local_path}")
    for v in reg.vlm_models:
        status = "builtin" if v.model_source == "builtin" else "opt-in"
        print(f"{v.model_id:<30}{'vlm':<22}{status:<15}{'-':<11}{v.model_source}")
    for m in getattr(reg, "models", []):
        p = C.REPO_ROOT / m.local_path
        size = C.human_size(C.file_size(p)) if p.exists() else (
            C.human_size(m.file_size_bytes) if m.file_size_bytes else "?"
        )
        print(f"{m.model_id:<30}{m.task:<22}{m.deployment_status:<15}{size:<11}{m.local_path}")

    print("\nInstall a model with: --install <model_id>")
    print("VLM models are opt-in via transformers (see backend/requirements/vlm.txt).")
    return 0


def _free_disk_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def _confirm_large(est_bytes: int, assume_yes: bool) -> None:
    if est_bytes <= FIVE_GB:
        return
    C.warn(f"this install may download ~{C.human_size(est_bytes)} (> 5 GB).")
    if assume_yes:
        C.info("proceeding (-y given).")
        return
    resp = input("Proceed? [y/N] ").strip().lower()
    if resp not in ("y", "yes"):
        C.die("aborted by user.", code=2)


def _install_adapter_model(entry, assume_yes: bool) -> int:
    """Install an adapter-architecture model plus every companion file it needs.

    Companion files (tokenizers, preprocessing configs) are not optional: a model
    whose tokenizer is missing would load and then produce silently wrong output,
    which is worse than failing. So the install is all-or-nothing.
    """
    target = C.REPO_ROOT / entry.local_path
    target.parent.mkdir(parents=True, exist_ok=True)

    if not entry.download_url:
        C.die(f"'{entry.model_id}' declares no download_url; install it manually "
              f"({entry.install_hint or 'no hint recorded'})")

    est = entry.file_size_bytes or (100 * 1024 * 1024)
    free = _free_disk_bytes(C.MODELS_DIR)
    C.info(f"estimated download: ~{C.human_size(est)}; free disk: {C.human_size(free)}")
    if free < est * 2:
        C.warn("low free disk space relative to model size.")
    _confirm_large(est, assume_yes)

    if target.exists():
        C.info(f"weights already present at {target}")
    else:
        _download_url(entry.download_url, target)
    if entry.checksum_sha256:
        _verify(target, entry.checksum_sha256)

    for companion in entry.companion_files:
        dest = target.parent / companion.file_name
        if dest.exists():
            C.info(f"companion already present: {companion.file_name}")
        else:
            C.info(f"fetching companion {companion.file_name} ({companion.purpose})")
            _download_url(companion.download_url, dest)
        if companion.checksum_sha256:
            _verify(dest, companion.checksum_sha256)

    C.ok(f"installed {entry.model_id} -> {target} ({C.human_size(C.file_size(target))})")
    return 0


def cmd_install(model_id: str, assume_yes: bool) -> int:
    reg = _load_registry()
    adapter_entry = reg.adapter_model(model_id) if hasattr(reg, "adapter_model") else None
    if adapter_entry is not None and adapter_entry.download_url:
        return _install_adapter_model(adapter_entry, assume_yes)

    det = reg.detection(model_id)
    vlm = reg.vlm(model_id)

    if vlm is not None:
        C.die(f"'{model_id}' is a VLM ({vlm.model_source}). VLMs are opt-in via transformers; "
              "install backend/requirements/vlm.txt and load through the VLM manager. "
              "This downloader handles detection models only.")
    if det is None:
        available = ", ".join(m.model_id for m in reg.detection_models)
        C.die(f"unknown model_id '{model_id}'. Detection models: {available}")

    target = C.REPO_ROOT / det.local_path
    if target.exists():
        C.info(f"{model_id} already installed at {target}")
        if det.checksum_sha256:
            _verify(target, det.checksum_sha256)
        return 0

    # Rough download estimate for the disk/large-file guard.
    est = det.file_size_bytes or (55 * 1024 * 1024)  # medium ~50MB upper bound
    free = _free_disk_bytes(C.MODELS_DIR)
    C.info(f"estimated download/produce size: ~{C.human_size(est)}; free disk: {C.human_size(free)}")
    if free < est * 2:
        C.warn("low free disk space relative to model size.")
    _confirm_large(est, assume_yes)

    # A published URL is preferred over a local export: it needs no ultralytics or
    # torch on the machine doing the install, which matters on a deployment host.
    if det.download_url:
        _download_url(det.download_url, target)
    elif model_id in ONNX_EXPORT_KEY:
        C.info(f"producing {model_id} via ultralytics export ...")
        import export_onnx

        rc = export_onnx.main(["--model", ONNX_EXPORT_KEY[model_id], "--size",
                               str(det.input_size), "--output", str(target)])
        if rc != 0:
            C.die("export failed")
    elif model_id in PT_WEIGHTS:
        _download_pt(PT_WEIGHTS[model_id], target)
    else:
        C.die(f"no installer wired for '{model_id}'.")

    if not target.exists():
        C.die(f"install finished but file missing: {target}")
    C.ok(f"installed {model_id} -> {target} ({C.human_size(C.file_size(target))})")

    if det.checksum_sha256:
        _verify(target, det.checksum_sha256)
    else:
        C.info("no checksum in registry for this entry; run scripts/checksum.py "
               "--update-registry to record one.")
    return 0


def _download_url(url: str, target: Path) -> None:
    """Fetch ``url`` to ``target``. Writes to a temp file first.

    A half-written file that a later run mistakes for an installed model is worse
    than no file at all, so nothing lands at ``target`` unless the transfer
    completed.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    C.info(f"downloading {url}")
    try:
        with urlopen(url) as resp:  # noqa: S310 — registry-controlled https URL
            total = int(resp.headers.get("Content-Length") or 0)
            written = 0
            with open(tmp, "wb") as fh:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    written += len(chunk)
            if total and written != total:
                raise OSError(f"truncated download: got {written} of {total} bytes")
    except Exception as exc:  # noqa: BLE001 — reported cleanly, no traceback
        tmp.unlink(missing_ok=True)
        C.die(f"download failed: {exc}")
    tmp.replace(target)
    C.ok(f"downloaded {C.human_size(C.file_size(target))}")


def _download_pt(weights_name: str, target: Path) -> None:
    try:
        from ultralytics import YOLO
    except Exception:
        C.die("ultralytics not installed; pip install -r backend/requirements/base.txt")
    C.info(f"downloading {weights_name} via ultralytics ...")
    model = YOLO(weights_name)  # triggers download into cwd/cache
    src = Path(getattr(model, "ckpt_path", weights_name))
    if not src.exists():
        src = Path(weights_name)
    if src.exists() and src.resolve() != target.resolve():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(target))


def _verify(path: Path, expected_sha: str) -> None:
    from app.models.registry import verify_checksum

    if verify_checksum(path, expected_sha):
        C.ok(f"checksum verified: {expected_sha}")
    else:
        actual = C.sha256_file(path)
        C.warn(f"checksum MISMATCH:\n  expected {expected_sha}\n  actual   {actual}\n"
               "  (export is not always bit-reproducible across tool versions; run "
               "scripts/validate_onnx.py to confirm output agreement.)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true", help="list models + install status")
    g.add_argument("--install", metavar="MODEL_ID", help="install a single model by id")
    p.add_argument("-y", "--yes", action="store_true",
                   help="assume yes for the >5GB confirmation guard")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list:
        return cmd_list()
    return cmd_install(args.install, args.yes)


if __name__ == "__main__":
    raise SystemExit(main())
