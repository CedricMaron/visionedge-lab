#!/usr/bin/env python3
"""Compute a file's SHA-256 + size, and optionally record it in models/registry.json.

With --update-registry, the matching registry entry (by file_name / local_path) has its
``checksum_sha256`` and ``file_size_bytes`` updated in place. Nothing else is touched.

Example:
    python scripts/checksum.py --input models/yolov8n.onnx
    python scripts/checksum.py --input models/yolov8s.onnx --update-registry
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _common as C

C.bootstrap_path()

REGISTRY_PATH = C.MODELS_DIR / "registry.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", required=True, type=Path, help="file to checksum")
    p.add_argument("--update-registry", action="store_true",
                   help="write sha256 + size into the matching registry.json entry")
    p.add_argument("--registry", type=Path, default=REGISTRY_PATH,
                   help=f"registry path (default {REGISTRY_PATH})")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.input.exists():
        C.die(f"file not found: {args.input}")

    sha = C.sha256_file(args.input)
    size = C.file_size(args.input)
    print(f"file:   {args.input}")
    print(f"size:   {size} bytes ({C.human_size(size)})")
    print(f"sha256: {sha}")

    if not args.update_registry:
        return 0

    if not args.registry.exists():
        C.die(f"registry not found: {args.registry}")
    try:
        data = json.loads(args.registry.read_text())
    except Exception as exc:  # noqa: BLE001
        C.die(f"registry is not valid JSON: {exc}")

    fname = args.input.name
    # Match by file_name (detection_models) or local_path suffix.
    matched = None
    for section in ("detection_models", "vlm_models"):
        for entry in data.get(section, []):
            if entry.get("file_name") == fname or str(entry.get("local_path", "")).endswith(fname):
                matched = entry
                break
        if matched:
            break

    if matched is None:
        C.die(f"no registry entry references {fname}; not updating. "
              "Add the entry manually or check the file name.")

    old_sha = matched.get("checksum_sha256")
    old_size = matched.get("file_size_bytes")
    matched["checksum_sha256"] = sha
    if "file_size_bytes" in matched or matched.get("format") != "vlm":
        matched["file_size_bytes"] = size
    args.registry.write_text(json.dumps(data, indent=2) + "\n")
    C.ok(f"updated registry entry '{matched.get('model_id')}'")
    C.info(f"sha256: {old_sha} -> {sha}")
    C.info(f"size:   {old_size} -> {size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
