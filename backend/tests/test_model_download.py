"""Fetching a model from a registry download_url.

The scripts/ tools have no test suite of their own, but this path is what a fresh
deployment depends on to get its weights, so it is covered here.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

download_models = pytest.importorskip("download_models")


def test_downloads_to_target_and_creates_parent_dirs(tmp_path, monkeypatch):
    payload = b"fake onnx bytes"

    class _Resp:
        headers = {"Content-Length": str(len(payload))}

        def read(self, n=-1):
            nonlocal payload_read
            if payload_read:
                return b""
            payload_read = True
            return payload

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    payload_read = False
    monkeypatch.setattr(download_models, "urlopen", lambda *a, **k: _Resp())

    target = tmp_path / "nested" / "model.onnx"
    download_models._download_url("https://example.invalid/model.onnx", target)

    assert target.exists()
    assert target.read_bytes() == payload


def test_partial_download_does_not_leave_a_corrupt_target(tmp_path, monkeypatch):
    def _boom(*a, **k):
        raise OSError("connection reset")

    monkeypatch.setattr(download_models, "urlopen", _boom)
    target = tmp_path / "model.onnx"

    with pytest.raises(SystemExit):
        download_models._download_url("https://example.invalid/model.onnx", target)

    # A half-written file that later passes as "installed" is worse than no file.
    assert not target.exists()


def test_checksum_helper_detects_a_mismatch(tmp_path):
    f = tmp_path / "m.onnx"
    f.write_bytes(b"content")
    good = hashlib.sha256(b"content").hexdigest()

    from app.models.registry import verify_checksum

    assert verify_checksum(f, good) is True
    assert verify_checksum(f, "0" * 64) is False
