"""Tests for the capability scanner. These assert honesty, not specific hardware."""
from __future__ import annotations

from app.capabilities.scanner import scan_capabilities


def test_scan_returns_consistent_shape():
    caps = scan_capabilities()
    assert caps.cpu_cores_logical >= 1
    assert caps.ram_total_mb > 0
    assert "fp32" in caps.supported_precisions
    # onnxruntime is installed in this environment
    assert caps.runtimes.onnxruntime is True
    # providers list is populated when ORT is present
    assert isinstance(caps.runtimes.onnxruntime_providers, list)


def test_cuda_claim_matches_providers():
    caps = scan_capabilities()
    # never claim onnxruntime CUDA unless the provider is actually listed
    if caps.runtimes.onnxruntime_cuda:
        assert "CUDAExecutionProvider" in caps.runtimes.onnxruntime_providers


def test_fp16_only_claimed_with_accelerator():
    caps = scan_capabilities()
    if "fp16" in caps.supported_precisions:
        assert caps.runtimes.pytorch_cuda or caps.runtimes.onnxruntime_cuda or caps.nvidia_gpu_present
