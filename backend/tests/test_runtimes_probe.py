"""Capability probes: nothing may be claimed that was not verified."""
from __future__ import annotations

import pytest

from app.runtimes.base import RuntimeCapability, SessionHandle
from app.runtimes.registry import (
    _DECLARED,
    available_runtime_ids,
    capability_matrix,
    get_adapter,
    probe_all,
)
from app.schemas.enums import DeviceKind, Precision


class TestCapabilityContract:
    def test_unavailable_without_reason_is_impossible(self):
        with pytest.raises(ValueError, match="without a reason"):
            RuntimeCapability(runtime_id="x", available=False)

    def test_every_probe_result_explains_itself(self):
        for cap in probe_all():
            if not cap.available:
                assert cap.unavailable_reason, f"{cap.runtime_id} gave no reason"

    def test_probe_never_raises(self):
        # A broken optional dependency must not take the application down.
        assert len(probe_all()) == len(_DECLARED) + 1  # declared + onnxruntime

    def test_available_ids_are_a_subset_of_probed(self):
        probed = {c.runtime_id for c in probe_all()}
        assert set(available_runtime_ids()).issubset(probed)


class TestOnnxRuntimeProbe:
    def test_onnxruntime_is_available_here(self):
        # This repository depends on ORT, so its absence is a real failure, not a skip.
        cap = next(c for c in probe_all() if c.runtime_id == "onnxruntime")
        assert cap.available and cap.version
        assert DeviceKind.CPU in cap.devices

    def test_precision_support_is_per_device(self):
        cap = next(c for c in probe_all() if c.runtime_id == "onnxruntime")
        # fp16 on CPU is slower than fp32 under ORT (cast nodes around fp32 kernels),
        # so offering it would push users toward a strictly worse configuration.
        assert Precision.FP16 not in cap.precisions_for(DeviceKind.CPU)
        assert Precision.FP32 in cap.precisions_for(DeviceKind.CPU)

    def test_declares_synchronization_semantics(self):
        cap = next(c for c in probe_all() if c.runtime_id == "onnxruntime")
        assert cap.supports_device_synchronization
        assert any("host-resident" in n for n in cap.notes)


class TestDeclaredRuntimes:
    def test_directml_is_not_claimed_just_because_ort_is_installed(self):
        # The regression this guards: probing the `onnxruntime` module made DirectML
        # report "installed" on Linux, where DmlExecutionProvider does not exist.
        cap = next(c for c in probe_all() if c.runtime_id == "onnxruntime-directml")
        assert not cap.available
        assert "DmlExecutionProvider" in cap.unavailable_reason

    def test_missing_dependency_and_missing_adapter_are_distinguishable(self):
        by_id = {c.runtime_id: c for c in probe_all()}
        assert "not installed" in by_id["tensorrt"].unavailable_reason
        assert "no execution adapter" in by_id["pytorch"].unavailable_reason

    def test_declared_runtime_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="no execution adapter"):
            get_adapter("vllm")

    def test_unknown_runtime_raises_key_error(self):
        with pytest.raises(KeyError):
            get_adapter("definitely-not-a-runtime")


class TestCapabilityMatrix:
    def test_every_unsupported_cell_has_a_reason(self):
        for row in capability_matrix():
            if not row["supported"]:
                assert row["reason"], f"{row} was refused without explanation"

    def test_supported_cells_have_no_reason(self):
        for row in capability_matrix():
            if row["supported"]:
                assert row["reason"] is None

    def test_fp16_on_cpu_is_refused_with_an_explanation(self):
        row = next(
            r for r in capability_matrix()
            if r["runtime_id"] == "onnxruntime" and r["device"] == "cpu" and r["precision"] == "fp16"
        )
        assert not row["supported"] and "fp16" in row["reason"]


class TestSessionHonesty:
    def test_mismatch_is_detected(self):
        handle = SessionHandle(
            session=object(), runtime_id="onnxruntime", runtime_version="1.20.1",
            execution_provider="CPUExecutionProvider",
            effective_device=DeviceKind.CPU, effective_precision=Precision.FP32,
            requested_device=DeviceKind.CUDA,
        )
        assert not handle.honored
        assert "misattribute" in handle.mismatch_message()

    def test_match_is_honored(self):
        handle = SessionHandle(
            session=object(), runtime_id="onnxruntime", runtime_version="1.20.1",
            execution_provider="CPUExecutionProvider",
            effective_device=DeviceKind.CPU, effective_precision=Precision.FP32,
            requested_device=DeviceKind.CPU,
        )
        assert handle.honored
