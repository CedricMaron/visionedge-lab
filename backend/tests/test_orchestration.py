"""Tests for the orchestration subsystem (pure logic)."""
from __future__ import annotations


from app.orchestration.execution_planner import (
    PRESETS,
    STAGES,
    ExecutionPlanner,
    preset_balanced,
)
from app.orchestration.invocation_policy import (
    InvocationConfig,
    Signals,
    VLMInvocationPolicy,
)
from app.orchestration.resource_manager import (
    PRIORITY_REALTIME_DETECTION,
    PRIORITY_TRAINING,
    PRIORITY_VLM,
    ModelResourceManager,
)


# --- invocation policy truth table ----------------------------------------
def test_invocation_policy_truth_table():
    policy = VLMInvocationPolicy(InvocationConfig(anomaly_threshold=0.7, cooldown_seconds=2.0))

    # User request always wins, even inside cooldown.
    ok, reason = policy.decide(Signals(user_asked=True, seconds_since_last_invocation=0.0))
    assert ok and "user" in reason.lower()

    # Cooldown suppresses automatic triggers.
    ok, reason = policy.decide(
        Signals(detector_event=True, seconds_since_last_invocation=0.5)
    )
    assert not ok and "cooldown" in reason.lower()

    # Anomaly above threshold triggers.
    ok, reason = policy.decide(Signals(anomaly_score=0.9))
    assert ok and "anomaly" in reason.lower()

    # Anomaly below threshold, nothing else -> no invoke.
    ok, reason = policy.decide(Signals(anomaly_score=0.3))
    assert not ok

    # Detector event triggers.
    ok, reason = policy.decide(Signals(detector_event=True))
    assert ok and "detector" in reason.lower()

    # Selected object triggers.
    ok, reason = policy.decide(Signals(selected_object_present=True))
    assert ok

    # Scene change triggers.
    ok, reason = policy.decide(Signals(scene_changed=True))
    assert ok and "scene" in reason.lower()

    # Timer heartbeat triggers.
    ok, reason = policy.decide(Signals(timer_elapsed=True))
    assert ok and "timer" in reason.lower()

    # No signals -> no invoke.
    ok, reason = policy.decide(Signals())
    assert not ok


def test_invocation_policy_selected_object_gate():
    cfg = InvocationConfig(require_selected_object_for_events=True)
    policy = VLMInvocationPolicy(cfg)
    # Detector event but no selected object -> gated off.
    ok, _ = policy.decide(Signals(detector_event=True, selected_object_present=False))
    assert not ok
    # With selected object present -> allowed.
    ok, _ = policy.decide(Signals(detector_event=True, selected_object_present=True))
    assert ok


# --- resource manager ------------------------------------------------------
def test_resource_manager_fits_without_eviction():
    mgr = ModelResourceManager(budget_mb=1000)
    d = mgr.request_load("detector", 200, priority=PRIORITY_REALTIME_DETECTION)
    assert d.admitted and not d.unloaded
    assert mgr.used_mb() == 200


def test_resource_manager_evicts_lru_low_priority():
    mgr = ModelResourceManager(budget_mb=1000)
    mgr.request_load("detector", 400, priority=PRIORITY_REALTIME_DETECTION)
    mgr.request_load("vlm", 400, priority=PRIORITY_VLM)
    # Use vlm so detector-equivalent LRU is clear; now load a big model needing space.
    d = mgr.request_load("encoder", 400, priority=PRIORITY_VLM)
    assert d.admitted
    # The real-time detector must never be evicted.
    assert "detector" not in d.unloaded
    assert "vlm" in d.unloaded  # lower-recency, evictable
    assert any("UNLOAD" in line for line in d.log)


def test_resource_manager_pauses_training_not_kills():
    mgr = ModelResourceManager(budget_mb=500)
    mgr.request_load("trainer", 400, priority=PRIORITY_TRAINING, pausable=True)
    d = mgr.request_load("detector", 300, priority=PRIORITY_REALTIME_DETECTION)
    assert d.admitted
    assert "trainer" in d.paused
    assert any("PAUSE" in line and "not silently killed" in line for line in d.log)


def test_resource_manager_rejects_too_big():
    mgr = ModelResourceManager(budget_mb=100)
    d = mgr.request_load("huge", 200)
    assert not d.admitted
    assert any("REJECT" in line for line in d.log)


def test_resource_manager_protects_realtime():
    mgr = ModelResourceManager(budget_mb=500)
    mgr.request_load("detector", 500, priority=PRIORITY_REALTIME_DETECTION)
    # Cannot evict the protected real-time detector to fit a new model.
    d = mgr.request_load("vlm", 100, priority=PRIORITY_VLM)
    assert not d.admitted


# --- execution planner -----------------------------------------------------
def test_all_presets_return_full_plans_with_reasons():
    for name, fn in PRESETS.items():
        plan = fn(metrics={"vlm": "measured 800ms remote round-trip"})
        assert plan.preset == name
        stages = {p.stage for p in plan.placements}
        assert stages == set(STAGES)
        for p in plan.placements:
            assert p.reason  # every placement explains itself
            assert p.location in ("pc_local", "local_server", "remote_server")
        # Measured tradeoff is surfaced verbatim, never fabricated.
        vlm = next(p for p in plan.placements if p.stage == "vlm")
        assert vlm.measured_tradeoff == "measured 800ms remote round-trip"


def test_planner_privacy_forbids_remote():
    planner = ExecutionPlanner(capabilities={"privacy_mode": True, "gpu_vram_mb": 0})
    plan = planner.recommend()
    for p in plan.placements:
        assert p.location != "remote_server"
    assert any("privacy" in n.lower() for n in plan.notes)


def test_balanced_preset_offloads_vlm_remote():
    plan = preset_balanced()
    vlm = next(p for p in plan.placements if p.stage == "vlm")
    assert vlm.location == "remote_server"
