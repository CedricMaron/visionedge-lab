"""Execution planner: decide where each pipeline stage should run.

Given a description of the available backends (GPU VRAM, RAM, runtimes, network latency,
privacy mode, target FPS, VLM invocation frequency, battery mode) the planner recommends
a placement (pc / local-server / remote) for each pipeline stage together with a
human-readable reason.

There is no universally-optimal placement. This planner encodes *heuristics* and always
attaches the reasoning and the (caller-supplied) measured tradeoffs it used. Any latency
or throughput number in a plan comes from the passed-in ``metrics`` dict -- the planner
never invents benchmark figures.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Placement values mirror ExecutionLocation-ish strings used across the UI.
PC = "pc_local"
LOCAL_SERVER = "local_server"
REMOTE = "remote_server"

STAGES = ("detector", "visual_encoder", "future_predictor", "vlm")


@dataclass
class StagePlacement:
    """Where one stage runs and why."""

    stage: str
    location: str
    reason: str
    measured_tradeoff: Optional[str] = None


@dataclass
class ExecutionPlan:
    """A full plan: placement per stage plus the preset name and notes."""

    preset: str
    placements: List[StagePlacement] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Dict[str, Optional[str]]]:
        return {
            p.stage: {
                "location": p.location,
                "reason": p.reason,
                "measured_tradeoff": p.measured_tradeoff,
            }
            for p in self.placements
        }


def _tradeoff(metrics: Optional[Dict[str, str]], key: str) -> Optional[str]:
    """Fetch a measured tradeoff string for ``key`` from caller metrics, or None."""
    if not metrics:
        return None
    val = metrics.get(key)
    return str(val) if val is not None else None


class ExecutionPlanner:
    """Heuristic planner over a capabilities dict.

    Expected ``capabilities`` keys (all optional, with safe defaults):
        gpu_vram_mb, ram_mb, runtimes (list[str]), network_latency_ms, privacy_mode
        (bool), target_fps, vlm_invocation_hz, battery_mode (bool).

    ``metrics`` (optional) maps a stage/decision key to a measured tradeoff string that
    is surfaced verbatim in the plan (never fabricated).
    """

    def __init__(
        self,
        capabilities: Optional[Dict] = None,
        metrics: Optional[Dict[str, str]] = None,
    ) -> None:
        self.caps = dict(capabilities or {})
        self.metrics = dict(metrics or {})

    # --- helpers -------------------------------------------------------------
    def _cap(self, key: str, default):
        val = self.caps.get(key)
        return default if val is None else val

    def _has_local_gpu(self) -> bool:
        return float(self._cap("gpu_vram_mb", 0)) >= 2000.0

    def _privacy(self) -> bool:
        return bool(self._cap("privacy_mode", False))

    def _remote_allowed(self) -> bool:
        # Privacy mode forbids sending frames off-device.
        return not self._privacy()

    # --- generic recommendation ---------------------------------------------
    def recommend(self) -> ExecutionPlan:
        """Recommend placements from the raw capabilities (no preset)."""
        plan = ExecutionPlan(preset="custom")
        privacy = self._privacy()
        gpu = self._has_local_gpu()
        latency = float(self._cap("network_latency_ms", 100.0))
        battery = bool(self._cap("battery_mode", False))

        if privacy:
            plan.notes.append("privacy_mode on: no stage may run on a remote server")

        # Detector: latency-critical, keep local for real-time.
        det_loc = PC if gpu or battery else LOCAL_SERVER
        plan.placements.append(
            StagePlacement(
                "detector",
                det_loc,
                "real-time detector kept on-device to avoid per-frame network latency"
                + (" (GPU available)" if gpu else " (CPU)"),
                _tradeoff(self.metrics, "detector"),
            )
        )

        # Visual encoder: heavier; can offload to local server if no local GPU.
        enc_loc = PC if gpu else LOCAL_SERVER
        plan.placements.append(
            StagePlacement(
                "visual_encoder",
                enc_loc,
                "visual encoder placed where compute is cheapest without leaving the "
                "trust boundary" + (" (local GPU)" if gpu else " (offload to local server)"),
                _tradeoff(self.metrics, "visual_encoder"),
            )
        )

        # Future predictor: small; co-locate with encoder.
        plan.placements.append(
            StagePlacement(
                "future_predictor",
                enc_loc,
                "future predictor is small and reuses encoder embeddings, so it "
                "co-locates with the visual encoder",
                _tradeoff(self.metrics, "future_predictor"),
            )
        )

        # VLM: expensive & infrequent -> remote unless privacy forbids it.
        if self._remote_allowed() and not gpu:
            vlm_loc, vlm_reason = REMOTE, (
                "VLM is expensive and invoked infrequently; remote server avoids "
                "holding a large model in local memory"
            )
        elif gpu:
            vlm_loc, vlm_reason = PC, "local GPU can host the VLM without a round trip"
        else:
            vlm_loc, vlm_reason = LOCAL_SERVER, (
                "privacy_mode forbids remote; VLM runs on the local server"
                if privacy
                else "VLM runs on the local server"
            )
        if latency > 300.0 and vlm_loc == REMOTE:
            vlm_reason += f" (note: measured network latency {latency:.0f}ms is high)"
        plan.placements.append(
            StagePlacement("vlm", vlm_loc, vlm_reason, _tradeoff(self.metrics, "vlm"))
        )

        return plan


# --- Presets ----------------------------------------------------------------
# Each preset returns a full ExecutionPlan. Presets encode an intent, not a claim of
# optimality; the reason strings say what is being traded.

def _plan(preset: str, mapping: Dict[str, tuple], metrics: Optional[Dict[str, str]]) -> ExecutionPlan:
    plan = ExecutionPlan(preset=preset)
    for stage in STAGES:
        loc, reason = mapping[stage]
        plan.placements.append(
            StagePlacement(stage, loc, reason, _tradeoff(metrics, stage))
        )
    return plan


def preset_max_speed(metrics: Optional[Dict[str, str]] = None) -> ExecutionPlan:
    """Everything on the fastest available local compute; minimise round trips."""
    return _plan(
        "max_speed",
        {
            "detector": (PC, "keep the detector on local GPU/CPU to eliminate network latency per frame"),
            "visual_encoder": (PC, "run the encoder locally to avoid frame-transfer overhead"),
            "future_predictor": (PC, "tiny model co-located with the encoder for lowest latency"),
            "vlm": (PC, "accept higher local memory pressure to avoid remote round-trip latency"),
        },
        metrics,
    )


def preset_balanced(metrics: Optional[Dict[str, str]] = None) -> ExecutionPlan:
    """Latency-critical stages local, expensive infrequent VLM remote."""
    return _plan(
        "balanced",
        {
            "detector": (PC, "real-time stage stays local for responsiveness"),
            "visual_encoder": (LOCAL_SERVER, "offload steady encoder load to a nearby server"),
            "future_predictor": (LOCAL_SERVER, "co-locate with the encoder it consumes"),
            "vlm": (REMOTE, "expensive, infrequent VLM offloaded to keep local memory free"),
        },
        metrics,
    )


def preset_max_quality(metrics: Optional[Dict[str, str]] = None) -> ExecutionPlan:
    """Prefer the most capable (remote) models; trade latency for quality."""
    return _plan(
        "max_quality",
        {
            "detector": (LOCAL_SERVER, "run a larger detector on the server for accuracy"),
            "visual_encoder": (REMOTE, "use the largest available encoder remotely"),
            "future_predictor": (REMOTE, "co-locate with the remote encoder embeddings"),
            "vlm": (REMOTE, "use the strongest remote VLM; latency is acceptable for quality"),
        },
        metrics,
    )


def preset_battery_saver(metrics: Optional[Dict[str, str]] = None) -> ExecutionPlan:
    """Push heavy compute off the battery-powered device."""
    return _plan(
        "battery_saver",
        {
            "detector": (PC, "keep only the lightest stage on-device to preserve responsiveness"),
            "visual_encoder": (LOCAL_SERVER, "move sustained compute off the battery"),
            "future_predictor": (LOCAL_SERVER, "offload to save energy"),
            "vlm": (REMOTE, "never run the heaviest model on battery power"),
        },
        metrics,
    )


def preset_low_bandwidth(metrics: Optional[Dict[str, str]] = None) -> ExecutionPlan:
    """Minimise bytes over the network; keep frames on-device, send only text."""
    return _plan(
        "low_bandwidth",
        {
            "detector": (PC, "process frames locally so raw video never crosses the network"),
            "visual_encoder": (PC, "encode locally; only compact embeddings would ever be sent"),
            "future_predictor": (PC, "local prediction avoids streaming frames"),
            "vlm": (LOCAL_SERVER, "run the VLM on the LAN to avoid uploading frames over a metered link"),
        },
        metrics,
    )


PRESETS = {
    "max_speed": preset_max_speed,
    "balanced": preset_balanced,
    "max_quality": preset_max_quality,
    "battery_saver": preset_battery_saver,
    "low_bandwidth": preset_low_bandwidth,
}
