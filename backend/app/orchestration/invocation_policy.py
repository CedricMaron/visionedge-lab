"""Policy deciding when to invoke an expensive VLM.

A vision-language model call is costly (latency, memory, possibly network/privacy). This
pure-logic policy fuses cheap signals into a single ``(should_invoke, reason)`` decision
so the system spends VLM budget only when it is likely to matter. It has no side effects
and is fully unit-tested via a truth table.

Signals (all optional; unknown ones default to "no signal"):

* ``user_asked`` -- the user explicitly requested a description now (highest priority).
* ``detector_event`` -- the detector fired a relevant event.
* ``selected_object_present`` -- a user-selected object of interest is in frame.
* ``scene_changed`` -- scene-change detector flagged a change.
* ``anomaly_score`` -- 0..1 anomaly score from the temporal model.
* ``timer_elapsed`` -- the minimum-interval heartbeat elapsed.

A cooldown prevents back-to-back invocations except when the user explicitly asks.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InvocationConfig:
    """Thresholds and gates for :class:`VLMInvocationPolicy`."""

    anomaly_threshold: float = 0.7
    invoke_on_detector_event: bool = True
    invoke_on_scene_change: bool = True
    invoke_on_selected_object: bool = True
    invoke_on_timer: bool = True
    require_selected_object_for_events: bool = False
    cooldown_seconds: float = 2.0


@dataclass
class Signals:
    """Snapshot of the signals available at decision time."""

    user_asked: bool = False
    detector_event: bool = False
    selected_object_present: bool = False
    scene_changed: bool = False
    anomaly_score: float | None = None
    timer_elapsed: bool = False
    seconds_since_last_invocation: float | None = None


class VLMInvocationPolicy:
    """Decide whether to invoke the VLM given current signals."""

    def __init__(self, config: InvocationConfig | None = None) -> None:
        self.config = config or InvocationConfig()

    def _in_cooldown(self, signals: Signals) -> bool:
        s = signals.seconds_since_last_invocation
        return s is not None and s < self.config.cooldown_seconds

    def decide(self, signals: Signals) -> tuple[bool, str]:
        """Return ``(should_invoke, reason)``.

        Priority order: explicit user request > anomaly > detector event > selected
        object > scene change > timer heartbeat. A cooldown suppresses everything except
        an explicit user request.
        """
        cfg = self.config

        # 1) Explicit user request always wins, ignoring cooldown.
        if signals.user_asked:
            return True, "user explicitly requested a description"

        # 2) Cooldown gate for all automatic triggers.
        if self._in_cooldown(signals):
            return False, (
                f"in cooldown ({signals.seconds_since_last_invocation:.2f}s < "
                f"{cfg.cooldown_seconds:.2f}s since last invocation)"
            )

        # Optional gate: automatic events only fire when a selected object is present.
        if cfg.require_selected_object_for_events and not signals.selected_object_present:
            # Timer heartbeat is still allowed below; other event triggers suppressed.
            pass

        gate_ok = (not cfg.require_selected_object_for_events) or signals.selected_object_present

        # 3) Anomaly.
        if (
            signals.anomaly_score is not None
            and signals.anomaly_score >= cfg.anomaly_threshold
            and gate_ok
        ):
            return True, (
                f"anomaly score {signals.anomaly_score:.2f} >= threshold "
                f"{cfg.anomaly_threshold:.2f}"
            )

        # 4) Detector event.
        if cfg.invoke_on_detector_event and signals.detector_event and gate_ok:
            return True, "detector reported a relevant event"

        # 5) Selected object present.
        if cfg.invoke_on_selected_object and signals.selected_object_present:
            return True, "user-selected object is present in frame"

        # 6) Scene change.
        if cfg.invoke_on_scene_change and signals.scene_changed and gate_ok:
            return True, "scene changed significantly"

        # 7) Timer heartbeat (lowest priority; not subject to the selected-object gate).
        if cfg.invoke_on_timer and signals.timer_elapsed:
            return True, "periodic timer elapsed"

        return False, "no trigger active"
