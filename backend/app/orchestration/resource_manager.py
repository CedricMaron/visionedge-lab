"""Model resource manager: fit models into a memory budget.

Tracks which models are loaded and their estimated memory footprint, and decides which
ones to unload to make room for a new model. The policy combines:

* a *priority* per model (higher = keep). Real-time detection is highest and is never
  auto-evicted; a training model is *pausable* -- it may be selected for eviction but the
  manager emits a "pause & checkpoint" decision rather than silently killing it.
* *LRU* among equal priorities (least-recently-used goes first).

The manager is pure logic (no real allocation); it emits human-readable decision strings
so callers can log/audit why something was unloaded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# Conventional priority levels (higher = more protected).
PRIORITY_TRAINING = 10
PRIORITY_BACKGROUND = 20
PRIORITY_VLM = 40
PRIORITY_VISUAL_ENCODER = 50
PRIORITY_REALTIME_DETECTION = 100


@dataclass
class LoadedModel:
    """Bookkeeping for one loaded model."""

    model_id: str
    estimated_mb: float
    priority: int = PRIORITY_BACKGROUND
    pausable: bool = False
    last_used: float = 0.0  # monotonic-ish tick; larger = more recent


@dataclass
class ResourceDecision:
    """Outcome of an :meth:`ModelResourceManager.request_load`."""

    admitted: bool
    unloaded: List[str] = field(default_factory=list)
    paused: List[str] = field(default_factory=list)
    log: List[str] = field(default_factory=list)


class ModelResourceManager:
    """Track loaded models against a memory budget and plan evictions."""

    def __init__(self, budget_mb: float) -> None:
        if budget_mb <= 0:
            raise ValueError("budget_mb must be positive")
        self.budget_mb = float(budget_mb)
        self._models: Dict[str, LoadedModel] = {}
        self._tick = 0

    # --- introspection -------------------------------------------------------
    def used_mb(self) -> float:
        return sum(m.estimated_mb for m in self._models.values())

    def free_mb(self) -> float:
        return self.budget_mb - self.used_mb()

    def loaded(self) -> List[str]:
        return list(self._models.keys())

    def touch(self, model_id: str) -> None:
        """Mark a model as just used (updates LRU recency)."""
        if model_id in self._models:
            self._tick += 1
            self._models[model_id].last_used = self._tick

    # --- core policy ---------------------------------------------------------
    def _eviction_order(self, protect: Optional[str] = None) -> List[LoadedModel]:
        """Candidates ordered best-to-evict-first: low priority then least recent."""
        cands = [m for m in self._models.values() if m.model_id != protect]
        # Never auto-evict real-time detection (highest priority sentinel).
        cands = [m for m in cands if m.priority < PRIORITY_REALTIME_DETECTION]
        return sorted(cands, key=lambda m: (m.priority, m.last_used))

    def request_load(
        self,
        model_id: str,
        estimated_mb: float,
        priority: int = PRIORITY_BACKGROUND,
        pausable: bool = False,
    ) -> ResourceDecision:
        """Decide whether ``model_id`` can be loaded, evicting/pausing as needed.

        Returns a :class:`ResourceDecision`. When admitted, the manager's internal
        state is updated (evicted models removed, new model registered). When not
        admitted (cannot free enough even after evicting everything evictable), state is
        left unchanged.
        """
        decision = ResourceDecision(admitted=False)

        if estimated_mb > self.budget_mb:
            decision.log.append(
                f"REJECT {model_id}: needs {estimated_mb:.0f}MB > total budget "
                f"{self.budget_mb:.0f}MB"
            )
            return decision

        if model_id in self._models:
            self.touch(model_id)
            decision.admitted = True
            decision.log.append(f"ADMIT {model_id}: already loaded")
            return decision

        needed = estimated_mb - self.free_mb()
        if needed <= 0:
            self._register(model_id, estimated_mb, priority, pausable)
            decision.admitted = True
            decision.log.append(
                f"ADMIT {model_id}: {estimated_mb:.0f}MB fits in {self.free_mb() + estimated_mb:.0f}MB free"
            )
            return decision

        # Need to free space.
        freed = 0.0
        to_remove: List[LoadedModel] = []
        for cand in self._eviction_order(protect=model_id):
            if freed >= needed:
                break
            to_remove.append(cand)
            freed += cand.estimated_mb
            if cand.pausable:
                decision.paused.append(cand.model_id)
                decision.log.append(
                    f"PAUSE {cand.model_id} (priority {cand.priority}, {cand.estimated_mb:.0f}MB): "
                    "checkpoint and pause -- not silently killed"
                )
            else:
                decision.log.append(
                    f"UNLOAD {cand.model_id} (priority {cand.priority}, LRU tick "
                    f"{cand.last_used}, {cand.estimated_mb:.0f}MB)"
                )
            decision.unloaded.append(cand.model_id)

        if freed < needed:
            decision.log.append(
                f"REJECT {model_id}: cannot free enough (freed {freed:.0f}MB, "
                f"need {needed:.0f}MB); highest-priority models are protected"
            )
            return decision

        # Commit evictions and load.
        for m in to_remove:
            del self._models[m.model_id]
        self._register(model_id, estimated_mb, priority, pausable)
        decision.admitted = True
        decision.log.append(
            f"ADMIT {model_id}: freed {freed:.0f}MB by evicting "
            f"{len(to_remove)} model(s)"
        )
        return decision

    def unload(self, model_id: str) -> Optional[str]:
        """Explicitly unload a model. Returns a log string or ``None`` if absent."""
        if model_id in self._models:
            mb = self._models[model_id].estimated_mb
            del self._models[model_id]
            return f"UNLOAD {model_id}: released {mb:.0f}MB (explicit)"
        return None

    def _register(self, model_id: str, mb: float, priority: int, pausable: bool) -> None:
        self._tick += 1
        self._models[model_id] = LoadedModel(
            model_id=model_id,
            estimated_mb=float(mb),
            priority=int(priority),
            pausable=bool(pausable),
            last_used=self._tick,
        )
