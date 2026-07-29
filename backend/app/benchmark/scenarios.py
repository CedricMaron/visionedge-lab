"""Load and validate scenario definitions from ``benchmarks/scenarios/*.yaml``.

Scenarios are data, not code: they are versioned in the repository, readable in a
diff, and validated against :class:`~app.schemas.scenario.ScenarioSpec` on load, so
a malformed scenario fails at load time with a field-level message rather than
halfway through a benchmark.

``yaml.safe_load`` is used rather than ``yaml.load``: scenario files are ordinary
repository content, and full YAML can construct arbitrary Python objects.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from app.core.config import REPO_ROOT
from app.core.errors import ConfigInvalidError
from app.schemas.scenario import ScenarioSpec

SCENARIOS_DIR = REPO_ROOT / "benchmarks" / "scenarios"


def scenario_paths(directory: Path | None = None) -> list[Path]:
    directory = directory or SCENARIOS_DIR
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix in (".yaml", ".yml"))


def load_scenario_file(path: Path) -> ScenarioSpec:
    """Parse and validate one scenario file."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigInvalidError(f"{path.name} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigInvalidError(f"{path.name} must contain a YAML mapping at the top level")
    try:
        return ScenarioSpec.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 - re-raised as a domain error with the file name
        raise ConfigInvalidError(f"{path.name} is not a valid scenario: {exc}") from exc


def load_all(directory: Path | None = None) -> dict[str, ScenarioSpec]:
    """Load every scenario, keyed by id. Duplicate ids are an error, not a silent overwrite."""
    scenarios: dict[str, ScenarioSpec] = {}
    sources: dict[str, str] = {}
    for path in scenario_paths(directory):
        spec = load_scenario_file(path)
        if spec.id in scenarios:
            raise ConfigInvalidError(
                f"duplicate scenario id '{spec.id}' in {path.name} and {sources[spec.id]}"
            )
        scenarios[spec.id] = spec
        sources[spec.id] = path.name
    return scenarios


def get_scenario(scenario_id: str, directory: Path | None = None) -> ScenarioSpec:
    scenarios = load_all(directory)
    if scenario_id not in scenarios:
        available = ", ".join(sorted(scenarios)) or "none found"
        raise ConfigInvalidError(
            f"unknown scenario '{scenario_id}'. Available: {available}"
        )
    return scenarios[scenario_id]
