"""Result persistence, exports, scenario loading and the CLI."""
from __future__ import annotations

import json

import pytest

from app.adapters.base import LoadConfig
from app.adapters.mock import MockAdapter
from app.benchmark import BenchmarkEngine, EngineOptions
from app.benchmark.export import iterations_to_csv, summary_to_csv, to_json, to_markdown
from app.benchmark.scenarios import get_scenario, load_all, load_scenario_file, scenario_paths
from app.core.errors import ConfigInvalidError
from app.schemas.enums import DeviceKind, IterationPhaseGroup, Precision, Task
from app.schemas.environment import RuntimeReference
from app.schemas.run import BenchmarkRun
from app.schemas.scenario import ScenarioSpec
from app.storage.runs import RunStore


@pytest.fixture
def sample_run() -> BenchmarkRun:
    engine = BenchmarkEngine(EngineOptions(enable_sampler=False, label="fixture"))
    try:
        return engine.run(
            MockAdapter(latency_ms=1.0, load_ms=1.0, fail_on_iterations=(4,), allow_override=True),
            ScenarioSpec(
                id="mock-scenario", task=Task.IMAGE_CLASSIFICATION,
                warmup_iterations=2, measured_iterations=8,
            ),
            LoadConfig(runtime_id="mock"),
            RuntimeReference(
                runtime_id="mock", device=DeviceKind.CPU, precision=Precision.FP32
            ),
        )
    finally:
        engine.close()


@pytest.fixture
def store(tmp_path) -> RunStore:
    return RunStore(tmp_path / "runs.db")


class TestRunPersistence:
    def test_round_trips_without_loss(self, store, sample_run):
        store.save(sample_run)
        loaded = store.get(sample_run.identity.run_id)

        assert loaded is not None
        assert loaded.identity.run_id == sample_run.identity.run_id
        assert loaded.timings.total.p95_ms == pytest.approx(sample_run.timings.total.p95_ms)
        assert loaded.fingerprint.digest == sample_run.fingerprint.digest

    def test_raw_iterations_are_stored_not_just_aggregates(self, store, sample_run):
        store.save(sample_run)
        raw = store.iterations(sample_run.identity.run_id)

        assert len(raw) == len(sample_run.iterations)
        assert any(i.group is IterationPhaseGroup.WARMUP for i in raw)
        assert any(not i.succeeded for i in raw)

    def test_failed_iterations_survive_persistence(self, store, sample_run):
        store.save(sample_run)
        loaded = store.get(sample_run.identity.run_id)
        assert loaded.errors.failure_count == sample_run.errors.failure_count
        assert loaded.errors.failures[0].error_type == "RuntimeError"

    def test_warnings_survive_persistence(self, store, sample_run):
        store.save(sample_run)
        assert store.get(sample_run.identity.run_id).warnings == sample_run.warnings

    def test_unavailable_measurements_keep_their_reasons(self, store, sample_run):
        store.save(sample_run)
        energy = store.get(sample_run.identity.run_id).energy.total_energy_j
        assert not energy.available
        assert energy.unavailable_reason

    def test_missing_run_returns_none(self, store):
        assert store.get("run_does_not_exist") is None

    def test_listing_filters_by_task_and_model(self, store, sample_run):
        store.save(sample_run)
        assert len(store.list(task="image_classification")) == 1
        assert len(store.list(task="object_detection")) == 0
        assert len(store.list(model_id="mock-adapter")) == 1

    def test_resaving_replaces_children_rather_than_duplicating(self, store, sample_run):
        store.save(sample_run)
        store.save(sample_run)
        assert len(store.iterations(sample_run.identity.run_id)) == len(sample_run.iterations)

    def test_delete_cascades_to_iterations(self, store, sample_run):
        store.save(sample_run)
        assert store.delete(sample_run.identity.run_id) is True
        assert store.iterations(sample_run.identity.run_id) == []

    def test_fingerprint_groups_comparable_runs(self, store, sample_run):
        store.save(sample_run)
        group = store.comparable_group(sample_run.fingerprint.digest)
        assert len(group) == 1


class TestExports:
    def test_json_is_a_complete_document(self, sample_run):
        payload = json.loads(to_json(sample_run))
        assert payload["identity"]["run_id"] == sample_run.identity.run_id
        assert len(payload["iterations"]) == len(sample_run.iterations)
        # Re-validates, so the export is not lossy.
        assert BenchmarkRun.model_validate(payload)

    def test_iteration_csv_has_one_row_per_iteration(self, sample_run):
        lines = iterations_to_csv(sample_run).strip().splitlines()
        assert len(lines) == len(sample_run.iterations) + 1  # + header

    def test_iteration_csv_distinguishes_counted_rows(self, sample_run):
        csv_text = iterations_to_csv(sample_run)
        assert "warmup" in csv_text and "measured" in csv_text
        assert "False" in csv_text  # warm-up rows are not counted

    def test_summary_csv_carries_sample_count_and_warnings(self, sample_run):
        header, row = summary_to_csv([sample_run]).strip().splitlines()[:2]
        assert "n_measured" in header and "warnings" in header
        assert str(sample_run.timings.total.n) in row

    def test_markdown_leads_with_warnings(self, sample_run):
        md = to_markdown(sample_run)
        assert "## Warnings" in md
        assert md.index("## Warnings") < md.index("## Latency decomposition")

    def test_markdown_reports_unavailable_metrics_honestly(self, sample_run):
        md = to_markdown(sample_run)
        assert "_unavailable —" in md

    def test_markdown_states_failures_are_excluded(self, sample_run):
        md = to_markdown(sample_run)
        assert "Statistics below exclude these iterations." in md

    def test_markdown_includes_reproduction_metadata(self, sample_run):
        md = to_markdown(sample_run)
        assert "## Reproducibility" in md
        assert "Raw samples retained" in md


class TestScenarioLoading:
    def test_repository_scenarios_all_validate(self):
        scenarios = load_all()
        assert scenarios, "no scenarios found in benchmarks/scenarios/"
        for spec in scenarios.values():
            assert spec.measured_iterations >= 1

    def test_every_scenario_file_parses(self):
        for path in scenario_paths():
            assert load_scenario_file(path).id

    def test_unknown_scenario_lists_the_available_ones(self):
        with pytest.raises(ConfigInvalidError, match="Available:"):
            get_scenario("no-such-scenario")

    def test_invalid_yaml_names_the_file(self, tmp_path):
        bad = tmp_path / "broken.yaml"
        bad.write_text("id: [unclosed\n")
        with pytest.raises(ConfigInvalidError, match="broken.yaml"):
            load_scenario_file(bad)

    def test_non_mapping_is_rejected(self, tmp_path):
        bad = tmp_path / "list.yaml"
        bad.write_text("- one\n- two\n")
        with pytest.raises(ConfigInvalidError, match="mapping"):
            load_scenario_file(bad)

    def test_duplicate_ids_are_an_error_not_a_silent_overwrite(self, tmp_path):
        for name in ("a.yaml", "b.yaml"):
            (tmp_path / name).write_text("id: dupe\ntask: object_detection\n")
        with pytest.raises(ConfigInvalidError, match="duplicate scenario id"):
            load_all(tmp_path)

    def test_streaming_on_a_non_generative_task_is_rejected(self):
        with pytest.raises(ValueError, match="streaming"):
            ScenarioSpec(id="bad", task=Task.OBJECT_DETECTION, streaming=True)

    def test_thin_sample_scenarios_are_identifiable(self):
        assert not ScenarioSpec(id="thin", task=Task.OBJECT_DETECTION,
                                measured_iterations=3).has_sufficient_samples
        assert ScenarioSpec(id="ok", task=Task.OBJECT_DETECTION,
                            measured_iterations=20).has_sufficient_samples


class TestCli:
    def test_runtimes_command_runs(self, capsys):
        from app.cli import main

        assert main(["runtimes"]) == 0
        assert "onnxruntime" in capsys.readouterr().out

    def test_matrix_explains_unsupported_cells(self, capsys):
        from app.cli import main

        assert main(["matrix"]) == 0
        out = capsys.readouterr().out
        assert "no" in out and "precision" in out

    def test_scenarios_command_lists_repository_scenarios(self, capsys):
        from app.cli import main

        assert main(["scenarios"]) == 0
        assert "single-image-detection" in capsys.readouterr().out

    def test_system_command_reports_real_hardware(self, capsys):
        from app.cli import main

        assert main(["system"]) == 0
        out = capsys.readouterr().out
        assert "CPU" in out and "Python" in out

    def test_unknown_model_fails_with_a_helpful_message(self, capsys):
        from app.cli import main

        assert main([
            "benchmark", "run", "--scenario", "single-image-detection",
            "--model", "not-a-real-model", "--no-save",
        ]) == 2
        assert "has no adapter" in capsys.readouterr().err

    def test_parser_rejects_an_invalid_device(self):
        from app.cli import build_parser

        with pytest.raises(SystemExit):
            build_parser().parse_args([
                "benchmark", "run", "--scenario", "x", "--model", "y", "--device", "quantum",
            ])
