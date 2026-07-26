"""Backward compatibility across the VisionEdge Lab -> InferenceLab rename.

An existing deployment must survive the rename without editing its service
definition or losing its benchmark history.
"""
from __future__ import annotations

import sqlite3

from app.core.config import ENV_PREFIX, LEGACY_ENV_PREFIX, adopt_legacy_env
from app.storage.db import Database
from app.storage.migrations import (
    SCHEMA_VERSION,
    adopt_legacy_database,
    current_version,
    migrate,
)


class TestLegacyEnvironment:
    def test_legacy_prefix_is_carried_forward(self):
        env = {f"{LEGACY_ENV_PREFIX}PORT": "9000"}
        carried = adopt_legacy_env(env)
        assert env[f"{ENV_PREFIX}PORT"] == "9000"
        assert carried == [f"{LEGACY_ENV_PREFIX}PORT"]

    def test_explicit_new_value_wins_over_legacy(self):
        env = {f"{LEGACY_ENV_PREFIX}PORT": "9000", f"{ENV_PREFIX}PORT": "8000"}
        carried = adopt_legacy_env(env)
        assert env[f"{ENV_PREFIX}PORT"] == "8000"
        assert carried == []  # nothing was carried, so nothing is warned about

    def test_unrelated_variables_are_untouched(self):
        env = {"PATH": "/usr/bin", "HOME": "/root"}
        assert adopt_legacy_env(env) == []
        assert env == {"PATH": "/usr/bin", "HOME": "/root"}

    def test_multiple_variables_are_all_reported(self):
        env = {f"{LEGACY_ENV_PREFIX}HOST": "0.0.0.0", f"{LEGACY_ENV_PREFIX}LOG_JSON": "true"}
        assert adopt_legacy_env(env) == [
            f"{LEGACY_ENV_PREFIX}HOST", f"{LEGACY_ENV_PREFIX}LOG_JSON",
        ]


class TestDatabaseAdoption:
    def test_legacy_file_is_renamed(self, tmp_path):
        legacy = tmp_path / "visionedge.db"
        sqlite3.connect(legacy).close()
        new = tmp_path / "inferencelab.db"

        assert adopt_legacy_database(new, legacy) is True
        assert new.exists() and not legacy.exists()

    def test_existing_new_file_is_never_overwritten(self, tmp_path):
        legacy = tmp_path / "visionedge.db"
        new = tmp_path / "inferencelab.db"
        sqlite3.connect(legacy).close()
        with sqlite3.connect(new) as c:
            c.execute("CREATE TABLE keepme (x INTEGER)")

        assert adopt_legacy_database(new, legacy) is False
        # Live data survives, and the legacy file is left alone for manual review.
        with sqlite3.connect(new) as c:
            assert c.execute("SELECT name FROM sqlite_master WHERE name='keepme'").fetchone()
        assert legacy.exists()

    def test_no_legacy_file_is_a_no_op(self, tmp_path):
        assert adopt_legacy_database(tmp_path / "new.db", tmp_path / "absent.db") is False

    def test_wal_companions_follow_the_rename(self, tmp_path):
        # Leaving the -wal behind would discard committed transactions that had not
        # yet been checkpointed into the main file.
        legacy = tmp_path / "visionedge.db"
        sqlite3.connect(legacy).close()
        (tmp_path / "visionedge.db-wal").write_bytes(b"wal")
        (tmp_path / "visionedge.db-shm").write_bytes(b"shm")
        new = tmp_path / "inferencelab.db"

        assert adopt_legacy_database(new, legacy) is True
        assert (tmp_path / "inferencelab.db-wal").read_bytes() == b"wal"
        assert (tmp_path / "inferencelab.db-shm").read_bytes() == b"shm"


class TestMigrations:
    def test_fresh_database_reaches_current_version(self, tmp_path):
        db = Database(tmp_path / "fresh.db")
        assert db.schema_version == SCHEMA_VERSION
        assert db.applied_migrations == list(range(1, SCHEMA_VERSION + 1))

    def test_legacy_unversioned_database_is_upgraded_without_data_loss(self, tmp_path):
        # A VisionEdge-era file has the v1 tables but user_version = 0.
        path = tmp_path / "legacy.db"
        with sqlite3.connect(path) as c:
            c.executescript(
                "CREATE TABLE benchmarks (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL,"
                " backend TEXT, model_id TEXT, input_size INTEGER, precision TEXT, device TEXT,"
                " provider TEXT, runs INTEGER, fps REAL, latency_mean_ms REAL, latency_p50_ms REAL,"
                " latency_p95_ms REAL, latency_p99_ms REAL, memory_rss_mb REAL, hardware TEXT,"
                " os TEXT, runtime_versions TEXT, model_checksum TEXT, config TEXT, notes TEXT);"
                "CREATE TABLE sessions (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT UNIQUE,"
                " ts REAL NOT NULL, client_device TEXT, execution_location TEXT, model_id TEXT,"
                " runtime TEXT, last_seen REAL);"
            )
            c.execute("INSERT INTO benchmarks (ts, model_id, fps) VALUES (1.0, 'yolov8n-onnx', 42.0)")
            assert current_version(c) == 0

        db = Database(path)
        assert db.schema_version == SCHEMA_VERSION
        rows = db.list_benchmarks()
        assert len(rows) == 1 and rows[0]["model_id"] == "yolov8n-onnx"

    def test_migration_is_idempotent(self, tmp_path):
        path = tmp_path / "twice.db"
        Database(path)
        second = Database(path)
        assert second.applied_migrations == []  # nothing left to do
        assert second.schema_version == SCHEMA_VERSION

    def test_new_tables_exist_after_migration(self, tmp_path):
        db = Database(tmp_path / "v2.db")
        with sqlite3.connect(db.path) as c:
            names = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"benchmark_runs", "run_iterations", "run_utilization"} <= names

    def test_iterations_cascade_with_their_run(self, tmp_path):
        db = Database(tmp_path / "cascade.db")
        with sqlite3.connect(db.path) as c:
            c.execute("PRAGMA foreign_keys = ON")
            c.execute(
                "INSERT INTO benchmark_runs (run_id, schema_version, created_at, status, task,"
                " scenario_id, model_id, runtime_id, device, precision, mode, execution_location,"
                " fingerprint, run_json) VALUES ('r1',1,1.0,'completed','object_detection','s',"
                " 'm','onnxruntime','cpu','fp32','standard','in_process','fp','{}')"
            )
            c.execute(
                "INSERT INTO run_iterations (run_id, idx, phase_group, succeeded, total_ms)"
                " VALUES ('r1', 0, 'measured', 1, 10.0)"
            )
            c.execute("DELETE FROM benchmark_runs WHERE run_id='r1'")
            assert c.execute("SELECT COUNT(*) FROM run_iterations").fetchone()[0] == 0

    def test_migrate_returns_empty_on_current_database(self, tmp_path):
        path = tmp_path / "x.db"
        with sqlite3.connect(path) as c:
            migrate(c)
            assert migrate(c) == []
