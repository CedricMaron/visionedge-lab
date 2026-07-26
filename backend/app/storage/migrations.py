"""Versioned SQLite migrations, tracked with ``PRAGMA user_version``.

The original schema was created with bare ``CREATE TABLE IF NOT EXISTS`` and no
version marker, so an existing deployment's database reports ``user_version = 0``
while already having the v1 tables. Every migration is therefore written to be
idempotent, and v1 in particular must be safe to "apply" to a database that
already has it.

Migrations only ever add. Nothing here drops a table or a column, so a rollback to
a previous application version leaves the data readable.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

from app.core.logging import get_logger

log = get_logger("storage.migrations")

# --- v1: the schema inherited from VisionEdge Lab ---------------------------

_V1 = """
CREATE TABLE IF NOT EXISTS benchmarks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    backend TEXT, model_id TEXT, input_size INTEGER, precision TEXT, device TEXT,
    provider TEXT, runs INTEGER, fps REAL,
    latency_mean_ms REAL, latency_p50_ms REAL, latency_p95_ms REAL, latency_p99_ms REAL,
    memory_rss_mb REAL, hardware TEXT, os TEXT, runtime_versions TEXT, model_checksum TEXT,
    config TEXT, notes TEXT
);
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT UNIQUE, ts REAL NOT NULL, client_device TEXT,
    execution_location TEXT, model_id TEXT, runtime TEXT, last_seen REAL
);
"""

# --- v2: InferenceLab benchmark runs ----------------------------------------
#
# `run_json` holds the complete validated BenchmarkRun document. The flat columns
# beside it are a query index, not a second source of truth: they are projections
# of fields inside the document, used for listing and filtering without parsing
# every row. On any disagreement the document wins.
#
# Raw iterations live in their own table rather than inside the document blob so
# a run with 10,000 iterations can be listed cheaply and drilled into on demand.

_V2 = """
CREATE TABLE IF NOT EXISTS benchmark_runs (
    run_id TEXT PRIMARY KEY,
    schema_version INTEGER NOT NULL,
    created_at REAL NOT NULL,
    status TEXT NOT NULL,
    task TEXT NOT NULL,
    scenario_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    runtime_id TEXT NOT NULL,
    device TEXT NOT NULL,
    precision TEXT NOT NULL,
    mode TEXT NOT NULL,
    execution_location TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    batch_size INTEGER,
    concurrency INTEGER,
    measured_iterations INTEGER,
    failed_iterations INTEGER,
    latency_p50_ms REAL,
    latency_p95_ms REAL,
    throughput_per_s REAL,
    peak_rss_mb REAL,
    run_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_fingerprint ON benchmark_runs(fingerprint);
CREATE INDEX IF NOT EXISTS idx_runs_task_model ON benchmark_runs(task, model_id);
CREATE INDEX IF NOT EXISTS idx_runs_created ON benchmark_runs(created_at DESC);

CREATE TABLE IF NOT EXISTS run_iterations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES benchmark_runs(run_id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    phase_group TEXT NOT NULL,
    succeeded INTEGER NOT NULL,
    total_ms REAL,
    error_type TEXT,
    error_message TEXT,
    spans_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_iterations_run ON run_iterations(run_id, idx);

CREATE TABLE IF NOT EXISTS run_utilization (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES benchmark_runs(run_id) ON DELETE CASCADE,
    t_offset_ms REAL NOT NULL,
    sample_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_utilization_run ON run_utilization(run_id, t_offset_ms);
"""


def _apply_sql(sql: str) -> Callable[[sqlite3.Connection], None]:
    def _run(conn: sqlite3.Connection) -> None:
        conn.executescript(sql)

    return _run


#: (target_version, description, migration). Applied in order.
MIGRATIONS: list[tuple[int, str, Callable[[sqlite3.Connection], None]]] = [
    (1, "inherited benchmarks and sessions tables", _apply_sql(_V1)),
    (2, "InferenceLab benchmark runs, raw iterations and utilization samples", _apply_sql(_V2)),
]

SCHEMA_VERSION = MIGRATIONS[-1][0]


def current_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def migrate(conn: sqlite3.Connection) -> list[int]:
    """Bring ``conn`` up to :data:`SCHEMA_VERSION`. Returns the versions applied."""
    applied: list[int] = []
    version = current_version(conn)
    for target, description, migration in MIGRATIONS:
        if version >= target:
            continue
        log.info("migration_apply", version=target, description=description)
        migration(conn)
        # PRAGMA does not accept a bound parameter, and `target` is an int literal
        # from this module rather than external input.
        conn.execute(f"PRAGMA user_version = {int(target)}")
        conn.commit()
        applied.append(target)
        version = target
    return applied


def adopt_legacy_database(new_path: Path, legacy_path: Path) -> bool:
    """Rename a VisionEdge-era database file to its InferenceLab name.

    Only acts when the new file is absent and the legacy one exists, so it can
    never overwrite live data. Returns True when a file was adopted.

    A rename is used rather than a copy: it is atomic within a filesystem and does
    not leave two divergent databases behind for someone to pick the wrong one.
    """
    if new_path.exists() or not legacy_path.exists():
        return False
    legacy_path.rename(new_path)
    # SQLite's WAL companions must follow, or the adopted database loses any
    # committed-but-not-checkpointed transactions.
    for suffix in ("-wal", "-shm"):
        companion = legacy_path.with_name(legacy_path.name + suffix)
        if companion.exists():
            companion.rename(new_path.with_name(new_path.name + suffix))
    log.info("legacy_database_adopted", legacy=str(legacy_path), adopted_as=str(new_path))
    return True
