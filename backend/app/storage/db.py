"""SQLite persistence for benchmarks and sessions.

Uses the stdlib sqlite3 with a small connection helper. Writes are short and infrequent
(benchmark rows, session rows) so they don't block the event loop meaningfully; callers
that run them from request handlers use ``run_in_executor`` (see api layer).
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

_SCHEMA = """
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


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._init()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._conn() as c:
            c.executescript(_SCHEMA)

    def insert_benchmark(self, result: dict, meta: dict | None = None) -> int:
        meta = meta or {}
        with self._conn() as c:
            cur = c.execute(
                """INSERT INTO benchmarks
                (ts,backend,model_id,input_size,precision,device,provider,runs,fps,
                 latency_mean_ms,latency_p50_ms,latency_p95_ms,latency_p99_ms,memory_rss_mb,
                 hardware,os,runtime_versions,model_checksum,config,notes)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    time.time(), result.get("backend"), result.get("model_id"),
                    result.get("input_size"), result.get("precision"), result.get("device"),
                    result.get("provider"), result.get("runs"), result.get("fps"),
                    result.get("latency_mean_ms"), result.get("latency_p50_ms"),
                    result.get("latency_p95_ms"), result.get("latency_p99_ms"),
                    result.get("memory_rss_mb"),
                    meta.get("hardware"), meta.get("os"),
                    json.dumps(meta.get("runtime_versions")) if meta.get("runtime_versions") else None,
                    meta.get("model_checksum"),
                    json.dumps(meta.get("config")) if meta.get("config") else None,
                    result.get("notes", ""),
                ),
            )
            return int(cur.lastrowid)

    def list_benchmarks(self, limit: int = 100) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM benchmarks ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
            return [dict(r) for r in rows]

    def upsert_session(self, session_id: str, **fields) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT INTO sessions (session_id,ts,client_device,execution_location,model_id,runtime,last_seen)
                   VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(session_id) DO UPDATE SET last_seen=excluded.last_seen,
                     model_id=excluded.model_id, runtime=excluded.runtime""",
                (session_id, time.time(), fields.get("client_device"), fields.get("execution_location"),
                 fields.get("model_id"), fields.get("runtime"), time.time()),
            )

    def list_sessions(self, limit: int = 100) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM sessions ORDER BY last_seen DESC LIMIT ?", (limit,)).fetchall()
            return [dict(r) for r in rows]
