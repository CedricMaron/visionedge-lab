"""Persistence for :class:`~app.schemas.run.BenchmarkRun`.

The complete validated document is stored as JSON in ``benchmark_runs.run_json``;
the flat columns beside it are a query index, not a second source of truth. On any
disagreement the document wins, and :meth:`RunStore.get` reconstructs from the
document alone.

Raw iterations and utilization samples live in their own tables rather than inside
the document. A run with 10,000 iterations can then be listed without parsing
megabytes of JSON, while the raw evidence stays available on demand — which is the
point of §19's ban on storing only averages.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from app.core.logging import get_logger
from app.schemas.resources import UtilizationSample
from app.schemas.run import BenchmarkRun
from app.schemas.timing import IterationSample
from app.storage.migrations import migrate

log = get_logger("storage.runs")


class RunStore:
    """Stores and retrieves benchmark runs."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        with self._conn() as c:
            migrate(c)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # --- write ------------------------------------------------------------

    def save(self, run: BenchmarkRun, *, store_utilization: bool = True) -> str:
        """Persist a run and its raw evidence. Returns the run id."""
        document = run.model_dump(mode="json")

        with self._conn() as c:
            c.execute(
                """INSERT OR REPLACE INTO benchmark_runs
                   (run_id, schema_version, created_at, status, task, scenario_id, model_id,
                    runtime_id, device, precision, mode, execution_location, fingerprint,
                    batch_size, concurrency, measured_iterations, failed_iterations,
                    latency_p50_ms, latency_p95_ms, throughput_per_s, peak_rss_mb, run_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run.identity.run_id,
                    run.schema_version,
                    run.identity.created_at,
                    run.status.value,
                    run.task.value,
                    run.scenario.id,
                    run.model.model_id,
                    run.runtime.runtime_id,
                    run.runtime.device.value,
                    run.runtime.precision.value,
                    run.mode.value,
                    run.execution_location.value,
                    run.fingerprint.digest,
                    run.scenario.batch_size,
                    run.scenario.concurrency,
                    run.successful_iterations,
                    run.failed_iterations,
                    run.timings.total.p50_ms,
                    run.timings.total.p95_ms,
                    (
                        run.throughput.requests_per_second.value
                        if run.throughput.requests_per_second.available
                        else None
                    ),
                    (
                        run.memory.peak_process_rss_mb.value
                        if run.memory.peak_process_rss_mb.available
                        else None
                    ),
                    json.dumps(document),
                ),
            )

            # Replace children so a re-save cannot leave orphans from a prior attempt.
            c.execute("DELETE FROM run_iterations WHERE run_id = ?", (run.identity.run_id,))
            c.executemany(
                """INSERT INTO run_iterations
                   (run_id, idx, phase_group, succeeded, total_ms, error_type, error_message, spans_json)
                   VALUES (?,?,?,?,?,?,?,?)""",
                [
                    (
                        run.identity.run_id,
                        it.index,
                        it.group.value,
                        int(it.succeeded),
                        it.total_ms,
                        it.error_type,
                        it.error_message,
                        json.dumps([s.model_dump(mode="json") for s in it.spans]),
                    )
                    for it in run.iterations
                ],
            )

            c.execute("DELETE FROM run_utilization WHERE run_id = ?", (run.identity.run_id,))
            if store_utilization and run.utilization.samples:
                c.executemany(
                    "INSERT INTO run_utilization (run_id, t_offset_ms, sample_json) VALUES (?,?,?)",
                    [
                        (run.identity.run_id, s.t_offset_ms, json.dumps(s.model_dump(mode="json")))
                        for s in run.utilization.samples
                    ],
                )

        log.info(
            "run_saved",
            run_id=run.identity.run_id,
            iterations=len(run.iterations),
            status=run.status.value,
        )
        return run.identity.run_id

    # --- read -------------------------------------------------------------

    def get(self, run_id: str) -> BenchmarkRun | None:
        """Reconstruct a run from its stored document."""
        with self._conn() as c:
            row = c.execute(
                "SELECT run_json FROM benchmark_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        return BenchmarkRun.model_validate_json(row["run_json"])

    def list(
        self,
        limit: int = 50,
        task: str | None = None,
        model_id: str | None = None,
        fingerprint: str | None = None,
    ) -> list[dict]:
        """Summary rows for listing. Reads the index columns, not the documents."""
        clauses, params = [], []
        if task:
            clauses.append("task = ?")
            params.append(task)
        if model_id:
            clauses.append("model_id = ?")
            params.append(model_id)
        if fingerprint:
            clauses.append("fingerprint = ?")
            params.append(fingerprint)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)

        with self._conn() as c:
            rows = c.execute(
                f"""SELECT run_id, created_at, status, task, scenario_id, model_id, runtime_id,
                           device, precision, mode, fingerprint, batch_size, measured_iterations,
                           failed_iterations, latency_p50_ms, latency_p95_ms, throughput_per_s,
                           peak_rss_mb
                    FROM benchmark_runs {where}
                    ORDER BY created_at DESC LIMIT ?""",
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    def iterations(self, run_id: str) -> list[IterationSample]:
        """Raw per-iteration samples, loaded on demand."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM run_iterations WHERE run_id = ? ORDER BY idx", (run_id,)
            ).fetchall()
        return [
            IterationSample(
                index=r["idx"],
                group=r["phase_group"],
                total_ms=r["total_ms"],
                succeeded=bool(r["succeeded"]),
                error_type=r["error_type"],
                error_message=r["error_message"],
                spans=json.loads(r["spans_json"]) if r["spans_json"] else [],
            )
            for r in rows
        ]

    def utilization(self, run_id: str) -> list[UtilizationSample]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT sample_json FROM run_utilization WHERE run_id = ? ORDER BY t_offset_ms",
                (run_id,),
            ).fetchall()
        return [UtilizationSample.model_validate_json(r["sample_json"]) for r in rows]

    def delete(self, run_id: str) -> bool:
        with self._conn() as c:
            cur = c.execute("DELETE FROM benchmark_runs WHERE run_id = ?", (run_id,))
            return cur.rowcount > 0

    def comparable_group(self, fingerprint: str) -> list[dict]:
        """All runs sharing an environment fingerprint, i.e. safe to pool."""
        return self.list(limit=1000, fingerprint=fingerprint)
