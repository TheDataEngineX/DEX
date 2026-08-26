"""DexStore — SQLite-backed persistence layer for all project state.

Replaces DuckDB with SQLite (WAL mode) for metadata storage.

Why SQLite instead of DuckDB for metadata:
- WAL mode: N concurrent readers + 1 writer with retry instead of SQLITE_BUSY crash
- threading.local: each thread gets its own connection — no shared-state races
- Multi-process: CLI + dex-studio web server can coexist without a file lock error
- DuckDB's file lock blocks a second process entirely; SQLite WAL does not

Domain tables (unchanged interface from DuckDB version):
  pipeline_runs   — execution history
  lineage_events  — data lineage graph
  model_artifacts — ML model registry
  quality_runs    — data quality check history
  audit_log       — user/system action audit trail
  ai_memory       — long-term AI agent memory
  ai_episodes     — episodic agent memory
  catalog_entries — data catalog (lakehouse datasets)
"""

from __future__ import annotations

import contextlib
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from dataenginex import _json

logger = structlog.get_logger()

__all__ = ["DexStore"]

# ---------------------------------------------------------------------------
# Lightweight data records
# ---------------------------------------------------------------------------


@dataclass
class PipelineRunRecord:
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    pipeline_name: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat())
    success: bool = False
    rows_input: int = 0
    rows_output: int = 0
    steps_completed: int = 0
    duration_ms: float = 0.0
    error: str | None = None
    skipped: bool = False


@dataclass
class LineageEvent:
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_id: str | None = None
    operation: str = ""
    layer: str = ""
    source: str = ""
    destination: str = ""
    input_count: int = 0
    output_count: int = 0
    error_count: int = 0
    quality_score: float | None = None
    pipeline_name: str = ""
    step_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=UTC))


@dataclass
class AuditEvent:
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat())
    actor: str = "system"
    action: str = ""
    resource: str = ""
    resource_type: str = ""
    status: str = "success"
    details: dict[str, Any] = field(default_factory=dict)
    ip_address: str = ""


@dataclass
class MemoryEntry:
    content: str = ""
    role: str = "user"
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass
class Episode:
    task: str = ""
    steps: list[dict[str, Any]] = field(default_factory=list)
    outcome: str = ""
    reward: float = 0.0
    timestamp: float = 0.0


@dataclass
class CatalogEntry:
    name: str = ""
    layer: str = ""
    format: str = "parquet"
    location: str = ""
    record_count: int = 0
    schema_fields: list[str] = field(default_factory=list)
    description: str = ""
    owner: str = ""
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    metadata: dict[str, Any] = field(default_factory=dict)
    version: int = 1


@dataclass
class ModelArtifact:
    name: str = ""
    version: str = "0.1.0"
    stage: str = "development"
    artifact_path: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    promoted_at: datetime | None = None


# ---------------------------------------------------------------------------
# Schema DDL (SQLite dialect — REAL instead of DOUBLE, INTEGER for booleans)
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id          TEXT PRIMARY KEY,
    pipeline_name   TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    success         INTEGER NOT NULL DEFAULT 0,
    rows_input      INTEGER NOT NULL DEFAULT 0,
    rows_output     INTEGER NOT NULL DEFAULT 0,
    steps_completed INTEGER NOT NULL DEFAULT 0,
    duration_ms     REAL    NOT NULL DEFAULT 0,
    error           TEXT,
    skipped         INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS lineage_events (
    event_id      TEXT PRIMARY KEY,
    parent_id     TEXT,
    operation     TEXT NOT NULL DEFAULT '',
    layer         TEXT NOT NULL DEFAULT '',
    source        TEXT NOT NULL DEFAULT '',
    destination   TEXT NOT NULL DEFAULT '',
    input_count   INTEGER NOT NULL DEFAULT 0,
    output_count  INTEGER NOT NULL DEFAULT 0,
    error_count   INTEGER NOT NULL DEFAULT 0,
    quality_score REAL,
    pipeline_name TEXT NOT NULL DEFAULT '',
    step_name     TEXT NOT NULL DEFAULT '',
    metadata      TEXT NOT NULL DEFAULT '{}',
    timestamp     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_lineage_pipeline ON lineage_events (pipeline_name);
CREATE INDEX IF NOT EXISTS idx_lineage_parent   ON lineage_events (parent_id);

CREATE TABLE IF NOT EXISTS model_artifacts (
    name          TEXT NOT NULL,
    version       TEXT NOT NULL,
    stage         TEXT NOT NULL DEFAULT 'development',
    artifact_path TEXT NOT NULL DEFAULT '',
    metrics       TEXT NOT NULL DEFAULT '{}',
    parameters    TEXT NOT NULL DEFAULT '{}',
    description   TEXT NOT NULL DEFAULT '',
    tags          TEXT NOT NULL DEFAULT '[]',
    created_at    TEXT NOT NULL,
    promoted_at   TEXT,
    PRIMARY KEY (name, version)
);

CREATE TABLE IF NOT EXISTS quality_runs (
    run_id    TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    results   TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS audit_log (
    event_id      TEXT PRIMARY KEY,
    timestamp     TEXT NOT NULL,
    actor         TEXT NOT NULL DEFAULT 'system',
    action        TEXT NOT NULL DEFAULT '',
    resource      TEXT NOT NULL DEFAULT '',
    resource_type TEXT NOT NULL DEFAULT '',
    status        TEXT NOT NULL DEFAULT 'success',
    details       TEXT NOT NULL DEFAULT '{}',
    ip_address    TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS ai_memory (
    content   TEXT NOT NULL,
    role      TEXT NOT NULL DEFAULT 'user',
    metadata  TEXT NOT NULL DEFAULT '{}',
    timestamp REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS ai_episodes (
    task      TEXT NOT NULL,
    steps     TEXT NOT NULL DEFAULT '[]',
    outcome   TEXT NOT NULL DEFAULT '',
    reward    REAL NOT NULL DEFAULT 0,
    timestamp REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS catalog_entries (
    name          TEXT PRIMARY KEY,
    layer         TEXT NOT NULL,
    format        TEXT NOT NULL DEFAULT 'parquet',
    location      TEXT NOT NULL,
    record_count  INTEGER NOT NULL DEFAULT 0,
    schema_fields TEXT NOT NULL DEFAULT '[]',
    description   TEXT NOT NULL DEFAULT '',
    owner         TEXT NOT NULL DEFAULT '',
    tags          TEXT NOT NULL DEFAULT '[]',
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    metadata      TEXT NOT NULL DEFAULT '{}',
    version       INTEGER NOT NULL DEFAULT 1
);
"""


# ---------------------------------------------------------------------------
# DexStore
# ---------------------------------------------------------------------------


class DexStore:
    """SQLite-backed store for all project metadata.

    Thread-safe: per-thread connections via threading.local (file mode) or a
    single shared connection with a Lock (in-memory mode).
    Multi-process-safe: WAL journal mode serialises writes without blocking reads.

    Args:
        db_path: Path to the SQLite file.  Use ``Path(":memory:")`` for tests.
    """

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._in_memory = str(db_path) == ":memory:"
        self._lock = threading.Lock()

        if self._in_memory:
            # Single shared connection — all threads share one in-memory DB
            self._mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._init_schema(self._mem_conn)
        else:
            self._tls: threading.local = threading.local()
            self._init_schema(self._get_conn())

        logger.info(
            "DexStore ready",
            path=str(db_path),
            mode="memory" if self._in_memory else "file",
        )

    def _get_conn(self) -> sqlite3.Connection:
        """Return the per-thread SQLite connection, creating it on first use."""
        if self._in_memory:
            return self._mem_conn
        if not hasattr(self._tls, "conn") or self._tls.conn is None:
            # timeout=10: retry for up to 10 s before raising OperationalError
            conn = sqlite3.connect(str(self._db_path), timeout=10.0, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")  # safe with WAL, faster than FULL
            conn.execute("PRAGMA foreign_keys=ON")
            self._tls.conn = conn
        return self._tls.conn  # type: ignore[no-any-return]

    def _init_schema(self, conn: sqlite3.Connection) -> None:
        with conn:
            for stmt in _SCHEMA.split(";"):
                stmt = stmt.strip()
                if stmt:
                    conn.execute(stmt)
            self._apply_additive_migrations(conn)

    @staticmethod
    def _apply_additive_migrations(conn: sqlite3.Connection) -> None:
        """Add columns introduced after a database was first created.

        ``CREATE TABLE IF NOT EXISTS`` is a no-op on an existing table, so a
        column added to _SCHEMA later never reaches databases created before
        it and every write fails with "no such column". Reconcile the declared
        columns against the live table and add whatever is missing.
        """
        for table, column, ddl in (("pipeline_runs", "skipped", "INTEGER NOT NULL DEFAULT 0"),):
            cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
            if cols and column not in cols:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    def _execute(self, sql: str, params: list[Any] | None = None) -> sqlite3.Cursor:
        return self._get_conn().execute(sql, params or [])

    def _write(self, sql: str, params: list[Any] | None = None) -> None:
        """Execute a single write inside a transaction, serialised by _lock."""
        with self._lock, self._get_conn() as conn:
            conn.execute(sql, params or [])

    def _write_many(self, ops: list[tuple[str, list[Any]]]) -> None:
        """Execute multiple writes in a single atomic transaction."""
        with self._lock, self._get_conn() as conn:
            for sql, params in ops:
                conn.execute(sql, params)

    # =========================================================================
    # Pipeline runs
    # =========================================================================

    def record_pipeline_run(
        self,
        pipeline_name: str,
        success: bool,
        rows_input: int = 0,
        rows_output: int = 0,
        steps_completed: int = 0,
        duration_ms: float = 0.0,
        error: str | None = None,
        skipped: bool = False,
    ) -> PipelineRunRecord:
        rec = PipelineRunRecord(
            pipeline_name=pipeline_name,
            success=success,
            rows_input=rows_input,
            rows_output=rows_output,
            steps_completed=steps_completed,
            duration_ms=round(duration_ms, 2),
            error=error,
            skipped=skipped,
        )
        self._write(
            """INSERT INTO pipeline_runs
               (run_id, pipeline_name, timestamp, success,
                rows_input, rows_output, steps_completed, duration_ms, error, skipped)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            [
                rec.run_id,
                rec.pipeline_name,
                rec.timestamp,
                1 if rec.success else 0,
                rec.rows_input,
                rec.rows_output,
                rec.steps_completed,
                rec.duration_ms,
                rec.error,
                1 if rec.skipped else 0,
            ],
        )
        logger.info("pipeline run recorded", pipeline=pipeline_name, success=success)
        return rec

    def get_pipeline_runs(self, pipeline_name: str | None = None) -> list[PipelineRunRecord]:
        if pipeline_name:
            rows = self._execute(
                "SELECT * FROM pipeline_runs WHERE pipeline_name=? ORDER BY timestamp DESC",
                [pipeline_name],
            ).fetchall()
        else:
            rows = self._execute("SELECT * FROM pipeline_runs ORDER BY timestamp DESC").fetchall()
        return [self._row_to_run(r) for r in rows]

    def get_last_pipeline_run(self, pipeline_name: str) -> PipelineRunRecord | None:
        row = self._execute(
            "SELECT * FROM pipeline_runs WHERE pipeline_name=? ORDER BY timestamp DESC LIMIT 1",
            [pipeline_name],
        ).fetchone()
        return self._row_to_run(row) if row else None

    @staticmethod
    def _row_to_run(row: tuple[Any, ...]) -> PipelineRunRecord:
        return PipelineRunRecord(
            run_id=row[0],
            pipeline_name=row[1],
            timestamp=row[2],
            success=bool(row[3]),
            rows_input=row[4],
            rows_output=row[5],
            steps_completed=row[6],
            duration_ms=row[7],
            error=row[8],
            skipped=bool(row[9]) if len(row) > 9 else False,
        )

    # =========================================================================
    # Lineage events — satisfies LineageBackend duck-typing for PipelineRunner
    # =========================================================================

    def record(self, **kwargs: Any) -> LineageEvent:
        """Create and persist a lineage event."""
        event = LineageEvent(**kwargs)
        self._write(
            """INSERT INTO lineage_events
               (event_id, parent_id, operation, layer, source, destination,
                input_count, output_count, error_count, quality_score,
                pipeline_name, step_name, metadata, timestamp)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [
                event.event_id,
                event.parent_id,
                event.operation,
                event.layer,
                event.source,
                event.destination,
                event.input_count,
                event.output_count,
                event.error_count,
                event.quality_score,
                event.pipeline_name,
                event.step_name,
                _json.dumps(event.metadata),
                event.timestamp.isoformat(),
            ],
        )
        return event

    def get_lineage_event(self, event_id: str) -> LineageEvent | None:
        row = self._execute("SELECT * FROM lineage_events WHERE event_id=?", [event_id]).fetchone()
        return self._row_to_lineage(row) if row else None

    def get_lineage_children(self, parent_id: str) -> list[LineageEvent]:
        rows = self._execute(
            "SELECT * FROM lineage_events WHERE parent_id=? ORDER BY timestamp",
            [parent_id],
        ).fetchall()
        return [self._row_to_lineage(r) for r in rows]

    def get_lineage_by_pipeline(self, pipeline_name: str) -> list[LineageEvent]:
        rows = self._execute(
            "SELECT * FROM lineage_events WHERE pipeline_name=? ORDER BY timestamp DESC",
            [pipeline_name],
        ).fetchall()
        return [self._row_to_lineage(r) for r in rows]

    def get_lineage_by_layer(self, layer: str) -> list[LineageEvent]:
        rows = self._execute(
            "SELECT * FROM lineage_events WHERE layer=? ORDER BY timestamp DESC",
            [layer],
        ).fetchall()
        return [self._row_to_lineage(r) for r in rows]

    @property
    def all_events(self) -> list[LineageEvent]:
        rows = self._execute(
            "SELECT * FROM lineage_events ORDER BY timestamp DESC LIMIT 1000"
        ).fetchall()
        return [self._row_to_lineage(r) for r in rows]

    def lineage_summary(self) -> dict[str, Any]:
        _row = self._execute("SELECT COUNT(*) FROM lineage_events").fetchone()
        total = _row[0] if _row else 0
        layer_rows = self._execute(
            "SELECT layer, COUNT(*) FROM lineage_events GROUP BY layer"
        ).fetchall()
        op_rows = self._execute(
            "SELECT operation, COUNT(*) FROM lineage_events GROUP BY operation"
        ).fetchall()
        return {
            "total_events": total,
            "by_layer": {r[0]: r[1] for r in layer_rows},
            "by_operation": {r[0]: r[1] for r in op_rows},
        }

    @staticmethod
    def _row_to_lineage(row: tuple[Any, ...]) -> LineageEvent:
        meta: dict[str, Any] = _json.loads(row[12]) if row[12] else {}
        return LineageEvent(
            event_id=row[0],
            parent_id=row[1],
            operation=row[2],
            layer=row[3],
            source=row[4],
            destination=row[5],
            input_count=row[6],
            output_count=row[7],
            error_count=row[8],
            quality_score=row[9],
            pipeline_name=row[10],
            step_name=row[11],
            metadata=meta,
            timestamp=datetime.fromisoformat(row[13]) if row[13] else datetime.now(tz=UTC),
        )

    # =========================================================================
    # Model registry
    # =========================================================================

    def register_model(self, artifact: ModelArtifact) -> ModelArtifact:
        existing = self.get_model(artifact.name, artifact.version)
        if existing:
            msg = f"Model {artifact.name!r} v{artifact.version} already registered"
            raise ValueError(msg)
        self._write(
            """INSERT INTO model_artifacts
               (name, version, stage, artifact_path, metrics, parameters,
                description, tags, created_at, promoted_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            [
                artifact.name,
                artifact.version,
                artifact.stage,
                artifact.artifact_path,
                _json.dumps(artifact.metrics),
                _json.dumps(artifact.parameters),
                artifact.description,
                _json.dumps(artifact.tags),
                artifact.created_at.isoformat(),
                artifact.promoted_at.isoformat() if artifact.promoted_at else None,
            ],
        )
        logger.info("model registered", name=artifact.name, version=artifact.version)
        return artifact

    def get_model(self, name: str, version: str) -> ModelArtifact | None:
        row = self._execute(
            "SELECT * FROM model_artifacts WHERE name=? AND version=?", [name, version]
        ).fetchone()
        return self._row_to_model(row) if row else None

    def get_latest_model(self, name: str) -> ModelArtifact | None:
        row = self._execute(
            "SELECT * FROM model_artifacts WHERE name=? ORDER BY created_at DESC LIMIT 1",
            [name],
        ).fetchone()
        return self._row_to_model(row) if row else None

    def get_production_model(self, name: str) -> ModelArtifact | None:
        row = self._execute(
            "SELECT * FROM model_artifacts WHERE name=? AND stage='production' LIMIT 1",
            [name],
        ).fetchone()
        return self._row_to_model(row) if row else None

    def list_model_names(self) -> list[str]:
        rows = self._execute("SELECT DISTINCT name FROM model_artifacts ORDER BY name").fetchall()
        return [r[0] for r in rows]

    def list_model_versions(self, name: str) -> list[str]:
        rows = self._execute(
            "SELECT version FROM model_artifacts WHERE name=? ORDER BY created_at",
            [name],
        ).fetchall()
        return [r[0] for r in rows]

    def promote_model(self, name: str, version: str, stage: str) -> ModelArtifact:
        ops: list[tuple[str, list[Any]]] = []
        if stage == "production":
            ops.append(
                (
                    "UPDATE model_artifacts SET stage='archived'"
                    " WHERE name=? AND stage='production'",
                    [name],
                )
            )
        ops.append(
            (
                "UPDATE model_artifacts SET stage=?, promoted_at=? WHERE name=? AND version=?",
                [stage, datetime.now(tz=UTC).isoformat(), name, version],
            )
        )
        self._write_many(ops)
        artifact = self.get_model(name, version)
        if artifact is None:
            msg = f"Model {name!r} v{version} not found"
            raise ValueError(msg)
        return artifact

    def delete_model(self, name: str, version: str | None = None) -> None:
        if version:
            self._write("DELETE FROM model_artifacts WHERE name=? AND version=?", [name, version])
        else:
            self._write("DELETE FROM model_artifacts WHERE name=?", [name])

    @staticmethod
    def _row_to_model(row: tuple[Any, ...]) -> ModelArtifact:
        return ModelArtifact(
            name=row[0],
            version=row[1],
            stage=row[2],
            artifact_path=row[3],
            metrics=_json.loads(row[4]),
            parameters=_json.loads(row[5]),
            description=row[6],
            tags=_json.loads(row[7]),
            created_at=datetime.fromisoformat(row[8]) if row[8] else datetime.now(tz=UTC),
            promoted_at=datetime.fromisoformat(row[9]) if row[9] else None,
        )

    # =========================================================================
    # Quality history
    # =========================================================================

    def record_quality_run(self, results: dict[str, Any]) -> str:
        run_id = uuid.uuid4().hex[:8]
        timestamp = datetime.now(tz=UTC).isoformat()
        self._write_many(
            [
                (
                    "INSERT INTO quality_runs (run_id, timestamp, results) VALUES (?,?,?)",
                    [run_id, timestamp, _json.dumps(results)],
                ),
                (
                    """DELETE FROM quality_runs WHERE run_id NOT IN (
                   SELECT run_id FROM quality_runs ORDER BY timestamp DESC LIMIT 50)""",
                    [],
                ),
            ]
        )
        return run_id

    def get_quality_history(self, limit: int = 50) -> dict[str, Any]:
        rows = self._execute(
            "SELECT run_id, timestamp, results FROM quality_runs ORDER BY timestamp DESC LIMIT ?",
            [limit],
        ).fetchall()
        runs = [{"run_id": r[0], "timestamp": r[1], "results": _json.loads(r[2])} for r in rows]
        return {"runs": runs}

    # =========================================================================
    # Audit log
    # =========================================================================

    def log_audit(
        self,
        action: str,
        resource: str = "",
        resource_type: str = "",
        actor: str = "user",
        status: str = "success",
        details: dict[str, Any] | None = None,
        ip_address: str = "",
    ) -> AuditEvent:
        event = AuditEvent(
            action=action,
            resource=resource,
            resource_type=resource_type,
            actor=actor,
            status=status,
            details=details or {},
            ip_address=ip_address,
        )
        self._write(
            """INSERT INTO audit_log
               (event_id, timestamp, actor, action, resource, resource_type,
                status, details, ip_address)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            [
                event.event_id,
                event.timestamp,
                event.actor,
                event.action,
                event.resource,
                event.resource_type,
                event.status,
                _json.dumps(event.details),
                event.ip_address,
            ],
        )
        return event

    def get_audit_events(
        self,
        action: str | None = None,
        resource: str | None = None,
        actor: str | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        sql = "SELECT * FROM audit_log"
        params: list[Any] = []
        conditions: list[str] = []
        if action:
            conditions.append("action LIKE ?")
            params.append(f"%{action}%")
        if resource:
            conditions.append("resource LIKE ?")
            params.append(f"%{resource}%")
        if actor:
            conditions.append("actor LIKE ?")
            params.append(f"%{actor}%")
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        rows = self._execute(sql, params).fetchall()
        return [self._row_to_audit(r) for r in rows]

    @staticmethod
    def _row_to_audit(row: tuple[Any, ...]) -> AuditEvent:
        return AuditEvent(
            event_id=row[0],
            timestamp=row[1],
            actor=row[2],
            action=row[3],
            resource=row[4],
            resource_type=row[5],
            status=row[6],
            details=_json.loads(row[7]),
            ip_address=row[8],
        )

    # =========================================================================
    # AI memory
    # =========================================================================

    def save_memory(self, entry: MemoryEntry) -> None:
        self._write(
            "INSERT INTO ai_memory (content, role, metadata, timestamp) VALUES (?,?,?,?)",
            [entry.content, entry.role, _json.dumps(entry.metadata), entry.timestamp],
        )

    def get_recent_memory(self, n: int = 100) -> list[MemoryEntry]:
        rows = self._execute(
            "SELECT content, role, metadata, timestamp"
            " FROM ai_memory ORDER BY timestamp DESC LIMIT ?",
            [n],
        ).fetchall()
        return [
            MemoryEntry(content=r[0], role=r[1], metadata=_json.loads(r[2]), timestamp=r[3])
            for r in rows
        ]

    def search_memory(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        query_lower = query.lower()
        all_entries = self.get_recent_memory(1000)
        scored = [
            (sum(1 for w in query_lower.split() if w in e.content.lower()), e) for e in all_entries
        ]
        scored = [(s, e) for s, e in scored if s > 0]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    # =========================================================================
    # AI episodic memory
    # =========================================================================

    def add_episode(self, episode: Episode) -> None:
        self._write(
            "INSERT INTO ai_episodes (task, steps, outcome, reward, timestamp) VALUES (?,?,?,?,?)",
            [
                episode.task,
                _json.dumps(episode.steps),
                episode.outcome,
                episode.reward,
                episode.timestamp,
            ],
        )

    def recall_episodes(self, task: str, top_k: int = 5) -> list[Episode]:
        rows = self._execute(
            "SELECT task, steps, outcome, reward, timestamp"
            " FROM ai_episodes ORDER BY timestamp DESC LIMIT 500"
        ).fetchall()
        episodes = [
            Episode(
                task=r[0],
                steps=_json.loads(r[1]),
                outcome=r[2],
                reward=r[3],
                timestamp=r[4],
            )
            for r in rows
        ]
        task_lower = task.lower()
        scored = [(sum(1 for w in task_lower.split() if w in e.task.lower()), e) for e in episodes]
        scored = [(s, e) for s, e in scored if s > 0]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    # =========================================================================
    # Data catalog
    # =========================================================================

    def register_catalog(self, entry: CatalogEntry) -> CatalogEntry:
        existing = self.get_catalog(entry.name)
        if existing:
            entry.version = existing.version + 1
            entry.created_at = existing.created_at
        entry.updated_at = datetime.now(tz=UTC)
        self._write(
            """INSERT OR REPLACE INTO catalog_entries
               (name, layer, format, location, record_count, schema_fields,
                description, owner, tags, created_at, updated_at, metadata, version)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [
                entry.name,
                entry.layer,
                entry.format,
                entry.location,
                entry.record_count,
                _json.dumps(entry.schema_fields),
                entry.description,
                entry.owner,
                _json.dumps(entry.tags),
                entry.created_at.isoformat(),
                entry.updated_at.isoformat(),
                _json.dumps(entry.metadata),
                entry.version,
            ],
        )
        logger.info("catalog entry registered", name=entry.name, layer=entry.layer)
        return entry

    def get_catalog(self, name: str) -> CatalogEntry | None:
        row = self._execute("SELECT * FROM catalog_entries WHERE name=?", [name]).fetchone()
        return self._row_to_catalog(row) if row else None

    def search_catalog(
        self,
        layer: str | None = None,
        name_contains: str | None = None,
    ) -> list[CatalogEntry]:
        sql = "SELECT * FROM catalog_entries"
        params: list[Any] = []
        conditions: list[str] = []
        if layer:
            conditions.append("layer=?")
            params.append(layer)
        if name_contains:
            conditions.append("name LIKE ?")
            params.append(f"%{name_contains}%")
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY updated_at DESC"
        rows = self._execute(sql, params).fetchall()
        return [self._row_to_catalog(r) for r in rows]

    def all_catalog(self) -> list[CatalogEntry]:
        rows = self._execute("SELECT * FROM catalog_entries ORDER BY updated_at DESC").fetchall()
        return [self._row_to_catalog(r) for r in rows]

    def delete_catalog(self, name: str) -> None:
        """Remove a catalog entry. Silent when it is already absent."""
        self._write("DELETE FROM catalog_entries WHERE name=?", [name])

    @staticmethod
    def _row_to_catalog(row: tuple[Any, ...]) -> CatalogEntry:
        return CatalogEntry(
            name=row[0],
            layer=row[1],
            format=row[2],
            location=row[3],
            record_count=row[4],
            schema_fields=_json.loads(row[5]),
            description=row[6],
            owner=row[7],
            tags=_json.loads(row[8]),
            created_at=datetime.fromisoformat(row[9]) if row[9] else datetime.now(tz=UTC),
            updated_at=datetime.fromisoformat(row[10]) if row[10] else datetime.now(tz=UTC),
            metadata=_json.loads(row[11]),
            version=row[12],
        )

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def close(self) -> None:
        if self._in_memory:
            self._mem_conn.close()
        elif hasattr(self._tls, "conn") and self._tls.conn is not None:
            self._tls.conn.close()
            self._tls.conn = None
        logger.info("DexStore closed", path=str(self._db_path))

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()
