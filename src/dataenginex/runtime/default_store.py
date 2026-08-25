"""The default persistence binding for domains that need a store of their own.

``DataCatalog`` and ``ModelRegistry`` both accept an injected repository, which
is how a running system wires them: everything shares the control store, so
catalog state and run state cannot drift apart. But both are also used
standalone in scripts and tests, and those callers need *a* store without
choosing one.

That default belongs somewhere. Putting it in the domains meant each of them
imported a concrete store, which is the coupling §5.5 forbids; putting it in
``bootstrap/`` would make the convenience constructors unusable without a full
application assembly. So it lives here, in the runtime — one module to change
when the default changes, and the domains keep depending only on the Protocol.

What this replaces is worth stating. The default used to be ``DexStore``: one
SQLite file holding pipeline runs, lineage, audit, agent memory, episodes,
catalog entries and model artifacts together. Six concerns in one schema, and
five of them now belong to the control store (§8.2), where they are written
transactionally alongside the run that produced them. Keeping ``DexStore``
alive to serve two convenience constructors meant maintaining a second,
divergent copy of run and lineage history that nothing reconciled.

So the default is narrow on purpose: two tables, exactly the two ports
(``CatalogRepository``, ``ModelRepository``) that the standalone path needs. A
caller wanting durable run history injects the control store instead — that is
what the ``store=`` parameter is for.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = [
    "CatalogEntry",
    "LocalRepository",
    "ModelArtifact",
    "catalog_row",
    "model_row",
    "open_default_store",
]


@dataclass
class CatalogEntry:
    """A dataset registered in the catalog."""

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
    """One registered version of a model."""

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


_SCHEMA = """
CREATE TABLE IF NOT EXISTS catalog_entries (
    name           TEXT PRIMARY KEY,
    layer          TEXT NOT NULL DEFAULT '',
    format         TEXT NOT NULL DEFAULT 'parquet',
    location       TEXT NOT NULL DEFAULT '',
    record_count   INTEGER NOT NULL DEFAULT 0,
    schema_fields  TEXT NOT NULL DEFAULT '[]',
    description    TEXT NOT NULL DEFAULT '',
    owner          TEXT NOT NULL DEFAULT '',
    tags           TEXT NOT NULL DEFAULT '[]',
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    metadata       TEXT NOT NULL DEFAULT '{}',
    version        INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_catalog_layer ON catalog_entries(layer);

CREATE TABLE IF NOT EXISTS model_artifacts (
    name           TEXT NOT NULL,
    version        TEXT NOT NULL,
    stage          TEXT NOT NULL DEFAULT 'development',
    artifact_path  TEXT NOT NULL DEFAULT '',
    metrics        TEXT NOT NULL DEFAULT '{}',
    parameters     TEXT NOT NULL DEFAULT '{}',
    description    TEXT NOT NULL DEFAULT '',
    tags           TEXT NOT NULL DEFAULT '[]',
    created_at     TEXT NOT NULL,
    promoted_at    TEXT,
    PRIMARY KEY (name, version)
);
CREATE INDEX IF NOT EXISTS idx_models_name ON model_artifacts(name);
"""


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _parse(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


class LocalRepository:
    """A single-file SQLite store satisfying the catalog and model ports.

    Connections are per-thread. SQLite objects cannot cross threads, and a
    catalog is routinely read from a request handler while a script writes it,
    so the alternative to thread-local connections is an error that only shows
    up under concurrency.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = Path(path)
        self._local = threading.local()
        # An in-memory database is per-connection, so a thread-local pool would
        # give every thread its own empty catalog. One shared connection keeps
        # the ephemeral case coherent; the lock is what makes that safe.
        self._memory = str(path) == ":memory:"
        self._lock = threading.RLock()
        self._shared: sqlite3.Connection | None = None
        self._open: list[sqlite3.Connection] = []
        if not self._memory:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn().executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        if self._memory:
            if self._shared is None:
                self._shared = sqlite3.connect(":memory:", check_same_thread=False)
                self._shared.row_factory = sqlite3.Row
                self._open.append(self._shared)
            return self._shared
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
            # Tracked centrally as well as thread-locally. ``close()`` runs on
            # one thread and can only reach its own thread-local slot, so
            # without this list every other thread's connection is left to be
            # finalised by the garbage collector — which surfaces as a
            # ResourceWarning, and under a strict test runner as a failure.
            self._open.append(conn)
        return conn

    def _write(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        with self._lock:
            conn = self._conn()
            conn.execute(sql, params)
            conn.commit()

    def _query(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn().execute(sql, params).fetchall())

    # --- CatalogRepository --------------------------------------------------

    def register_catalog(self, entry: CatalogEntry) -> CatalogEntry:
        """Insert or update an entry, bumping ``version`` on update.

        Re-registering is the common case — a pipeline rewrites its output and
        re-registers the dataset — so this upserts rather than failing. The
        version counter is what lets a reader tell "same entry, refreshed" from
        "never changed".
        """
        existing = self.get_catalog(entry.name)
        if existing is not None:
            entry.version = existing.version + 1
            entry.created_at = existing.created_at
        entry.updated_at = datetime.now(tz=UTC)
        self._write(
            "INSERT INTO catalog_entries "
            "(name, layer, format, location, record_count, schema_fields, description, "
            " owner, tags, created_at, updated_at, metadata, version) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(name) DO UPDATE SET "
            " layer=excluded.layer, format=excluded.format, location=excluded.location, "
            " record_count=excluded.record_count, schema_fields=excluded.schema_fields, "
            " description=excluded.description, owner=excluded.owner, tags=excluded.tags, "
            " updated_at=excluded.updated_at, metadata=excluded.metadata, "
            " version=excluded.version",
            (
                entry.name,
                entry.layer,
                entry.format,
                entry.location,
                entry.record_count,
                json.dumps(list(entry.schema_fields)),
                entry.description,
                entry.owner,
                json.dumps(list(entry.tags)),
                _iso(entry.created_at),
                _iso(entry.updated_at),
                json.dumps(entry.metadata),
                entry.version,
            ),
        )
        return entry

    def get_catalog(self, name: str) -> CatalogEntry | None:
        rows = self._query("SELECT * FROM catalog_entries WHERE name = ?", (name,))
        return self._to_catalog(rows[0]) if rows else None

    def search_catalog(
        self, *, layer: str | None = None, name_contains: str | None = None
    ) -> list[CatalogEntry]:
        sql = "SELECT * FROM catalog_entries WHERE 1=1"
        params: list[Any] = []
        if layer:
            sql += " AND layer = ?"
            params.append(layer)
        if name_contains:
            sql += " AND name LIKE ?"
            params.append(f"%{name_contains}%")
        return [self._to_catalog(r) for r in self._query(sql + " ORDER BY name", tuple(params))]

    def all_catalog(self) -> list[CatalogEntry]:
        return [
            self._to_catalog(r) for r in self._query("SELECT * FROM catalog_entries ORDER BY name")
        ]

    def delete_catalog(self, name: str) -> None:
        self._write("DELETE FROM catalog_entries WHERE name = ?", (name,))

    @staticmethod
    def _to_catalog(row: sqlite3.Row) -> CatalogEntry:
        created = _parse(row["created_at"]) or datetime.now(tz=UTC)
        updated = _parse(row["updated_at"]) or created
        return CatalogEntry(
            name=row["name"],
            layer=row["layer"],
            format=row["format"],
            location=row["location"],
            record_count=row["record_count"],
            schema_fields=json.loads(row["schema_fields"]),
            description=row["description"],
            owner=row["owner"],
            tags=json.loads(row["tags"]),
            created_at=created,
            updated_at=updated,
            metadata=json.loads(row["metadata"]),
            version=row["version"],
        )

    # --- ModelRepository ----------------------------------------------------

    def register_model(self, artifact: ModelArtifact) -> ModelArtifact:
        self._write(
            "INSERT INTO model_artifacts "
            "(name, version, stage, artifact_path, metrics, parameters, description, "
            " tags, created_at, promoted_at) VALUES (?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(name, version) DO UPDATE SET "
            " stage=excluded.stage, artifact_path=excluded.artifact_path, "
            " metrics=excluded.metrics, parameters=excluded.parameters, "
            " description=excluded.description, tags=excluded.tags, "
            " promoted_at=excluded.promoted_at",
            (
                artifact.name,
                artifact.version,
                artifact.stage,
                artifact.artifact_path,
                json.dumps(artifact.metrics),
                json.dumps(artifact.parameters),
                artifact.description,
                json.dumps(list(artifact.tags)),
                _iso(artifact.created_at),
                _iso(artifact.promoted_at),
            ),
        )
        return artifact

    def get_model(self, name: str, version: str) -> ModelArtifact | None:
        rows = self._query(
            "SELECT * FROM model_artifacts WHERE name = ? AND version = ?", (name, version)
        )
        return self._to_model(rows[0]) if rows else None

    def get_latest_model(self, name: str) -> ModelArtifact | None:
        """The most recently created version, not the highest version string.

        Sorting by ``version`` as text puts "0.10.0" before "0.9.0", so recency
        is the honest ordering without a semver parser this store has no reason
        to carry.
        """
        rows = self._query(
            "SELECT * FROM model_artifacts WHERE name = ? ORDER BY created_at DESC LIMIT 1",
            (name,),
        )
        return self._to_model(rows[0]) if rows else None

    def get_production_model(self, name: str) -> ModelArtifact | None:
        rows = self._query(
            "SELECT * FROM model_artifacts WHERE name = ? AND stage = 'production' "
            "ORDER BY promoted_at DESC LIMIT 1",
            (name,),
        )
        return self._to_model(rows[0]) if rows else None

    def list_model_names(self) -> list[str]:
        return [r["name"] for r in self._query("SELECT DISTINCT name FROM model_artifacts")]

    def list_model_versions(self, name: str) -> list[str]:
        return [
            r["version"]
            for r in self._query(
                "SELECT version FROM model_artifacts WHERE name = ? ORDER BY created_at", (name,)
            )
        ]

    def promote_model(self, name: str, version: str, stage: str) -> ModelArtifact | None:
        """Move a version to *stage*, demoting whatever held it.

        Demotion happens in the same transaction as promotion. Two versions both
        marked production is not a state a caller can resolve — it makes "which
        one is serving?" unanswerable — so it never exists, even briefly.
        """
        now = datetime.now(tz=UTC)
        with self._lock:
            conn = self._conn()
            if stage == "production":
                conn.execute(
                    "UPDATE model_artifacts SET stage = 'archived' "
                    "WHERE name = ? AND stage = 'production' AND version != ?",
                    (name, version),
                )
            conn.execute(
                "UPDATE model_artifacts SET stage = ?, promoted_at = ? "
                "WHERE name = ? AND version = ?",
                (stage, _iso(now), name, version),
            )
            conn.commit()
        return self.get_model(name, version)

    @staticmethod
    def _to_model(row: sqlite3.Row) -> ModelArtifact:
        return ModelArtifact(
            name=row["name"],
            version=row["version"],
            stage=row["stage"],
            artifact_path=row["artifact_path"],
            metrics=json.loads(row["metrics"]),
            parameters=json.loads(row["parameters"]),
            description=row["description"],
            tags=json.loads(row["tags"]),
            created_at=_parse(row["created_at"]) or datetime.now(tz=UTC),
            promoted_at=_parse(row["promoted_at"]),
        )

    def close(self) -> None:
        """Close every connection this store opened, on whichever thread."""
        with self._lock:
            for conn in self._open:
                with contextlib.suppress(Exception):
                    conn.close()
            self._open.clear()
            self._shared = None
            if getattr(self._local, "conn", None) is not None:
                self._local.conn = None

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()


def open_default_store(persist_path: str | Path | None = None) -> LocalRepository:
    """Open the default store, on disk at *persist_path* or in memory.

    In-memory is the right default for the no-argument case: a script or test
    that never said where to persist has not asked for a file, and silently
    creating one in the working directory is worse than keeping it ephemeral.
    """
    return LocalRepository(persist_path if persist_path is not None else ":memory:")


def catalog_row(**fields: Any) -> CatalogEntry:
    """Build a catalog storage record from a domain entry's fields."""
    return CatalogEntry(**fields)


def model_row(**fields: Any) -> ModelArtifact:
    """Build a model storage record from a domain artifact's fields."""
    return ModelArtifact(**fields)
