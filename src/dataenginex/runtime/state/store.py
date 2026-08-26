"""Control store: SQLite (WAL) with migrations and a transactional outbox.

One store owns all control-plane state (§8.1). User analytical data lives in
DuckDB, artifacts on the filesystem or object storage, and secrets in the OS
keyring — the control store holds only references to those.

The most important method here is :meth:`ControlStore.transaction`. Every state
change goes through it, and events emitted inside it land in ``outbox_events``
in the *same* transaction as the state change (§8.3). That is what removes the
dual-write hole: either the run row and its ``RunStarted`` event both commit, or
neither does. A dispatcher drains the outbox afterwards, so a crash between
commit and publication loses nothing — the event is still in the table.

Concurrency: SQLite in WAL mode allows one writer with concurrent readers. That
is sufficient for Lite mode by design (ADR-0006), and ``BEGIN IMMEDIATE`` is
used for write transactions so a lock conflict surfaces at transaction start
rather than halfway through.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from dataenginex.foundation import AuditEvent, MetadataEvent, new_id, utcnow
from dataenginex.runtime.state.migrations import MIGRATIONS, Migration

__all__ = ["ControlStore", "OutboxRecord", "StoreError", "Transaction"]


class StoreError(RuntimeError):
    """Control-store failure that callers are expected to handle."""


class OutboxRecord:
    """One pending outbox row handed to a dispatcher.

    Deliberately plain: the dispatcher forwards ``payload`` verbatim and never
    interprets it, so validating it back into a domain model would cost work
    for no benefit.
    """

    __slots__ = (
        "attempts",
        "created_at",
        "event_kind",
        "event_type",
        "outbox_id",
        "payload",
    )

    def __init__(
        self,
        outbox_id: str,
        event_kind: str,
        event_type: str,
        payload: dict[str, Any],
        created_at: str,
        attempts: int,
    ) -> None:
        self.outbox_id = outbox_id
        self.event_kind = event_kind
        self.event_type = event_type
        self.payload = payload
        self.created_at = created_at
        self.attempts = attempts


def _json_default(value: object) -> str:
    """Serialize the few non-JSON types that reach the outbox."""
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


class Transaction:
    """Handle for statements and events inside one control-store transaction.

    Not constructed directly — obtained from :meth:`ControlStore.transaction`.
    Its whole purpose is that ``execute`` and ``emit_*`` share a transaction, so
    an event can never be published for a state change that rolled back.
    """

    __slots__ = ("_conn",)

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        return self._conn.execute(sql, params)

    def executemany(self, sql: str, params: Sequence[Sequence[Any]]) -> sqlite3.Cursor:
        return self._conn.executemany(sql, params)

    def emit_metadata(self, event: MetadataEvent) -> str:
        """Persist a metadata event and queue it for dispatch."""
        env = event.envelope
        self._conn.execute(
            "INSERT INTO metadata_events (event_id, occurred_at, producer, "
            "installation_id, workspace_id, project_id, revision_id, principal_id, "
            "correlation_id, schema_version, event_type, subject_id, subject_type, "
            "payload_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                env.event_id,
                env.occurred_at.isoformat(),
                env.producer,
                env.installation_id,
                env.workspace_id,
                env.project_id,
                env.revision_id,
                env.principal_id,
                env.correlation_id,
                env.schema_version,
                event.event_type,
                event.subject_id,
                event.subject_type,
                json.dumps(event.payload, default=_json_default),
            ),
        )
        return self._enqueue("metadata", event.event_type, event.model_dump(mode="json"))

    def emit_audit(self, event: AuditEvent) -> str:
        """Persist an audit event and queue it for dispatch.

        Insert only — invariant 9 means there is no update path, and the schema
        triggers reject one even if a caller tried.
        """
        env = event.envelope
        self._conn.execute(
            "INSERT INTO audit_events (event_id, occurred_at, producer, "
            "installation_id, workspace_id, project_id, revision_id, principal_id, "
            "correlation_id, schema_version, event_type, action, outcome, target_id, "
            "target_type, destination, policy_decision_id, detail_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                env.event_id,
                env.occurred_at.isoformat(),
                env.producer,
                env.installation_id,
                env.workspace_id,
                env.project_id,
                env.revision_id,
                env.principal_id,
                env.correlation_id,
                env.schema_version,
                event.event_type.value,
                event.action,
                event.outcome,
                event.target_id,
                event.target_type,
                event.destination,
                event.policy_decision_id,
                json.dumps(event.detail, default=_json_default),
            ),
        )
        return self._enqueue("audit", event.event_type.value, event.model_dump(mode="json"))

    def _enqueue(self, kind: str, event_type: str, payload: dict[str, Any]) -> str:
        outbox_id = new_id("obx")
        self._conn.execute(
            "INSERT INTO outbox_events (outbox_id, event_kind, event_type, "
            "payload_json, created_at) VALUES (?,?,?,?,?)",
            (
                outbox_id,
                kind,
                event_type,
                json.dumps(payload, default=_json_default),
                utcnow().isoformat(),
            ),
        )
        return outbox_id


class ControlStore:
    """The control plane's durable state (§8.1-8.3).

    Usage::

        store = ControlStore(Path(".dex/control.db"))
        store.migrate()
        with store.transaction() as tx:
            tx.execute("UPDATE runs SET state = ? WHERE run_id = ?", ("queued", rid))
            tx.emit_metadata(event)
    """

    def __init__(self, path: Path, *, timeout: float = 30.0) -> None:
        self.path = path
        self._timeout = timeout
        # SQLite connections are not safe to share across threads, and the
        # runtime has a scheduler thread plus workers. One connection per
        # thread, created on demand.
        self._local = threading.local()
        self._write_lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    # --- connection handling ------------------------------------------------

    @property
    def _connection(self) -> sqlite3.Connection:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(
                self.path,
                timeout=self._timeout,
                # Transactions are managed explicitly below; autocommit mode
                # would defeat the outbox guarantee.
                isolation_level=None,
                check_same_thread=False,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode = WAL")
            # NORMAL is the documented-safe pairing with WAL: it can lose the
            # last commits on power loss but never corrupts, and FULL costs an
            # fsync per transaction that Lite mode does not need.
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute(f"PRAGMA busy_timeout = {int(self._timeout * 1000)}")
            self._local.conn = conn
        return conn

    def close(self) -> None:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # --- migrations (§16.6) -------------------------------------------------

    def migrate(self) -> int:
        """Apply pending migrations in order. Returns the resulting version.

        Each migration runs in its own transaction, so a failure leaves the
        database at the last fully-applied version rather than half-migrated.
        """
        conn = self._connection
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version     INTEGER PRIMARY KEY,
                name        TEXT NOT NULL,
                applied_at  TEXT NOT NULL
            )
            """
        )
        applied = {
            int(row["version"]) for row in conn.execute("SELECT version FROM schema_migrations")
        }

        for migration in sorted(MIGRATIONS, key=lambda m: m.version):
            if migration.version in applied:
                continue
            self._apply(conn, migration)

        return self.schema_version

    def _apply(self, conn: sqlite3.Connection, migration: Migration) -> None:
        with self._write_lock:
            conn.execute("BEGIN IMMEDIATE")
            try:
                for statement in migration.statements:
                    conn.execute(statement)
                conn.execute(
                    "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
                    (migration.version, migration.name, utcnow().isoformat()),
                )
            except Exception as exc:
                conn.execute("ROLLBACK")
                raise StoreError(
                    f"migration {migration.version} ({migration.name}) failed: {exc}"
                ) from exc
            conn.execute("COMMIT")

    @property
    def schema_version(self) -> int:
        row = self._connection.execute(
            "SELECT COALESCE(MAX(version), 0) AS v FROM schema_migrations"
        ).fetchone()
        return int(row["v"])

    # --- transactions and the outbox (§8.3) --------------------------------

    @contextmanager
    def transaction(self) -> Iterator[Transaction]:
        """Run a write transaction. State and events commit together.

        ``BEGIN IMMEDIATE`` takes the write lock up front so a busy database
        fails fast instead of after the caller has done its work.
        """
        conn = self._connection
        with self._write_lock:
            conn.execute("BEGIN IMMEDIATE")
            tx = Transaction(conn)
            try:
                yield tx
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        """Read rows outside a transaction. WAL allows this during writes."""
        return list(self._connection.execute(sql, params))

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        row: sqlite3.Row | None = self._connection.execute(sql, params).fetchone()
        return row

    # --- outbox dispatch ----------------------------------------------------

    def pending_outbox(self, limit: int = 100) -> list[OutboxRecord]:
        """Undispatched events, oldest first."""
        rows = self.query(
            "SELECT outbox_id, event_kind, event_type, payload_json, created_at, "
            "attempts FROM outbox_events WHERE dispatched_at IS NULL "
            "ORDER BY created_at LIMIT ?",
            (limit,),
        )
        return [
            OutboxRecord(
                outbox_id=row["outbox_id"],
                event_kind=row["event_kind"],
                event_type=row["event_type"],
                payload=json.loads(row["payload_json"]),
                created_at=row["created_at"],
                attempts=int(row["attempts"]),
            )
            for row in rows
        ]

    def mark_dispatched(self, outbox_ids: Sequence[str]) -> None:
        if not outbox_ids:
            return
        now = utcnow().isoformat()
        with self.transaction() as tx:
            for outbox_id in outbox_ids:
                tx.execute(
                    "UPDATE outbox_events SET dispatched_at = ? WHERE outbox_id = ?",
                    (now, outbox_id),
                )

    def mark_dispatch_failed(self, outbox_id: str, error: str) -> None:
        """Record a failed dispatch without marking the event delivered.

        The row stays pending so the next drain retries it. ``attempts`` grows
        so a permanently-failing sink can be spotted rather than retried
        forever in silence.
        """
        with self.transaction() as tx:
            tx.execute(
                "UPDATE outbox_events SET attempts = attempts + 1, last_error = ? "
                "WHERE outbox_id = ?",
                (error[:1000], outbox_id),
            )
