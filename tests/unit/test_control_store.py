"""Control store: migrations, transactional outbox, and schema guarantees.

The outbox tests are the point of this file. §8.3 exists to close a dual-write
hole, and a test that only checks "event was written" would pass just as
happily against the broken design. The tests here check that a rollback takes
the event *with* it.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from dataenginex.foundation import (
    AuditEvent,
    AuditEventType,
    EventEnvelope,
    MetadataEvent,
    PrincipalId,
    ProjectId,
)
from dataenginex.runtime.state import ControlStore, StoreError, latest_version

PROJECT = ProjectId("proj_test")
PRINCIPAL = PrincipalId("prin_test")
TS = "2026-08-03T00:00:00+00:00"


@pytest.fixture
def store(tmp_path: Path) -> Iterator[ControlStore]:
    with ControlStore(tmp_path / "control.db") as s:
        s.migrate()
        yield s


def make_metadata_event(event_type: str = "ResourceRegistered") -> MetadataEvent:
    return MetadataEvent(
        envelope=EventEnvelope(producer="test", project_id=PROJECT),
        event_type=event_type,
        subject_id="res_1",
        subject_type="resource",
        payload={"name": "customers"},
    )


def make_audit_event() -> AuditEvent:
    return AuditEvent(
        envelope=EventEnvelope(producer="test", project_id=PROJECT, principal_id=PRINCIPAL),
        event_type=AuditEventType.EXTERNAL_TRANSMISSION,
        action="email.send",
        outcome="permitted",
        destination="gmail.googleapis.com",
    )


def seed_project(store: ControlStore) -> None:
    """Insert the installation/workspace/project chain the FKs require."""
    with store.transaction() as tx:
        tx.execute(
            "INSERT INTO installations (installation_id, name, created_at) "
            "VALUES ('inst_1', 'test', ?)",
            (TS,),
        )
        tx.execute(
            "INSERT INTO workspaces (workspace_id, installation_id, name, created_at) "
            "VALUES ('ws_1', 'inst_1', 'default', ?)",
            (TS,),
        )
        tx.execute(
            "INSERT INTO projects (project_id, workspace_id, name, created_at) "
            "VALUES (?, 'ws_1', 'demo', ?)",
            (PROJECT, TS),
        )


def seed_revision(store: ControlStore) -> None:
    with store.transaction() as tx:
        tx.execute(
            "INSERT INTO project_revisions (revision_id, project_id, content_hash, "
            "created_by, created_at, manifest_schema_version, status) VALUES "
            "('rev_1', ?, 'sha256:x', ?, ?, 'dex/v1alpha1', 'published')",
            (PROJECT, PRINCIPAL, TS),
        )


# --- migrations ------------------------------------------------------------


def test_migrate_reaches_latest_version(store: ControlStore) -> None:
    assert store.schema_version == latest_version()


def test_migrate_is_idempotent(store: ControlStore) -> None:
    before = store.schema_version
    assert store.migrate() == before
    rows = store.query("SELECT version FROM schema_migrations")
    assert len(rows) == len({r["version"] for r in rows})


def test_all_schema_areas_exist(store: ControlStore) -> None:
    # §8.2 lists these by name; a missing one means a later block has nowhere
    # to write.
    expected = {
        "installations",
        "workspaces",
        "principals",
        "memberships",
        "projects",
        "project_revisions",
        "revision_files",
        "resources",
        "resource_versions",
        "resource_grants",
        "workload_definitions",
        "schedules",
        "triggers",
        "runs",
        "task_runs",
        "attempts",
        "queue_items",
        "workers",
        "worker_capabilities",
        "leases",
        "heartbeats",
        "policies",
        "policy_decisions",
        "approvals",
        "artifact_records",
        "checkpoint_records",
        "environment_records",
        "metadata_events",
        "audit_events",
        "outbox_events",
    }
    rows = store.query("SELECT name FROM sqlite_master WHERE type = 'table'")
    assert expected <= {r["name"] for r in rows}


def test_wal_mode_is_enabled(store: ControlStore) -> None:
    row = store.query_one("PRAGMA journal_mode")
    assert row is not None
    assert row[0].lower() == "wal"


def test_foreign_keys_are_enforced(store: ControlStore) -> None:
    with pytest.raises(sqlite3.IntegrityError), store.transaction() as tx:
        tx.execute(
            "INSERT INTO workspaces (workspace_id, installation_id, name, created_at) "
            "VALUES ('ws_x', 'inst_missing', 'orphan', ?)",
            (TS,),
        )


# --- transactional outbox (§8.3) -------------------------------------------


def test_event_and_state_commit_together(store: ControlStore) -> None:
    seed_project(store)
    with store.transaction() as tx:
        tx.execute(
            "INSERT INTO resources (resource_id, project_id, revision_id, "
            "resource_type, name, classification, lifecycle_state, created_at) "
            "VALUES ('res_1', ?, 'rev_1', 'dataset', 'customers', 'internal', "
            "'active', ?)",
            (PROJECT, TS),
        )
        tx.emit_metadata(make_metadata_event())

    assert store.query("SELECT 1 FROM resources WHERE resource_id = 'res_1'")
    assert len(store.pending_outbox()) == 1


def test_rollback_discards_the_event_too(store: ControlStore) -> None:
    # The dual-write hole: without a shared transaction, the event would
    # survive a failed state change and announce something that never happened.
    seed_project(store)
    with pytest.raises(RuntimeError, match="deliberate"), store.transaction() as tx:
        tx.execute(
            "INSERT INTO resources (resource_id, project_id, revision_id, "
            "resource_type, name, classification, lifecycle_state, created_at) "
            "VALUES ('res_2', ?, 'rev_1', 'dataset', 'orders', 'internal', "
            "'active', ?)",
            (PROJECT, TS),
        )
        tx.emit_metadata(make_metadata_event())
        raise RuntimeError("deliberate failure after both writes")

    assert store.query("SELECT 1 FROM resources WHERE resource_id = 'res_2'") == []
    assert store.pending_outbox() == []
    assert store.query("SELECT 1 FROM metadata_events") == []


def test_audit_event_reaches_outbox_and_table(store: ControlStore) -> None:
    seed_project(store)
    with store.transaction() as tx:
        tx.emit_audit(make_audit_event())

    assert len(store.query("SELECT 1 FROM audit_events")) == 1
    pending = store.pending_outbox()
    assert len(pending) == 1
    assert pending[0].event_kind == "audit"
    assert pending[0].payload["action"] == "email.send"


def test_dispatch_marks_events_delivered(store: ControlStore) -> None:
    seed_project(store)
    with store.transaction() as tx:
        tx.emit_metadata(make_metadata_event())

    pending = store.pending_outbox()
    store.mark_dispatched([p.outbox_id for p in pending])
    assert store.pending_outbox() == []


def test_failed_dispatch_stays_pending(store: ControlStore) -> None:
    # A sink that is down must not cause silent event loss.
    seed_project(store)
    with store.transaction() as tx:
        tx.emit_metadata(make_metadata_event())

    pending = store.pending_outbox()
    store.mark_dispatch_failed(pending[0].outbox_id, "connection refused")

    retried = store.pending_outbox()
    assert len(retried) == 1
    assert retried[0].attempts == 1


def test_outbox_drains_in_creation_order(store: ControlStore) -> None:
    seed_project(store)
    for i in range(5):
        with store.transaction() as tx:
            tx.emit_metadata(make_metadata_event(f"Event{i}"))

    pending = store.pending_outbox()
    assert [p.event_type for p in pending] == [f"Event{i}" for i in range(5)]


# --- invariant 9: audit events are append-only ------------------------------


def test_audit_events_cannot_be_updated(store: ControlStore) -> None:
    seed_project(store)
    with store.transaction() as tx:
        tx.emit_audit(make_audit_event())

    with pytest.raises(sqlite3.IntegrityError, match="append-only"), store.transaction() as tx:
        tx.execute("UPDATE audit_events SET outcome = 'denied'")


def test_audit_events_cannot_be_deleted(store: ControlStore) -> None:
    seed_project(store)
    with store.transaction() as tx:
        tx.emit_audit(make_audit_event())

    with pytest.raises(sqlite3.IntegrityError, match="append-only"), store.transaction() as tx:
        tx.execute("DELETE FROM audit_events")


# --- invariant 4: artifacts are not silently overwritten --------------------


def test_one_location_cannot_hold_two_digests(store: ControlStore) -> None:
    seed_project(store)
    with store.transaction() as tx:
        tx.execute(
            "INSERT INTO artifact_records (artifact_id, project_id, revision_id, "
            "logical_name, digest, size_bytes, media_type, provider, provider_uri, "
            "classification, retention_state, created_at) VALUES "
            "('art_1', ?, 'rev_1', 'silver', 'sha256:aaa', 10, 'application/parquet',"
            " 'filesystem', 'file:///a', 'internal', 'active', ?)",
            (PROJECT, TS),
        )

    with pytest.raises(sqlite3.IntegrityError), store.transaction() as tx:
        tx.execute(
            "INSERT INTO artifact_records (artifact_id, project_id, revision_id, "
            "logical_name, digest, size_bytes, media_type, provider, provider_uri, "
            "classification, retention_state, created_at) VALUES "
            "('art_2', ?, 'rev_1', 'silver', 'sha256:bbb', 10, 'application/parquet',"
            " 'filesystem', 'file:///a', 'internal', 'active', ?)",
            (PROJECT, TS),
        )


# --- §13.4: idempotency keys deduplicate submissions ------------------------


def test_duplicate_idempotency_key_is_rejected(store: ControlStore) -> None:
    seed_project(store)
    seed_revision(store)
    with store.transaction() as tx:
        tx.execute(
            "INSERT INTO runs (run_id, project_id, revision_id, workload_name, kind, "
            "state, trigger_type, requested_by, created_at, idempotency_key) VALUES "
            "('run_1', ?, 'rev_1', 'clean', 'batch', 'requested', 'manual', ?, ?, "
            "'key-1')",
            (PROJECT, PRINCIPAL, TS),
        )

    with pytest.raises(sqlite3.IntegrityError), store.transaction() as tx:
        tx.execute(
            "INSERT INTO runs (run_id, project_id, revision_id, workload_name, kind, "
            "state, trigger_type, requested_by, created_at, idempotency_key) VALUES "
            "('run_2', ?, 'rev_1', 'clean', 'batch', 'requested', 'manual', ?, ?, "
            "'key-1')",
            (PROJECT, PRINCIPAL, TS),
        )


def test_null_idempotency_keys_do_not_collide(store: ControlStore) -> None:
    # A partial index must let unkeyed runs coexist.
    seed_project(store)
    seed_revision(store)
    with store.transaction() as tx:
        for run_id in ("run_a", "run_b"):
            tx.execute(
                "INSERT INTO runs (run_id, project_id, revision_id, workload_name, "
                "kind, state, trigger_type, requested_by, created_at) VALUES "
                "(?, ?, 'rev_1', 'clean', 'batch', 'requested', 'manual', ?, ?)",
                (run_id, PROJECT, PRINCIPAL, TS),
            )
    assert len(store.query("SELECT 1 FROM runs")) == 2


# --- run/attempt separation (§4.10) -----------------------------------------


def test_retries_insert_attempts_rather_than_overwriting(store: ControlStore) -> None:
    seed_project(store)
    seed_revision(store)
    with store.transaction() as tx:
        tx.execute(
            "INSERT INTO runs (run_id, project_id, revision_id, workload_name, kind, "
            "state, trigger_type, requested_by, created_at) VALUES "
            "('run_1', ?, 'rev_1', 'train', 'batch', 'running', 'manual', ?, ?)",
            (PROJECT, PRINCIPAL, TS),
        )
        for number, state in ((1, "failed"), (2, "succeeded")):
            tx.execute(
                "INSERT INTO attempts (attempt_id, run_id, project_id, revision_id, "
                "attempt_number, state, principal_id) VALUES (?, 'run_1', ?, 'rev_1',"
                " ?, ?, ?)",
                (f"att_{number}", PROJECT, number, state, PRINCIPAL),
            )

    attempts = store.query(
        "SELECT state FROM attempts WHERE run_id = 'run_1' ORDER BY attempt_number"
    )
    # The failed attempt is still there — that is the whole point of the split.
    assert [a["state"] for a in attempts] == ["failed", "succeeded"]


def test_attempt_numbers_are_unique_per_run(store: ControlStore) -> None:
    seed_project(store)
    seed_revision(store)
    with store.transaction() as tx:
        tx.execute(
            "INSERT INTO runs (run_id, project_id, revision_id, workload_name, kind, "
            "state, trigger_type, requested_by, created_at) VALUES "
            "('run_1', ?, 'rev_1', 'train', 'batch', 'running', 'manual', ?, ?)",
            (PROJECT, PRINCIPAL, TS),
        )
        tx.execute(
            "INSERT INTO attempts (attempt_id, run_id, project_id, revision_id, "
            "attempt_number, state, principal_id) VALUES ('att_1', 'run_1', ?, "
            "'rev_1', 1, 'running', ?)",
            (PROJECT, PRINCIPAL),
        )

    with pytest.raises(sqlite3.IntegrityError), store.transaction() as tx:
        tx.execute(
            "INSERT INTO attempts (attempt_id, run_id, project_id, revision_id, "
            "attempt_number, state, principal_id) VALUES ('att_dup', 'run_1', ?, "
            "'rev_1', 1, 'running', ?)",
            (PROJECT, PRINCIPAL),
        )


# --- failure handling -------------------------------------------------------


def test_broken_migration_raises_store_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataenginex.runtime.state import migrations as mig
    from dataenginex.runtime.state import store as store_module

    # store.py binds MIGRATIONS at import, so patch the name it actually reads.
    monkeypatch.setattr(
        store_module,
        "MIGRATIONS",
        (mig.Migration(1, "broken", ("CREATE TABLE ((;",)),),
    )
    store = ControlStore(tmp_path / "broken.db")
    try:
        with pytest.raises(StoreError, match="migration 1"):
            store.migrate()
    finally:
        store.close()


def test_store_closes_cleanly(tmp_path: Path) -> None:
    with ControlStore(tmp_path / "c.db") as store:
        store.migrate()
        assert store.schema_version == latest_version()
