"""Tests for runtime control plane: coordinator and recovery."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from dataenginex.foundation.ids import AttemptId, RunId, new_id
from dataenginex.foundation.workloads import RunState
from dataenginex.runtime.control_plane.coordinator import ControlPlaneCoordinator
from dataenginex.runtime.control_plane.recovery import RecoveryManager
from dataenginex.runtime.state import ControlStore


@pytest.fixture()
def store(tmp_path: Path) -> Iterator[ControlStore]:
    with ControlStore(tmp_path / "control.db") as s:
        s.migrate()
        yield s


def _insert_run(
    store: ControlStore, run_id: str, state: str = "queued",
    project_id: str = "proj_1", revision_id: str = "rev_1",
    kind: str = "batch", attempt_count: int = 0,
) -> None:
    from dataenginex.foundation.projects import utcnow
    now = utcnow().isoformat()
    # Ensure full FK chain exists
    store.query_one(
        "INSERT OR IGNORE INTO installations "
        "(installation_id, name, created_at) VALUES (?, ?, ?)",
        ("inst_test", "test", now),
    )
    store.query_one(
        "INSERT OR IGNORE INTO workspaces "
        "(workspace_id, installation_id, name, created_at) "
        "VALUES (?, ?, ?, ?)",
        ("ws_test", "inst_test", "default", now),
    )
    store.query_one(
        "INSERT OR IGNORE INTO projects "
        "(project_id, workspace_id, name, created_at) "
        "VALUES (?, ?, ?, ?)",
        (project_id, "ws_test", "test_project", now),
    )
    store.query_one(
        "INSERT OR IGNORE INTO project_revisions "
        "(revision_id, project_id, content_hash, created_by, "
        "created_at, manifest_schema_version, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (revision_id, project_id, "abc123", "test", now, "dex/v0.7", "published"),
    )
    store.query_one(
        "INSERT INTO runs "
        "(run_id, project_id, revision_id, workload_name, "
        "state, kind, trigger_type, requested_by, "
        "created_at, attempt_count) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, project_id, revision_id, "test_workload",
         state, kind, "manual", "test", now, attempt_count),
    )


class TestControlPlaneCoordinator:
    def test_admit_run(self, store: ControlStore) -> None:
        run_id = RunId(new_id("run"))
        _insert_run(store, run_id, state="requested")
        coord = ControlPlaneCoordinator(store)
        result = coord.admit_run(run_id)
        assert result == RunState.AWAITING_POLICY

    def test_admit_run_invalid_state(self, store: ControlStore) -> None:
        run_id = RunId(new_id("run"))
        _insert_run(store, run_id, state="completed")
        coord = ControlPlaneCoordinator(store)
        with pytest.raises(RuntimeError, match="cannot be admitted"):
            coord.admit_run(run_id)

    def test_admit_run_not_found(self, store: ControlStore) -> None:
        coord = ControlPlaneCoordinator(store)
        with pytest.raises(RuntimeError, match="not found"):
            coord.admit_run(RunId("run_nonexistent"))

    def test_complete_policy_approved(self, store: ControlStore) -> None:
        run_id = RunId(new_id("run"))
        _insert_run(store, run_id, state="awaiting_policy")
        coord = ControlPlaneCoordinator(store)
        result = coord.complete_policy(run_id, approved=True)
        assert result == RunState.PLANNING

    def test_complete_policy_rejected(self, store: ControlStore) -> None:
        run_id = RunId(new_id("run"))
        _insert_run(store, run_id, state="awaiting_policy")
        coord = ControlPlaneCoordinator(store)
        result = coord.complete_policy(run_id, approved=False)
        assert result == RunState.FAILED

    def test_enqueue_run(self, store: ControlStore) -> None:
        run_id = RunId(new_id("run"))
        _insert_run(store, run_id, state="planning")
        coord = ControlPlaneCoordinator(store)
        result = coord.enqueue_run(run_id)
        assert result == RunState.QUEUED

    def test_claim_run(self, store: ControlStore) -> None:
        run_id = RunId(new_id("run"))
        _insert_run(store, run_id, state="queued")
        coord = ControlPlaneCoordinator(store)
        attempt_id = AttemptId(new_id("att"))
        result = coord.claim_run(run_id, "worker_1", attempt_id)
        assert result == RunState.LEASED

    def test_start_execution(self, store: ControlStore) -> None:
        run_id = RunId(new_id("run"))
        _insert_run(store, run_id, state="leased")
        coord = ControlPlaneCoordinator(store)
        result = coord.start_execution(run_id)
        assert result == RunState.RUNNING

    def test_commit_outputs(self, store: ControlStore) -> None:
        run_id = RunId(new_id("run"))
        _insert_run(store, run_id, state="running")
        coord = ControlPlaneCoordinator(store)
        result = coord.commit_outputs(run_id)
        assert result == RunState.COMPLETED

    def test_fail_run(self, store: ControlStore) -> None:
        run_id = RunId(new_id("run"))
        _insert_run(store, run_id, state="running")
        coord = ControlPlaneCoordinator(store)
        result = coord.fail_run(run_id, "something broke")
        assert result == RunState.FAILED

    def test_cancel_run(self, store: ControlStore) -> None:
        run_id = RunId(new_id("run"))
        _insert_run(store, run_id, state="queued")
        coord = ControlPlaneCoordinator(store)
        result = coord.cancel_run(run_id)
        assert result == RunState.CANCELLED


class TestRecoveryManager:
    def test_recover_empty_store(self, store: ControlStore) -> None:
        mgr = RecoveryManager(store)
        stats = mgr.recover()
        assert stats["outbox_replayed"] == 0
        assert stats["leases_expired"] == 0
        assert stats["attempts_reclaimed"] == 0
        assert stats["streams_restarted"] == 0
        assert stats["revisions_verified"] == 0

    def test_replay_outbox(self, store: ControlStore) -> None:
        mgr = RecoveryManager(store)
        stats = mgr.recover()
        assert stats["outbox_replayed"] == 0

    def test_restart_streams_empty(self, store: ControlStore) -> None:
        mgr = RecoveryManager(store)
        stats = mgr.recover()
        assert stats["streams_restarted"] == 0

    def test_verify_revision_refs_empty(self, store: ControlStore) -> None:
        mgr = RecoveryManager(store)
        stats = mgr.recover()
        assert stats["revisions_verified"] == 0
