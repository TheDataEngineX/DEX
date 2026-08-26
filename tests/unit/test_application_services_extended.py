"""Tests for application services: approval, catalog, export, workspace."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from dataenginex.application.approval_service import (
    ApprovalAlreadyDecided,
    ApprovalService,
)
from dataenginex.application.catalog_service import CatalogService
from dataenginex.application.export_service import ExportService, ImportError
from dataenginex.application.workspace_service import WorkspaceService
from dataenginex.foundation.ids import PrincipalId, ProjectId, new_id
from dataenginex.foundation.projects import utcnow
from dataenginex.foundation.resources import LifecycleState
from dataenginex.runtime.state import ControlStore


@pytest.fixture()
def store(tmp_path: Path) -> Iterator[ControlStore]:
    with ControlStore(tmp_path / "control.db") as s:
        s.migrate()
        now = utcnow().isoformat()
        s.query_one(
            "INSERT INTO installations "
            "(installation_id, name, created_at) VALUES (?, ?, ?)",
            ("inst_test", "test", now),
        )
        s.query_one(
            "INSERT INTO workspaces "
            "(workspace_id, installation_id, name, created_at) "
            "VALUES (?, ?, ?, ?)",
            ("ws_test", "inst_test", "default", now),
        )
        yield s


class TestApprovalService:
    def test_request_approval(self, store: ControlStore) -> None:
        svc = ApprovalService(store)
        project_id = ProjectId(new_id("proj"))
        store.query_one(
            "INSERT INTO projects (project_id, workspace_id, name, created_at) VALUES (?, ?, ?, ?)",
            (project_id, "ws_test", "test", utcnow().isoformat()),
        )
        approval_id = svc.request_approval(
            project_id=project_id,
            action="deploy",
            requested_by=PrincipalId("user:alice"),
        )
        assert str(approval_id).startswith("apr_")

    def test_approve_and_get(self, store: ControlStore) -> None:
        svc = ApprovalService(store)
        project_id = ProjectId(new_id("proj"))
        store.query_one(
            "INSERT INTO projects (project_id, workspace_id, name, created_at) VALUES (?, ?, ?, ?)",
            (project_id, "ws_test", "test", utcnow().isoformat()),
        )
        approval_id = svc.request_approval(
            project_id=project_id,
            action="deploy",
            requested_by=PrincipalId("user:alice"),
        )
        svc.approve(approval_id, PrincipalId("user:bob"))
        view = svc.get_approval(approval_id)
        assert view.state == "approved"
        assert view.decided_by == "user:bob"

    def test_approve_already_decided(self, store: ControlStore) -> None:
        svc = ApprovalService(store)
        project_id = ProjectId(new_id("proj"))
        store.query_one(
            "INSERT INTO projects (project_id, workspace_id, name, created_at) VALUES (?, ?, ?, ?)",
            (project_id, "ws_test", "test", utcnow().isoformat()),
        )
        approval_id = svc.request_approval(
            project_id=project_id,
            action="deploy",
            requested_by=PrincipalId("user:alice"),
        )
        svc.approve(approval_id, PrincipalId("user:bob"))
        with pytest.raises(ApprovalAlreadyDecided):
            svc.approve(approval_id, PrincipalId("user:charlie"))

    def test_reject(self, store: ControlStore) -> None:
        svc = ApprovalService(store)
        project_id = ProjectId(new_id("proj"))
        store.query_one(
            "INSERT INTO projects (project_id, workspace_id, name, created_at) VALUES (?, ?, ?, ?)",
            (project_id, "ws_test", "test", utcnow().isoformat()),
        )
        approval_id = svc.request_approval(
            project_id=project_id,
            action="deploy",
            requested_by=PrincipalId("user:alice"),
        )
        svc.reject(approval_id, PrincipalId("user:bob"), reason="not ready")
        view = svc.get_approval(approval_id)
        assert view.state == "rejected"

    def test_reject_already_decided(self, store: ControlStore) -> None:
        svc = ApprovalService(store)
        project_id = ProjectId(new_id("proj"))
        store.query_one(
            "INSERT INTO projects (project_id, workspace_id, name, created_at) VALUES (?, ?, ?, ?)",
            (project_id, "ws_test", "test", utcnow().isoformat()),
        )
        approval_id = svc.request_approval(
            project_id=project_id,
            action="deploy",
            requested_by=PrincipalId("user:alice"),
        )
        svc.reject(approval_id, PrincipalId("user:bob"))
        with pytest.raises(ApprovalAlreadyDecided):
            svc.reject(approval_id, PrincipalId("user:charlie"))

    def test_list_pending(self, store: ControlStore) -> None:
        svc = ApprovalService(store)
        project_id = ProjectId(new_id("proj"))
        store.query_one(
            "INSERT INTO projects (project_id, workspace_id, name, created_at) VALUES (?, ?, ?, ?)",
            (project_id, "ws_test", "test", utcnow().isoformat()),
        )
        svc.request_approval(
            project_id=project_id,
            action="deploy",
            requested_by=PrincipalId("user:alice"),
        )
        pending = svc.list_pending(project_id)
        assert len(pending) == 1

    def test_list_pending_all(self, store: ControlStore) -> None:
        svc = ApprovalService(store)
        project_id = ProjectId(new_id("proj"))
        store.query_one(
            "INSERT INTO projects (project_id, workspace_id, name, created_at) VALUES (?, ?, ?, ?)",
            (project_id, "ws_test", "test", utcnow().isoformat()),
        )
        svc.request_approval(
            project_id=project_id,
            action="deploy",
            requested_by=PrincipalId("user:alice"),
        )
        pending = svc.list_pending()
        assert len(pending) >= 1

    def test_check_digest_match(self, store: ControlStore) -> None:
        svc = ApprovalService(store)
        project_id = ProjectId(new_id("proj"))
        store.query_one(
            "INSERT INTO projects (project_id, workspace_id, name, created_at) VALUES (?, ?, ?, ?)",
            (project_id, "ws_test", "test", utcnow().isoformat()),
        )
        approval_id = svc.request_approval(
            project_id=project_id,
            action="deploy",
            requested_by=PrincipalId("user:alice"),
            operation_digest="abc123",
        )
        assert svc.check_digest(approval_id, "abc123") is True

    def test_check_digest_mismatch(self, store: ControlStore) -> None:
        svc = ApprovalService(store)
        project_id = ProjectId(new_id("proj"))
        store.query_one(
            "INSERT INTO projects (project_id, workspace_id, name, created_at) VALUES (?, ?, ?, ?)",
            (project_id, "ws_test", "test", utcnow().isoformat()),
        )
        approval_id = svc.request_approval(
            project_id=project_id,
            action="deploy",
            requested_by=PrincipalId("user:alice"),
            operation_digest="abc123",
        )
        assert svc.check_digest(approval_id, "def456") is False

    def test_check_digest_no_digest(self, store: ControlStore) -> None:
        svc = ApprovalService(store)
        project_id = ProjectId(new_id("proj"))
        store.query_one(
            "INSERT INTO projects (project_id, workspace_id, name, created_at) VALUES (?, ?, ?, ?)",
            (project_id, "ws_test", "test", utcnow().isoformat()),
        )
        approval_id = svc.request_approval(
            project_id=project_id,
            action="deploy",
            requested_by=PrincipalId("user:alice"),
        )
        assert svc.check_digest(approval_id, "anything") is True


class TestCatalogService:
    def test_register_and_get(self, store: ControlStore) -> None:
        svc = CatalogService(store)
        project_id = ProjectId(new_id("proj"))
        store.query_one(
            "INSERT INTO projects (project_id, workspace_id, name, created_at) VALUES (?, ?, ?, ?)",
            (project_id, "ws_test", "test", utcnow().isoformat()),
        )
        resource_id = svc.register_resource(
            name="users",
            resource_type="dataset",
            project_id=project_id,
        )
        entry = svc.get_resource(resource_id)
        assert entry.name == "users"
        assert entry.resource_type == "dataset"

    def test_search_resources(self, store: ControlStore) -> None:
        svc = CatalogService(store)
        project_id = ProjectId(new_id("proj"))
        store.query_one(
            "INSERT INTO projects (project_id, workspace_id, name, created_at) VALUES (?, ?, ?, ?)",
            (project_id, "ws_test", "test", utcnow().isoformat()),
        )
        svc.register_resource(name="ds1", resource_type="dataset", project_id=project_id)
        svc.register_resource(name="m1", resource_type="model", project_id=project_id)
        results = svc.search_resources(project_id=project_id, resource_type="dataset")
        assert len(results) == 1
        assert results[0].resource_type == "dataset"

    def test_list_resources(self, store: ControlStore) -> None:
        svc = CatalogService(store)
        project_id = ProjectId(new_id("proj"))
        store.query_one(
            "INSERT INTO projects (project_id, workspace_id, name, created_at) VALUES (?, ?, ?, ?)",
            (project_id, "ws_test", "test", utcnow().isoformat()),
        )
        svc.register_resource(name="ds1", resource_type="dataset", project_id=project_id)
        results = svc.list_resources(project_id)
        assert len(results) == 1

    def test_update_resource(self, store: ControlStore) -> None:
        svc = CatalogService(store)
        project_id = ProjectId(new_id("proj"))
        store.query_one(
            "INSERT INTO projects (project_id, workspace_id, name, created_at) VALUES (?, ?, ?, ?)",
            (project_id, "ws_test", "test", utcnow().isoformat()),
        )
        resource_id = svc.register_resource(
            name="ds1", resource_type="dataset", project_id=project_id,
        )
        svc.update_resource(resource_id, classification="confidential")
        entry = svc.get_resource(resource_id)
        assert entry.classification == "confidential"

    def test_update_resource_no_changes(self, store: ControlStore) -> None:
        svc = CatalogService(store)
        project_id = ProjectId(new_id("proj"))
        store.query_one(
            "INSERT INTO projects (project_id, workspace_id, name, created_at) VALUES (?, ?, ?, ?)",
            (project_id, "ws_test", "test", utcnow().isoformat()),
        )
        resource_id = svc.register_resource(
            name="ds1", resource_type="dataset", project_id=project_id,
        )
        svc.update_resource(resource_id)

    def test_delete_resource(self, store: ControlStore) -> None:
        svc = CatalogService(store)
        project_id = ProjectId(new_id("proj"))
        store.query_one(
            "INSERT INTO projects (project_id, workspace_id, name, created_at) VALUES (?, ?, ?, ?)",
            (project_id, "ws_test", "test", utcnow().isoformat()),
        )
        resource_id = svc.register_resource(
            name="ds1", resource_type="dataset", project_id=project_id,
        )
        svc.delete_resource(resource_id)
        entry = svc.get_resource(resource_id)
        assert entry.lifecycle_state == LifecycleState.DELETED.value


class TestExportService:
    def test_export_and_import(self, store: ControlStore, tmp_path: Path) -> None:
        svc = ExportService(store)
        project_id = ProjectId(new_id("proj"))
        store.query_one(
            "INSERT INTO projects "
            "(project_id, workspace_id, name, description, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (project_id, "ws_test", "my_project", "test desc", utcnow().isoformat()),
        )
        revision_id = new_id("rev")
        store.query_one(
            "INSERT INTO project_revisions "
            "(revision_id, project_id, content_hash, created_by, "
            "created_at, manifest_schema_version, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (revision_id, project_id, "abc123", "test",
             utcnow().isoformat(), "dex/v0.7", "published"),
        )
        store.query_one(
            "UPDATE projects SET active_revision_id = ? WHERE project_id = ?",
            (revision_id, project_id),
        )

        output_dir = tmp_path / "export"
        output_dir.mkdir()
        project_dir = svc.export_project(project_id, output_dir)
        assert (project_dir / "dex.yaml").exists()
        assert (project_dir / "revision.json").exists()

        # Create target workspace for import
        store.query_one(
            "INSERT OR IGNORE INTO workspaces "
            "(workspace_id, installation_id, name, created_at) "
            "VALUES (?, ?, ?, ?)",
            ("ws_export", "inst_test", "export_ws", utcnow().isoformat()),
        )
        imported_id = svc.import_project(project_dir, workspace_id="ws_export")
        assert str(imported_id).startswith("proj_")

    def test_import_missing_manifest(self, store: ControlStore, tmp_path: Path) -> None:
        svc = ExportService(store)
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(ImportError, match="No dex.yaml"):
            svc.import_project(empty_dir, workspace_id="ws_1")


class TestWorkspaceService:
    def test_create_workspace(self, store: ControlStore) -> None:
        svc = WorkspaceService(store)
        ws = svc.create_workspace("analytics", PrincipalId("user:alice"))
        assert ws.name == "analytics"

    def test_get_workspace(self, store: ControlStore) -> None:
        svc = WorkspaceService(store)
        ws = svc.create_workspace("analytics", PrincipalId("user:alice"))
        fetched = svc.get_workspace(ws.workspace_id)
        assert fetched.name == "analytics"

    def test_list_workspaces(self, store: ControlStore) -> None:
        svc = WorkspaceService(store)
        svc.create_workspace("ws1", PrincipalId("user:alice"))
        svc.create_workspace("ws2", PrincipalId("user:bob"))
        views = svc.list_workspaces()
        assert len(views) >= 2  # fixture creates ws_test too

    def test_add_remove_member(self, store: ControlStore) -> None:
        svc = WorkspaceService(store)
        ws = svc.create_workspace("analytics", PrincipalId("user:alice"))
        # Insert the principal first (FK requirement)
        store.query_one(
            "INSERT OR IGNORE INTO principals "
            "(principal_id, principal_type, name, display_name, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("user:bob", "user", "bob", "Bob", utcnow().isoformat()),
        )
        svc.add_member(ws.workspace_id, PrincipalId("user:bob"))
        svc.remove_member(ws.workspace_id, PrincipalId("user:bob"))

    def test_assign_project(self, store: ControlStore) -> None:
        svc = WorkspaceService(store)
        ws = svc.create_workspace("analytics", PrincipalId("user:alice"))
        project_id = ProjectId(new_id("proj"))
        store.query_one(
            "INSERT INTO projects (project_id, workspace_id, name, created_at) VALUES (?, ?, ?, ?)",
            (project_id, "ws_test", "test", utcnow().isoformat()),
        )
        svc.assign_project(ws.workspace_id, project_id)
