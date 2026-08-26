"""Workspace service (§4.3).

Manages workspace lifecycle, members, and cross-project grants.
"""

from __future__ import annotations

from dataenginex.application.services import Service
from dataenginex.foundation.ids import InstallationId, PrincipalId, ProjectId, WorkspaceId
from dataenginex.foundation.projects import utcnow
from dataenginex.foundation.workspaces import Workspace, WorkspaceMember, WorkspaceMemberRole

__all__ = ["WorkspaceService", "WorkspaceView"]


class WorkspaceView:
    """Read-only projection of a workspace for APIs/UI."""

    def __init__(self, ws: Workspace) -> None:
        self.workspace_id = ws.workspace_id
        self.name = ws.name
        self.description = ws.description
        self.member_count = len(ws.members)
        self.project_count = len(ws.project_ids)
        self.budgets = dict(ws.budgets)


class WorkspaceService(Service):
    """Manages workspace CRUD and membership (§4.3)."""

    def create_workspace(
        self,
        name: str,
        owner_id: PrincipalId,
        *,
        description: str = "",
    ) -> Workspace:
        """Create a workspace with the given principal as owner."""
        ws = Workspace(
            installation_id=self._installation_id(),
            name=name,
            description=description,
            members=(
                WorkspaceMember(
                    principal_id=owner_id,
                    role=WorkspaceMemberRole.OWNER,
                ),
            ),
        )
        self.store.query_one(
            "INSERT INTO workspaces (workspace_id, installation_id, name, created_at) "
            "VALUES (?, ?, ?, ?)",
            (ws.workspace_id, ws.installation_id, ws.name, utcnow().isoformat()),
        )
        return ws

    def get_workspace(self, workspace_id: WorkspaceId) -> Workspace:
        """Fetch a workspace by ID."""
        row = self.require_row(
            "SELECT * FROM workspaces WHERE workspace_id = ?",
            (workspace_id,),
            subject=f"workspace {workspace_id}",
        )
        return Workspace(
            workspace_id=row["workspace_id"],
            installation_id=row["installation_id"],
            name=row["name"],
        )

    def list_workspaces(self) -> list[WorkspaceView]:
        """List all workspaces."""
        rows = self.store.query("SELECT * FROM workspaces ORDER BY name")
        return [
            WorkspaceView(
                Workspace(
                    workspace_id=r["workspace_id"],
                    installation_id=r["installation_id"],
                    name=r["name"],
                )
            )
            for r in rows
        ]

    def add_member(
        self,
        workspace_id: WorkspaceId,
        principal_id: PrincipalId,
        role: WorkspaceMemberRole = WorkspaceMemberRole.MEMBER,
    ) -> None:
        """Add a member to a workspace."""
        from dataenginex.foundation.projects import utcnow
        self.store.query_one(
            "INSERT OR IGNORE INTO memberships (workspace_id, principal_id, role, created_at) "
            "VALUES (?, ?, ?, ?)",
            (workspace_id, principal_id, role.value, utcnow().isoformat()),
        )

    def remove_member(self, workspace_id: WorkspaceId, principal_id: PrincipalId) -> None:
        """Remove a member from a workspace."""
        self.store.query_one(
            "DELETE FROM memberships WHERE workspace_id = ? AND principal_id = ?",
            (workspace_id, principal_id),
        )

    def assign_project(self, workspace_id: WorkspaceId, project_id: ProjectId) -> None:
        """Assign a project to a workspace."""
        self.store.query_one(
            "UPDATE projects SET workspace_id = ? WHERE project_id = ?",
            (workspace_id, project_id),
        )

    def _installation_id(self) -> InstallationId:
        """Get the current installation ID."""
        row = self.store.query_one("SELECT installation_id FROM installations LIMIT 1", ())
        return InstallationId(row["installation_id"] if row else "inst_default")
