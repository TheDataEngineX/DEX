"""Tests for v0.7 Application Services."""


from dataenginex.application.workspace_service import WorkspaceView
from dataenginex.foundation.workspaces import Workspace


class TestWorkspaceView:
    def test_workspace_view(self) -> None:
        ws = Workspace(
            installation_id="inst:123",
            name="analytics",
        )
        view = WorkspaceView(ws)
        assert view.name == "analytics"
        assert view.member_count == 0

    def test_workspace_view_with_members(self) -> None:
        from dataenginex.foundation.workspaces import WorkspaceMember, WorkspaceMemberRole
        ws = Workspace(
            installation_id="inst:123",
            name="analytics",
            members=(
                WorkspaceMember(
                    principal_id="user:alice",
                    role=WorkspaceMemberRole.OWNER,
                ),
                WorkspaceMember(
                    principal_id="user:bob",
                    role=WorkspaceMemberRole.MEMBER,
                ),
            ),
        )
        view = WorkspaceView(ws)
        assert view.member_count == 2
