"""Tests for v0.7 Foundation types: Workspace and SecretReference."""

import pytest

from dataenginex.foundation.ids import InstallationId, new_id
from dataenginex.foundation.secrets import SecretReferenceV2, SecretRotationPolicy
from dataenginex.foundation.workspaces import (
    Workspace,
    WorkspaceMember,
    WorkspaceMemberRole,
)


class TestWorkspace:
    def test_workspace_creation(self) -> None:
        ws = Workspace(
            installation_id=InstallationId(new_id("inst")),
            name="analytics",
        )
        assert ws.name == "analytics"
        assert ws.members == ()

    def test_workspace_frozen(self) -> None:
        ws = Workspace(
            installation_id=InstallationId(new_id("inst")),
            name="analytics",
        )
        with pytest.raises(Exception, match="frozen"):
            ws.name = "changed"  # type: ignore[misc]

    def test_workspace_with_members(self) -> None:
        member = WorkspaceMember(
            principal_id="user:alice",
            role=WorkspaceMemberRole.MEMBER,
        )
        ws = Workspace(
            installation_id=InstallationId(new_id("inst")),
            name="analytics",
            members=(member,),
        )
        assert len(ws.members) == 1
        assert ws.members[0].principal_id == "user:alice"


class TestWorkspaceMemberRole:
    def test_roles_are_distinct(self) -> None:
        assert WorkspaceMemberRole.OWNER != WorkspaceMemberRole.MEMBER
        assert WorkspaceMemberRole.MEMBER != WorkspaceMemberRole.VIEWER
        assert WorkspaceMemberRole.OWNER != WorkspaceMemberRole.VIEWER


class TestSecretReferenceV2:
    def test_secret_reference_creation(self) -> None:
        ref = SecretReferenceV2(
            project_id="proj:123",
            name="db-password",
            provider="vault",
        )
        assert ref.name == "db-password"
        assert ref.provider == "vault"

    def test_secret_reference_frozen(self) -> None:
        ref = SecretReferenceV2(
            project_id="proj:123",
            name="db-password",
        )
        with pytest.raises(Exception, match="frozen"):
            ref.name = "changed"  # type: ignore[misc]

    def test_secret_rotation_policy(self) -> None:
        policy = SecretRotationPolicy.SCHEDULED
        assert policy == SecretRotationPolicy.SCHEDULED


class TestWorkspaceIntegration:
    def test_workspace_with_multiple_members(self) -> None:
        members = (
            WorkspaceMember(principal_id="user:alice", role=WorkspaceMemberRole.OWNER),
            WorkspaceMember(principal_id="user:bob", role=WorkspaceMemberRole.MEMBER),
            WorkspaceMember(principal_id="user:carol", role=WorkspaceMemberRole.VIEWER),
        )
        ws = Workspace(
            installation_id=InstallationId(new_id("inst")),
            name="ml-team",
            members=members,
        )
        assert len(ws.members) == 3
        roles = [m.role for m in ws.members]
        assert WorkspaceMemberRole.OWNER in roles
        assert WorkspaceMemberRole.VIEWER in roles
