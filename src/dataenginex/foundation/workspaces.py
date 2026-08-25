"""Workspace types (§4.3).

A workspace is an administrative and sharing boundary. It owns members,
shared policies, resource budgets, cross-project grants, and projects.
A single personal installation typically has one default workspace.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field

from dataenginex.foundation.ids import (
    InstallationId,
    PrincipalId,
    ProjectId,
    WorkspaceId,
    new_id,
)
from dataenginex.foundation.projects import FrozenModel, utcnow

__all__ = [
    "Workspace",
    "WorkspaceMember",
    "WorkspaceMemberRole",
]


class WorkspaceMemberRole(StrEnum):
    """Roles within a workspace."""

    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


class WorkspaceMember(FrozenModel):
    """A principal with a role in a workspace (§4.3)."""

    principal_id: PrincipalId
    role: WorkspaceMemberRole = WorkspaceMemberRole.MEMBER
    granted_at: datetime = Field(default_factory=utcnow)
    granted_by: PrincipalId | None = None


class Workspace(FrozenModel):
    """Administrative and sharing boundary (§4.3).

    Owns members, shared policies, resource budgets, cross-project grants,
    and projects. A personal installation typically has exactly one.
    """

    workspace_id: WorkspaceId = Field(default_factory=lambda: WorkspaceId(new_id("ws")))
    installation_id: InstallationId
    name: str
    description: str = ""
    created_at: datetime = Field(default_factory=utcnow)
    members: tuple[WorkspaceMember, ...] = ()
    # Budget keys are resource dimensions ("cpu_seconds", "memory_mb", "egress_bytes").
    budgets: dict[str, int] = Field(default_factory=dict)
    # Cross-project resource grants: project_id -> tuple of granted project IDs
    project_grants: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    # Shared policies applied to all projects in this workspace
    shared_policy_ids: tuple[str, ...] = ()
    # Projects belonging to this workspace
    project_ids: tuple[ProjectId, ...] = ()
