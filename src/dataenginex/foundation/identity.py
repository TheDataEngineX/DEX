"""Principals and delegated capability tokens (§4.11, §9.4).

A principal is anything that can request or perform an action. The rule that
shapes this module: a workload executes under a *delegated* principal with fewer
permissions than the human who initiated it. An AI assistant asked to draft a
reply gets read access to the approved documents and nothing else — not the
mailbox, and not the send permission its operator holds.

Capability tokens make that narrowing explicit and short-lived. They bind to an
exact project, revision, run, operation, resource scope, action set, expiry, and
optional destination, so a leaked token is bounded in both blast radius and time.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import StrEnum

from pydantic import Field

from dataenginex.foundation.ids import (
    AttemptId,
    OperationId,
    PrincipalId,
    ProjectId,
    RevisionId,
    RunId,
    WorkspaceId,
    new_id,
)
from dataenginex.foundation.projects import FrozenModel, utcnow

__all__ = [
    "CapabilityToken",
    "Principal",
    "PrincipalType",
    "SecretReference",
    "TrustLevel",
    "issue_capability",
]


class PrincipalType(StrEnum):
    """Kinds of actor (§4.11)."""

    HUMAN = "human"
    OS_USER = "os_user"
    SERVICE_ACCOUNT = "service_account"
    WORKER = "worker"
    WORKLOAD = "workload"
    AGENT = "agent"
    CONNECTOR = "connector"
    DEVICE = "device"
    PLUGIN = "plugin"


class TrustLevel(StrEnum):
    """How much a principal's declarations are believed (§10.8).

    Applies mainly to plugins and agents: an untrusted plugin's manifest claims
    are validated against observed behavior rather than taken at face value.
    """

    UNTRUSTED = "untrusted"
    COMMUNITY = "community"
    VERIFIED = "verified"
    FIRST_PARTY = "first_party"


class Principal(FrozenModel):
    """An actor that can request or perform actions (§4.11).

    ``delegated_from`` records the chain back to the initiating human. Without
    it, an audit trail shows an agent acting on its own authority and loses the
    accountability link.
    """

    principal_id: PrincipalId = Field(default_factory=lambda: PrincipalId(new_id("prin")))
    principal_type: PrincipalType
    name: str
    display_name: str = ""
    trust_level: TrustLevel = TrustLevel.UNTRUSTED
    delegated_from: PrincipalId | None = None
    roles: tuple[str, ...] = ()
    created_at: datetime = Field(default_factory=utcnow)
    disabled: bool = False


class SecretReference(FrozenModel):
    """A pointer to a secret — never the secret itself (§9.6).

    Invariant 5: secret *values* never appear in revisions, metadata, logs,
    lineage, or exports. Only this reference does. There is deliberately no
    field on this type that could hold a value.
    """

    name: str
    project_id: ProjectId
    provider: str = "keyring"
    # Principals permitted to resolve this reference; empty means project-scoped
    # default rules apply.
    permitted_consumers: tuple[PrincipalId, ...] = ()
    rotation_due: datetime | None = None


class CapabilityToken(FrozenModel):
    """A short-lived, narrowly scoped grant (§9.4).

    Issued per attempt. The binding fields are not optional decoration: they are
    what lets the control plane reject a token replayed against a different run,
    a newer revision, or an undeclared destination.
    """

    token_id: str = Field(default_factory=lambda: new_id("cap"))
    principal_id: PrincipalId
    workspace_id: WorkspaceId | None = None
    project_id: ProjectId
    revision_id: RevisionId
    run_id: RunId | None = None
    attempt_id: AttemptId | None = None
    operation_id: OperationId | None = None
    operation_type: str | None = None
    # Resource IDs or glob patterns this token may touch.
    resource_scope: tuple[str, ...] = ()
    actions: tuple[str, ...] = ()
    # Network destinations permitted, checked against egress policy (§9.7).
    destinations: tuple[str, ...] = ()
    secret_refs: tuple[str, ...] = ()
    issued_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime

    def is_expired(self, now: datetime | None = None) -> bool:
        return (now or utcnow()) >= self.expires_at

    def permits(self, action: str, resource_id: str | None = None) -> bool:
        """Whether this token allows an action, ignoring expiry.

        Fails closed: an empty ``actions`` tuple grants nothing. Callers must
        check :meth:`is_expired` separately — keeping the time check out of here
        makes the expiry decision explicit at the call site rather than hidden
        inside a boolean.
        """
        if action not in self.actions:
            return False
        if resource_id is None:
            return True
        if not self.resource_scope:
            return False
        return any(
            scope == "*"
            or scope == resource_id
            or (scope.endswith("*") and resource_id.startswith(scope[:-1]))
            for scope in self.resource_scope
        )

    @classmethod
    def from_run_context(
        cls,
        *,
        principal_id: PrincipalId,
        workspace_id: WorkspaceId,
        project_id: ProjectId,
        revision_id: RevisionId,
        run_id: RunId,
        attempt_id: AttemptId,
        resource_scope: tuple[str, ...] = (),
        actions: tuple[str, ...] = (),
        destinations: tuple[str, ...] = (),
        secret_refs: tuple[str, ...] = (),
        ttl: timedelta = timedelta(minutes=15),
    ) -> CapabilityToken:
        """Factory: mint a token bound to a specific run context (§9.4)."""
        now = utcnow()
        return cls(
            principal_id=principal_id,
            workspace_id=workspace_id,
            project_id=project_id,
            revision_id=revision_id,
            run_id=run_id,
            attempt_id=attempt_id,
            resource_scope=resource_scope,
            actions=actions,
            destinations=destinations,
            secret_refs=secret_refs,
            issued_at=now,
            expires_at=now + ttl,
        )


def issue_capability(
    *,
    principal_id: PrincipalId,
    project_id: ProjectId,
    revision_id: RevisionId,
    ttl: timedelta = timedelta(minutes=15),
    run_id: RunId | None = None,
    operation_type: str | None = None,
    resource_scope: tuple[str, ...] = (),
    actions: tuple[str, ...] = (),
    destinations: tuple[str, ...] = (),
    secret_refs: tuple[str, ...] = (),
) -> CapabilityToken:
    """Mint a token expiring after ``ttl``.

    The default TTL is deliberately short. A token outliving the attempt it was
    minted for is a standing grant, which is the thing §9.4 exists to prevent.
    """
    now = utcnow()
    return CapabilityToken(
        principal_id=principal_id,
        project_id=project_id,
        revision_id=revision_id,
        run_id=run_id,
        operation_type=operation_type,
        resource_scope=resource_scope,
        actions=actions,
        destinations=destinations,
        secret_refs=secret_refs,
        issued_at=now,
        expires_at=now + ttl,
    )
