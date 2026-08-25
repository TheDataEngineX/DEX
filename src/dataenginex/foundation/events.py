"""Events and lineage edges (§4.14-4.15).

Two event families share one envelope. Metadata events record system and domain
facts — resource registration, schema observation, quality evaluation, artifact
production. Audit events record security-relevant facts — authentication,
secret access, external transmission, policy change, destructive action.

They are separated because they have different retention, different access
control, and different tamper requirements (invariant 9: audit events are not
editable through ordinary application APIs). They share an envelope because
correlating "what happened" with "who was allowed to do it" across two schemas
is how audit trails become unusable.

Events are append-only. Nothing in this module offers mutation.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import Field

from dataenginex.foundation.ids import (
    InstallationId,
    PrincipalId,
    ProjectId,
    RevisionId,
    WorkspaceId,
    new_id,
)
from dataenginex.foundation.projects import FrozenModel, utcnow

__all__ = [
    "AuditEvent",
    "AuditEventType",
    "EventEnvelope",
    "LineageEdge",
    "LineageNodeType",
    "LineageRelation",
    "MetadataEvent",
]


class EventEnvelope(FrozenModel):
    """Common header on every event (§4.14).

    ``correlation_id`` is what stitches a user action to the runs, policy
    decisions, and artifacts it caused. Without it each table has to be joined
    by timestamp guesswork.
    """

    event_id: str = Field(default_factory=lambda: new_id("evt"))
    occurred_at: datetime = Field(default_factory=utcnow)
    producer: str
    installation_id: InstallationId | None = None
    workspace_id: WorkspaceId | None = None
    project_id: ProjectId | None = None
    revision_id: RevisionId | None = None
    principal_id: PrincipalId | None = None
    correlation_id: str | None = None
    schema_version: str = "dex/v1alpha1"


class MetadataEvent(FrozenModel):
    """A system or domain fact (§4.14).

    ``payload`` is intentionally loose: metadata events describe an open set of
    domain facts, and forcing every one into a closed schema at the foundation
    layer would put domain knowledge in the wrong layer. Audit events, which
    carry security weight, are typed more tightly below.
    """

    envelope: EventEnvelope
    event_type: str
    subject_id: str | None = None
    subject_type: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class AuditEventType(StrEnum):
    """Security-relevant facts (§4.14)."""

    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    SECRET_ACCESS = "secret_access"
    EXTERNAL_TRANSMISSION = "external_transmission"
    PROJECT_SHARING = "project_sharing"
    POLICY_CHANGE = "policy_change"
    DESTRUCTIVE_ACTION = "destructive_action"
    APPROVAL_DECISION = "approval_decision"
    CAPABILITY_ISSUED = "capability_issued"


class AuditEvent(FrozenModel):
    """An append-only security record (§4.14).

    ``outcome`` is required. An audit trail that records only attempts, or only
    successes, cannot answer the question it exists for.
    """

    envelope: EventEnvelope
    event_type: AuditEventType
    action: str
    outcome: str
    target_id: str | None = None
    target_type: str | None = None
    destination: str | None = None
    policy_decision_id: str | None = None
    # Free-form detail. Never a place for secret values (invariant 5).
    detail: dict[str, str] = Field(default_factory=dict)


class LineageNodeType(StrEnum):
    RESOURCE = "resource"
    ARTIFACT = "artifact"
    OPERATION = "operation"
    RUN = "run"
    ATTEMPT = "attempt"
    AGENT = "agent"
    PROMPT = "prompt"
    MODEL = "model"
    APPROVAL = "approval"
    POLICY = "policy"


class LineageRelation(StrEnum):
    """Typed edges (§4.15).

    Typed rather than generic "depends on" because the questions asked of the
    graph are type-specific: retention needs DERIVED_FROM, leakage checks need
    TRAINED_ON, and incident review needs TRANSMITTED_TO.
    """

    CONSUMED = "consumed"
    PRODUCED = "produced"
    DERIVED_FROM = "derived_from"
    TRAINED_ON = "trained_on"
    EVALUATED_WITH = "evaluated_with"
    PROMPTED_BY = "prompted_by"
    RETRIEVED_FROM = "retrieved_from"
    APPROVED_BY = "approved_by"
    TRANSMITTED_TO = "transmitted_to"
    SUPERSEDES = "supersedes"
    INVALIDATES = "invalidates"
    DELETED_BECAUSE_OF = "deleted_because_of"


class LineageEdge(FrozenModel):
    """One typed relationship in the provenance graph (§4.15).

    Carries the revision that produced it, so a graph can be filtered to what a
    specific published definition actually did rather than blending every
    version of a project together.
    """

    edge_id: str = Field(default_factory=lambda: new_id("lin"))
    source_id: str
    source_type: LineageNodeType
    target_id: str
    target_type: LineageNodeType
    relation: LineageRelation
    project_id: ProjectId
    revision_id: RevisionId | None = None
    run_id: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    attributes: dict[str, str] = Field(default_factory=dict)
