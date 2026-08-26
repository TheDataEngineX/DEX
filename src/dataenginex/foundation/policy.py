"""Policy, decisions, and approvals (§4.12-4.13, §9.3).

Policy sits *in* the execution path, not beside it. Every authorization asks the
same question — principal + action + resource + project + revision + destination
+ classification + workload kind + risk level + environment — and gets back one
of four answers: permit, deny, require approval, or permit with obligations.

The fourth answer is the one that earns its keep. Redaction, row filtering, and
destination restriction let a request proceed in a reduced form instead of
forcing a binary allow/deny that pushes users toward disabling policy entirely.

A ``PolicyDecision`` is immutable evidence. It records the input digest rather
than the raw context so a decision can be audited without re-exposing whatever
sensitive values were in scope at the time.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from urllib.parse import urlparse

from pydantic import Field

from dataenginex.foundation.ids import (
    PolicyDecisionId,
    PrincipalId,
    ProjectId,
    ResourceId,
    RevisionId,
    new_id,
)
from dataenginex.foundation.operations import RiskLevel
from dataenginex.foundation.projects import FrozenModel, utcnow
from dataenginex.foundation.resources import Classification
from dataenginex.foundation.workloads import WorkloadKind

__all__ = [
    "Approval",
    "ApprovalState",
    "AuthorizationRequest",
    "NetworkDestination",
    "Obligation",
    "ObligationType",
    "Policy",
    "PolicyDecision",
    "PolicyEffect",
    "extract_host",
]


class PolicyEffect(StrEnum):
    """The four possible outcomes (§9.3)."""

    PERMIT = "permit"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"
    PERMIT_WITH_OBLIGATIONS = "permit_with_obligations"


class ObligationType(StrEnum):
    """Conditions attached to a conditional permit (§9.3)."""

    REDACT_FIELDS = "redact_fields"
    FILTER_ROWS = "filter_rows"
    RESTRICT_DESTINATION = "restrict_destination"
    ENHANCED_AUDIT = "enhanced_audit"
    MASK_VALUES = "mask_values"


class Obligation(FrozenModel):
    """One condition the caller must honor.

    Obligations are enforced by the caller, so they are recorded on the decision
    and audited — an unfulfilled obligation is a policy violation, not a hint.
    """

    obligation_type: ObligationType
    # Interpretation depends on the type: field names for REDACT_FIELDS, a
    # predicate for FILTER_ROWS, allowed hosts for RESTRICT_DESTINATION.
    parameters: tuple[str, ...] = ()


class NetworkDestination(FrozenModel):
    """An external destination subject to egress policy (§9.7)."""

    host: str
    purpose: str = ""
    operations: tuple[str, ...] = ()


class AuthorizationRequest(FrozenModel):
    """The full context a policy decision is made against (§9.3).

    Every dimension the spec lists is present in the type rather than passed as
    loose context, because a policy that silently sees a missing classification
    fails open.
    """

    principal_id: PrincipalId
    action: str
    project_id: ProjectId
    revision_id: RevisionId | None = None
    resource_id: ResourceId | None = None
    classification: Classification = Classification.INTERNAL
    workload_kind: WorkloadKind = WorkloadKind.BATCH
    risk_level: RiskLevel = RiskLevel.READ_PROJECT_DATA
    destination: NetworkDestination | None = None
    environment: str = "default"
    requested_at: datetime = Field(default_factory=utcnow)


class Policy(FrozenModel):
    """A rule over the authorization context (§4.12).

    Unset match dimensions match anything. Higher ``priority`` wins, and deny
    beats permit at equal priority — the engine in block 5 implements that
    ordering.
    """

    name: str
    version: str = "1"
    project_id: ProjectId | None = None
    effect: PolicyEffect = PolicyEffect.DENY
    actions: tuple[str, ...] = ()
    resource_patterns: tuple[str, ...] = ()
    classifications: tuple[Classification, ...] = ()
    workload_kinds: tuple[WorkloadKind, ...] = ()
    max_risk_level: RiskLevel | None = None
    destinations: tuple[str, ...] = ()
    obligations: tuple[Obligation, ...] = ()
    priority: int = Field(default=100, ge=0)
    description: str = ""


class PolicyDecision(FrozenModel):
    """Immutable evidence of one evaluation (§4.12).

    Stores ``input_context_digest`` instead of the context itself: the decision
    must remain auditable and comparable without becoming a second copy of the
    sensitive data it was protecting.
    """

    decision_id: PolicyDecisionId = Field(default_factory=lambda: PolicyDecisionId(new_id("dec")))
    policy_set_version: str
    input_context_digest: str
    effect: PolicyEffect
    obligations: tuple[Obligation, ...] = ()
    matched_policies: tuple[str, ...] = ()
    reason: str = ""
    evaluated_by: str = "builtin"
    evaluated_at: datetime = Field(default_factory=utcnow)

    @property
    def allowed(self) -> bool:
        """Whether execution may proceed now.

        ``REQUIRE_APPROVAL`` is not allowed — it becomes allowed only once an
        approval is granted and a fresh decision is issued.
        """
        return self.effect in (
            PolicyEffect.PERMIT,
            PolicyEffect.PERMIT_WITH_OBLIGATIONS,
        )


class ApprovalState(StrEnum):
    PENDING = "pending"
    GRANTED = "granted"
    DENIED = "denied"
    EXPIRED = "expired"
    INVALIDATED = "invalidated"


class Approval(FrozenModel):
    """A first-class record of human confirmation (§4.13).

    ``operation_digest`` is what makes this safe: it pins the exact operation
    approved. If the operation changes at all, the approval no longer matches
    and is invalidated rather than silently covering different work.
    """

    approval_id: str = Field(default_factory=lambda: new_id("appr"))
    project_id: ProjectId
    requested_by: PrincipalId
    action_summary: str
    operation_digest: str
    affected_resource_ids: tuple[ResourceId, ...] = ()
    destination: NetworkDestination | None = None
    risk_level: RiskLevel = RiskLevel.MODIFY_EXTERNAL
    state: ApprovalState = ApprovalState.PENDING
    approver_id: PrincipalId | None = None
    requested_at: datetime = Field(default_factory=utcnow)
    decided_at: datetime | None = None
    expires_at: datetime | None = None
    reason: str = ""

    def covers(self, operation_digest: str, now: datetime | None = None) -> bool:
        """Whether this approval authorizes exactly the given operation.

        Fails closed on every axis: wrong digest, not granted, or past expiry
        all return False.
        """
        if self.state is not ApprovalState.GRANTED:
            return False
        if self.operation_digest != operation_digest:
            return False
        return not (self.expires_at is not None and (now or utcnow()) >= self.expires_at)


def extract_host(target: str) -> str:
    """Host portion of a URL, hostname, or ``host:port`` string.

    Egress rules are written against hosts, but callers hold URLs. Normalising
    here keeps every call site from doing it slightly differently — and a
    slightly-different parse is how a rule gets bypassed.
    """
    candidate = target.strip()
    if not candidate:
        return ""

    if "://" in candidate:
        parsed = urlparse(candidate)
        return (parsed.hostname or "").lower()

    # Bare "host:port" — urlparse needs a scheme to populate .hostname.
    parsed = urlparse(f"//{candidate}")
    return (parsed.hostname or candidate).lower()
