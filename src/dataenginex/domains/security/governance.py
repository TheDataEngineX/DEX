"""Governance: decisions, approvals, and capability issuance (§9.3-9.5).

The policy engine decides; this module makes the decision *durable* and acts on
it. Three responsibilities that belong together because they share one
transaction:

* **Record every decision.** A permit is as interesting as a denial when
  reconstructing an incident, so both are written to ``policy_decisions`` and
  audited. Recording only refusals produces a log that cannot answer "what was
  this allowed to do?".
* **Mediate approvals.** ``REQUIRE_APPROVAL`` creates a pending approval bound
  to the exact context digest. Granting it does not itself authorize anything —
  the caller re-evaluates, and the engine then finds the approval. That
  round-trip is what stops "approved once" becoming "approved forever".
* **Issue capability tokens** only after a permit, scoped to what was actually
  decided (§9.4). A token minted before the decision, or wider than it, is the
  hole the token design exists to close.

Every write goes through :meth:`ControlStore.transaction`, so the decision row
and its audit event commit together or not at all (§8.3).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from datetime import datetime, timedelta

from dataenginex.domains.security.engine import StaticPolicyEngine, context_digest
from dataenginex.foundation import (
    Approval,
    ApprovalState,
    AuditEvent,
    AuditEventType,
    AuthorizationRequest,
    CapabilityToken,
    EventEnvelope,
    NetworkDestination,
    Policy,
    PolicyDecision,
    PolicyEffect,
    Principal,
    PrincipalId,
    ProjectId,
    RiskLevel,
    RunId,
    issue_capability,
    utcnow,
)
from dataenginex.runtime.state import ControlStore

__all__ = ["ApprovalRequired", "GovernanceError", "GovernanceService"]

PRODUCER = "dex.governance"


class GovernanceError(RuntimeError):
    """A governance operation could not be completed."""


class ApprovalRequired(GovernanceError):
    """Raised when work cannot proceed without a human decision.

    Carries the pending approval so a caller can surface its id rather than
    telling the user only that something was blocked.
    """

    def __init__(self, approval: Approval, decision: PolicyDecision) -> None:
        super().__init__(f"approval {approval.approval_id} required: {approval.action_summary}")
        self.approval = approval
        self.decision = decision


class GovernanceService:
    """Durable policy decisions, approvals, and capability issuance.

    The engine is rebuilt per evaluation rather than held as mutable state: an
    engine that could be mutated mid-evaluation is an engine whose decisions
    cannot be explained afterwards.
    """

    def __init__(
        self,
        store: ControlStore,
        *,
        policies: Sequence[Policy] | None = None,
        allowed_actions: Sequence[str] = (),
    ) -> None:
        self.store = store
        self._policies = tuple(policies) if policies is not None else None
        self._allowed_actions = tuple(allowed_actions)

    # --- evaluation ---------------------------------------------------------

    def engine(self, project_id: ProjectId | None = None) -> StaticPolicyEngine:
        """Build an engine over current approvals and principals.

        Approvals are loaded per evaluation because an approval granted a second
        ago must count. Caching them would create exactly the window where a
        just-granted approval is invisible.
        """
        return StaticPolicyEngine(
            self._policies,
            allowed_actions=self._allowed_actions,
            approvals=self._load_approvals(project_id),
            principals=self._load_principals(),
        )

    def authorize(
        self,
        request: AuthorizationRequest,
        *,
        auto_request_approval: bool = True,
    ) -> PolicyDecision:
        """Evaluate, persist, and audit one authorization.

        Returns the decision — including denials. Raises only
        :class:`ApprovalRequired`, and only when ``auto_request_approval`` has
        created something actionable for the user to respond to.
        """
        decision = self.engine(request.project_id).evaluate(request)
        self._record(decision, request)

        if decision.effect is PolicyEffect.REQUIRE_APPROVAL and auto_request_approval:
            approval = self.request_approval(request, decision)
            raise ApprovalRequired(approval, decision)

        return decision

    def _record(self, decision: PolicyDecision, request: AuthorizationRequest) -> None:
        """Persist a decision and its audit event in one transaction."""
        with self.store.transaction() as tx:
            tx.execute(
                "INSERT INTO policy_decisions (decision_id, policy_set_version, "
                "input_context_digest, effect, obligations_json, matched_policies_json, "
                "reason, evaluated_by, evaluated_at, project_id, principal_id, action) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    decision.decision_id,
                    decision.policy_set_version,
                    decision.input_context_digest,
                    decision.effect.value,
                    json.dumps([o.model_dump(mode="json") for o in decision.obligations]),
                    json.dumps(list(decision.matched_policies)),
                    decision.reason,
                    decision.evaluated_by,
                    decision.evaluated_at.isoformat(),
                    request.project_id,
                    request.principal_id,
                    request.action,
                ),
            )
            tx.emit_audit(
                AuditEvent(
                    envelope=EventEnvelope(
                        producer=PRODUCER,
                        project_id=request.project_id,
                        revision_id=request.revision_id,
                        principal_id=request.principal_id,
                    ),
                    event_type=AuditEventType.AUTHORIZATION,
                    action=request.action,
                    outcome=decision.effect.value,
                    target_id=request.resource_id,
                    target_type="resource" if request.resource_id else None,
                    destination=request.destination.host if request.destination else None,
                    policy_decision_id=decision.decision_id,
                    detail={"reason": decision.reason},
                )
            )

    # --- approvals (§4.13, §9.5) --------------------------------------------

    def request_approval(
        self,
        request: AuthorizationRequest,
        decision: PolicyDecision | None = None,
        *,
        summary: str = "",
        ttl: timedelta = timedelta(hours=24),
    ) -> Approval:
        """Create a pending approval bound to this exact context.

        The digest is the binding. Changing anything about the request produces
        a different digest, so a granted approval cannot drift onto work the
        approver never saw.
        """
        digest = decision.input_context_digest if decision else context_digest(request)
        approval = Approval(
            project_id=request.project_id,
            requested_by=request.principal_id,
            action_summary=summary or f"{request.action} (risk {int(request.risk_level)})",
            operation_digest=digest,
            affected_resource_ids=(request.resource_id,) if request.resource_id else (),
            destination=request.destination,
            risk_level=request.risk_level,
            expires_at=utcnow() + ttl,
        )
        self._insert_approval(approval)
        return approval

    def _insert_approval(self, approval: Approval) -> None:
        with self.store.transaction() as tx:
            tx.execute(
                "INSERT INTO approvals (approval_id, project_id, requested_by, "
                "action_summary, operation_digest, affected_resources_json, "
                "destination_json, risk_level, state, approver_id, requested_at, "
                "decided_at, expires_at, reason) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    approval.approval_id,
                    approval.project_id,
                    approval.requested_by,
                    approval.action_summary,
                    approval.operation_digest,
                    json.dumps(list(approval.affected_resource_ids)),
                    approval.destination.model_dump_json() if approval.destination else None,
                    int(approval.risk_level),
                    approval.state.value,
                    approval.approver_id,
                    approval.requested_at.isoformat(),
                    approval.decided_at.isoformat() if approval.decided_at else None,
                    approval.expires_at.isoformat() if approval.expires_at else None,
                    approval.reason,
                ),
            )
            tx.emit_audit(
                AuditEvent(
                    envelope=EventEnvelope(
                        producer=PRODUCER,
                        project_id=approval.project_id,
                        principal_id=approval.requested_by,
                    ),
                    event_type=AuditEventType.APPROVAL_DECISION,
                    action="approval.requested",
                    outcome="pending",
                    target_id=approval.approval_id,
                    target_type="approval",
                    detail={"summary": approval.action_summary},
                )
            )

    def decide_approval(
        self,
        approval_id: str,
        *,
        approver_id: PrincipalId,
        granted: bool,
        reason: str = "",
        now: datetime | None = None,
    ) -> Approval:
        """Grant or deny a pending approval.

        Refuses to decide anything not currently pending, so a decision cannot
        be overwritten and an expired request cannot be quietly revived.
        """
        moment = now or utcnow()
        approval = self.get_approval(approval_id)
        if approval is None:
            raise GovernanceError(f"no approval {approval_id!r}")
        if approval.state is not ApprovalState.PENDING:
            raise GovernanceError(f"approval {approval_id} is already {approval.state.value}")

        state = ApprovalState.GRANTED if granted else ApprovalState.DENIED
        if approval.expires_at is not None and moment >= approval.expires_at:
            # An expired request is not granted late — it is recorded as expired
            # so the approver's intent is not silently applied to stale work.
            state = ApprovalState.EXPIRED

        with self.store.transaction() as tx:
            tx.execute(
                "UPDATE approvals SET state = ?, approver_id = ?, decided_at = ?, "
                "reason = ? WHERE approval_id = ? AND state = ?",
                (
                    state.value,
                    approver_id,
                    moment.isoformat(),
                    reason,
                    approval_id,
                    ApprovalState.PENDING.value,
                ),
            )
            tx.emit_audit(
                AuditEvent(
                    envelope=EventEnvelope(
                        producer=PRODUCER,
                        project_id=approval.project_id,
                        principal_id=approver_id,
                    ),
                    event_type=AuditEventType.APPROVAL_DECISION,
                    action="approval.decided",
                    outcome=state.value,
                    target_id=approval_id,
                    target_type="approval",
                    detail={"reason": reason},
                )
            )

        return approval.model_copy(
            update={"state": state, "approver_id": approver_id, "decided_at": moment}
        )

    def get_approval(self, approval_id: str) -> Approval | None:
        row = self.store.query_one("SELECT * FROM approvals WHERE approval_id = ?", (approval_id,))
        return _row_to_approval(row) if row is not None else None

    def pending_approvals(self, project_id: ProjectId | None = None) -> tuple[Approval, ...]:
        if project_id is None:
            rows = self.store.query(
                "SELECT * FROM approvals WHERE state = ? ORDER BY requested_at",
                (ApprovalState.PENDING.value,),
            )
        else:
            rows = self.store.query(
                "SELECT * FROM approvals WHERE state = ? AND project_id = ? ORDER BY requested_at",
                (ApprovalState.PENDING.value, project_id),
            )
        return tuple(_row_to_approval(row) for row in rows)

    # --- capability issuance (§9.4) -----------------------------------------

    def issue_for(
        self,
        request: AuthorizationRequest,
        decision: PolicyDecision,
        *,
        run_id: RunId | None = None,
        resource_scope: Sequence[str] = (),
        actions: Sequence[str] = (),
        secret_refs: Sequence[str] = (),
        ttl: timedelta = timedelta(minutes=15),
    ) -> CapabilityToken:
        """Mint a token for a permitted request (§9.4).

        Refuses on a non-permitting decision. Issuing a token for a denial would
        make the decision advisory, which is the failure mode this layer exists
        to prevent.
        """
        if not decision.allowed:
            raise GovernanceError(
                f"cannot issue a capability for a {decision.effect.value} decision"
            )
        if request.revision_id is None:
            raise GovernanceError("cannot issue a capability without a pinned revision")

        token = issue_capability(
            principal_id=request.principal_id,
            project_id=request.project_id,
            revision_id=request.revision_id,
            run_id=run_id,
            operation_type=request.action,
            resource_scope=tuple(resource_scope),
            # Defaults to exactly the action decided, never a wider set.
            actions=tuple(actions) or (request.action,),
            destinations=(request.destination.host,) if request.destination else (),
            secret_refs=tuple(secret_refs),
            ttl=ttl,
        )

        with self.store.transaction() as tx:
            tx.emit_audit(
                AuditEvent(
                    envelope=EventEnvelope(
                        producer=PRODUCER,
                        project_id=request.project_id,
                        revision_id=request.revision_id,
                        principal_id=request.principal_id,
                    ),
                    event_type=AuditEventType.CAPABILITY_ISSUED,
                    action="capability.issued",
                    outcome="granted",
                    target_id=token.token_id,
                    target_type="capability",
                    policy_decision_id=decision.decision_id,
                    detail={
                        "actions": ",".join(token.actions),
                        "expires_at": token.expires_at.isoformat(),
                    },
                )
            )
        return token

    # --- loading ------------------------------------------------------------

    def _load_approvals(self, project_id: ProjectId | None) -> tuple[Approval, ...]:
        """Granted approvals only — the engine has no use for the rest."""
        if project_id is None:
            rows = self.store.query(
                "SELECT * FROM approvals WHERE state = ?", (ApprovalState.GRANTED.value,)
            )
        else:
            rows = self.store.query(
                "SELECT * FROM approvals WHERE state = ? AND project_id = ?",
                (ApprovalState.GRANTED.value, project_id),
            )
        return tuple(_row_to_approval(row) for row in rows)

    def _load_principals(self) -> tuple[Principal, ...]:
        rows = self.store.query("SELECT * FROM principals")
        return tuple(
            Principal(
                principal_id=PrincipalId(row["principal_id"]),
                principal_type=row["principal_type"],
                name=row["name"],
                display_name=row["display_name"],
                trust_level=row["trust_level"],
                delegated_from=row["delegated_from"],
                roles=tuple(json.loads(row["roles_json"])),
                created_at=datetime.fromisoformat(row["created_at"]),
                disabled=bool(row["disabled"]),
            )
            for row in rows
        )


def _row_to_approval(row: sqlite3.Row) -> Approval:
    """Rebuild an approval from its stored row."""
    destination = (
        NetworkDestination.model_validate_json(row["destination_json"])
        if row["destination_json"]
        else None
    )
    return Approval(
        approval_id=row["approval_id"],
        project_id=ProjectId(row["project_id"]),
        requested_by=PrincipalId(row["requested_by"]),
        action_summary=row["action_summary"],
        operation_digest=row["operation_digest"],
        affected_resource_ids=tuple(json.loads(row["affected_resources_json"])),
        destination=destination,
        risk_level=RiskLevel(int(row["risk_level"])),
        state=ApprovalState(row["state"]),
        approver_id=PrincipalId(row["approver_id"]) if row["approver_id"] else None,
        requested_at=datetime.fromisoformat(row["requested_at"]),
        decided_at=datetime.fromisoformat(row["decided_at"]) if row["decided_at"] else None,
        expires_at=datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None,
        reason=row["reason"],
    )
