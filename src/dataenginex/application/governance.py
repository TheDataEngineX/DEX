"""Governance read services — policies, decisions, and the audit trail (§9.3, §9.5).

These answer the questions a security page asks: what rules are in force, what
did they decide, and what happened as a result.

The distinction from the old design is worth stating, because it changes what
these pages *mean*. Studio previously rendered ``config.secops.policies`` and
``config.secops.alerts`` — Pydantic models parsed from the manifest that nothing
ever evaluated. A page showing them told the user their data was governed by
rules that were, in fact, inert.

What is read here is the live policy set the engine actually evaluates, and the
decisions it actually made. A rule shown on this page is a rule that ran.
"""

from __future__ import annotations

import json
from typing import Any

from dataenginex.application.services import Service
from dataenginex.foundation import FrozenModel, PolicyEffect, PrincipalId, ProjectId
from dataenginex.foundation.operations import RiskLevel

__all__ = [
    "AuditEventView",
    "DecisionView",
    "GovernanceQueryService",
    "PolicyView",
]


class PolicyView(FrozenModel):
    """A policy as the security pages display it (§4.12).

    ``enabled`` is carried rather than filtered on, because "this rule exists
    but is switched off" is exactly what an operator needs to see. Hiding
    disabled rules makes a gap in enforcement look like a gap in configuration.
    """

    policy_id: str
    name: str
    effect: PolicyEffect
    priority: int = 0
    enabled: bool = True
    project_id: ProjectId | None = None
    description: str = ""
    actions: tuple[str, ...] = ()
    max_risk_level: RiskLevel | None = None
    created_at: str = ""


class DecisionView(FrozenModel):
    """One recorded authorization decision (§4.12).

    ``reason`` and ``matched_policies`` are both kept: an operator asking "why
    was this denied?" needs the rule that fired, not only the verdict.
    """

    decision_id: str
    effect: PolicyEffect
    action: str = ""
    principal_id: PrincipalId | None = None
    project_id: ProjectId | None = None
    reason: str = ""
    matched_policies: tuple[str, ...] = ()
    evaluated_at: str = ""


class AuditEventView(FrozenModel):
    """One audit record (§4.15).

    ``outcome`` is separate from ``action`` on purpose. "Attempted to export"
    and "exported" are different facts, and a trail that conflates them cannot
    answer whether a control held.
    """

    event_id: str
    occurred_at: str
    action: str
    outcome: str
    event_type: str = ""
    principal_id: PrincipalId | None = None
    project_id: ProjectId | None = None
    target_id: str | None = None
    target_type: str | None = None
    destination: str | None = None
    policy_decision_id: str | None = None


class GovernanceQueryService(Service):
    """Read-only views over the policy engine and audit trail (§9.3, §9.5)."""

    def list_policies(self, project_id: ProjectId | None = None) -> list[PolicyView]:
        """Policies in force, highest priority first.

        Installation-wide policies (``project_id IS NULL``) are included
        alongside the project's own, because both are evaluated. Showing only
        project rules would tell an operator a run is unconstrained when an
        installation rule is what actually governs it.
        """
        rows = self.store.query(
            "SELECT * FROM policies WHERE project_id IS NULL OR project_id = ? "
            "ORDER BY priority DESC, name",
            (project_id,),
        )
        return [_row_to_policy(row) for row in rows]

    def list_decisions(
        self, project_id: ProjectId | None = None, *, limit: int = 100
    ) -> list[DecisionView]:
        """Recent authorization decisions, newest first."""
        rows = self.store.query(
            "SELECT * FROM policy_decisions WHERE project_id = ? "
            "ORDER BY evaluated_at DESC LIMIT ?",
            (project_id, limit),
        )
        return [_row_to_decision(row) for row in rows]

    def list_denials(
        self, project_id: ProjectId | None = None, *, limit: int = 100
    ) -> list[DecisionView]:
        """Only the refusals.

        Its own query rather than a client-side filter over ``list_decisions``:
        denials are rare relative to permits, so filtering in Python would page
        through mostly-permits and could return an empty page while denials
        exist further back.
        """
        rows = self.store.query(
            "SELECT * FROM policy_decisions WHERE project_id = ? AND effect = ? "
            "ORDER BY evaluated_at DESC LIMIT ?",
            (project_id, PolicyEffect.DENY.value, limit),
        )
        return [_row_to_decision(row) for row in rows]

    def list_audit_events(
        self,
        project_id: ProjectId | None = None,
        *,
        action: str | None = None,
        limit: int = 100,
    ) -> list[AuditEventView]:
        """The audit trail, newest first (§4.15)."""
        clauses = ["project_id = ?"]
        params: list[Any] = [project_id]
        if action is not None:
            clauses.append("action = ?")
            params.append(action)
        params.append(limit)

        rows = self.store.query(
            f"SELECT * FROM audit_events WHERE {' AND '.join(clauses)} "  # noqa: S608 - literals
            "ORDER BY occurred_at DESC LIMIT ?",
            params,
        )
        return [_row_to_audit(row) for row in rows]


def _row_to_policy(row: Any) -> PolicyView:
    data = dict(row)
    # The definition holds whatever the policy model carries beyond the indexed
    # columns. Unpacked by key so that adding a field to the model does not
    # require a migration to display it.
    definition = json.loads(data["definition_json"] or "{}")
    return PolicyView(
        policy_id=data["policy_id"],
        name=data["name"],
        effect=PolicyEffect(data["effect"]),
        priority=int(data["priority"]),
        enabled=bool(data["enabled"]),
        project_id=ProjectId(data["project_id"]) if data["project_id"] else None,
        description=definition.get("description", ""),
        actions=tuple(definition.get("actions", ())),
        created_at=data["created_at"],
    )


def _row_to_decision(row: Any) -> DecisionView:
    data = dict(row)
    return DecisionView(
        decision_id=data["decision_id"],
        effect=PolicyEffect(data["effect"]),
        action=data["action"] or "",
        principal_id=PrincipalId(data["principal_id"]) if data["principal_id"] else None,
        project_id=ProjectId(data["project_id"]) if data["project_id"] else None,
        reason=data["reason"],
        matched_policies=tuple(json.loads(data["matched_policies_json"] or "[]")),
        evaluated_at=data["evaluated_at"],
    )


def _row_to_audit(row: Any) -> AuditEventView:
    data = dict(row)
    return AuditEventView(
        event_id=data["event_id"],
        occurred_at=data["occurred_at"],
        action=data["action"],
        outcome=data["outcome"],
        event_type=data["event_type"],
        principal_id=PrincipalId(data["principal_id"]) if data["principal_id"] else None,
        project_id=ProjectId(data["project_id"]) if data["project_id"] else None,
        target_id=data["target_id"],
        target_type=data["target_type"],
        destination=data["destination"],
        policy_decision_id=data["policy_decision_id"],
    )
