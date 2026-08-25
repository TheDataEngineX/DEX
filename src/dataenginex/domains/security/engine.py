"""Policy engine: the single authorization decision point (§9.3).

Every authorization asks one question — principal + action + resource + project
+ revision + destination + classification + workload kind + risk level +
environment — and gets one of four answers back. This module is the only place
that question is answered, which is what makes "policy sits in the execution
path" enforceable rather than aspirational.

Evaluation order matters and is deliberate:

1. **Baseline denials** run first and cannot be overridden by a permit rule.
   A disabled principal or a malformed destination is denied regardless of what
   any project policy says. These are the rules a misconfigured project must not
   be able to switch off.
2. **Risk floor** (§9.5). Level 3 requires the action to be explicitly
   configured; levels 4-5 require a human approval that names the exact
   operation digest.
3. **Matching policies**, highest priority first. At equal priority deny beats
   permit, so an accidental overlap fails closed.
4. **Default deny.** No matching policy is a denial, not a permit.

The result is a :class:`~dataenginex.foundation.policy.PolicyDecision`, which is
immutable evidence: it records the digest of the input context rather than the
context itself, so a decision stays auditable without becoming a second copy of
whatever sensitive values were in scope.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
from collections.abc import Callable, Iterable, Sequence

from dataenginex.foundation import (
    Approval,
    AuthorizationRequest,
    Classification,
    Obligation,
    ObligationType,
    Policy,
    PolicyDecision,
    PolicyEffect,
    Principal,
    RiskLevel,
)

__all__ = [
    "DEFAULT_POLICY_SET",
    "PolicyEngineError",
    "PolicySet",
    "StaticPolicyEngine",
    "context_digest",
]


class PolicyEngineError(RuntimeError):
    """Policy evaluation could not be completed.

    Raised only for malformed inputs. A *decision* to refuse is never an
    exception — it is a ``PolicyDecision`` with ``DENY``, because denials must
    be recorded and audited like any other outcome.
    """


def context_digest(request: AuthorizationRequest) -> str:
    """Stable digest of an authorization context (§4.12).

    Canonical JSON so the same logical request always digests identically.
    ``requested_at`` is excluded: including it would make every decision unique
    and destroy the ability to recognise that the same question was asked twice.
    """
    payload = request.model_dump(mode="json", exclude={"requested_at"})
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


class PolicySet:
    """An ordered, versioned collection of policies.

    The version is part of every decision. Without it a stored decision cannot
    be explained later, because the rules that produced it may have changed.
    """

    __slots__ = ("policies", "version")

    def __init__(self, policies: Sequence[Policy], *, version: str | None = None) -> None:
        # Highest priority first; deny before permit at equal priority so an
        # ambiguous overlap resolves to the safer answer.
        self.policies = tuple(
            sorted(
                policies,
                key=lambda p: (-p.priority, p.effect is not PolicyEffect.DENY, p.name),
            )
        )
        self.version = version or self._derive_version()

    def _derive_version(self) -> str:
        """Content-hash the rules so a changed rule set gets a new version."""
        canonical = json.dumps(
            [p.model_dump(mode="json") for p in self.policies],
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def matching(self, request: AuthorizationRequest) -> tuple[Policy, ...]:
        """Policies whose match dimensions all accept this request."""
        return tuple(p for p in self.policies if _matches(p, request))


def _match_project(policy: Policy, request: AuthorizationRequest) -> bool:
    return policy.project_id is None or policy.project_id == request.project_id


def _match_action(policy: Policy, request: AuthorizationRequest) -> bool:
    return not policy.actions or any(fnmatch.fnmatch(request.action, a) for a in policy.actions)


def _match_classification(policy: Policy, request: AuthorizationRequest) -> bool:
    return not policy.classifications or request.classification in policy.classifications


def _match_workload_kind(policy: Policy, request: AuthorizationRequest) -> bool:
    return not policy.workload_kinds or request.workload_kind in policy.workload_kinds


def _match_risk(policy: Policy, request: AuthorizationRequest) -> bool:
    return policy.max_risk_level is None or request.risk_level <= policy.max_risk_level


def _match_resource(policy: Policy, request: AuthorizationRequest) -> bool:
    if not policy.resource_patterns:
        return True
    resource = request.resource_id or ""
    return any(fnmatch.fnmatch(resource, p) for p in policy.resource_patterns)


def _match_destination(policy: Policy, request: AuthorizationRequest) -> bool:
    if not policy.destinations:
        return True
    # A policy scoped to destinations speaks only to requests that have one.
    if request.destination is None:
        return False
    return any(fnmatch.fnmatch(request.destination.host, d) for d in policy.destinations)


# One predicate per match dimension. A table rather than a chain of ifs so a new
# dimension is a new entry, not another branch in a function that already does
# too much.
_DIMENSIONS: tuple[Callable[[Policy, AuthorizationRequest], bool], ...] = (
    _match_project,
    _match_action,
    _match_classification,
    _match_workload_kind,
    _match_risk,
    _match_resource,
    _match_destination,
)


def _matches(policy: Policy, request: AuthorizationRequest) -> bool:
    """Whether every set dimension of ``policy`` accepts ``request``.

    An unset dimension matches anything — that is what lets a policy speak to
    one axis without having to enumerate all the others.
    """
    return all(dimension(policy, request) for dimension in _DIMENSIONS)


# Baseline rules every installation gets. Written as explicit policies rather
# than hidden in code so a user can be shown why something was refused, and so
# they appear in the policy-set version hash.
DEFAULT_POLICY_SET: tuple[Policy, ...] = (
    Policy(
        name="deny-restricted-external-transmission",
        effect=PolicyEffect.DENY,
        classifications=(Classification.RESTRICTED,),
        destinations=("*",),
        priority=1000,
        description="Restricted data never leaves the installation (§9.8).",
    ),
    Policy(
        name="redact-confidential-egress",
        effect=PolicyEffect.PERMIT_WITH_OBLIGATIONS,
        classifications=(Classification.CONFIDENTIAL,),
        destinations=("*",),
        obligations=(
            Obligation(obligation_type=ObligationType.REDACT_FIELDS),
            Obligation(obligation_type=ObligationType.ENHANCED_AUDIT),
        ),
        priority=800,
        description="Confidential data may egress only redacted and audited (§9.3).",
    ),
    Policy(
        name="permit-project-reads",
        effect=PolicyEffect.PERMIT,
        actions=("read", "read:*", "list", "list:*", "query", "query:*"),
        max_risk_level=RiskLevel.READ_PROJECT_DATA,
        priority=100,
        description="Reading a project's own data is the ordinary case.",
    ),
    Policy(
        name="permit-local-artifacts",
        effect=PolicyEffect.PERMIT,
        actions=("write:local", "write:artifact", "checkpoint", "checkpoint:*"),
        max_risk_level=RiskLevel.CREATE_LOCAL_ARTIFACT,
        priority=100,
        description="Producing local artifacts stays inside the installation.",
    ),
    # Without this a default installation can open a project and never run it:
    # nothing names ``run:<workload>``, so every request is refused by default
    # deny, and no code path offers the operator a way to add a rule. Running a
    # project's own declared workload is the ordinary case, and it produces the
    # same local artifacts the policy above already permits.
    #
    # Capped at level 2 deliberately. A workload that transmits externally or
    # deletes is level 3+ and still needs explicit configuration or human
    # approval, and the restricted-data and confidential-egress denials outrank
    # this rule on priority, so they continue to win.
    Policy(
        name="permit-project-workloads",
        effect=PolicyEffect.PERMIT,
        actions=("run:*", "interactive:*"),
        max_risk_level=RiskLevel.CREATE_LOCAL_ARTIFACT,
        priority=100,
        description="Running a project's own declared workloads is the ordinary case (§9.5).",
    ),
)


class StaticPolicyEngine:
    """Evaluates a fixed policy set (§9.3).

    Implements the ``PolicyEngine`` protocol from ``foundation.contracts``.
    "Static" means the rules are supplied at construction rather than fetched
    per call — a decision must not depend on a network round-trip that can fail
    open under load.

    ``allowed_actions`` is the §9.5 level-3 gate: transmitting externally is
    refused unless the action was explicitly configured. Approvals are consulted
    for levels 4-5 and must name the exact operation digest.
    """

    def __init__(
        self,
        policies: Sequence[Policy] | PolicySet | None = None,
        *,
        allowed_actions: Iterable[str] = (),
        approvals: Sequence[Approval] = (),
        principals: Sequence[Principal] = (),
        evaluated_by: str = "static",
    ) -> None:
        if policies is None:
            self.policy_set = PolicySet(DEFAULT_POLICY_SET, version="default-v1")
        elif isinstance(policies, PolicySet):
            self.policy_set = policies
        else:
            self.policy_set = PolicySet(policies)

        self._allowed_actions = frozenset(allowed_actions)
        self._approvals = tuple(approvals)
        self._principals = {p.principal_id: p for p in principals}
        self._evaluated_by = evaluated_by

    # --- protocol -----------------------------------------------------------

    def evaluate(self, request: AuthorizationRequest) -> PolicyDecision:
        """Decide one authorization request.

        Never raises for a refusal — a denial is a returned decision so that it
        can be persisted and audited alongside permits.
        """
        digest = context_digest(request)

        baseline = self._baseline_denial(request)
        if baseline is not None:
            return self._decision(digest, PolicyEffect.DENY, reason=baseline)

        risk = self._risk_gate(request, digest)
        if risk is not None:
            return self._decision(digest, risk[0], reason=risk[1])

        matched = self.policy_set.matching(request)
        if not matched:
            return self._decision(
                digest,
                PolicyEffect.DENY,
                reason=f"no policy permits action {request.action!r}; default deny",
            )

        # Ordered highest-priority-first with deny ahead of permit, so the first
        # match is the governing one.
        winner = matched[0]
        return self._decision(
            digest,
            winner.effect,
            obligations=winner.obligations,
            matched=tuple(p.name for p in matched),
            reason=winner.description or f"matched policy {winner.name!r}",
        )

    # --- gates --------------------------------------------------------------

    def _baseline_denial(self, request: AuthorizationRequest) -> str | None:
        """Refusals no project policy may override."""
        principal = self._principals.get(request.principal_id)
        if principal is not None and principal.disabled:
            return f"principal {principal.name!r} is disabled"

        # Invariant 7: an external transmission always requires a destination
        # policy can see. A blank host is an undeclared destination, not a
        # default one.
        if request.destination is not None and not request.destination.host.strip():
            return "destination declared with an empty host"

        return None

    def _risk_gate(
        self, request: AuthorizationRequest, digest: str
    ) -> tuple[PolicyEffect, str] | None:
        """Apply the §9.5 action-risk ladder.

        Returns ``None`` when risk does not by itself decide the request, which
        leaves the ordinary policy match to run.
        """
        if request.risk_level >= RiskLevel.MODIFY_EXTERNAL:
            if self._find_approval(request, digest) is None:
                return (
                    PolicyEffect.REQUIRE_APPROVAL,
                    f"risk level {int(request.risk_level)} requires human approval (§9.5)",
                )
            return None

        if (
            request.risk_level == RiskLevel.TRANSMIT_EXTERNAL
            and request.action not in self._allowed_actions
        ):
            return (
                PolicyEffect.DENY,
                f"action {request.action!r} transmits externally and is not "
                "explicitly configured (§9.5 level 3)",
            )

        return None

    def _find_approval(self, request: AuthorizationRequest, digest: str) -> Approval | None:
        """A granted, unexpired approval covering exactly this operation.

        Matched on the context digest, so an approval granted for one operation
        cannot be reused to authorize a different one.
        """
        for approval in self._approvals:
            if approval.project_id != request.project_id:
                continue
            if approval.covers(digest, request.requested_at):
                return approval
        return None

    # --- helpers ------------------------------------------------------------

    def _decision(
        self,
        digest: str,
        effect: PolicyEffect,
        *,
        obligations: tuple[Obligation, ...] = (),
        matched: tuple[str, ...] = (),
        reason: str = "",
    ) -> PolicyDecision:
        return PolicyDecision(
            policy_set_version=self.policy_set.version,
            input_context_digest=digest,
            effect=effect,
            obligations=obligations,
            matched_policies=matched,
            reason=reason,
            evaluated_by=self._evaluated_by,
        )
