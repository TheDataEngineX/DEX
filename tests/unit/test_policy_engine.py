"""Security, policy, and governance (§9).

These tests target the ways authorization fails *open*, because that is the only
failure mode that matters here. A policy engine that permits the happy path but
also permits an expired token, a disabled principal, or an undeclared
destination is worse than no policy engine — it produces an audit trail that
says everything was fine.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import pytest

from dataenginex.domains.security import (
    EgressGuard,
    GovernanceService,
    MemorySecretProvider,
    PolicySet,
    SecretAccessError,
    SecretNotFoundError,
    StaticPolicyEngine,
    context_digest,
    extract_host,
)
from dataenginex.domains.security.governance import ApprovalRequired, GovernanceError
from dataenginex.foundation import (
    ApprovalState,
    AuthorizationRequest,
    CapabilityToken,
    Classification,
    NetworkDestination,
    Obligation,
    ObligationType,
    Policy,
    PolicyEffect,
    Principal,
    PrincipalId,
    PrincipalType,
    ProjectId,
    RevisionId,
    RiskLevel,
    SecretReference,
    WorkloadKind,
    issue_capability,
    utcnow,
)
from dataenginex.runtime.state import ControlStore

PROJECT = ProjectId("proj_test")
REVISION = RevisionId("rev_test")
ALICE = PrincipalId("prin_alice")


def make_request(**overrides: object) -> AuthorizationRequest:
    """An ordinary low-risk read, unless a test says otherwise."""
    defaults: dict[str, object] = {
        "principal_id": ALICE,
        "action": "read",
        "project_id": PROJECT,
        "revision_id": REVISION,
        "risk_level": RiskLevel.READ_PROJECT_DATA,
    }
    defaults.update(overrides)
    return AuthorizationRequest(**defaults)  # type: ignore[arg-type]


def token_for(*, secret_refs: tuple[str, ...] = ("api_key",), **kwargs: object) -> CapabilityToken:
    defaults: dict[str, object] = {
        "principal_id": ALICE,
        "project_id": PROJECT,
        "revision_id": REVISION,
        "secret_refs": secret_refs,
    }
    defaults.update(kwargs)
    return issue_capability(**defaults)  # type: ignore[arg-type]


# --- the default-deny posture ---------------------------------------------


def test_unmatched_action_is_denied_not_permitted() -> None:
    """The single most important property: no rule means no.

    An engine that returned PERMIT here would make every policy set a
    blocklist, and blocklists are wrong by default.
    """
    decision = StaticPolicyEngine().evaluate(make_request(action="drop_everything"))

    assert decision.effect is PolicyEffect.DENY
    assert not decision.allowed
    assert "default deny" in decision.reason


def test_empty_policy_set_denies_everything() -> None:
    engine = StaticPolicyEngine([])

    assert engine.evaluate(make_request()).effect is PolicyEffect.DENY


def test_ordinary_read_is_permitted() -> None:
    """Default deny must not mean nothing works."""
    decision = StaticPolicyEngine().evaluate(make_request(action="read"))

    assert decision.effect is PolicyEffect.PERMIT
    assert decision.allowed


def test_deny_beats_permit_at_equal_priority() -> None:
    """An accidental overlap resolves to the safer answer."""
    policies = [
        Policy(name="permissive", effect=PolicyEffect.PERMIT, actions=("export",), priority=500),
        Policy(name="restrictive", effect=PolicyEffect.DENY, actions=("export",), priority=500),
    ]
    decision = StaticPolicyEngine(policies).evaluate(make_request(action="export"))

    assert decision.effect is PolicyEffect.DENY
    assert decision.matched_policies[0] == "restrictive"


def test_higher_priority_permit_beats_lower_priority_deny() -> None:
    """Priority still orders rules — deny only wins at a *tie*."""
    policies = [
        Policy(name="broad-deny", effect=PolicyEffect.DENY, actions=("*",), priority=10),
        Policy(name="narrow-permit", effect=PolicyEffect.PERMIT, actions=("read",), priority=900),
    ]
    decision = StaticPolicyEngine(policies).evaluate(make_request(action="read"))

    assert decision.effect is PolicyEffect.PERMIT


# --- baseline denials no project policy may override ------------------------


def test_disabled_principal_is_denied_despite_a_permit_all_policy() -> None:
    """A permit-everything rule must not resurrect a disabled account."""
    permit_all = [Policy(name="permit-all", effect=PolicyEffect.PERMIT, actions=("*",))]
    engine = StaticPolicyEngine(
        permit_all,
        principals=[
            Principal(
                principal_id=ALICE,
                principal_type=PrincipalType.HUMAN,
                name="alice",
                disabled=True,
            )
        ],
    )

    decision = engine.evaluate(make_request())

    assert decision.effect is PolicyEffect.DENY
    assert "disabled" in decision.reason


def test_blank_destination_host_is_denied() -> None:
    """A destination policy cannot evaluate is not a default-allowed one."""
    engine = StaticPolicyEngine(
        [Policy(name="permit-all", effect=PolicyEffect.PERMIT, actions=("*",))]
    )

    decision = engine.evaluate(
        make_request(action="send", destination=NetworkDestination(host="   "))
    )

    assert decision.effect is PolicyEffect.DENY
    assert "empty host" in decision.reason


# --- the §9.5 risk ladder ---------------------------------------------------


def test_level_three_transmission_denied_without_explicit_configuration() -> None:
    """L3 requires opt-in, so an unconfigured egress action is refused."""
    decision = StaticPolicyEngine().evaluate(
        make_request(
            action="send_email",
            risk_level=RiskLevel.TRANSMIT_EXTERNAL,
            destination=NetworkDestination(host="smtp.example.com"),
        )
    )

    assert decision.effect is PolicyEffect.DENY
    assert "not explicitly configured" in decision.reason


def test_level_three_permitted_once_explicitly_configured() -> None:
    engine = StaticPolicyEngine(
        [
            Policy(
                name="allow-smtp",
                effect=PolicyEffect.PERMIT,
                actions=("send_email",),
                destinations=("smtp.example.com",),
            )
        ],
        allowed_actions=["send_email"],
    )

    decision = engine.evaluate(
        make_request(
            action="send_email",
            risk_level=RiskLevel.TRANSMIT_EXTERNAL,
            destination=NetworkDestination(host="smtp.example.com"),
        )
    )

    assert decision.effect is PolicyEffect.PERMIT


@pytest.mark.parametrize("risk", [RiskLevel.MODIFY_EXTERNAL, RiskLevel.CONSEQUENTIAL])
def test_high_risk_requires_approval_even_with_permit_all(risk: RiskLevel) -> None:
    """Levels 4-5 need a human, and no policy may waive that."""
    engine = StaticPolicyEngine(
        [Policy(name="permit-all", effect=PolicyEffect.PERMIT, actions=("*",), priority=9999)],
        allowed_actions=["delete_account"],
    )

    decision = engine.evaluate(make_request(action="delete_account", risk_level=risk))

    assert decision.effect is PolicyEffect.REQUIRE_APPROVAL
    assert not decision.allowed


def test_require_approval_is_not_allowed() -> None:
    """``allowed`` must be false until an approval actually exists."""
    decision = StaticPolicyEngine().evaluate(
        make_request(action="wire_transfer", risk_level=RiskLevel.CONSEQUENTIAL)
    )

    assert decision.effect is PolicyEffect.REQUIRE_APPROVAL
    assert decision.allowed is False


# --- classification ---------------------------------------------------------


def test_restricted_data_cannot_leave_the_installation() -> None:
    decision = StaticPolicyEngine(allowed_actions=["upload"]).evaluate(
        make_request(
            action="upload",
            classification=Classification.RESTRICTED,
            risk_level=RiskLevel.TRANSMIT_EXTERNAL,
            destination=NetworkDestination(host="api.vendor.com"),
        )
    )

    assert decision.effect is PolicyEffect.DENY


def test_confidential_egress_permitted_only_with_obligations() -> None:
    """The fourth effect earns its keep: reduced, not refused."""
    decision = StaticPolicyEngine(allowed_actions=["upload"]).evaluate(
        make_request(
            action="upload",
            classification=Classification.CONFIDENTIAL,
            risk_level=RiskLevel.TRANSMIT_EXTERNAL,
            destination=NetworkDestination(host="api.vendor.com"),
        )
    )

    assert decision.effect is PolicyEffect.PERMIT_WITH_OBLIGATIONS
    assert decision.allowed
    kinds = {o.obligation_type for o in decision.obligations}
    assert ObligationType.REDACT_FIELDS in kinds
    assert ObligationType.ENHANCED_AUDIT in kinds


def test_destination_scoped_policy_ignores_requests_without_a_destination() -> None:
    """A rule about egress must not silently govern local work."""
    policies = [
        Policy(
            name="egress-only",
            effect=PolicyEffect.DENY,
            destinations=("*",),
            priority=999,
        ),
        Policy(name="permit-local", effect=PolicyEffect.PERMIT, actions=("read",)),
    ]
    decision = StaticPolicyEngine(policies).evaluate(make_request(action="read"))

    assert decision.effect is PolicyEffect.PERMIT


def test_workload_kind_narrows_a_policy() -> None:
    policies = [
        Policy(
            name="batch-only",
            effect=PolicyEffect.PERMIT,
            actions=("read",),
            workload_kinds=(WorkloadKind.BATCH,),
        )
    ]
    engine = StaticPolicyEngine(policies)

    assert engine.evaluate(make_request(workload_kind=WorkloadKind.BATCH)).allowed
    assert not engine.evaluate(make_request(workload_kind=WorkloadKind.SPARK_STREAM)).allowed


# --- decisions as evidence --------------------------------------------------


def test_decision_records_a_digest_not_the_raw_context() -> None:
    """Invariant: a decision must not become a second copy of the data."""
    request = make_request(resource_id="res_customers_pii")
    decision = StaticPolicyEngine().evaluate(request)

    assert decision.input_context_digest.startswith("sha256:")
    assert "res_customers_pii" not in decision.model_dump_json()


def test_identical_requests_digest_identically_across_time() -> None:
    """Excluding the timestamp is what makes decisions comparable."""
    assert context_digest(make_request()) == context_digest(make_request())


def test_any_change_to_the_request_changes_the_digest() -> None:
    base = context_digest(make_request())

    assert context_digest(make_request(action="write")) != base
    assert context_digest(make_request(resource_id="res_other")) != base
    assert context_digest(make_request(classification=Classification.RESTRICTED)) != base


def test_policy_set_version_changes_when_rules_change() -> None:
    """A stored decision is unexplainable if the version does not move."""
    one = PolicySet([Policy(name="a", effect=PolicyEffect.PERMIT, actions=("read",))])
    two = PolicySet([Policy(name="a", effect=PolicyEffect.DENY, actions=("read",))])

    assert one.version != two.version


def test_policy_set_version_is_stable_for_the_same_rules_in_any_order() -> None:
    rules = [
        Policy(name="a", effect=PolicyEffect.PERMIT, actions=("read",)),
        Policy(name="b", effect=PolicyEffect.DENY, actions=("write",)),
    ]

    assert PolicySet(rules).version == PolicySet(list(reversed(rules))).version


# --- egress (§9.7) ----------------------------------------------------------


@pytest.fixture
def guard() -> EgressGuard:
    return EgressGuard(
        [
            NetworkDestination(host="api.example.com", purpose="pricing", operations=("fetch",)),
            NetworkDestination(host="*.cdn.example.com", purpose="assets"),
        ]
    )


def test_undeclared_destination_is_denied(guard: EgressGuard) -> None:
    decision = guard.check("https://evil.example.net/steal")

    assert not decision.allowed
    assert "not a declared destination" in decision.reason


def test_declared_destination_is_allowed(guard: EgressGuard) -> None:
    decision = guard.check("https://api.example.com/v1/prices")

    assert decision.allowed
    assert decision.host == "api.example.com"


def test_wildcard_subdomain_does_not_grant_the_apex(guard: EgressGuard) -> None:
    """``*.cdn.example.com`` is not ``cdn.example.com`` — a different service."""
    assert guard.check("https://img.cdn.example.com/a.png").allowed
    assert not guard.check("https://cdn.example.com/a.png").allowed


def test_metadata_endpoint_is_blocked_even_if_declared() -> None:
    """SSRF's favourite target, refused ahead of any project declaration."""
    permissive = EgressGuard([NetworkDestination(host="169.254.169.254")])

    decision = permissive.check("http://169.254.169.254/latest/meta-data/")

    assert not decision.allowed
    assert "blocked" in decision.reason


def test_declared_host_still_refused_for_an_undeclared_operation(guard: EgressGuard) -> None:
    assert guard.check("https://api.example.com", operation="fetch").allowed
    assert not guard.check("https://api.example.com", operation="delete").allowed


def test_token_without_the_destination_cannot_egress(guard: EgressGuard) -> None:
    """Declaration and token scope are both required (§9.4 + §9.7)."""
    token = issue_capability(
        principal_id=ALICE,
        project_id=PROJECT,
        revision_id=REVISION,
        destinations=("other.example.com",),
    )

    decision = guard.check("https://api.example.com", capability=token)

    assert not decision.allowed
    assert "destination scope" in decision.reason


def test_token_with_no_destinations_grants_no_egress(guard: EgressGuard) -> None:
    """Unspecified is not unlimited."""
    token = issue_capability(principal_id=ALICE, project_id=PROJECT, revision_id=REVISION)

    assert not guard.check("https://api.example.com", capability=token).allowed


def test_token_carrying_the_destination_is_allowed(guard: EgressGuard) -> None:
    token = issue_capability(
        principal_id=ALICE,
        project_id=PROJECT,
        revision_id=REVISION,
        destinations=("api.example.com",),
    )

    assert guard.check("https://api.example.com/v1", capability=token).allowed


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("https://Api.Example.com/path", "api.example.com"),
        ("api.example.com:8443", "api.example.com"),
        ("http://user:pw@api.example.com/x", "api.example.com"),
        ("api.example.com", "api.example.com"),
        ("", ""),
    ],
)
def test_host_extraction_normalises_forms(target: str, expected: str) -> None:
    """Every call site must derive the same host, or rules can be bypassed."""
    assert extract_host(target) == expected


def test_unparseable_target_is_denied(guard: EgressGuard) -> None:
    assert not guard.check("   ").allowed


# --- secrets (§9.6) ---------------------------------------------------------


@pytest.fixture
def secret_ref() -> SecretReference:
    return SecretReference(name="api_key", project_id=PROJECT)


@pytest.fixture
def provider() -> MemorySecretProvider:
    return MemorySecretProvider({"api_key": "s3cr3t-value"})


def test_secret_resolves_with_a_scoped_token(
    provider: MemorySecretProvider, secret_ref: SecretReference
) -> None:
    lease = provider.resolve(secret_ref, token_for())

    assert lease.value == "s3cr3t-value"
    assert lease.reference_name == "api_key"


def test_token_not_naming_the_secret_cannot_resolve_it(
    provider: MemorySecretProvider, secret_ref: SecretReference
) -> None:
    """An empty or unrelated secret scope grants nothing."""
    with pytest.raises(SecretAccessError, match="secret scope"):
        provider.resolve(secret_ref, token_for(secret_refs=()))


def test_expired_token_cannot_resolve_a_secret(
    provider: MemorySecretProvider, secret_ref: SecretReference
) -> None:
    expired = token_for(ttl=timedelta(seconds=-1))

    with pytest.raises(SecretAccessError, match="expired"):
        provider.resolve(secret_ref, expired)


def test_token_from_another_project_cannot_resolve_a_secret(
    provider: MemorySecretProvider, secret_ref: SecretReference
) -> None:
    other = token_for(project_id=ProjectId("proj_other"))

    with pytest.raises(SecretAccessError, match="different project"):
        provider.resolve(secret_ref, other)


def test_principal_outside_permitted_consumers_is_refused(
    provider: MemorySecretProvider,
) -> None:
    restricted = SecretReference(
        name="api_key",
        project_id=PROJECT,
        permitted_consumers=(PrincipalId("prin_bob"),),
    )

    with pytest.raises(SecretAccessError, match="permitted consumer"):
        provider.resolve(restricted, token_for())


def test_missing_secret_raises_without_leaking_anything(
    secret_ref: SecretReference,
) -> None:
    with pytest.raises(SecretNotFoundError) as excinfo:
        MemorySecretProvider().resolve(secret_ref, token_for())

    assert "api_key" in str(excinfo.value)


def test_lease_never_renders_its_value(
    provider: MemorySecretProvider, secret_ref: SecretReference
) -> None:
    """A secret must not reach a log line, traceback, or REPL echo."""
    lease = provider.resolve(secret_ref, token_for())

    assert "s3cr3t-value" not in repr(lease)
    assert "s3cr3t-value" not in str(lease)
    assert "s3cr3t-value" not in f"{lease}"


# --- governance: durable decisions and approvals ----------------------------


@pytest.fixture
def store(tmp_path: Path) -> Iterator[ControlStore]:
    with ControlStore(tmp_path / "control.db") as s:
        s.migrate()
        with s.transaction() as tx:
            tx.execute(
                "INSERT INTO installations (installation_id, name, created_at) VALUES (?,?,?)",
                ("inst_1", "test", utcnow().isoformat()),
            )
            tx.execute(
                "INSERT INTO workspaces (workspace_id, installation_id, name, created_at) "
                "VALUES (?,?,?,?)",
                ("ws_1", "inst_1", "default", utcnow().isoformat()),
            )
            tx.execute(
                "INSERT INTO projects (project_id, workspace_id, name, created_at) "
                "VALUES (?,?,?,?)",
                (PROJECT, "ws_1", "test-project", utcnow().isoformat()),
            )
            tx.execute(
                "INSERT INTO principals (principal_id, principal_type, name, created_at) "
                "VALUES (?,?,?,?)",
                (ALICE, PrincipalType.HUMAN.value, "alice", utcnow().isoformat()),
            )
        yield s


@pytest.fixture
def governance(store: ControlStore) -> GovernanceService:
    return GovernanceService(store)


def test_permitted_decision_is_persisted(
    governance: GovernanceService, store: ControlStore
) -> None:
    decision = governance.authorize(make_request(action="read"))

    row = store.query_one(
        "SELECT * FROM policy_decisions WHERE decision_id = ?", (decision.decision_id,)
    )
    assert row is not None
    assert row["effect"] == PolicyEffect.PERMIT.value


def test_denials_are_persisted_too(governance: GovernanceService, store: ControlStore) -> None:
    """A log of permits only cannot answer what was refused."""
    governance.authorize(make_request(action="unknown_action"))

    rows = store.query("SELECT * FROM policy_decisions WHERE effect = ?", ("deny",))
    assert len(rows) == 1


def test_every_decision_writes_an_audit_event(
    governance: GovernanceService, store: ControlStore
) -> None:
    decision = governance.authorize(make_request(action="read"))

    row = store.query_one(
        "SELECT * FROM audit_events WHERE policy_decision_id = ?", (decision.decision_id,)
    )
    assert row is not None
    assert row["event_type"] == "authorization"
    assert row["outcome"] == PolicyEffect.PERMIT.value


def test_decision_and_audit_event_commit_together(
    governance: GovernanceService, store: ControlStore
) -> None:
    """§8.3: no decision row without its audit event."""
    governance.authorize(make_request(action="read"))

    decisions = store.query("SELECT decision_id FROM policy_decisions")
    audits = store.query("SELECT policy_decision_id FROM audit_events WHERE action = 'read'")

    assert len(decisions) == len(audits) == 1
    assert decisions[0]["decision_id"] == audits[0]["policy_decision_id"]


def test_high_risk_raises_approval_required_with_a_pending_record(
    governance: GovernanceService,
) -> None:
    with pytest.raises(ApprovalRequired) as excinfo:
        governance.authorize(
            make_request(action="delete_account", risk_level=RiskLevel.CONSEQUENTIAL)
        )

    approval = excinfo.value.approval
    assert approval.state is ApprovalState.PENDING
    assert governance.pending_approvals(PROJECT)[0].approval_id == approval.approval_id


def test_granted_approval_unblocks_exactly_the_approved_operation(
    governance: GovernanceService,
) -> None:
    """The round-trip: approve, re-evaluate, and only then proceed."""
    request = make_request(action="delete_account", risk_level=RiskLevel.MODIFY_EXTERNAL)

    with pytest.raises(ApprovalRequired) as excinfo:
        governance.authorize(request)

    governance.decide_approval(excinfo.value.approval.approval_id, approver_id=ALICE, granted=True)

    # Re-evaluated with the same context, the engine now finds the approval.
    decision = governance.authorize(request, auto_request_approval=False)
    assert decision.effect is not PolicyEffect.REQUIRE_APPROVAL


def test_approval_does_not_cover_a_different_operation(
    governance: GovernanceService,
) -> None:
    """The digest binding is the whole point — no drift onto unseen work."""
    approved = make_request(action="delete_account", risk_level=RiskLevel.MODIFY_EXTERNAL)
    with pytest.raises(ApprovalRequired) as excinfo:
        governance.authorize(approved)
    governance.decide_approval(excinfo.value.approval.approval_id, approver_id=ALICE, granted=True)

    other = make_request(action="wire_transfer", risk_level=RiskLevel.MODIFY_EXTERNAL)
    decision = governance.authorize(other, auto_request_approval=False)

    assert decision.effect is PolicyEffect.REQUIRE_APPROVAL


def test_denied_approval_keeps_the_operation_blocked(governance: GovernanceService) -> None:
    request = make_request(action="delete_account", risk_level=RiskLevel.MODIFY_EXTERNAL)
    with pytest.raises(ApprovalRequired) as excinfo:
        governance.authorize(request)

    governance.decide_approval(excinfo.value.approval.approval_id, approver_id=ALICE, granted=False)

    assert (
        governance.authorize(request, auto_request_approval=False).effect
        is PolicyEffect.REQUIRE_APPROVAL
    )


def test_an_approval_cannot_be_decided_twice(governance: GovernanceService) -> None:
    """Otherwise a denial could be quietly overwritten by a grant."""
    with pytest.raises(ApprovalRequired) as excinfo:
        governance.authorize(
            make_request(action="delete_account", risk_level=RiskLevel.CONSEQUENTIAL)
        )
    approval_id = excinfo.value.approval.approval_id

    governance.decide_approval(approval_id, approver_id=ALICE, granted=False)

    with pytest.raises(GovernanceError, match="already denied"):
        governance.decide_approval(approval_id, approver_id=ALICE, granted=True)


def test_expired_approval_records_as_expired_not_granted(
    governance: GovernanceService,
) -> None:
    """Late intent must not be applied to stale work."""
    with pytest.raises(ApprovalRequired) as excinfo:
        governance.authorize(
            make_request(action="delete_account", risk_level=RiskLevel.CONSEQUENTIAL)
        )
    approval = excinfo.value.approval
    assert approval.expires_at is not None

    decided = governance.decide_approval(
        approval.approval_id,
        approver_id=ALICE,
        granted=True,
        now=approval.expires_at + timedelta(seconds=1),
    )

    assert decided.state is ApprovalState.EXPIRED


def test_deciding_an_unknown_approval_fails(governance: GovernanceService) -> None:
    with pytest.raises(GovernanceError, match="no approval"):
        governance.decide_approval("appr_missing", approver_id=ALICE, granted=True)


def test_disabled_principal_in_the_store_is_refused(
    governance: GovernanceService, store: ControlStore
) -> None:
    """Governance reads principals from the store, not from memory."""
    with store.transaction() as tx:
        tx.execute("UPDATE principals SET disabled = 1 WHERE principal_id = ?", (ALICE,))

    assert governance.authorize(make_request(action="read")).effect is PolicyEffect.DENY


# --- capability issuance (§9.4) ---------------------------------------------


def test_capability_is_not_issued_for_a_denial(governance: GovernanceService) -> None:
    """Issuing on a denial would make the decision advisory."""
    request = make_request(action="unknown_action")
    decision = governance.authorize(request)

    with pytest.raises(GovernanceError, match="cannot issue"):
        governance.issue_for(request, decision)


def test_capability_defaults_to_exactly_the_decided_action(
    governance: GovernanceService,
) -> None:
    request = make_request(action="read")
    decision = governance.authorize(request)

    token = governance.issue_for(request, decision)

    assert token.actions == ("read",)
    assert token.permits("read")
    assert not token.permits("write")


def test_capability_requires_a_pinned_revision(governance: GovernanceService) -> None:
    """ADR-0003: every run pins exactly one revision."""
    request = make_request(action="read", revision_id=None)
    decision = governance.authorize(request)

    with pytest.raises(GovernanceError, match="pinned revision"):
        governance.issue_for(request, decision)


def test_issued_capability_is_audited(governance: GovernanceService, store: ControlStore) -> None:
    request = make_request(action="read")
    decision = governance.authorize(request)

    token = governance.issue_for(request, decision)

    row = store.query_one("SELECT * FROM audit_events WHERE target_id = ?", (token.token_id,))
    assert row is not None
    assert row["event_type"] == "capability_issued"


def test_capability_expires(governance: GovernanceService) -> None:
    """A token outliving its attempt is a standing grant."""
    request = make_request(action="read")
    decision = governance.authorize(request)

    token = governance.issue_for(request, decision, ttl=timedelta(seconds=-1))

    assert token.is_expired()


def test_capability_resource_scope_is_enforced(governance: GovernanceService) -> None:
    request = make_request(action="read")
    decision = governance.authorize(request)

    token = governance.issue_for(request, decision, resource_scope=("res_allowed*",))

    assert token.permits("read", "res_allowed_1")
    assert not token.permits("read", "res_other")


def test_approval_survives_a_round_trip_through_the_store(
    governance: GovernanceService,
) -> None:
    """Reload must preserve every field the engine matches on."""
    with pytest.raises(ApprovalRequired) as excinfo:
        governance.authorize(
            make_request(
                action="delete_account",
                risk_level=RiskLevel.CONSEQUENTIAL,
                resource_id="res_target",
                destination=NetworkDestination(host="api.example.com", purpose="deletion"),
            )
        )
    original = excinfo.value.approval

    reloaded = governance.get_approval(original.approval_id)

    assert reloaded is not None
    assert reloaded.operation_digest == original.operation_digest
    assert reloaded.risk_level == original.risk_level
    assert reloaded.destination is not None
    assert reloaded.destination.host == "api.example.com"
    assert reloaded.affected_resource_ids == ("res_target",)


def test_obligations_survive_persistence(store: ControlStore) -> None:
    """An unfulfilled obligation is a violation, so it must be recorded."""
    service = GovernanceService(store, allowed_actions=["upload"])
    decision = service.authorize(
        make_request(
            action="upload",
            classification=Classification.CONFIDENTIAL,
            risk_level=RiskLevel.TRANSMIT_EXTERNAL,
            destination=NetworkDestination(host="api.vendor.com"),
        )
    )

    row = store.query_one(
        "SELECT obligations_json FROM policy_decisions WHERE decision_id = ?",
        (decision.decision_id,),
    )
    assert row is not None
    stored = json.loads(row["obligations_json"])
    assert {o["obligation_type"] for o in stored} == {
        ObligationType.REDACT_FIELDS.value,
        ObligationType.ENHANCED_AUDIT.value,
    }


def test_custom_policies_replace_the_defaults(store: ControlStore) -> None:
    """Supplying policies must not silently keep the built-in permits."""
    service = GovernanceService(
        store,
        policies=[Policy(name="only-write", effect=PolicyEffect.PERMIT, actions=("write",))],
    )

    assert service.authorize(make_request(action="read")).effect is PolicyEffect.DENY
    assert service.authorize(make_request(action="write")).effect is PolicyEffect.PERMIT


def test_obligation_parameters_are_preserved() -> None:
    """Redaction with no field list is not the same as redacting nothing."""
    obligation = Obligation(
        obligation_type=ObligationType.REDACT_FIELDS, parameters=("email", "ssn")
    )
    policies = [
        Policy(
            name="redact",
            effect=PolicyEffect.PERMIT_WITH_OBLIGATIONS,
            actions=("read",),
            obligations=(obligation,),
            priority=999,
        )
    ]

    decision = StaticPolicyEngine(policies).evaluate(make_request(action="read"))

    assert decision.obligations[0].parameters == ("email", "ssn")
