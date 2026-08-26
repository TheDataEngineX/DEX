"""Foundation layer: invariants, state machine, and identity behavior.

Covers the logic that has branches — the §4.16 invariant checks, the §7.4 run
state machine, capability scoping, and secret-leak resistance. Plain data
classes with no behavior are exercised only where a default matters.
"""

from __future__ import annotations

from datetime import timedelta
from itertools import pairwise

import pytest
from pydantic import ValidationError

from dataenginex.foundation import (
    Approval,
    ApprovalState,
    Artifact,
    AttemptState,
    CapabilityToken,
    Digest,
    ExecutionAttempt,
    IdempotencyStrategy,
    InvariantViolation,
    Operation,
    PolicyDecision,
    PolicyEffect,
    PrincipalId,
    Project,
    ProjectId,
    ProjectRevision,
    PublicationStatus,
    Resource,
    ResourceType,
    RevisionId,
    RiskLevel,
    RunState,
    SecretLease,
    SideEffectClass,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
    WorkloadRun,
    WorkspaceId,
    can_transition,
    check_active_revision_published,
    check_artifacts_not_overwritten,
    check_attempt_belongs_to_run,
    check_cross_project_access_granted,
    check_external_action_authorized,
    check_invariants,
    check_run_pins_published_revision,
    check_worker_scope_minimal,
    digest_bytes,
    is_terminal,
    issue_capability,
    utcnow,
    uuid7,
)

PRINCIPAL = PrincipalId("prin_test")
WORKSPACE = WorkspaceId("ws_test")


def make_revision(
    project_id: ProjectId,
    *,
    status: PublicationStatus = PublicationStatus.PUBLISHED,
) -> ProjectRevision:
    return ProjectRevision(
        project_id=project_id,
        content_hash="sha256:abc",
        created_by=PRINCIPAL,
        status=status,
    )


def make_run(project_id: ProjectId, revision_id: RevisionId) -> WorkloadRun:
    return WorkloadRun(
        project_id=project_id,
        revision_id=revision_id,
        workload_name="clean_users",
        requested_by=PRINCIPAL,
    )


def make_artifact(
    project_id: ProjectId, revision_id: RevisionId, digest: str
) -> Artifact:
    return Artifact(
        project_id=project_id,
        revision_id=revision_id,
        logical_name="silver_users",
        digest=Digest(value=digest),
        size_bytes=10,
        provider="filesystem",
        provider_uri=f"file:///artifacts/{digest}",
    )


# --- ids -------------------------------------------------------------------


def test_uuid7_is_sortable_and_versioned() -> None:
    ids = [uuid7() for _ in range(500)]
    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids)
    assert all(i[14] == "7" for i in ids)


# --- invariant 1: active revision is published -----------------------------


def test_active_revision_must_be_published() -> None:
    project_id = ProjectId("proj_1")
    draft = make_revision(project_id, status=PublicationStatus.DRAFT)
    project = Project(
        project_id=project_id,
        workspace_id=WORKSPACE,
        name="demo",
        active_revision_id=draft.revision_id,
    )
    problems = check_active_revision_published(project, [draft])
    assert problems and "expected published" in problems[0]


def test_new_project_without_revisions_is_legal() -> None:
    project = Project(workspace_id=WORKSPACE, name="fresh")
    assert check_active_revision_published(project, []) == []


def test_project_with_revisions_but_no_active_pointer_is_a_violation() -> None:
    project_id = ProjectId("proj_2")
    published = make_revision(project_id)
    project = Project(project_id=project_id, workspace_id=WORKSPACE, name="demo")
    assert check_active_revision_published(project, [published])


def test_archived_project_may_have_no_active_revision() -> None:
    project_id = ProjectId("proj_3")
    published = make_revision(project_id)
    project = Project(
        project_id=project_id, workspace_id=WORKSPACE, name="old", archived=True
    )
    assert check_active_revision_published(project, [published]) == []


# --- invariant 2: runs pin a published revision ----------------------------


def test_run_against_draft_revision_is_rejected() -> None:
    project_id = ProjectId("proj_4")
    draft = make_revision(project_id, status=PublicationStatus.DRAFT)
    run = make_run(project_id, draft.revision_id)
    problems = check_run_pins_published_revision(run, draft)
    assert problems and "draft revision" in problems[0]


def test_run_against_superseded_revision_is_allowed() -> None:
    # History must stay executable after a newer revision ships.
    project_id = ProjectId("proj_5")
    old = make_revision(project_id, status=PublicationStatus.SUPERSEDED)
    run = make_run(project_id, old.revision_id)
    assert check_run_pins_published_revision(run, old) == []


def test_run_with_missing_revision_is_rejected() -> None:
    run = make_run(ProjectId("proj_6"), RevisionId("rev_gone"))
    assert check_run_pins_published_revision(run, None)


# --- invariant 3: attempts belong to one run -------------------------------


def test_attempt_must_match_its_run() -> None:
    project_id = ProjectId("proj_7")
    revision = make_revision(project_id)
    run = make_run(project_id, revision.revision_id)
    foreign = ExecutionAttempt(
        run_id=run.run_id,
        project_id=ProjectId("proj_other"),
        revision_id=revision.revision_id,
        principal_id=PRINCIPAL,
    )
    problems = check_attempt_belongs_to_run(foreign, run)
    assert problems and "project" in problems[0]


def test_matching_attempt_passes() -> None:
    project_id = ProjectId("proj_8")
    revision = make_revision(project_id)
    run = make_run(project_id, revision.revision_id)
    attempt = ExecutionAttempt(
        run_id=run.run_id,
        project_id=project_id,
        revision_id=revision.revision_id,
        principal_id=PRINCIPAL,
    )
    assert check_attempt_belongs_to_run(attempt, run) == []


# --- invariant 4: artifacts are not overwritten ----------------------------


def test_same_location_with_two_digests_is_a_violation() -> None:
    project_id = ProjectId("proj_9")
    revision_id = make_revision(project_id).revision_id
    first = make_artifact(project_id, revision_id, "aaa")
    clobbered = first.model_copy(update={"digest": Digest(value="bbb")})
    problems = check_artifacts_not_overwritten([first, clobbered])
    assert problems and "two digests" in problems[0]


def test_new_version_of_same_logical_name_is_allowed() -> None:
    project_id = ProjectId("proj_10")
    revision_id = make_revision(project_id).revision_id
    v1 = make_artifact(project_id, revision_id, "aaa")
    v2 = make_artifact(project_id, revision_id, "bbb")
    assert check_artifacts_not_overwritten([v1, v2]) == []


# --- invariant 6: cross-project access needs a grant -----------------------


def test_cross_project_access_denied_without_grant() -> None:
    resource = Resource(
        project_id=ProjectId("proj_owner"),
        revision_id=RevisionId("rev_1"),
        resource_type=ResourceType.DATASET,
        name="customers",
    )
    assert check_cross_project_access_granted(resource, "proj_other")
    assert (
        check_cross_project_access_granted(resource, "proj_other", ["proj_other"]) == []
    )
    assert check_cross_project_access_granted(resource, "proj_owner") == []


# --- invariant 7: external actions need authorization + destination --------


def test_external_action_requires_a_decision() -> None:
    assert check_external_action_authorized(None, "api.example.com")


def test_external_action_requires_a_destination() -> None:
    decision = PolicyDecision(
        policy_set_version="1",
        input_context_digest="sha256:ctx",
        effect=PolicyEffect.PERMIT,
    )
    problems = check_external_action_authorized(decision, None)
    assert problems and "no destination" in problems[0]


def test_denied_decision_blocks_external_action() -> None:
    decision = PolicyDecision(
        policy_set_version="1",
        input_context_digest="sha256:ctx",
        effect=PolicyEffect.DENY,
    )
    assert check_external_action_authorized(decision, "api.example.com")


def test_require_approval_is_not_allowed() -> None:
    decision = PolicyDecision(
        policy_set_version="1",
        input_context_digest="sha256:ctx",
        effect=PolicyEffect.REQUIRE_APPROVAL,
    )
    assert not decision.allowed


def test_permit_with_obligations_must_list_obligations() -> None:
    decision = PolicyDecision(
        policy_set_version="1",
        input_context_digest="sha256:ctx",
        effect=PolicyEffect.PERMIT_WITH_OBLIGATIONS,
    )
    problems = check_external_action_authorized(decision, "api.example.com")
    assert problems and "lists none" in problems[0]


# --- invariant 8: worker scope is minimal ----------------------------------


def test_capability_bound_to_another_project_is_rejected() -> None:
    project_id = ProjectId("proj_11")
    revision_id = RevisionId("rev_11")
    attempt = ExecutionAttempt(
        run_id="run_11",  # type: ignore[arg-type]
        project_id=project_id,
        revision_id=revision_id,
        principal_id=PRINCIPAL,
    )
    token = issue_capability(
        principal_id=PRINCIPAL,
        project_id=ProjectId("proj_elsewhere"),
        revision_id=revision_id,
    )
    problems = check_worker_scope_minimal(token, attempt)
    assert problems and "project" in problems[0]


def test_expired_capability_is_rejected() -> None:
    project_id = ProjectId("proj_12")
    revision_id = RevisionId("rev_12")
    attempt = ExecutionAttempt(
        run_id="run_12",  # type: ignore[arg-type]
        project_id=project_id,
        revision_id=revision_id,
        principal_id=PRINCIPAL,
    )
    token = issue_capability(
        principal_id=PRINCIPAL,
        project_id=project_id,
        revision_id=revision_id,
        ttl=timedelta(seconds=-1),
    )
    problems = check_worker_scope_minimal(token, attempt)
    assert problems and "expired" in problems[0]


def test_capability_permits_fails_closed() -> None:
    token = issue_capability(
        principal_id=PRINCIPAL,
        project_id=ProjectId("proj_13"),
        revision_id=RevisionId("rev_13"),
        actions=("read",),
        resource_scope=("res_allowed*",),
    )
    assert token.permits("read")
    assert not token.permits("write")
    assert token.permits("read", "res_allowed_1")
    assert not token.permits("read", "res_other")


def test_capability_without_scope_denies_resource_access() -> None:
    token = issue_capability(
        principal_id=PRINCIPAL,
        project_id=ProjectId("proj_14"),
        revision_id=RevisionId("rev_14"),
        actions=("read",),
    )
    assert not token.permits("read", "res_anything")


def test_default_capability_ttl_is_short() -> None:
    token = issue_capability(
        principal_id=PRINCIPAL,
        project_id=ProjectId("proj_15"),
        revision_id=RevisionId("rev_15"),
    )
    assert token.expires_at - token.issued_at <= timedelta(minutes=15)
    assert not token.is_expired()


# --- run state machine (§7.4) ----------------------------------------------


def test_terminal_states_have_no_exits() -> None:
    for state in (RunState.COMPLETED, RunState.FAILED, RunState.CANCELLED):
        assert is_terminal(state)
        assert not any(can_transition(state, target) for target in RunState)


def test_running_can_requeue_for_retry() -> None:
    assert can_transition(RunState.RUNNING, RunState.QUEUED)


def test_requested_cannot_skip_the_policy_gate() -> None:
    # Reaching execution without policy evaluation is the bypass §9.3 forbids.
    assert not can_transition(RunState.REQUESTED, RunState.RUNNING)
    assert not can_transition(RunState.REQUESTED, RunState.QUEUED)
    assert can_transition(RunState.REQUESTED, RunState.AWAITING_POLICY)


def test_policy_gate_may_require_approval_or_proceed() -> None:
    assert can_transition(RunState.AWAITING_POLICY, RunState.AWAITING_APPROVAL)
    assert can_transition(RunState.AWAITING_POLICY, RunState.PLANNING)
    assert can_transition(RunState.AWAITING_POLICY, RunState.FAILED)


def test_full_happy_path_is_walkable() -> None:
    path = [
        RunState.REQUESTED,
        RunState.AWAITING_POLICY,
        RunState.PLANNING,
        RunState.QUEUED,
        RunState.LEASED,
        RunState.RUNNING,
        RunState.COMMITTING,
        RunState.COMPLETED,
    ]
    for current, target in pairwise(path):
        assert can_transition(current, target), f"{current} -> {target}"


def test_expired_lease_returns_run_to_queue() -> None:
    assert can_transition(RunState.LEASED, RunState.QUEUED)


def test_committing_cannot_be_cancelled() -> None:
    # Interrupting a half-written commit is what §14.3 exists to prevent.
    assert not can_transition(RunState.COMMITTING, RunState.CANCELLED)
    assert can_transition(RunState.COMMITTING, RunState.COMPLETED)
    assert can_transition(RunState.COMMITTING, RunState.FAILED)


def test_non_terminal_states_are_cancellable() -> None:
    for state in (
        RunState.REQUESTED,
        RunState.AWAITING_POLICY,
        RunState.AWAITING_APPROVAL,
        RunState.PLANNING,
        RunState.QUEUED,
        RunState.LEASED,
        RunState.RUNNING,
    ):
        assert can_transition(state, RunState.CANCELLED), state


# --- operations ------------------------------------------------------------


def test_external_write_without_idempotency_is_not_retry_safe() -> None:
    op = Operation(
        operation_type="notify",
        side_effect_class=SideEffectClass.EXTERNAL_WRITE,
        idempotency=IdempotencyStrategy.NONE,
    )
    assert not op.retry_safe


def test_local_read_is_retry_safe() -> None:
    assert Operation(operation_type="ingest").retry_safe


def test_high_risk_operations_require_approval() -> None:
    assert Operation(
        operation_type="delete", risk_level=RiskLevel.CONSEQUENTIAL
    ).requires_approval
    assert not Operation(
        operation_type="read", risk_level=RiskLevel.READ_PROJECT_DATA
    ).requires_approval


# --- approvals -------------------------------------------------------------


def test_approval_does_not_cover_a_changed_operation() -> None:
    approval = Approval(
        project_id=ProjectId("proj_16"),
        requested_by=PRINCIPAL,
        action_summary="send email",
        operation_digest="sha256:original",
        state=ApprovalState.GRANTED,
    )
    assert approval.covers("sha256:original")
    assert not approval.covers("sha256:modified")


def test_pending_approval_covers_nothing() -> None:
    approval = Approval(
        project_id=ProjectId("proj_17"),
        requested_by=PRINCIPAL,
        action_summary="send email",
        operation_digest="sha256:x",
    )
    assert not approval.covers("sha256:x")


def test_expired_approval_covers_nothing() -> None:
    approval = Approval(
        project_id=ProjectId("proj_18"),
        requested_by=PRINCIPAL,
        action_summary="send email",
        operation_digest="sha256:x",
        state=ApprovalState.GRANTED,
        expires_at=utcnow() - timedelta(seconds=1),
    )
    assert not approval.covers("sha256:x")


# --- validation gate -------------------------------------------------------


def test_revision_with_errors_is_not_publishable() -> None:
    report = ValidationReport(
        issues=(
            ValidationIssue(
                severity=ValidationSeverity.ERROR, code="E1", message="bad source"
            ),
        )
    )
    revision = ProjectRevision(
        project_id=ProjectId("proj_19"),
        content_hash="sha256:x",
        created_by=PRINCIPAL,
        validation_report=report,
    )
    assert not revision.publishable


def test_warnings_do_not_block_publication() -> None:
    report = ValidationReport(
        issues=(
            ValidationIssue(
                severity=ValidationSeverity.WARNING, code="W1", message="slow query"
            ),
        )
    )
    revision = ProjectRevision(
        project_id=ProjectId("proj_20"),
        content_hash="sha256:x",
        created_by=PRINCIPAL,
        validation_report=report,
    )
    assert revision.publishable


# --- secrets ---------------------------------------------------------------


def test_secret_lease_never_renders_its_value() -> None:
    lease = SecretLease(
        reference_name="gmail_token",
        value="super-secret-value",
        expires_at=utcnow() + timedelta(minutes=1),
    )
    assert "super-secret-value" not in repr(lease)
    assert "super-secret-value" not in str(lease)
    assert "super-secret-value" not in f"{lease}"


# --- misc ------------------------------------------------------------------


def test_digest_is_self_describing() -> None:
    assert str(digest_bytes(b"hello")).startswith("sha256:")


def test_foundation_models_are_immutable() -> None:
    project = Project(workspace_id=WORKSPACE, name="demo")
    with pytest.raises(ValidationError):
        project.name = "renamed"  # type: ignore[misc]


def test_check_invariants_raises_with_every_violation() -> None:
    with pytest.raises(InvariantViolation) as exc:
        check_invariants(["first problem", "second problem"])
    assert len(exc.value.violations) == 2


def test_check_invariants_passes_when_clean() -> None:
    check_invariants([])


def test_attempt_defaults_to_pending() -> None:
    attempt = ExecutionAttempt(
        run_id="run_x",  # type: ignore[arg-type]
        project_id=ProjectId("proj_21"),
        revision_id=RevisionId("rev_21"),
        principal_id=PRINCIPAL,
    )
    assert attempt.state is AttemptState.PENDING
    assert attempt.attempt_number == 1


def test_capability_token_is_frozen() -> None:
    token: CapabilityToken = issue_capability(
        principal_id=PRINCIPAL,
        project_id=ProjectId("proj_22"),
        revision_id=RevisionId("rev_22"),
    )
    with pytest.raises(ValidationError):
        token.actions = ("write",)  # type: ignore[misc]
