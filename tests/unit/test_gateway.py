"""The gateway contract (§13.2-13.8).

The properties that matter here are the ones a client depends on: a retried
command does not create a second run, the policy gate cannot be skipped, a run
always pins a published revision, and error codes stay stable enough to branch
on.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from dataenginex.domains.security import GovernanceService
from dataenginex.foundation import (
    Policy,
    PolicyEffect,
    PrincipalId,
    PrincipalType,
    ProjectId,
    ResourceType,
    RevisionId,
    RunId,
    RunState,
    utcnow,
)
from dataenginex.interfaces import (
    Command,
    DexGateway,
    EmbeddedGateway,
    ErrorCode,
    GatewayError,
    Query,
)
from dataenginex.runtime.state import ControlStore

PROJECT = ProjectId("proj_test")
REVISION = RevisionId("rev_test")
ALICE = PrincipalId("prin_alice")


@pytest.fixture
def store(tmp_path: Path) -> Iterator[ControlStore]:
    with ControlStore(tmp_path / "control.db") as s:
        s.migrate()
        now = utcnow().isoformat()
        with s.transaction() as tx:
            tx.execute(
                "INSERT INTO installations (installation_id, name, created_at) VALUES (?,?,?)",
                ("inst_1", "test", now),
            )
            tx.execute(
                "INSERT INTO workspaces (workspace_id, installation_id, name, created_at) "
                "VALUES (?,?,?,?)",
                ("ws_1", "inst_1", "default", now),
            )
            tx.execute(
                "INSERT INTO projects (project_id, workspace_id, name, created_at) "
                "VALUES (?,?,?,?)",
                (PROJECT, "ws_1", "test-project", now),
            )
            tx.execute(
                "INSERT INTO project_revisions (revision_id, project_id, content_hash, "
                "created_by, created_at, manifest_schema_version, status) "
                "VALUES (?,?,?,?,?,?,?)",
                (REVISION, PROJECT, "sha256:abc", ALICE, now, "dex/v1alpha1", "published"),
            )
            tx.execute(
                "UPDATE projects SET active_revision_id = ? WHERE project_id = ?",
                (REVISION, PROJECT),
            )
            tx.execute(
                "INSERT INTO principals (principal_id, principal_type, name, created_at) "
                "VALUES (?,?,?,?)",
                (ALICE, PrincipalType.HUMAN.value, "alice", now),
            )
        yield s


@pytest.fixture
def gateway(store: ControlStore) -> EmbeddedGateway:
    """A gateway whose policy permits ordinary runs."""
    governance = GovernanceService(
        store,
        policies=[
            Policy(name="permit-runs", effect=PolicyEffect.PERMIT, actions=("run:*",)),
        ],
    )
    return EmbeddedGateway(store, governance=governance)


def command(**overrides: object) -> Command:
    defaults: dict[str, object] = {"principal_id": ALICE, "project_id": PROJECT}
    defaults.update(overrides)
    return Command(**defaults)  # type: ignore[arg-type]


# --- the protocol ------------------------------------------------------------


def test_embedded_gateway_satisfies_the_protocol(gateway: EmbeddedGateway) -> None:
    """Clients written against DexGateway must accept this implementation."""
    assert isinstance(gateway, DexGateway)


def test_error_codes_are_stable_strings() -> None:
    """Clients branch on these; renaming one is a breaking change."""
    assert ErrorCode.POLICY_DENIED.value == "E_POLICY_DENIED"
    assert ErrorCode.NOT_FOUND.value == "E_NOT_FOUND"
    assert ErrorCode.APPROVAL_REQUIRED.value == "E_APPROVAL_REQUIRED"


def test_gateway_error_serializes_for_any_transport() -> None:
    error = GatewayError(
        ErrorCode.POLICY_DENIED, "denied", details={"decision_id": "dec_1"}
    )

    assert error.to_dict() == {
        "code": "E_POLICY_DENIED",
        "message": "denied",
        "details": {"decision_id": "dec_1"},
    }


# --- starting runs -----------------------------------------------------------


def test_start_run_creates_a_queued_run(gateway: EmbeddedGateway) -> None:
    result = gateway.start_run(command(), workload="daily")

    assert result.accepted
    assert result.subject_id is not None

    summary = gateway.get_run(Query(principal_id=ALICE), run_id=RunId(result.subject_id))
    assert summary.state is RunState.QUEUED
    assert summary.workload_name == "daily"


def test_a_run_pins_the_published_revision(gateway: EmbeddedGateway) -> None:
    """ADR-0003: 'what definition did this run use?' always has an answer."""
    result = gateway.start_run(command(), workload="daily")

    summary = gateway.get_run(Query(principal_id=ALICE), run_id=RunId(result.subject_id or ""))

    assert summary.revision_id == REVISION


def test_a_project_without_a_published_revision_cannot_run(
    store: ControlStore, gateway: EmbeddedGateway
) -> None:
    """Running against working files would make provenance unanswerable."""
    with store.transaction() as tx:
        tx.execute(
            "UPDATE projects SET active_revision_id = NULL WHERE project_id = ?", (PROJECT,)
        )

    with pytest.raises(GatewayError) as excinfo:
        gateway.start_run(command(), workload="daily")

    assert excinfo.value.code is ErrorCode.REVISION_NOT_PUBLISHED


def test_a_command_without_a_project_is_rejected(gateway: EmbeddedGateway) -> None:
    with pytest.raises(GatewayError) as excinfo:
        gateway.start_run(command(project_id=None), workload="daily")

    assert excinfo.value.code is ErrorCode.INVALID_REQUEST


def test_an_unknown_project_is_not_found(gateway: EmbeddedGateway) -> None:
    with pytest.raises(GatewayError) as excinfo:
        gateway.start_run(command(project_id=ProjectId("proj_ghost")), workload="daily")

    assert excinfo.value.code is ErrorCode.NOT_FOUND


# --- idempotency (§13.4) -----------------------------------------------------


def test_a_replayed_command_returns_the_original_run(gateway: EmbeddedGateway) -> None:
    """A client retrying after a timeout must not create a second run."""
    first = gateway.start_run(command(idempotency_key="key-1"), workload="daily")
    second = gateway.start_run(command(idempotency_key="key-1"), workload="daily")

    assert second.subject_id == first.subject_id
    assert second.replayed
    assert not first.replayed


def test_different_keys_create_different_runs(gateway: EmbeddedGateway) -> None:
    first = gateway.start_run(command(idempotency_key="key-1"), workload="daily")
    second = gateway.start_run(command(idempotency_key="key-2"), workload="daily")

    assert first.subject_id != second.subject_id


def test_commands_without_a_key_are_not_deduplicated(gateway: EmbeddedGateway) -> None:
    """Absent a key there is no basis for calling two requests the same."""
    first = gateway.start_run(command(), workload="daily")
    second = gateway.start_run(command(), workload="daily")

    assert first.subject_id != second.subject_id


# --- the policy gate ---------------------------------------------------------


def test_a_denied_run_is_never_created(store: ControlStore) -> None:
    """The bypass this layer exists to close: no run row on a denial."""
    denying = GovernanceService(
        store, policies=[Policy(name="deny-all", effect=PolicyEffect.DENY, actions=("*",))]
    )
    gateway = EmbeddedGateway(store, governance=denying)

    with pytest.raises(GatewayError) as excinfo:
        gateway.start_run(command(), workload="daily")

    assert excinfo.value.code is ErrorCode.POLICY_DENIED
    assert store.query("SELECT run_id FROM runs") == []


def test_a_denial_carries_the_decision_id(store: ControlStore) -> None:
    """So a user can be shown why, not just that it failed."""
    denying = GovernanceService(
        store, policies=[Policy(name="deny-all", effect=PolicyEffect.DENY, actions=("*",))]
    )

    with pytest.raises(GatewayError) as excinfo:
        EmbeddedGateway(store, governance=denying).start_run(command(), workload="daily")

    assert "decision_id" in excinfo.value.details


def test_every_run_leaves_a_policy_decision(
    gateway: EmbeddedGateway, store: ControlStore
) -> None:
    gateway.start_run(command(), workload="daily")

    decisions = store.query("SELECT * FROM policy_decisions WHERE action LIKE 'run:%'")

    assert len(decisions) == 1
    assert decisions[0]["effect"] == PolicyEffect.PERMIT.value


# --- cancellation (§14.7) ----------------------------------------------------


def test_a_queued_run_can_be_cancelled(gateway: EmbeddedGateway) -> None:
    started = gateway.start_run(command(), workload="daily")
    run_id = RunId(started.subject_id or "")

    gateway.cancel_run(command(), run_id=run_id)

    summary = gateway.get_run(Query(principal_id=ALICE), run_id=run_id)
    assert summary.state is RunState.CANCELLED
    assert summary.is_terminal


def test_a_completed_run_cannot_be_cancelled(
    gateway: EmbeddedGateway, store: ControlStore
) -> None:
    started = gateway.start_run(command(), workload="daily")
    run_id = RunId(started.subject_id or "")
    with store.transaction() as tx:
        tx.execute(
            "UPDATE runs SET state = ? WHERE run_id = ?", (RunState.COMPLETED.value, run_id)
        )

    with pytest.raises(GatewayError) as excinfo:
        gateway.cancel_run(command(), run_id=run_id)

    assert excinfo.value.code is ErrorCode.CONFLICT


def test_a_committing_run_cannot_be_cancelled(
    gateway: EmbeddedGateway, store: ControlStore
) -> None:
    """§14.3: interrupting a commit orphans the artifact it was registering."""
    started = gateway.start_run(command(), workload="daily")
    run_id = RunId(started.subject_id or "")
    with store.transaction() as tx:
        tx.execute(
            "UPDATE runs SET state = ? WHERE run_id = ?", (RunState.COMMITTING.value, run_id)
        )

    with pytest.raises(GatewayError) as excinfo:
        gateway.cancel_run(command(), run_id=run_id)

    assert excinfo.value.code is ErrorCode.CONFLICT


def test_cancelling_an_unknown_run_is_not_found(gateway: EmbeddedGateway) -> None:
    with pytest.raises(GatewayError) as excinfo:
        gateway.cancel_run(command(), run_id=RunId("run_ghost"))

    assert excinfo.value.code is ErrorCode.NOT_FOUND


# --- queries and pagination (§13.8) ------------------------------------------


def test_get_run_reports_an_unknown_run(gateway: EmbeddedGateway) -> None:
    with pytest.raises(GatewayError) as excinfo:
        gateway.get_run(Query(principal_id=ALICE), run_id=RunId("run_ghost"))

    assert excinfo.value.code is ErrorCode.NOT_FOUND


def test_list_runs_returns_newest_first(gateway: EmbeddedGateway) -> None:
    for index in range(3):
        gateway.start_run(command(idempotency_key=f"key-{index}"), workload=f"w{index}")

    page = gateway.list_runs(Query(principal_id=ALICE, project_id=PROJECT))

    assert len(page.items) == 3
    assert [item.workload_name for item in page.items] == ["w2", "w1", "w0"]


def test_list_runs_paginates_with_a_cursor(gateway: EmbeddedGateway) -> None:
    for index in range(5):
        gateway.start_run(command(idempotency_key=f"key-{index}"), workload=f"w{index}")

    first = gateway.list_runs(Query(principal_id=ALICE, project_id=PROJECT, limit=2))

    assert len(first.items) == 2
    assert first.has_more
    assert first.next_cursor is not None

    second = gateway.list_runs(
        Query(principal_id=ALICE, project_id=PROJECT, limit=2, cursor=first.next_cursor)
    )

    assert len(second.items) == 2
    # No overlap between pages — the failure mode offsets have.
    assert not {i.run_id for i in first.items} & {i.run_id for i in second.items}


def test_the_last_page_reports_no_more(gateway: EmbeddedGateway) -> None:
    gateway.start_run(command(), workload="only")

    page = gateway.list_runs(Query(principal_id=ALICE, project_id=PROJECT, limit=10))

    assert not page.has_more
    assert page.next_cursor is None


def test_list_runs_filters_by_state(gateway: EmbeddedGateway) -> None:
    first = gateway.start_run(command(idempotency_key="a"), workload="a")
    gateway.start_run(command(idempotency_key="b"), workload="b")
    gateway.cancel_run(command(), run_id=RunId(first.subject_id or ""))

    cancelled = gateway.list_runs(
        Query(principal_id=ALICE, project_id=PROJECT), state=RunState.CANCELLED
    )

    assert [item.workload_name for item in cancelled.items] == ["a"]


def test_list_runs_scopes_to_a_project(store: ControlStore, gateway: EmbeddedGateway) -> None:
    """A project must not see another's runs."""
    gateway.start_run(command(), workload="mine")

    page = gateway.list_runs(Query(principal_id=ALICE, project_id=ProjectId("proj_other")))

    assert page.items == ()


# --- queries do not mutate ---------------------------------------------------


def test_queries_change_nothing(gateway: EmbeddedGateway, store: ControlStore) -> None:
    """§13.3: a read endpoint that mutates makes auditing unsound."""
    gateway.start_run(command(), workload="daily")
    before = store.query("SELECT run_id, state FROM runs")

    gateway.list_runs(Query(principal_id=ALICE, project_id=PROJECT))
    gateway.list_approvals(Query(principal_id=ALICE, project_id=PROJECT))

    after = store.query("SELECT run_id, state FROM runs")
    assert [dict(r) for r in before] == [dict(r) for r in after]


def test_queries_write_no_audit_events(
    gateway: EmbeddedGateway, store: ControlStore
) -> None:
    gateway.start_run(command(), workload="daily")
    before = len(store.query("SELECT event_id FROM audit_events"))

    gateway.list_runs(Query(principal_id=ALICE, project_id=PROJECT))

    assert len(store.query("SELECT event_id FROM audit_events")) == before


# --- approvals ---------------------------------------------------------------


def test_deciding_an_unknown_approval_conflicts(gateway: EmbeddedGateway) -> None:
    with pytest.raises(GatewayError) as excinfo:
        gateway.decide_approval(command(), approval_id="appr_ghost", granted=True)

    assert excinfo.value.code is ErrorCode.CONFLICT


def test_publishing_an_invalid_project_reports_every_issue(
    gateway: EmbeddedGateway, tmp_path: Path
) -> None:
    """§6.8 fails closed, and the gateway carries the report across (§13.5).

    ``details`` holds the structured issues rather than a sentence, so a UI can
    point at the manifest location that failed instead of asking the user to
    parse prose. An empty directory has no manifest at all, which is the
    simplest way for compilation to legitimately refuse.
    """
    with pytest.raises(GatewayError) as excinfo:
        gateway.publish_revision(command(), source=str(tmp_path))

    assert excinfo.value.code is ErrorCode.VALIDATION_FAILED
    assert excinfo.value.details["issues"]


# --- the query surface --------------------------------------------------------
#
# These are what Studio reads on almost every page (§13.3). The properties worth
# asserting are that a query never leaves the project it was scoped to, that a
# project with nothing published says so with a code the caller can branch on,
# and that reads stay reads.


def seed_catalog(store: ControlStore) -> None:
    """Two resources and one workload on the active revision."""
    now = utcnow().isoformat()
    with store.transaction() as tx:
        for resource_id, name, resource_type in (
            ("res_orders", "orders", "table"),
            ("res_orders_csv", "orders_csv", "dataset"),
            ("res_churn", "churn", "model"),
        ):
            tx.execute(
                "INSERT INTO resources (resource_id, project_id, revision_id, resource_type, "
                "name, classification, lifecycle_state, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (resource_id, PROJECT, REVISION, resource_type, name, "internal", "active", now),
            )
        tx.execute(
            "INSERT INTO workload_definitions (workload_id, project_id, revision_id, name, "
            "kind, definition_json, continuous, created_at) VALUES (?,?,?,?,?,?,?,?)",
            ("wl_daily", PROJECT, REVISION, "daily_load", "batch", '{"operations": []}', 0, now),
        )


def test_get_project_reports_the_active_revision(gateway: EmbeddedGateway) -> None:
    project = gateway.get_project(Query(principal_id=ALICE, project_id=PROJECT))

    assert project.name == "test-project"
    assert project.active_revision_id == REVISION
    assert project.content_hash == "sha256:abc"
    assert project.is_published


def test_a_project_with_nothing_published_says_so(
    gateway: EmbeddedGateway, store: ControlStore
) -> None:
    """``is_published`` is what gates the run button.

    A project exists before anything is published, and a UI that cannot tell the
    difference offers the user an action that will always fail.
    """
    with store.transaction() as tx:
        tx.execute(
            "INSERT INTO projects (project_id, workspace_id, name, created_at) VALUES (?,?,?,?)",
            ("proj_empty", "ws_1", "empty", utcnow().isoformat()),
        )

    project = gateway.get_project(Query(principal_id=ALICE, project_id=ProjectId("proj_empty")))

    assert project.active_revision_id is None
    assert not project.is_published


def test_an_unknown_project_is_not_found_by_query(gateway: EmbeddedGateway) -> None:
    with pytest.raises(GatewayError) as excinfo:
        gateway.get_project(Query(principal_id=ALICE, project_id=ProjectId("proj_ghost")))

    assert excinfo.value.code is ErrorCode.NOT_FOUND


def test_a_query_without_a_project_is_rejected(gateway: EmbeddedGateway) -> None:
    """Answering across every project would leak one workspace into another."""
    with pytest.raises(GatewayError) as excinfo:
        gateway.list_resources(Query(principal_id=ALICE))

    assert excinfo.value.code is ErrorCode.INVALID_REQUEST


def test_list_resources_returns_the_declared_catalog(
    gateway: EmbeddedGateway, store: ControlStore
) -> None:
    seed_catalog(store)

    page = gateway.list_resources(Query(principal_id=ALICE, project_id=PROJECT))

    assert {r.name for r in page.items} == {"orders", "orders_csv", "churn"}


def test_list_resources_filters_by_type(gateway: EmbeddedGateway, store: ControlStore) -> None:
    """The typed filter is the whole point — a UI asks for models, not a string."""
    seed_catalog(store)

    page = gateway.list_resources(
        Query(principal_id=ALICE, project_id=PROJECT), resource_type=ResourceType.MODEL
    )

    assert [r.name for r in page.items] == ["churn"]


def test_list_resources_scopes_to_the_project(
    gateway: EmbeddedGateway, store: ControlStore
) -> None:
    seed_catalog(store)
    with store.transaction() as tx:
        tx.execute(
            "INSERT INTO projects (project_id, workspace_id, name, created_at) VALUES (?,?,?,?)",
            ("proj_other", "ws_1", "other", utcnow().isoformat()),
        )
        tx.execute(
            "INSERT INTO resources (resource_id, project_id, revision_id, resource_type, "
            "name, classification, lifecycle_state, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (
                "res_secret",
                "proj_other",
                REVISION,
                "table",
                "secret",
                "restricted",
                "active",
                utcnow().isoformat(),
            ),
        )

    page = gateway.list_resources(Query(principal_id=ALICE, project_id=PROJECT))

    assert "secret" not in {r.name for r in page.items}


def test_get_resource_finds_one_by_name(gateway: EmbeddedGateway, store: ControlStore) -> None:
    seed_catalog(store)

    resource = gateway.get_resource(Query(principal_id=ALICE, project_id=PROJECT), name="orders")

    assert resource.resource_type is ResourceType.TABLE


def test_an_unknown_resource_is_not_found(gateway: EmbeddedGateway, store: ControlStore) -> None:
    seed_catalog(store)

    with pytest.raises(GatewayError) as excinfo:
        gateway.get_resource(Query(principal_id=ALICE, project_id=PROJECT), name="ghost")

    assert excinfo.value.code is ErrorCode.NOT_FOUND


def test_list_workloads_reports_the_active_revision(
    gateway: EmbeddedGateway, store: ControlStore
) -> None:
    seed_catalog(store)

    page = gateway.list_workloads(Query(principal_id=ALICE, project_id=PROJECT))

    assert [w.name for w in page.items] == ["daily_load"]


def test_list_workloads_folds_in_the_last_run_state(
    gateway: EmbeddedGateway, store: ControlStore
) -> None:
    """Every pipelines list shows it, so fetching it per row is the N+1 to avoid."""
    seed_catalog(store)
    gateway.start_run(command(), workload="daily_load")

    page = gateway.list_workloads(Query(principal_id=ALICE, project_id=PROJECT))

    assert page.items[0].last_run_state == RunState.QUEUED.value


def test_get_workload_definition_returns_what_the_manifest_said(
    gateway: EmbeddedGateway, store: ControlStore
) -> None:
    seed_catalog(store)

    definition = gateway.get_workload_definition(
        Query(principal_id=ALICE, project_id=PROJECT), name="daily_load"
    )

    assert definition == {"operations": []}


def test_an_unknown_workload_is_not_found(gateway: EmbeddedGateway, store: ControlStore) -> None:
    seed_catalog(store)

    with pytest.raises(GatewayError) as excinfo:
        gateway.get_workload(Query(principal_id=ALICE, project_id=PROJECT), name="ghost")

    assert excinfo.value.code is ErrorCode.NOT_FOUND


def test_workloads_of_an_unpublished_project_report_the_publish_problem(
    gateway: EmbeddedGateway, store: ControlStore
) -> None:
    """"Publish first" and "wrong id" are different fixes, so they get different codes."""
    with store.transaction() as tx:
        tx.execute(
            "INSERT INTO projects (project_id, workspace_id, name, created_at) VALUES (?,?,?,?)",
            ("proj_empty", "ws_1", "empty", utcnow().isoformat()),
        )

    with pytest.raises(GatewayError) as excinfo:
        gateway.list_workloads(Query(principal_id=ALICE, project_id=ProjectId("proj_empty")))

    assert excinfo.value.code is ErrorCode.REVISION_NOT_PUBLISHED


def test_get_revision_defaults_to_the_active_one(gateway: EmbeddedGateway) -> None:
    revision = gateway.get_revision(Query(principal_id=ALICE, project_id=PROJECT))

    assert revision.revision_id == REVISION
    assert revision.is_active


def test_list_revisions_is_the_rollback_menu(gateway: EmbeddedGateway) -> None:
    page = gateway.list_revisions(Query(principal_id=ALICE, project_id=PROJECT))

    assert [r.revision_id for r in page.items] == [REVISION]


def test_rollback_cannot_activate_an_unpublished_revision(
    gateway: EmbeddedGateway, store: ControlStore
) -> None:
    """Rolling onto a draft would put uncompiled content into production."""
    with store.transaction() as tx:
        tx.execute(
            "INSERT INTO project_revisions (revision_id, project_id, content_hash, created_by, "
            "created_at, manifest_schema_version, status) VALUES (?,?,?,?,?,?,?)",
            (
                "rev_draft",
                PROJECT,
                "sha256:draft",
                ALICE,
                utcnow().isoformat(),
                "dex/v1alpha1",
                "draft",
            ),
        )

    with pytest.raises(GatewayError) as excinfo:
        gateway.rollback_revision(command(), revision_id=RevisionId("rev_draft"))

    assert excinfo.value.code is ErrorCode.CONFLICT


def test_rollback_re_points_without_mutating_history(
    gateway: EmbeddedGateway, store: ControlStore
) -> None:
    """The superseded revision stays addressable — rolling forward must work."""
    with store.transaction() as tx:
        tx.execute(
            "INSERT INTO project_revisions (revision_id, project_id, content_hash, created_by, "
            "created_at, manifest_schema_version, status) VALUES (?,?,?,?,?,?,?)",
            (
                "rev_old",
                PROJECT,
                "sha256:old",
                ALICE,
                utcnow().isoformat(),
                "dex/v1alpha1",
                "published",
            ),
        )

    result = gateway.rollback_revision(command(), revision_id=RevisionId("rev_old"))

    assert result.subject_id == "rev_old"
    assert gateway.get_revision(Query(principal_id=ALICE, project_id=PROJECT)).revision_id == (
        RevisionId("rev_old")
    )
    assert len(gateway.list_revisions(Query(principal_id=ALICE, project_id=PROJECT)).items) == 2


def test_the_query_surface_writes_nothing(gateway: EmbeddedGateway, store: ControlStore) -> None:
    """A read that mutates makes both caching and auditing unsound (§13.3)."""
    seed_catalog(store)
    before = store.query("SELECT COUNT(*) AS n FROM audit_events")[0]["n"]
    query = Query(principal_id=ALICE, project_id=PROJECT)

    gateway.get_project(query)
    gateway.list_projects(query)
    gateway.get_revision(query)
    gateway.list_revisions(query)
    gateway.list_resources(query)
    gateway.get_resource(query, name="orders")
    gateway.list_workloads(query)
    gateway.get_workload(query, name="daily_load")
    gateway.get_workload_definition(query, name="daily_load")

    assert store.query("SELECT COUNT(*) AS n FROM audit_events")[0]["n"] == before


# --- governance queries -------------------------------------------------------
#
# These replace pages that used to render manifest config nothing evaluated
# (§9.3). The property that matters is that what is shown is what ran: a policy
# on this page is in the engine's live set, and a decision is one it recorded.


def seed_governance(store: ControlStore) -> None:
    """One installation-wide policy, one project policy, a permit and a denial."""
    now = utcnow().isoformat()
    with store.transaction() as tx:
        tx.execute(
            "INSERT INTO policies (policy_id, project_id, name, version, effect, "
            "definition_json, priority, created_at, enabled) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                "pol_global",
                None,
                "deny-restricted-egress",
                "1",
                "deny",
                '{"description": "installation-wide", "actions": ["egress:*"]}',
                900,
                now,
                1,
            ),
        )
        tx.execute(
            "INSERT INTO policies (policy_id, project_id, name, version, effect, "
            "definition_json, priority, created_at, enabled) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                "pol_project",
                PROJECT,
                "permit-runs",
                "1",
                "permit",
                '{"description": "project rule", "actions": ["run:*"]}',
                100,
                now,
                1,
            ),
        )
        for decision_id, effect, action in (
            ("dec_permit", "permit", "run:daily_load"),
            ("dec_deny", "deny", "egress:s3"),
        ):
            tx.execute(
                "INSERT INTO policy_decisions (decision_id, policy_set_version, "
                "input_context_digest, effect, obligations_json, matched_policies_json, "
                "reason, evaluated_by, evaluated_at, project_id, principal_id, action) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    decision_id,
                    "v1",
                    "sha256:ctx",
                    effect,
                    "[]",
                    '["pol_project"]',
                    f"{effect} by policy",
                    "engine",
                    now,
                    PROJECT,
                    ALICE,
                    action,
                ),
            )
        tx.execute(
            "INSERT INTO audit_events (event_id, occurred_at, producer, schema_version, "
            "event_type, action, outcome, detail_json, project_id, principal_id) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                "evt_1",
                now,
                "gateway",
                "v1",
                "run.requested",
                "run:daily_load",
                "success",
                "{}",
                PROJECT,
                ALICE,
            ),
        )


def test_list_policies_includes_installation_wide_rules(
    gateway: EmbeddedGateway, store: ControlStore
) -> None:
    """A project rule is not the only thing governing a project.

    Showing only project-scoped policies would tell an operator a run is
    unconstrained when an installation rule is what actually governs it.
    """
    seed_governance(store)

    page = gateway.list_policies(Query(principal_id=ALICE, project_id=PROJECT))

    assert [p.name for p in page.items] == ["deny-restricted-egress", "permit-runs"]


def test_policies_carry_their_definition(
    gateway: EmbeddedGateway, store: ControlStore
) -> None:
    seed_governance(store)

    page = gateway.list_policies(Query(principal_id=ALICE, project_id=PROJECT))
    project_policy = next(p for p in page.items if p.name == "permit-runs")

    assert project_policy.actions == ("run:*",)
    assert project_policy.description == "project rule"
    assert project_policy.enabled


def test_list_decisions_returns_what_the_engine_recorded(
    gateway: EmbeddedGateway, store: ControlStore
) -> None:
    seed_governance(store)

    page = gateway.list_decisions(Query(principal_id=ALICE, project_id=PROJECT))

    assert {d.decision_id for d in page.items} == {"dec_permit", "dec_deny"}


def test_denied_only_filters_in_the_query(
    gateway: EmbeddedGateway, store: ControlStore
) -> None:
    """Filtering in SQL, not Python: denials are rare relative to permits."""
    seed_governance(store)

    page = gateway.list_decisions(
        Query(principal_id=ALICE, project_id=PROJECT), denied_only=True
    )

    assert [d.decision_id for d in page.items] == ["dec_deny"]
    assert page.items[0].matched_policies == ("pol_project",)


def test_list_audit_events_separates_action_from_outcome(
    gateway: EmbeddedGateway, store: ControlStore
) -> None:
    """"Attempted" and "did" are different facts (§4.15)."""
    seed_governance(store)

    page = gateway.list_audit_events(Query(principal_id=ALICE, project_id=PROJECT))

    assert page.items[0].action == "run:daily_load"
    assert page.items[0].outcome == "success"


def test_audit_events_filter_by_action(
    gateway: EmbeddedGateway, store: ControlStore
) -> None:
    seed_governance(store)
    query = Query(principal_id=ALICE, project_id=PROJECT)

    assert gateway.list_audit_events(query, action="run:daily_load").items
    assert not gateway.list_audit_events(query, action="egress:s3").items


def test_governance_queries_scope_to_the_project(
    gateway: EmbeddedGateway, store: ControlStore
) -> None:
    """One project's decisions must not appear on another's security page."""
    seed_governance(store)
    other = Query(principal_id=ALICE, project_id=ProjectId("proj_other"))

    assert not gateway.list_decisions(other).items
    assert not gateway.list_audit_events(other).items
    # The installation-wide policy still applies, which is the point of it.
    assert [p.name for p in gateway.list_policies(other).items] == ["deny-restricted-egress"]
