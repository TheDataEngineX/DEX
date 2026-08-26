"""Application services (§5.3, §5.4, §6.3, §7.4).

The claims under test are the ones the old design got wrong:

* a config change is a new revision, not an in-place edit with a ``.yaml.bak``
* a draft that fails validation publishes nothing
* rollback re-points without destroying history
* a run is authorized before it exists, and reaches the queue only via §7.4
* a replayed command returns the original run instead of creating a second
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from textwrap import dedent

import pytest

from dataenginex.application import (
    NotFoundError,
    PolicyDenied,
    ProjectService,
    PublishRejected,
    ResourceService,
    RunService,
    WorkloadService,
)
from dataenginex.domains.security import DEFAULT_POLICY_SET, GovernanceService
from dataenginex.foundation import (
    Policy,
    PolicyEffect,
    PrincipalId,
    ProjectId,
    ResourceType,
    RevisionId,
    RiskLevel,
    RunState,
    utcnow,
)
from dataenginex.runtime.state import ControlStore

PROJECT = ProjectId("proj_app")
ALICE = PrincipalId("prin_alice")

MANIFEST = dedent("""\
    apiVersion: dex/v1alpha1
    kind: Project
    metadata:
      name: app-fixture
    spec:
      profile: lite
      capabilities:
        required: [data.batch]
      limits:
        cpu: 1
        memory: 1GiB
        working_storage: 1GiB
      resources:
        - name: orders_csv
          type: dataset
          classification: internal
          config:
            path: data/orders.csv
            format: csv
      workloads:
        - name: load_orders
          kind: batch
          operations:
            - type: ingest
              name: read_orders
              outputs: [orders_csv]
""")


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
                (PROJECT, "ws_1", "app-fixture", now),
            )
        yield s


@pytest.fixture
def source(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    (root / "dex.yaml").write_text(MANIFEST)
    (root / "dex.lock").write_text(
        'python_version = "3.13"\n\n[dependencies]\ndataenginex = "0.6.0"\n'
    )
    return root


def permissive_runs(store: ControlStore) -> RunService:
    """A run service that permits the fixture workload and nothing else.

    Appends to the default policy set rather than replacing it, so the
    restricted-data and egress rules still apply — a test that disabled them
    would be testing a system nobody ships.
    """
    permit = Policy(
        name="test-permit-run",
        effect=PolicyEffect.PERMIT,
        actions=("run:load_orders",),
        max_risk_level=RiskLevel.CREATE_LOCAL_ARTIFACT,
        priority=500,
    )
    return RunService(
        store, governance=GovernanceService(store, policies=(*DEFAULT_POLICY_SET, permit))
    )


# --- publishing (§6.3) ------------------------------------------------------


def test_publishing_compiles_and_activates(store: ControlStore, source: Path) -> None:
    revision = ProjectService(store).publish(PROJECT, source, principal_id=ALICE)

    assert revision.is_active
    assert revision.content_hash.startswith("sha256:")
    # The lock is hashed into the revision, so "same revision" also means
    # "same declared dependencies".
    assert revision.dependency_lock_hash


def test_identical_content_does_not_fork_history(store: ControlStore, source: Path) -> None:
    """Republishing unchanged content reuses the revision.

    A new id per publish would make history unreadable — you could not tell a
    real change from someone clicking save twice.
    """
    service = ProjectService(store)
    first = service.publish(PROJECT, source, principal_id=ALICE)
    second = service.publish(PROJECT, source, principal_id=ALICE)

    assert first.revision_id == second.revision_id
    assert len(service.list_revisions(PROJECT)) == 1


def test_changed_content_creates_a_new_revision(store: ControlStore, source: Path) -> None:
    service = ProjectService(store)
    first = service.publish(PROJECT, source, principal_id=ALICE)

    (source / "dex.yaml").write_text(MANIFEST.replace("app-fixture", "app-fixture-2"))
    second = service.publish(PROJECT, source, principal_id=ALICE)

    assert second.revision_id != first.revision_id
    assert second.is_active
    assert len(service.list_revisions(PROJECT)) == 2


def test_an_invalid_draft_publishes_nothing(store: ControlStore, source: Path) -> None:
    """Fail closed (§6.8).

    The old engine discarded ``validate_config``'s errors and published anyway.
    Here the store is untouched and the report survives for the caller to render.
    """
    service = ProjectService(store)
    (source / "dex.yaml").write_text(MANIFEST.replace("profile: lite", "profile: nonsense"))

    with pytest.raises(PublishRejected) as exc:
        service.publish(PROJECT, source, principal_id=ALICE)

    assert exc.value.report.issues
    assert service.list_revisions(PROJECT) == []


def test_a_failed_publish_leaves_the_previous_revision_serving(
    store: ControlStore, source: Path
) -> None:
    """A bad edit must not take a working project down."""
    service = ProjectService(store)
    good = service.publish(PROJECT, source, principal_id=ALICE)

    (source / "dex.yaml").write_text("not: a: valid: manifest")
    with pytest.raises(PublishRejected):
        service.publish(PROJECT, source, principal_id=ALICE)

    assert service.active_revision_summary(PROJECT).revision_id == good.revision_id


def test_rollback_repoints_without_destroying_history(store: ControlStore, source: Path) -> None:
    """§6.3: rollback re-points, it does not mutate.

    The abandoned revision stays addressable — rolling forward must remain
    possible, and an incident review needs to see what caused it.
    """
    service = ProjectService(store)
    first = service.publish(PROJECT, source, principal_id=ALICE)
    (source / "dex.yaml").write_text(MANIFEST.replace("app-fixture", "app-fixture-2"))
    second = service.publish(PROJECT, source, principal_id=ALICE)

    rolled = service.rollback(PROJECT, first.revision_id)

    assert rolled.revision_id == first.revision_id
    assert rolled.is_active
    # Both revisions still exist; nothing was deleted to make room.
    assert {r.revision_id for r in service.list_revisions(PROJECT)} == {
        first.revision_id,
        second.revision_id,
    }


def test_rollback_leaves_the_statuses_agreeing(store: ControlStore, source: Path) -> None:
    """The active revision must not be recorded as superseded.

    Publishing supersedes the previous revision, so the one a user rolls back to
    is superseded by definition. Moving only the pointer left the store saying
    two things at once, and requiring ``published`` to roll back made rollback
    impossible — the only revision with that status is the active one.
    """
    service = ProjectService(store)
    first = service.publish(PROJECT, source, principal_id=ALICE)
    (source / "dex.yaml").write_text(MANIFEST.replace("app-fixture", "app-fixture-2"))
    second = service.publish(PROJECT, source, principal_id=ALICE)

    service.rollback(PROJECT, first.revision_id)

    by_id = {r.revision_id: r for r in service.list_revisions(PROJECT)}
    assert by_id[first.revision_id].is_active
    assert not by_id[second.revision_id].is_active
    # And rolling forward again still works.
    assert service.rollback(PROJECT, second.revision_id).is_active


def test_rollback_refuses_a_foreign_revision(store: ControlStore, source: Path) -> None:
    service = ProjectService(store)
    service.publish(PROJECT, source, principal_id=ALICE)

    with pytest.raises(NotFoundError):
        service.rollback(PROJECT, RevisionId("rev_from_another_project"))


# --- resources and workloads (§4.6, §4.9) -----------------------------------


def test_workloads_are_scoped_to_the_active_revision(store: ControlStore, source: Path) -> None:
    """A list from a superseded revision would offer a run button for a
    definition that is no longer current."""
    ProjectService(store).publish(PROJECT, source, principal_id=ALICE)
    revision = ProjectService(store).active_revision_summary(PROJECT)

    # No hand-written row: publishing writes the workload. It did not until the
    # gateway's publish path was made to go through ``RevisionService``, and
    # inserting one here is what hid that.
    workloads = WorkloadService(store).list_workloads(PROJECT)
    assert [w.name for w in workloads] == ["load_orders"]
    assert workloads[0].revision_id == revision.revision_id


def test_a_workload_carries_its_last_run_state(store: ControlStore, source: Path) -> None:
    """Folded into the list query. Fetching it per row is the N+1 that makes a
    pipelines page slow."""
    ProjectService(store).publish(PROJECT, source, principal_id=ALICE)
    revision = ProjectService(store).active_revision_summary(PROJECT)
    now = utcnow().isoformat()

    with store.transaction() as tx:
        tx.execute(
            "INSERT INTO runs (run_id, project_id, revision_id, workload_name, kind, state, "
            "trigger_type, requested_by, created_at, attempt_count) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                "run_1",
                PROJECT,
                revision.revision_id,
                "load_orders",
                "batch",
                RunState.COMPLETED.value,
                "manual",
                ALICE,
                now,
                1,
            ),
        )

    assert WorkloadService(store).list_workloads(PROJECT)[0].last_run_state == "completed"


def test_searching_resources_is_typed(store: ControlStore, source: Path) -> None:
    """An untyped filter string would be a SQL-injection surface the gateway
    cannot validate."""
    ProjectService(store).publish(PROJECT, source, principal_id=ALICE)

    # Publishing writes the resource. The manifest declares ``orders_csv`` as a
    # ``csv`` resource — a *connector* kind, which used to be stored in
    # ``resource_type`` and made every read of it raise "'csv' is not a valid
    # ResourceType". It classifies as a dataset; the connector lives in facets.
    found = ResourceService(store).list_by_type(PROJECT, ResourceType.DATASET)
    assert [r.name for r in found] == ["orders_csv"]
    assert ResourceService(store).get_by_name(PROJECT, "orders_csv").name == "orders_csv"


# --- runs (§7.4, §13.4) -----------------------------------------------------


def test_a_run_is_queued_not_executed(store: ControlStore, source: Path) -> None:
    """§17 Phase 1: no workload runs in the calling process.

    The service records intent and hands off. Reaching QUEUED without executing
    is the whole point.
    """
    ProjectService(store).publish(PROJECT, source, principal_id=ALICE)
    service = permissive_runs(store)

    accepted = service.request_run(PROJECT, "load_orders", principal_id=ALICE)

    assert service.get_run(accepted.run_id).state is RunState.QUEUED
    assert accepted.decision_id, "a run must cite the decision that permitted it"


def test_a_replayed_command_returns_the_original_run(store: ControlStore, source: Path) -> None:
    """§13.4. Without this a client that retries after a timeout starts a
    second run and double-counts."""
    ProjectService(store).publish(PROJECT, source, principal_id=ALICE)
    service = permissive_runs(store)

    first = service.request_run(PROJECT, "load_orders", principal_id=ALICE, idempotency_key="k")
    second = service.request_run(PROJECT, "load_orders", principal_id=ALICE, idempotency_key="k")

    assert second.run_id == first.run_id
    assert second.replayed and not first.replayed


def test_a_denied_request_leaves_no_run_behind(store: ControlStore, source: Path) -> None:
    """A refused request must leave no run behind, or the run list becomes a
    list of things that never happened.

    Denied by an explicit rule rather than by default deny: the default set
    permits a project its own workloads, since an installation that cannot run
    the project it was just given is not usable. What is asserted here is the
    consequence of a denial, whatever produced it.
    """
    ProjectService(store).publish(PROJECT, source, principal_id=ALICE)
    deny = Policy(
        name="test-deny-load",
        effect=PolicyEffect.DENY,
        actions=("run:load_orders",),
        priority=900,
    )
    service = RunService(
        store, governance=GovernanceService(store, policies=(*DEFAULT_POLICY_SET, deny))
    )

    with pytest.raises(PolicyDenied):
        service.request_run(PROJECT, "load_orders", principal_id=ALICE)

    runs, _ = service.list_runs(PROJECT)
    assert runs == []


def test_a_run_requires_a_published_revision(store: ControlStore) -> None:
    """ADR-0003. Running against working files makes "which definition ran?"
    unanswerable, so it is refused rather than guessed."""
    service = permissive_runs(store)

    with pytest.raises(NotFoundError):
        service.request_run(PROJECT, "load_orders", principal_id=ALICE)


def test_cancelling_a_queued_run_works(store: ControlStore, source: Path) -> None:
    ProjectService(store).publish(PROJECT, source, principal_id=ALICE)
    service = permissive_runs(store)
    accepted = service.request_run(PROJECT, "load_orders", principal_id=ALICE)

    assert service.cancel_run(accepted.run_id).state is RunState.CANCELLED


def test_cancelling_a_finished_run_is_refused(store: ControlStore, source: Path) -> None:
    """A terminal run has nothing to cancel, and pretending otherwise would let
    a UI report a completed run as cancelled."""
    ProjectService(store).publish(PROJECT, source, principal_id=ALICE)
    service = permissive_runs(store)
    accepted = service.request_run(PROJECT, "load_orders", principal_id=ALICE)

    with store.transaction() as tx:
        tx.execute(
            "UPDATE runs SET state = ? WHERE run_id = ?",
            (RunState.COMPLETED.value, accepted.run_id),
        )

    with pytest.raises(Exception, match="cannot be cancelled"):
        service.cancel_run(accepted.run_id)


def test_run_listing_paginates_by_cursor(store: ControlStore, source: Path) -> None:
    """§13.8. Offsets skip or repeat rows when new runs arrive mid-pagination,
    which for a run list is the normal case rather than an edge one."""
    ProjectService(store).publish(PROJECT, source, principal_id=ALICE)
    service = permissive_runs(store)
    for i in range(5):
        service.request_run(PROJECT, "load_orders", principal_id=ALICE, idempotency_key=f"k{i}")

    page, cursor = service.list_runs(PROJECT, limit=2)
    assert len(page) == 2
    assert cursor is not None

    rest, _ = service.list_runs(PROJECT, cursor=cursor, limit=10)
    assert {r.run_id for r in rest}.isdisjoint({r.run_id for r in page})
