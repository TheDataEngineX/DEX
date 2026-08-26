"""Lineage is written when a run commits (§8.5).

``LineageService`` was complete and had no caller, so ``lineage_edges`` stayed
empty and every lineage view rendered a blank graph — the same defect the old
design had, where ``parent_id`` was never set. These tests hold the wiring in
place: a successful run leaves edges behind, a failed one does not, and the
edges say what was read and what was written.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import pytest

from dataenginex.application.runs import RunService
from dataenginex.bootstrap.lite import build_lite_backend
from dataenginex.cli.worker import _execute
from dataenginex.domains.governance.lineage import LineageService
from dataenginex.domains.security import DEFAULT_POLICY_SET, GovernanceService
from dataenginex.foundation import (
    LineageRelation,
    Policy,
    PolicyEffect,
    PrincipalId,
    ProjectId,
    RevisionId,
    RiskLevel,
)
from dataenginex.runtime.compiler.revisions import RevisionService
from dataenginex.runtime.queue import DurableQueue, Scheduler
from dataenginex.runtime.state import ControlStore

PROJECT = ProjectId("proj_lineage")
PRINCIPAL = PrincipalId("prin_test")
TS = "2026-08-08T00:00:00+00:00"

ORDERS_CSV = "order_id,amount\n1,10\n2,20\n"

MANIFEST = """apiVersion: dex/v1alpha1
kind: Project
metadata:
  name: lineage
spec:
  resources:
    - name: orders_csv
      type: csv
      config:
        path: "{csv_path}"
  workloads:
    - name: load
      kind: batch
      operations:
        - name: load-orders
          type: ingest
          inputs: [orders_csv]
          outputs: [orders]
"""


@pytest.fixture
def store(tmp_path: Path) -> Iterator[ControlStore]:
    with ControlStore(tmp_path / "control.db") as s:
        s.migrate()
        with s.transaction() as tx:
            tx.execute(
                "INSERT INTO installations (installation_id, name, created_at) "
                "VALUES ('inst_1', 'test', ?)",
                (TS,),
            )
            tx.execute(
                "INSERT INTO workspaces (workspace_id, installation_id, name, created_at) "
                "VALUES ('ws_1', 'inst_1', 'default', ?)",
                (TS,),
            )
            tx.execute(
                "INSERT INTO projects (project_id, workspace_id, name, created_at) "
                "VALUES (?, 'ws_1', 'lineage', ?)",
                (PROJECT, TS),
            )
        yield s


@pytest.fixture(autouse=True)
def workspace_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "workspaces"
    monkeypatch.setenv("DEX_WORKSPACE_DIR", str(root))
    return root


@pytest.fixture
def csv_path(tmp_path: Path) -> Path:
    """Where the source points. The failure case deletes it."""
    path = tmp_path / "orders.csv"
    path.write_text(ORDERS_CSV)
    return path


@pytest.fixture
def revision(store: ControlStore, tmp_path: Path, csv_path: Path) -> RevisionId:
    root = tmp_path / "project"
    root.mkdir()
    (root / "dex.yaml").write_text(MANIFEST.format(csv_path=csv_path))
    service = RevisionService(store)
    draft, compiled = service.create_draft(PROJECT, root, PRINCIPAL)
    assert compiled.report.ok, compiled.report.errors
    service.publish(draft.revision_id, PRINCIPAL)
    return draft.revision_id


def _run_load(store: ControlStore, revision: RevisionId) -> str:
    """Queue the ``load`` workload and drain it as a worker would."""
    queue = DurableQueue(store, lease_duration=timedelta(minutes=5))
    queue.register_worker("worker-1", pool="batch")
    scheduler = Scheduler(store, queue)
    permit = Policy(
        name="test-permit-load",
        effect=PolicyEffect.PERMIT,
        actions=("run:load",),
        max_risk_level=RiskLevel.CREATE_LOCAL_ARTIFACT,
        priority=500,
    )
    runs = RunService(
        store,
        governance=GovernanceService(store, policies=(*DEFAULT_POLICY_SET, permit)),
        queue=queue,
    )
    accepted = runs.request_run(PROJECT, "load", principal_id=PRINCIPAL, revision_id=revision)

    backend = build_lite_backend(store)
    while (claimed := scheduler.dispatch("worker-1")) is not None:
        _execute(store, queue, backend, claimed, worker_id="worker-1")
    return str(accepted.run_id)


def test_a_committed_run_records_what_it_read_and_wrote(
    store: ControlStore, revision: RevisionId
) -> None:
    """The regression this fixes: the graph was always empty."""
    run_id = _run_load(store, revision)

    edges = LineageService(store).edges_for(run_id)

    assert edges, "a successful run left no lineage behind"
    consumed = {e.target_id for e in edges if e.relation is LineageRelation.CONSUMED}
    produced = {e.target_id for e in edges if e.relation is LineageRelation.PRODUCED}
    assert consumed == {"orders_csv"}
    assert produced == {"orders"}


def test_the_output_is_linked_to_its_source_directly(
    store: ControlStore, revision: RevisionId
) -> None:
    """Answering "where did this come from" must not require walking run nodes."""
    _run_load(store, revision)

    derived = LineageService(store).edges_for(
        "orders", direction="upstream", relations=(LineageRelation.DERIVED_FROM,)
    )

    assert [(e.source_id, e.target_id) for e in derived] == [("orders", "orders_csv")]


def test_the_edges_pin_the_revision_that_produced_them(
    store: ControlStore, revision: RevisionId
) -> None:
    """A graph that cannot say which revision made an edge cannot be audited."""
    run_id = _run_load(store, revision)

    assert {e.revision_id for e in LineageService(store).edges_for(run_id)} == {revision}


def test_an_edge_names_the_workload_that_made_it(
    store: ControlStore, revision: RevisionId
) -> None:
    """The lineage view filters by workload, so the edge has to carry it."""
    run_id = _run_load(store, revision)

    attributes = [e.attributes.get("workload") for e in LineageService(store).edges_for(run_id)]

    assert attributes and set(attributes) == {"load"}


def test_the_gateway_will_not_return_another_project_s_edges(
    store: ControlStore, revision: RevisionId
) -> None:
    """Invariant 6. Node ids are bare names, so two projects can collide on one."""
    from dataenginex.bootstrap import build_lite_gateway
    from dataenginex.interfaces import Query

    _run_load(store, revision)
    gateway = build_lite_gateway(store)

    mine = gateway.list_lineage(Query(principal_id=PRINCIPAL, project_id=PROJECT), node="orders")
    theirs = gateway.list_lineage(
        Query(principal_id=PRINCIPAL, project_id=ProjectId("proj_other")), node="orders"
    )

    assert mine.items
    assert theirs.items == ()


def test_a_failed_run_records_nothing(
    store: ControlStore, revision: RevisionId, csv_path: Path
) -> None:
    """Lineage must describe what happened, not what was attempted.

    With the source file gone the ingest fails. Recording edges anyway would
    claim ``orders`` exists and was derived from ``orders_csv`` — a graph entry
    pointing at a table no reader can find.
    """
    csv_path.unlink()

    run_id = _run_load(store, revision)

    assert LineageService(store).edges_for(run_id) == ()
