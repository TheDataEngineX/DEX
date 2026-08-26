"""A real workload, end to end: CSV on disk in, checked table out.

The companion to ``test_revision_to_execution``, which proved the control plane
could carry a run from queued to completed with *fake* handlers. This proves the
real ones do real work — a file is read, rows are filtered, and a quality gate
either passes or fails the run.

That distinction is the whole point of this file. Registering handlers and never
asserting on their output would reproduce the exact gap this work exists to
close: a path that reports success without having done anything.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import pytest

from dataenginex.bootstrap.lite import build_lite_backend, open_connector
from dataenginex.cli.worker import _execute
from dataenginex.domains.execution.handlers import workspace_path
from dataenginex.foundation import (
    PrincipalId,
    ProjectId,
    RevisionId,
    RunId,
    RunState,
    WorkloadKind,
    utcnow,
)
from dataenginex.runtime.compiler.revisions import RevisionService
from dataenginex.runtime.planning.planner import plan_attempt
from dataenginex.runtime.queue import DurableQueue, Scheduler
from dataenginex.runtime.state import ControlStore

PROJECT = ProjectId("proj_handlers")
PRINCIPAL = PrincipalId("prin_test")
TS = "2026-08-07T00:00:00+00:00"

ORDERS_CSV = """order_id,customer,amount
1,ada,120.0
2,grace,15.0
3,alan,300.0
"""

MANIFEST = """apiVersion: dex/v1alpha1
kind: Project
metadata:
  name: handlers
spec:
  resources:
    - name: orders_csv
      type: csv
      config:
        path: "{csv_path}"
  workloads:
    - name: nightly
      kind: batch
      operations:
        - name: load-orders
          type: ingest
          inputs: [orders_csv]
          outputs: [orders_raw]
        - name: big-orders
          type: transform
          inputs: [orders_raw]
          outputs: [orders_big]
          parameters:
            transform: filter
            condition: "amount > 100"
        - name: check-orders
          type: validate
          inputs: [orders_big]
          parameters:
            required: "order_id,customer"
            row_count_min: "1"
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
                "VALUES (?, 'ws_1', ?, ?)",
                (PROJECT, PROJECT, TS),
            )
        yield s


@pytest.fixture(autouse=True)
def workspace_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep attempt workspaces inside the test's own directory."""
    root = tmp_path / "workspaces"
    monkeypatch.setenv("DEX_WORKSPACE_DIR", str(root))
    return root


def _project(tmp_path: Path, manifest: str = MANIFEST) -> Path:
    root = tmp_path / "project"
    root.mkdir(exist_ok=True)
    csv_path = root / "orders.csv"
    csv_path.write_text(ORDERS_CSV)
    (root / "dex.yaml").write_text(manifest.format(csv_path=csv_path))
    return root


def _revision(store: ControlStore, root: Path) -> RevisionId:
    draft, compiled = RevisionService(store).create_draft(PROJECT, root, PRINCIPAL)
    assert compiled.report.ok, compiled.report.errors
    return draft.revision_id


def _queue(
    store: ControlStore, revision: RevisionId, run_id: str
) -> tuple[DurableQueue, Scheduler]:
    queue = DurableQueue(store, lease_duration=timedelta(minutes=5))
    queue.register_worker("worker-1", pool="batch")
    with store.transaction() as tx:
        tx.execute(
            "INSERT INTO runs (run_id, project_id, revision_id, workload_name, kind, "
            "state, trigger_type, requested_by, created_at) VALUES (?,?,?,?,?,?,'manual',?,?)",
            (
                run_id,
                PROJECT,
                revision,
                "nightly",
                WorkloadKind.BATCH.value,
                RunState.PLANNING.value,
                PRINCIPAL,
                TS,
            ),
        )
    queue.enqueue(
        RunId(run_id),
        project_id=PROJECT,
        revision_id=revision,
        workload_kind=WorkloadKind.BATCH,
    )
    return queue, Scheduler(store, queue)


def test_a_csv_pipeline_runs_end_to_end(store: ControlStore, tmp_path: Path) -> None:
    """Ingest, transform, and validate against a file that really exists."""
    revision = _revision(store, _project(tmp_path))
    queue, scheduler = _queue(store, revision, "run_ok")
    claimed = scheduler.dispatch("worker-1", kinds=(WorkloadKind.BATCH,))
    assert claimed is not None

    _, context = plan_attempt(store, claimed.attempt_id)
    _execute(
        store,
        queue,
        build_lite_backend(),
        claimed,
        worker_id="worker-1",
    )

    row = store.query_one("SELECT state, error FROM runs WHERE run_id = 'run_ok'")
    assert row is not None
    assert row["state"] == RunState.COMPLETED.value, row["error"]

    # The filter really ran: two of three orders are over 100.
    import duckdb

    connection = duckdb.connect(str(workspace_path(context)))
    try:
        assert connection.execute('SELECT count(*) FROM "orders_raw"').fetchone() == (3,)
        assert connection.execute('SELECT count(*) FROM "orders_big"').fetchone() == (2,)
    finally:
        connection.close()


def test_resources_reach_the_plan(store: ControlStore, tmp_path: Path) -> None:
    """The declared config must survive compile, storage, and planning.

    Every one of those steps used to drop it, and a handler with no config has
    nothing to read from.
    """
    revision = _revision(store, _project(tmp_path))
    queue, scheduler = _queue(store, revision, "run_plan")
    del queue
    claimed = scheduler.dispatch("worker-1", kinds=(WorkloadKind.BATCH,))
    assert claimed is not None

    plan, _ = plan_attempt(store, claimed.attempt_id)

    assert "orders_csv" in plan.inputs, "the bound resource must reach the worker"
    assert "orders.csv" in plan.inputs["orders_csv"]
    # And only what this workload named — not every resource in the project.
    assert set(plan.inputs) == {"orders_csv"}

    named = [(op.name, op.operation_type, op.bound_inputs) for op in plan.operations]
    assert named == [
        ("load-orders", "ingest", ("orders_csv",)),
        ("big-orders", "transform", ("orders_raw",)),
        ("check-orders", "validate", ("orders_big",)),
    ]


def test_a_failing_quality_gate_fails_the_run(store: ControlStore, tmp_path: Path) -> None:
    """A gate that cannot be met must fail the run, not log and pass."""
    strict = MANIFEST.replace('row_count_min: "1"', 'row_count_min: "99"')
    revision = _revision(store, _project(tmp_path, strict))
    queue, scheduler = _queue(store, revision, "run_gate")
    claimed = scheduler.dispatch("worker-1", kinds=(WorkloadKind.BATCH,))
    assert claimed is not None

    _execute(
        store,
        queue,
        build_lite_backend(),
        claimed,
        worker_id="worker-1",
    )

    row = store.query_one("SELECT state, error FROM runs WHERE run_id = 'run_gate'")
    assert row is not None
    assert row["state"] == RunState.FAILED.value
    assert "quality gate" in (row["error"] or "")


def test_an_undeclared_resource_never_compiles(store: ControlStore, tmp_path: Path) -> None:
    """Binding to a resource nothing declares must fail at compile, not at run.

    Catching it here means the project cannot be published at all, which is
    strictly better than a run that gets queued, claimed, and leased before
    discovering it had nothing to read.
    """
    broken = MANIFEST.replace("inputs: [orders_csv]", "inputs: [missing_source]")
    _, compiled = RevisionService(store).create_draft(
        PROJECT, _project(tmp_path, broken), PRINCIPAL
    )

    assert not compiled.report.ok
    codes = {issue.code for issue in compiled.report.errors}
    assert codes == {"E_UNKNOWN_INPUT"}
    assert any("missing_source" in issue.message for issue in compiled.report.errors)


def test_a_handler_refuses_a_resource_the_plan_lacks() -> None:
    """The handler's own guard, for the case the compiler cannot catch.

    A revision compiles against the resources it declared; a plan is built later
    from what the store holds. If those ever disagree the handler must say which
    resource is missing rather than reading from a default.
    """
    from dataenginex.domains.execution.handlers import HandlerError, handle_ingest
    from dataenginex.foundation import (
        CapabilityToken,
        ExecutionContext,
        ExecutionPlan,
        Operation,
    )

    plan = ExecutionPlan(
        attempt_id="att_x",
        project_id=PROJECT,
        revision_id=RevisionId("rev_x"),
        operations=(
            Operation(operation_type="ingest", name="load", bound_inputs=("gone",)),
        ),
        inputs={},
    )
    context = ExecutionContext(
        attempt_id="att_x",
        capability=CapabilityToken(
            principal_id=PRINCIPAL,
            project_id=PROJECT,
            revision_id=RevisionId("rev_x"),
            run_id=RunId("run_x"),
            expires_at=utcnow() + timedelta(minutes=5),
        ),
        artifact_namespace=f"{PROJECT}/run_x/att_x",
    )

    with pytest.raises(HandlerError, match="gone"):
        handle_ingest(plan, context, connectors=open_connector)


def test_runs_of_one_revision_share_a_workspace(store: ControlStore, tmp_path: Path) -> None:
    """Two runs of the same revision must see the same tables.

    This is what makes a project more than one workload: an ingest lands a
    table, and a later transform, quality check, or SQL preview reads it. Keying
    the workspace per attempt — which an earlier version did — gave every run an
    empty database and made cross-workload work impossible.
    """
    revision = _revision(store, _project(tmp_path))
    queue, scheduler = _queue(store, revision, "run_iso")
    first = scheduler.dispatch("worker-1", kinds=(WorkloadKind.BATCH,))
    assert first is not None
    _, first_context = plan_attempt(store, first.attempt_id)

    _execute(store, queue, build_lite_backend(), first, worker_id="worker-1")

    _, scheduler2 = _queue(store, revision, "run_iso_2")
    second = scheduler2.dispatch("worker-1", kinds=(WorkloadKind.BATCH,))
    assert second is not None
    _, second_context = plan_attempt(store, second.attempt_id)

    assert workspace_path(first_context) == workspace_path(second_context)

    # But the *attempt* namespaces still differ, which is where §14.3's
    # isolation actually lives — output commit, not file layout.
    assert first_context.artifact_namespace != second_context.artifact_namespace
