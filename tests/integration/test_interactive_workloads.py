"""Interactive workloads, end to end (§7.3, §17 Phase 1).

The point of this file is the thing that is *not* here: nowhere does a query run
in the process that asked for it. A SQL preview is submitted, queued, claimed by
a worker, executed under a lease, and its rows read back from the control store —
the same path a nightly batch job takes, only faster and smaller.

That indirection looks like overhead until you name what it buys: the preview
runs under a capability token scoped to one project, against a pinned revision,
inside a resource ceiling, and it can be cancelled. A function call in the web
process has none of those properties.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import pytest

from dataenginex.application.runs import RunService
from dataenginex.bootstrap.lite import build_lite_backend
from dataenginex.cli.worker import _execute
from dataenginex.domains.execution import InProcessBackend
from dataenginex.domains.security import DEFAULT_POLICY_SET, GovernanceService
from dataenginex.foundation import (
    InteractiveRequest,
    Operation,
    Policy,
    PolicyEffect,
    PrincipalId,
    ProjectId,
    RevisionId,
    RiskLevel,
    RunId,
    RunState,
    WorkloadKind,
)
from dataenginex.runtime.compiler.revisions import RevisionService
from dataenginex.runtime.planning import plan_attempt, purge_expired_results
from dataenginex.runtime.queue import DurableQueue, Scheduler
from dataenginex.runtime.state import ControlStore

PROJECT = ProjectId("proj_interactive")
OTHER_PROJECT = ProjectId("proj_other")
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
  name: interactive
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
            for project in (PROJECT, OTHER_PROJECT):
                tx.execute(
                    "INSERT INTO projects (project_id, workspace_id, name, created_at) "
                    "VALUES (?, 'ws_1', ?, ?)",
                    (project, project, TS),
                )
        yield s


@pytest.fixture(autouse=True)
def workspace_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "workspaces"
    monkeypatch.setenv("DEX_WORKSPACE_DIR", str(root))
    return root


@pytest.fixture
def revision(store: ControlStore, tmp_path: Path) -> RevisionId:
    root = tmp_path / "project"
    root.mkdir()
    (root / "orders.csv").write_text(ORDERS_CSV)
    (root / "dex.yaml").write_text(MANIFEST.format(csv_path=root / "orders.csv"))
    service = RevisionService(store)
    draft, compiled = service.create_draft(PROJECT, root, PRINCIPAL)
    assert compiled.report.ok, compiled.report.errors
    service.publish(draft.revision_id, PRINCIPAL)
    return draft.revision_id


def _worker(store: ControlStore) -> tuple[DurableQueue, Scheduler]:
    queue = DurableQueue(store, lease_duration=timedelta(minutes=5))
    queue.register_worker("worker-1", pool="interactive")
    return queue, Scheduler(store, queue)


def _runs(store: ControlStore, queue: DurableQueue) -> RunService:
    """A run service that permits this fixture's work and nothing else.

    Appended to the default policy set rather than replacing it, so the
    restricted-data and egress rules still apply — a test that disabled those
    would be testing a system nobody ships.
    """
    permit = Policy(
        name="test-permit-interactive",
        effect=PolicyEffect.PERMIT,
        actions=(
            "run:load",
            "interactive:sql_preview",
            "interactive:schema",
            "interactive:stats",
            "interactive:inventory",
        ),
        max_risk_level=RiskLevel.CREATE_LOCAL_ARTIFACT,
        priority=500,
    )
    return RunService(
        store, governance=GovernanceService(store, policies=(*DEFAULT_POLICY_SET, permit)),
        queue=queue,
    )


def _drain(store: ControlStore, queue: DurableQueue, scheduler: Scheduler) -> None:
    """Run every queued unit to completion, as a worker loop would."""
    backend = build_lite_backend(store)
    while (claimed := scheduler.dispatch("worker-1")) is not None:
        _execute(store, queue, backend, claimed, worker_id="worker-1")


def _ingest_orders(store: ControlStore, revision: RevisionId) -> None:
    """Land the CSV in the workspace so a preview has something to read."""
    queue, scheduler = _worker(store)
    _runs(store, queue).request_run(
        PROJECT, "load", principal_id=PRINCIPAL, revision_id=revision
    )
    _drain(store, queue, scheduler)


def _preview(sql: str, *, max_rows: int = 1000) -> InteractiveRequest:
    return InteractiveRequest(
        operations=(
            Operation(operation_type="sql_preview", name="preview", parameters={"sql": sql}),
        ),
        label="sql_preview",
        max_rows=max_rows,
    )


def test_a_sql_preview_runs_on_a_worker_and_returns_rows(
    store: ControlStore, revision: RevisionId
) -> None:
    """The whole path: submit, queue, claim, execute, read the result back."""
    _ingest_orders(store, revision)

    queue, scheduler = _worker(store)
    runs = _runs(store, queue)
    accepted = runs.request_interactive(
        PROJECT,
        _preview('SELECT customer, amount FROM "orders" ORDER BY amount'),
        principal_id=PRINCIPAL,
        revision_id=revision,
    )

    # Nothing has run yet. The gateway accepted the work; it did not do it.
    assert runs.interactive_result(accepted.run_id) is None

    _drain(store, queue, scheduler)

    row = store.query_one("SELECT state, error FROM runs WHERE run_id = ?", (accepted.run_id,))
    assert row is not None
    assert row["state"] == RunState.COMPLETED.value, row["error"]

    result = runs.interactive_result(accepted.run_id)
    assert result is not None
    assert result.payload["columns"] == ["customer", "amount"]
    assert result.payload["rows"] == [["grace", 15.0], ["ada", 120.0], ["alan", 300.0]]
    assert result.row_count == 3
    assert not result.truncated


def test_an_interactive_run_is_queued_as_interactive(
    store: ControlStore, revision: RevisionId
) -> None:
    """Kind and priority decide whether a preview waits behind a batch backlog."""
    queue, _ = _worker(store)
    accepted = _runs(store, queue).request_interactive(
        PROJECT, _preview("SELECT 1"), principal_id=PRINCIPAL, revision_id=revision
    )

    run = store.query_one(
        "SELECT kind, revision_id FROM runs WHERE run_id = ?", (accepted.run_id,)
    )
    assert run is not None
    assert run["kind"] == WorkloadKind.INTERACTIVE.value
    # Pinned like any other run (§17 Phase 1), so a preview cannot reach
    # anything the revision did not declare.
    assert run["revision_id"] == revision

    item = store.query_one(
        "SELECT workload_kind, priority, resource_request_json FROM queue_items WHERE run_id = ?",
        (accepted.run_id,),
    )
    assert item is not None
    assert item["workload_kind"] == WorkloadKind.INTERACTIVE.value
    assert item["priority"] == 10, "interactive must outrank batch's 100"
    assert '"timeout_seconds":30' in item["resource_request_json"].replace(" ", "")


def test_the_row_cap_is_enforced_and_reported(
    store: ControlStore, revision: RevisionId
) -> None:
    """Truncation must be a stated fact, not something a caller has to infer."""
    _ingest_orders(store, revision)

    queue, scheduler = _worker(store)
    runs = _runs(store, queue)
    accepted = runs.request_interactive(
        PROJECT,
        _preview('SELECT * FROM "orders"', max_rows=2),
        principal_id=PRINCIPAL,
        revision_id=revision,
    )
    _drain(store, queue, scheduler)

    result = runs.interactive_result(accepted.run_id)
    assert result is not None
    assert len(result.payload["rows"]) == 2
    assert result.truncated, "a capped result must say so"


def test_a_preview_cannot_write(store: ControlStore, revision: RevisionId) -> None:
    """The connection is read-only, so a preview cannot modify project data.

    Enforced by the connection rather than by inspecting the SQL: a blocklist
    over SQL text loses to the next syntax nobody anticipated.
    """
    _ingest_orders(store, revision)

    queue, scheduler = _worker(store)
    runs = _runs(store, queue)
    accepted = runs.request_interactive(
        PROJECT,
        _preview('DELETE FROM "orders"'),
        principal_id=PRINCIPAL,
        revision_id=revision,
    )
    _drain(store, queue, scheduler)

    row = store.query_one("SELECT state, error FROM runs WHERE run_id = ?", (accepted.run_id,))
    assert row is not None
    assert row["state"] == RunState.FAILED.value
    assert runs.interactive_result(accepted.run_id) is None


def test_schema_inspection_describes_a_declared_resource(
    store: ControlStore, revision: RevisionId
) -> None:
    """Schema inspection reads the source, so it works before any ingest."""
    queue, scheduler = _worker(store)
    runs = _runs(store, queue)
    accepted = runs.request_interactive(
        PROJECT,
        InteractiveRequest(
            operations=(
                Operation(
                    operation_type="schema_inspect",
                    name="describe",
                    bound_inputs=("orders_csv",),
                ),
            ),
            label="schema",
        ),
        principal_id=PRINCIPAL,
        revision_id=revision,
    )
    _drain(store, queue, scheduler)

    result = runs.interactive_result(accepted.run_id)
    assert result is not None, "schema inspection must not need a prior ingest"
    assert result.payload["resource"] == "orders_csv"
    assert [c["name"] for c in result.payload["columns"]] == [
        "order_id",
        "customer",
        "amount",
    ]


def test_table_stats_counts_rows_and_nulls(
    store: ControlStore, revision: RevisionId
) -> None:
    _ingest_orders(store, revision)

    queue, scheduler = _worker(store)
    runs = _runs(store, queue)
    accepted = runs.request_interactive(
        PROJECT,
        InteractiveRequest(
            operations=(
                Operation(
                    operation_type="table_stats", name="stats", bound_inputs=("orders",)
                ),
            ),
            label="stats",
        ),
        principal_id=PRINCIPAL,
        revision_id=revision,
    )
    _drain(store, queue, scheduler)

    result = runs.interactive_result(accepted.run_id)
    assert result is not None
    assert result.payload["row_count"] == 3
    assert {c["name"]: c["nulls"] for c in result.payload["columns"]} == {
        "order_id": 0,
        "customer": 0,
        "amount": 0,
    }


def _inventory(lakehouse: Path | None) -> InteractiveRequest:
    parameters = {"lakehouse_root": str(lakehouse)} if lakehouse is not None else {}
    return InteractiveRequest(
        operations=(
            Operation(
                operation_type="lakehouse_inventory",
                name="inventory",
                parameters=parameters,
            ),
        ),
        label="inventory",
    )


def test_lakehouse_inventory_describes_what_is_on_disk(
    store: ControlStore, revision: RevisionId, tmp_path: Path
) -> None:
    """The warehouse and catalog pages' data, produced on a worker.

    Those pages used to glob ``.dex/lakehouse`` and open DuckDB in the web
    process. The root now travels on the plan, so a worker never has to guess
    at a path nobody granted it.
    """
    import duckdb

    lakehouse = tmp_path / "lakehouse"
    (lakehouse / "bronze").mkdir(parents=True)
    (lakehouse / "gold").mkdir(parents=True)

    connection = duckdb.connect(":memory:")
    try:
        connection.execute("CREATE TABLE t AS SELECT * FROM range(4) AS r(id)")
        connection.execute(
            f"COPY t TO '{lakehouse / 'bronze' / 'orders.parquet'}' (FORMAT PARQUET)"
        )
    finally:
        connection.close()

    queue, scheduler = _worker(store)
    runs = _runs(store, queue)
    accepted = runs.request_interactive(
        PROJECT, _inventory(lakehouse), principal_id=PRINCIPAL, revision_id=revision
    )
    _drain(store, queue, scheduler)

    result = runs.interactive_result(accepted.run_id)
    assert result is not None
    assert {layer["name"]: layer["table_count"] for layer in result.payload["layers"]} == {
        "bronze": 1,
        "silver": 0,
        "gold": 0,
    }
    table = result.payload["tables"][0]
    assert table["name"] == "orders"
    assert table["layer"] == "bronze"
    assert table["row_count"] == 4
    assert table["format"] == "parquet"


def test_lakehouse_inventory_refuses_without_a_root(
    store: ControlStore, revision: RevisionId
) -> None:
    """No root means no default. Guessing one reads somewhere nobody granted."""
    queue, scheduler = _worker(store)
    runs = _runs(store, queue)
    accepted = runs.request_interactive(
        PROJECT, _inventory(None), principal_id=PRINCIPAL, revision_id=revision
    )
    _drain(store, queue, scheduler)

    row = store.query_one("SELECT state, error FROM runs WHERE run_id = ?", (accepted.run_id,))
    assert row is not None
    assert row["state"] == RunState.FAILED.value
    assert "lakehouse_root" in (row["error"] or "")


def test_an_expired_result_is_not_served(store: ControlStore, revision: RevisionId) -> None:
    """§7.3 calls these ephemeral; expiry must be enforced on read.

    A sweeper that has not run yet must not be the reason a stale preview is
    shown as current.
    """
    _ingest_orders(store, revision)

    queue, scheduler = _worker(store)
    runs = _runs(store, queue)
    accepted = runs.request_interactive(
        PROJECT,
        _preview('SELECT * FROM "orders"'),
        principal_id=PRINCIPAL,
        revision_id=revision,
    )
    _drain(store, queue, scheduler)
    assert runs.interactive_result(accepted.run_id) is not None

    with store.transaction() as tx:
        tx.execute(
            "UPDATE interactive_results SET expires_at = ? WHERE run_id = ?",
            ("2020-01-01T00:00:00+00:00", accepted.run_id),
        )

    assert runs.interactive_result(accepted.run_id) is None, "expired results must not be served"
    assert purge_expired_results(store) == 1


def test_a_worker_without_a_store_refuses_interactive_work(
    store: ControlStore, revision: RevisionId
) -> None:
    """A worker that cannot store a result must not pretend to have produced one.

    Running the query and dropping the rows would leave the user waiting on a
    result that completed successfully and does not exist.
    """
    queue, scheduler = _worker(store)
    runs = _runs(store, queue)
    accepted = runs.request_interactive(
        PROJECT, _preview("SELECT 1"), principal_id=PRINCIPAL, revision_id=revision
    )

    claimed = scheduler.dispatch("worker-1")
    assert claimed is not None
    _execute(store, queue, InProcessBackend(), claimed, worker_id="worker-1")

    row = store.query_one("SELECT state, error FROM runs WHERE run_id = ?", (accepted.run_id,))
    assert row is not None
    assert row["state"] == RunState.FAILED.value
    assert "no handler registered" in (row["error"] or "")


def test_the_plan_comes_from_the_request_not_the_revision(
    store: ControlStore, revision: RevisionId
) -> None:
    """An ad-hoc plan must not need — or create — a declared workload."""
    queue, scheduler = _worker(store)
    accepted = _runs(store, queue).request_interactive(
        PROJECT, _preview("SELECT 42"), principal_id=PRINCIPAL, revision_id=revision
    )
    claimed = scheduler.dispatch("worker-1")
    assert claimed is not None
    assert claimed.run_id == accepted.run_id

    plan, context = plan_attempt(store, claimed.attempt_id)
    assert [op.operation_type for op in plan.operations] == ["sql_preview"]
    assert plan.parameters["max_rows"] == "1000"
    # Scoped exactly like a batch attempt.
    assert context.capability.revision_id == revision
    assert context.capability.resource_scope == (f"{PROJECT}/*",)

    # And the revision's declared workload set is untouched: a preview is not a
    # pipeline, and must never appear in one's list.
    names = {
        r["name"]
        for r in store.query(
            "SELECT name FROM workload_definitions WHERE revision_id = ?", (revision,)
        )
    }
    assert names == {"load"}


def test_a_result_is_not_readable_from_another_project(
    store: ControlStore, revision: RevisionId
) -> None:
    """Invariant 6: a run id from elsewhere must not return this project's rows."""
    from dataenginex.bootstrap import build_lite_gateway
    from dataenginex.interfaces import Query

    _ingest_orders(store, revision)
    queue, scheduler = _worker(store)
    runs = _runs(store, queue)
    accepted = runs.request_interactive(
        PROJECT,
        _preview('SELECT * FROM "orders"'),
        principal_id=PRINCIPAL,
        revision_id=revision,
    )
    _drain(store, queue, scheduler)

    gateway = build_lite_gateway(store)
    mine = Query(principal_id=PRINCIPAL, project_id=PROJECT)
    theirs = Query(principal_id=PRINCIPAL, project_id=OTHER_PROJECT)

    assert gateway.get_interactive_result(mine, run_id=RunId(accepted.run_id)) is not None
    assert gateway.get_interactive_result(theirs, run_id=RunId(accepted.run_id)) is None
