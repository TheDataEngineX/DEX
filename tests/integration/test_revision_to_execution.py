"""The whole §17 Phase 1 path: publish a revision, queue a run, execute it.

This test exists because every layer of it passed its own unit tests while the
path as a whole did nothing. ``workload_definitions`` had five readers and no
writer, so a published revision carried no runnable workloads; and nothing in
the tree called ``Scheduler.dispatch``, so queued runs were never claimed. Both
were invisible to tests that asserted on queue *state* rather than on work
actually completing.

So the assertions here are deliberately end-to-end: a handler really runs, and
the run really reaches ``completed``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from dataenginex.cli.worker import _execute
from dataenginex.domains.execution.backends import InProcessBackend
from dataenginex.foundation import (
    PrincipalId,
    ProjectId,
    RevisionId,
    RunId,
    RunState,
    WorkloadKind,
)
from dataenginex.runtime.compiler.revisions import RevisionService
from dataenginex.runtime.planning.planner import PlanningError, plan_attempt
from dataenginex.runtime.queue import DurableQueue, Scheduler
from dataenginex.runtime.state import ControlStore

PROJECT = ProjectId("proj_demo")
PRINCIPAL = PrincipalId("prin_test")
TS = "2026-08-03T00:00:00+00:00"

MANIFEST = """apiVersion: dex/v1alpha1
kind: Project
metadata:
  name: demo
spec:
  workloads:
    - name: nightly
      kind: batch
      operations:
        - name: ingest-orders
          type: ingest
        - name: clean
          type: transform
    - name: preview
      kind: interactive
      operations:
        - name: peek
          type: ingest
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


@pytest.fixture
def revision(store: ControlStore, tmp_path: Path) -> RevisionId:
    root = tmp_path / "project"
    root.mkdir()
    (root / "dex.yaml").write_text(MANIFEST)
    draft, _ = RevisionService(store).create_draft(PROJECT, root, PRINCIPAL)
    return draft.revision_id


def _queue_run(
    store: ControlStore,
    queue: DurableQueue,
    revision: RevisionId,
    *,
    run_id: str,
    workload: str,
    kind: WorkloadKind = WorkloadKind.BATCH,
) -> RunId:
    with store.transaction() as tx:
        tx.execute(
            "INSERT INTO runs (run_id, project_id, revision_id, workload_name, kind, "
            "state, trigger_type, requested_by, created_at) VALUES (?,?,?,?,?,?,'manual',?,?)",
            (
                run_id,
                PROJECT,
                revision,
                workload,
                kind.value,
                RunState.PLANNING.value,
                PRINCIPAL,
                TS,
            ),
        )
    queue.enqueue(RunId(run_id), project_id=PROJECT, revision_id=revision, workload_kind=kind)
    return RunId(run_id)


def _ready_worker(store: ControlStore) -> tuple[DurableQueue, Scheduler]:
    queue = DurableQueue(store, lease_duration=timedelta(minutes=5))
    queue.register_worker("worker-1", pool="batch")
    return queue, Scheduler(store, queue)


def test_publishing_a_revision_persists_runnable_workloads(
    store: ControlStore, revision: RevisionId
) -> None:
    """Compiling must write ``workload_definitions``, or nothing can be run."""
    rows = store.query(
        "SELECT name, kind FROM workload_definitions WHERE revision_id = ? ORDER BY name",
        (revision,),
    )
    assert {r["name"]: r["kind"] for r in rows} == {
        "nightly": "batch",
        "preview": "interactive",
    }


def test_a_queued_run_executes_and_completes(store: ControlStore, revision: RevisionId) -> None:
    queue, scheduler = _ready_worker(store)
    _queue_run(store, queue, revision, run_id="run_1", workload="nightly")

    claimed = scheduler.dispatch("worker-1", kinds=(WorkloadKind.BATCH,))
    assert claimed is not None, "a queued run must be claimable"

    ran: list[str] = []

    def _handler(name: str) -> Callable[[Any, Any], tuple[str, ...]]:
        def handle(_plan: Any, _ctx: Any) -> tuple[str, ...]:
            ran.append(name)
            return (f"{name}.parquet",)

        return handle

    backend = InProcessBackend()
    backend.register("ingest", _handler("ingest"))
    backend.register("transform", _handler("transform"))

    _execute(store, queue, backend, claimed, worker_id="worker-1")

    assert ran == ["ingest", "transform"], "operations run in declared order"
    row = store.query_one("SELECT state, error FROM runs WHERE run_id = 'run_1'")
    assert row is not None
    assert row["state"] == RunState.COMPLETED.value, row["error"]


def test_the_plan_is_scoped_to_its_own_attempt(store: ControlStore, revision: RevisionId) -> None:
    """Token and namespaces must not outlive or exceed the attempt (§7.8, §9.4)."""
    queue, scheduler = _ready_worker(store)
    _queue_run(store, queue, revision, run_id="run_2", workload="nightly")
    claimed = scheduler.dispatch("worker-1", kinds=(WorkloadKind.BATCH,))
    assert claimed is not None

    plan, context = plan_attempt(store, claimed.attempt_id)

    assert [op.operation_type for op in plan.operations] == ["ingest", "transform"]
    assert context.capability.run_id == "run_2"
    assert context.capability.revision_id == revision
    assert context.capability.expires_at > context.capability.issued_at
    # Scoped to this project, and to this attempt's own output namespace: a
    # retry must not be able to see or overwrite the previous attempt's files.
    assert context.capability.resource_scope == (f"{PROJECT}/*",)
    assert context.artifact_namespace.endswith(claimed.attempt_id)


def test_a_missing_workload_refuses_to_plan(store: ControlStore, revision: RevisionId) -> None:
    """A run naming a workload the revision lacks must fail, not run something else.

    Substituting a default here would commit a result for work nobody declared.
    """
    queue, scheduler = _ready_worker(store)
    _queue_run(store, queue, revision, run_id="run_3", workload="deleted-workload")
    claimed = scheduler.dispatch("worker-1", kinds=(WorkloadKind.BATCH,))
    assert claimed is not None

    with pytest.raises(PlanningError, match="declares no workload"):
        plan_attempt(store, claimed.attempt_id)


def test_an_unregistered_operation_fails_the_run(
    store: ControlStore, revision: RevisionId
) -> None:
    """A backend with no handler must fail the run rather than commit an empty success."""
    queue, scheduler = _ready_worker(store)
    _queue_run(store, queue, revision, run_id="run_4", workload="nightly")
    claimed = scheduler.dispatch("worker-1", kinds=(WorkloadKind.BATCH,))
    assert claimed is not None

    _execute(store, queue, InProcessBackend(), claimed, worker_id="worker-1")

    row = store.query_one("SELECT state, error FROM runs WHERE run_id = 'run_4'")
    assert row is not None
    assert row["state"] == RunState.FAILED.value
    assert "no handler registered" in (row["error"] or "")
