"""Durable queue, leases, and scheduler.

The tests that carry weight here are the concurrency ones: a double claim, a
late completion after reclaim, and a cancelled run that a live worker then tries
to finish. Those are the failure modes §7.6 and §14.3 exist to prevent, and they
are invisible to a test that only exercises the happy path.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import pytest

from dataenginex.foundation import (
    AttemptState,
    PrincipalId,
    ProjectId,
    ResourceRequest,
    RevisionId,
    RunId,
    RunState,
    WorkloadKind,
    utcnow,
)
from dataenginex.runtime.queue import (
    Capacity,
    DurableQueue,
    QueueError,
    Scheduler,
    SchedulerPolicy,
)
from dataenginex.runtime.state import ControlStore

PROJECT = ProjectId("proj_test")
OTHER_PROJECT = ProjectId("proj_other")
REVISION = RevisionId("rev_1")
PRINCIPAL = PrincipalId("prin_test")
TS = "2026-08-03T00:00:00+00:00"


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
                "INSERT INTO workspaces (workspace_id, installation_id, name, "
                "created_at) VALUES ('ws_1', 'inst_1', 'default', ?)",
                (TS,),
            )
            for project in (PROJECT, OTHER_PROJECT):
                tx.execute(
                    "INSERT INTO projects (project_id, workspace_id, name, "
                    "created_at) VALUES (?, 'ws_1', ?, ?)",
                    (project, project, TS),
                )
            tx.execute(
                "INSERT INTO project_revisions (revision_id, project_id, "
                "content_hash, created_by, created_at, manifest_schema_version, "
                "status) VALUES (?, ?, 'sha256:x', ?, ?, 'dex/v1alpha1', 'published')",
                (REVISION, PROJECT, PRINCIPAL, TS),
            )
        yield s


@pytest.fixture
def queue(store: ControlStore) -> DurableQueue:
    q = DurableQueue(store, lease_duration=timedelta(minutes=5))
    # Workers must exist before they can hold a lease; registering the ones the
    # tests use keeps each test focused on queue behaviour.
    for worker in ("worker-1", "worker-2", "worker-impostor", "batch-worker"):
        q.register_worker(worker)
    q.register_worker("stream-worker", pool="stream")
    return q


def make_run(
    store: ControlStore,
    run_id: str,
    *,
    project_id: ProjectId = PROJECT,
    state: RunState = RunState.PLANNING,
    kind: WorkloadKind = WorkloadKind.BATCH,
) -> RunId:
    """Insert a run.

    Defaults to ``PLANNING`` because that is the state a run reaches after
    policy evaluation and before the queue — enqueueing straight from
    ``REQUESTED`` is the bypass the state machine refuses.
    """
    with store.transaction() as tx:
        tx.execute(
            "INSERT INTO runs (run_id, project_id, revision_id, workload_name, kind, "
            "state, trigger_type, requested_by, created_at) "
            "VALUES (?,?,?,'work',?,?,'manual',?,?)",
            (run_id, project_id, REVISION, kind.value, state.value, PRINCIPAL, TS),
        )
    return RunId(run_id)


# --- enqueue ----------------------------------------------------------------


def test_enqueue_makes_work_claimable(store: ControlStore, queue: DurableQueue) -> None:
    run_id = make_run(store, "run_1")
    queue.enqueue(run_id, project_id=PROJECT, revision_id=REVISION)

    assert queue.depth() == 1
    row = store.query_one("SELECT state FROM runs WHERE run_id = ?", (run_id,))
    assert row is not None
    assert row["state"] == RunState.QUEUED.value


def test_enqueue_rejects_unknown_run(queue: DurableQueue) -> None:
    with pytest.raises(QueueError, match="does not exist"):
        queue.enqueue(RunId("run_ghost"), project_id=PROJECT, revision_id=REVISION)


def test_cancelled_run_cannot_be_requeued(store: ControlStore, queue: DurableQueue) -> None:
    # A stray enqueue must not resurrect a run the user cancelled.
    run_id = make_run(store, "run_1", state=RunState.CANCELLED)
    with pytest.raises(QueueError, match="cannot be queued"):
        queue.enqueue(run_id, project_id=PROJECT, revision_id=REVISION)


def test_run_cannot_skip_the_policy_gate_into_the_queue(
    store: ControlStore, queue: DurableQueue
) -> None:
    # Enqueueing straight from REQUESTED would put work on a worker without
    # policy evaluation — the §9.3 bypass. The state machine refuses it.
    run_id = make_run(store, "run_1", state=RunState.REQUESTED)
    with pytest.raises(QueueError, match="cannot be queued"):
        queue.enqueue(run_id, project_id=PROJECT, revision_id=REVISION)


# --- claim ------------------------------------------------------------------


def test_claim_returns_work_and_creates_an_attempt(
    store: ControlStore, queue: DurableQueue
) -> None:
    run_id = make_run(store, "run_1")
    queue.enqueue(run_id, project_id=PROJECT, revision_id=REVISION)

    work = queue.claim("worker-1")
    assert work is not None
    assert work.run_id == run_id
    assert work.attempt_number == 1
    assert work.commit_token

    row = store.query_one(
        "SELECT state, worker_id FROM attempts WHERE attempt_id = ?",
        (work.attempt_id,),
    )
    assert row is not None
    assert row["state"] == AttemptState.LEASED.value
    assert row["worker_id"] == "worker-1"


def test_claim_on_empty_queue_returns_none(queue: DurableQueue) -> None:
    assert queue.claim("worker-1") is None


def test_two_workers_cannot_claim_the_same_item(store: ControlStore, queue: DurableQueue) -> None:
    # The core safety property of the durable queue.
    run_id = make_run(store, "run_1")
    queue.enqueue(run_id, project_id=PROJECT, revision_id=REVISION)

    first = queue.claim("worker-1")
    second = queue.claim("worker-2")

    assert first is not None
    assert second is None


def test_claim_respects_priority(store: ControlStore, queue: DurableQueue) -> None:
    low = make_run(store, "run_low")
    high = make_run(store, "run_high")
    queue.enqueue(low, project_id=PROJECT, revision_id=REVISION, priority=500)
    queue.enqueue(high, project_id=PROJECT, revision_id=REVISION, priority=1)

    work = queue.claim("worker-1")
    assert work is not None
    assert work.run_id == high


def test_claim_respects_not_before(store: ControlStore, queue: DurableQueue) -> None:
    run_id = make_run(store, "run_1")
    queue.enqueue(
        run_id,
        project_id=PROJECT,
        revision_id=REVISION,
        not_before=utcnow() + timedelta(hours=1),
    )

    assert queue.claim("worker-1") is None


def test_claim_filters_by_workload_kind(store: ControlStore, queue: DurableQueue) -> None:
    batch = make_run(store, "run_batch", kind=WorkloadKind.BATCH)
    queue.enqueue(
        batch,
        project_id=PROJECT,
        revision_id=REVISION,
        workload_kind=WorkloadKind.BATCH,
    )

    assert queue.claim("stream-worker", kinds=(WorkloadKind.SPARK_STREAM,)) is None
    assert queue.claim("batch-worker", kinds=(WorkloadKind.BATCH,)) is not None


# --- heartbeat --------------------------------------------------------------


def test_heartbeat_extends_the_lease(store: ControlStore, queue: DurableQueue) -> None:
    run_id = make_run(store, "run_1")
    queue.enqueue(run_id, project_id=PROJECT, revision_id=REVISION)
    work = queue.claim("worker-1")
    assert work is not None

    later = utcnow() + timedelta(minutes=1)
    assert queue.heartbeat(work.attempt_id, "worker-1", later) is True

    row = store.query_one("SELECT expires_at FROM leases WHERE attempt_id = ?", (work.attempt_id,))
    assert row is not None
    assert row["expires_at"] > work.lease_expires_at.isoformat()


def test_heartbeat_from_the_wrong_worker_fails(store: ControlStore, queue: DurableQueue) -> None:
    run_id = make_run(store, "run_1")
    queue.enqueue(run_id, project_id=PROJECT, revision_id=REVISION)
    work = queue.claim("worker-1")
    assert work is not None

    assert queue.heartbeat(work.attempt_id, "worker-impostor") is False


def test_heartbeat_moves_run_to_running(store: ControlStore, queue: DurableQueue) -> None:
    run_id = make_run(store, "run_1")
    queue.enqueue(run_id, project_id=PROJECT, revision_id=REVISION)
    work = queue.claim("worker-1")
    assert work is not None
    queue.heartbeat(work.attempt_id, "worker-1")

    row = store.query_one("SELECT state FROM runs WHERE run_id = ?", (run_id,))
    assert row is not None
    assert row["state"] == RunState.RUNNING.value


# --- completion and fencing (§14.3) -----------------------------------------


def test_complete_with_valid_token_succeeds(store: ControlStore, queue: DurableQueue) -> None:
    run_id = make_run(store, "run_1")
    queue.enqueue(run_id, project_id=PROJECT, revision_id=REVISION)
    work = queue.claim("worker-1")
    assert work is not None

    assert queue.complete(work.attempt_id, work.commit_token, succeeded=True) is True

    row = store.query_one("SELECT state FROM runs WHERE run_id = ?", (run_id,))
    assert row is not None
    assert row["state"] == RunState.COMPLETED.value


def test_complete_with_wrong_token_is_rejected(store: ControlStore, queue: DurableQueue) -> None:
    run_id = make_run(store, "run_1")
    queue.enqueue(run_id, project_id=PROJECT, revision_id=REVISION)
    work = queue.claim("worker-1")
    assert work is not None

    assert queue.complete(work.attempt_id, "ctk_forged", succeeded=True) is False


def test_reclaimed_attempt_cannot_commit_late(store: ControlStore, queue: DurableQueue) -> None:
    # The §7.6 guarantee: "the previous worker's late completion cannot
    # silently overwrite the new attempt".
    run_id = make_run(store, "run_1")
    queue.enqueue(run_id, project_id=PROJECT, revision_id=REVISION)
    work = queue.claim("worker-1")
    assert work is not None

    # Lease expires; the control plane reclaims while worker-1 is still alive.
    queue.reclaim_expired(now=work.lease_expires_at + timedelta(seconds=1))

    # worker-1 finishes late and tries to commit its stale result.
    assert queue.complete(work.attempt_id, work.commit_token, succeeded=True) is False


def test_failed_completion_records_the_error(store: ControlStore, queue: DurableQueue) -> None:
    run_id = make_run(store, "run_1")
    queue.enqueue(run_id, project_id=PROJECT, revision_id=REVISION)
    work = queue.claim("worker-1")
    assert work is not None

    queue.complete(
        work.attempt_id,
        work.commit_token,
        succeeded=False,
        error="boom",
        error_class="ValueError",
    )

    row = store.query_one("SELECT state, error FROM runs WHERE run_id = ?", (run_id,))
    assert row is not None
    assert row["state"] == RunState.FAILED.value
    assert row["error"] == "boom"


# --- reclaim ----------------------------------------------------------------


def test_expired_lease_marks_attempt_lost_not_failed(
    store: ControlStore, queue: DurableQueue
) -> None:
    # LOST is not FAILED: the control plane does not know what happened.
    run_id = make_run(store, "run_1")
    queue.enqueue(run_id, project_id=PROJECT, revision_id=REVISION)
    work = queue.claim("worker-1")
    assert work is not None

    reclaimed = queue.reclaim_expired(now=work.lease_expires_at + timedelta(seconds=1))
    assert reclaimed == (work.attempt_id,)

    row = store.query_one(
        "SELECT state, commit_token FROM attempts WHERE attempt_id = ?",
        (work.attempt_id,),
    )
    assert row is not None
    assert row["state"] == AttemptState.LOST.value
    assert row["commit_token"] is None


def test_live_lease_is_not_reclaimed(store: ControlStore, queue: DurableQueue) -> None:
    run_id = make_run(store, "run_1")
    queue.enqueue(run_id, project_id=PROJECT, revision_id=REVISION)
    work = queue.claim("worker-1")
    assert work is not None

    assert queue.reclaim_expired(now=work.lease_expires_at - timedelta(seconds=1)) == ()


def test_requeue_creates_a_second_attempt(store: ControlStore, queue: DurableQueue) -> None:
    run_id = make_run(store, "run_1")
    queue.enqueue(run_id, project_id=PROJECT, revision_id=REVISION)
    first = queue.claim("worker-1")
    assert first is not None
    queue.reclaim_expired(now=first.lease_expires_at + timedelta(seconds=1))

    queue.requeue(
        run_id,
        project_id=PROJECT,
        revision_id=REVISION,
        retry_count=1,
        backoff=timedelta(0),
    )
    second = queue.claim("worker-2")
    assert second is not None
    assert second.attempt_number == 2

    # Both attempts survive — retry history is not overwritten (§4.10).
    attempts = store.query(
        "SELECT state FROM attempts WHERE run_id = ? ORDER BY attempt_number",
        (run_id,),
    )
    assert [a["state"] for a in attempts] == [
        AttemptState.LOST.value,
        AttemptState.LEASED.value,
    ]


# --- cancellation (§14.7) ---------------------------------------------------


def test_cancel_stops_a_queued_run(store: ControlStore, queue: DurableQueue) -> None:
    run_id = make_run(store, "run_1")
    queue.enqueue(run_id, project_id=PROJECT, revision_id=REVISION)

    assert queue.cancel(run_id) is True
    assert queue.claim("worker-1") is None


def test_cancel_fences_an_in_flight_attempt(store: ControlStore, queue: DurableQueue) -> None:
    run_id = make_run(store, "run_1")
    queue.enqueue(run_id, project_id=PROJECT, revision_id=REVISION)
    work = queue.claim("worker-1")
    assert work is not None

    queue.cancel(run_id)

    # The worker cannot finish a cancelled run out from under the user.
    assert queue.complete(work.attempt_id, work.commit_token, succeeded=True) is False


def test_cancelling_a_completed_run_is_refused(store: ControlStore, queue: DurableQueue) -> None:
    run_id = make_run(store, "run_1")
    queue.enqueue(run_id, project_id=PROJECT, revision_id=REVISION)
    work = queue.claim("worker-1")
    assert work is not None
    queue.complete(work.attempt_id, work.commit_token, succeeded=True)

    assert queue.cancel(run_id) is False


# --- scheduler (§7.5) -------------------------------------------------------


def test_scheduler_dispatches_queued_work(store: ControlStore, queue: DurableQueue) -> None:
    scheduler = Scheduler(store, queue)
    run_id = make_run(store, "run_1")
    queue.enqueue(run_id, project_id=PROJECT, revision_id=REVISION)

    work = scheduler.dispatch("worker-1")
    assert work is not None
    assert work.run_id == run_id


def test_scheduler_enforces_per_project_concurrency(
    store: ControlStore, queue: DurableQueue
) -> None:
    scheduler = Scheduler(store, queue, policy=SchedulerPolicy(max_concurrent_per_project=1))
    for i in range(2):
        queue.enqueue(make_run(store, f"run_{i}"), project_id=PROJECT, revision_id=REVISION)

    assert scheduler.dispatch("worker-1") is not None
    # Second dispatch blocked: the project is already at its cap.
    assert scheduler.dispatch("worker-2") is None


def test_scheduler_alternates_between_projects(store: ControlStore, queue: DurableQueue) -> None:
    # Weighted fairness: a deep backlog in one project must not starve another.
    scheduler = Scheduler(
        store,
        queue,
        capacity=Capacity(max_concurrent=10),
        policy=SchedulerPolicy(max_concurrent_per_project=5),
    )
    for i in range(3):
        queue.enqueue(make_run(store, f"busy_{i}"), project_id=PROJECT, revision_id=REVISION)
    queue.enqueue(
        make_run(store, "quiet_0", project_id=OTHER_PROJECT),
        project_id=OTHER_PROJECT,
        revision_id=REVISION,
    )

    first = scheduler.dispatch("worker-1")
    second = scheduler.dispatch("worker-2")
    assert first is not None
    assert second is not None
    assert first.project_id != second.project_id


def test_admission_rejects_oversized_requests(store: ControlStore, queue: DurableQueue) -> None:
    scheduler = Scheduler(store, queue, capacity=Capacity(cpu_cores=4, memory_mb=4096))
    assert scheduler.can_admit(ResourceRequest(cpu_cores=1, memory_mb=512))
    assert not scheduler.can_admit(ResourceRequest(cpu_cores=1, memory_mb=65536))


def test_batch_cannot_use_the_continuous_reservation(
    store: ControlStore, queue: DurableQueue
) -> None:
    # §7.5: reserved capacity for continuous workloads. A batch request sized
    # to the full pool must be refused while a stream request fits.
    scheduler = Scheduler(
        store,
        queue,
        capacity=Capacity(cpu_cores=4, memory_mb=4000),
        policy=SchedulerPolicy(continuous_reservation=0.25),
    )
    full_pool = ResourceRequest(cpu_cores=4, memory_mb=4000)

    assert not scheduler.can_admit(full_pool, WorkloadKind.BATCH)
    assert scheduler.can_admit(full_pool, WorkloadKind.SPARK_STREAM)


def test_quiet_hours_defer_batch_but_not_streams(store: ControlStore, queue: DurableQueue) -> None:
    moment = utcnow().replace(hour=3)
    scheduler = Scheduler(store, queue, policy=SchedulerPolicy(quiet_hours=(3,)))
    queue.enqueue(
        make_run(store, "run_batch"),
        project_id=PROJECT,
        revision_id=REVISION,
        workload_kind=WorkloadKind.BATCH,
    )

    assert scheduler.in_quiet_hours(moment)
    assert scheduler.dispatch("worker-1", now=moment) is None


def test_reclaim_and_retry_requeues_lost_work(store: ControlStore, queue: DurableQueue) -> None:
    scheduler = Scheduler(store, queue)
    run_id = make_run(store, "run_1")
    queue.enqueue(run_id, project_id=PROJECT, revision_id=REVISION)
    work = queue.claim("worker-1")
    assert work is not None

    reclaimed, requeued = scheduler.reclaim_and_retry(
        now=work.lease_expires_at + timedelta(seconds=1)
    )
    assert (reclaimed, requeued) == (1, 1)


def test_retries_are_bounded(store: ControlStore, queue: DurableQueue) -> None:
    # An infinitely retried workload is indistinguishable from a stuck one.
    scheduler = Scheduler(store, queue)
    run_id = make_run(store, "run_1")
    queue.enqueue(run_id, project_id=PROJECT, revision_id=REVISION)

    now = utcnow()
    for _ in range(4):
        work = queue.claim("worker-1", now=now)
        if work is None:
            break
        now = work.lease_expires_at + timedelta(seconds=1)
        scheduler.reclaim_and_retry(max_retries=2, backoff_base=timedelta(0), now=now)

    row = store.query_one("SELECT state FROM runs WHERE run_id = ?", (run_id,))
    assert row is not None
    assert row["state"] == RunState.FAILED.value


def test_interactive_beats_a_batch_backlog_in_another_project(
    store: ControlStore, queue: DurableQueue
) -> None:
    """A SQL preview does not wait its turn behind someone else's batch queue.

    The scheduler's fairness rotation picks a project first and claims only
    within it, so an interactive run in a project that is not "next" would sit
    behind an unrelated backlog. §7.3 gives interactive work high priority and
    §7.7 gives it its own pool; line 1431 names this exact starvation.
    """
    scheduler = Scheduler(
        store,
        queue,
        capacity=Capacity(max_concurrent=10),
        policy=SchedulerPolicy(max_concurrent_per_project=5),
    )
    # Deep backlog in PROJECT, which fairness would otherwise serve first.
    for i in range(5):
        queue.enqueue(make_run(store, f"batch_{i}"), project_id=PROJECT, revision_id=REVISION)
    queue.enqueue(
        make_run(store, "preview", project_id=OTHER_PROJECT, kind=WorkloadKind.INTERACTIVE),
        project_id=OTHER_PROJECT,
        revision_id=REVISION,
        workload_kind=WorkloadKind.INTERACTIVE,
    )

    claimed = scheduler.dispatch("worker-1")
    assert claimed is not None
    assert claimed.run_id == "preview", "interactive work must be served first"


def test_interactive_claim_does_not_cost_the_project_its_turn(
    store: ControlStore, queue: DurableQueue
) -> None:
    """Serving a preview must not push that project's batch work back.

    If an interactive claim updated the fairness clock, a project running many
    previews would keep demoting its own queued batch work — punishing it for
    using the feature.
    """
    scheduler = Scheduler(
        store,
        queue,
        capacity=Capacity(max_concurrent=10),
        policy=SchedulerPolicy(max_concurrent_per_project=5),
    )
    queue.enqueue(
        make_run(store, "preview", kind=WorkloadKind.INTERACTIVE),
        project_id=PROJECT,
        revision_id=REVISION,
        workload_kind=WorkloadKind.INTERACTIVE,
    )
    queue.enqueue(make_run(store, "batch_a"), project_id=PROJECT, revision_id=REVISION)
    queue.enqueue(
        make_run(store, "batch_b", project_id=OTHER_PROJECT),
        project_id=OTHER_PROJECT,
        revision_id=REVISION,
    )

    assert scheduler.dispatch("worker-1").run_id == "preview"
    # PROJECT has not been "served" for fairness purposes, so it still ranks
    # equal-first and its batch work is reachable on the next dispatch.
    remaining = {scheduler.dispatch("worker-2").run_id, scheduler.dispatch("batch-worker").run_id}
    assert remaining == {"batch_a", "batch_b"}


def test_priority_defaults_follow_workload_kind(store: ControlStore, queue: DurableQueue) -> None:
    """Kind sets priority; a flat default is what lets batch outrank a preview."""
    queue.enqueue(
        make_run(store, "preview", kind=WorkloadKind.INTERACTIVE),
        project_id=PROJECT,
        revision_id=REVISION,
        workload_kind=WorkloadKind.INTERACTIVE,
    )
    queue.enqueue(make_run(store, "nightly"), project_id=PROJECT, revision_id=REVISION)

    rows = {
        r["run_id"]: r["priority"]
        for r in store.query("SELECT run_id, priority FROM queue_items")
    }
    assert rows["preview"] < rows["nightly"]


def test_quiet_hours_still_serve_interactive(store: ControlStore, queue: DurableQueue) -> None:
    """Quiet hours defer batch, not a person waiting at a keyboard (§7.5)."""
    # Enqueue first, then read the clock. ``enqueue`` stamps ``not_before`` from
    # the real clock and ``claim`` requires ``not_before <= now``, so a moment
    # captured earlier — even by microseconds — makes the item invisible.
    queue.enqueue(make_run(store, "nightly"), project_id=PROJECT, revision_id=REVISION)
    queue.enqueue(
        make_run(store, "preview", kind=WorkloadKind.INTERACTIVE),
        project_id=PROJECT,
        revision_id=REVISION,
        workload_kind=WorkloadKind.INTERACTIVE,
    )
    moment = utcnow()
    scheduler = Scheduler(
        store,
        queue,
        capacity=Capacity(max_concurrent=10),
        policy=SchedulerPolicy(quiet_hours=(moment.hour,)),
    )

    # Interactive is served despite quiet hours; batch is what gets deferred.
    claimed = scheduler.dispatch("worker-2", now=moment)
    assert claimed is not None
    assert claimed.run_id == "preview"
    assert scheduler.dispatch("worker-1", now=moment) is None, "batch must stay deferred"


def test_queue_depth_by_project(store: ControlStore, queue: DurableQueue) -> None:
    scheduler = Scheduler(store, queue)
    for name in ("a", "b"):
        queue.enqueue(make_run(store, name), project_id=PROJECT, revision_id=REVISION)
    queue.enqueue(
        make_run(store, "c", project_id=OTHER_PROJECT),
        project_id=OTHER_PROJECT,
        revision_id=REVISION,
    )

    assert scheduler.queue_depth_by_project() == {PROJECT: 2, OTHER_PROJECT: 1}
