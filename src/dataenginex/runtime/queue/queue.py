"""Durable DB-backed queue with transactional leases (§7.6, ADR-0006).

The control database *is* the queue in Lite mode. No broker, no Redis. That is a
deliberate choice: a single-node install should not need a second daemon to run
a pipeline, and SQLite in WAL mode handles this load comfortably.

Two mechanisms make it correct:

**Transactional claim.** ``claim`` finds a ready item and marks it leased in one
transaction. Two workers racing for the same item cannot both win, because the
loser's ``UPDATE ... WHERE state = 'ready'`` matches zero rows.

**Lease expiry with attempt fencing.** A worker holds a time-limited lease and
heartbeats it. If the lease expires the attempt becomes ``LOST`` — not failed,
because the control plane genuinely does not know what happened — and a new
attempt may be queued. The old worker can still be alive and finish later, so
each attempt carries a commit token; :meth:`reclaim_expired` clears it, which is
what stops a late completion from overwriting the newer attempt's result
(§7.6 final bullet, §14.3).
"""

from __future__ import annotations

from datetime import datetime, timedelta

from dataenginex.foundation import (
    AttemptId,
    AttemptState,
    EventEnvelope,
    MetadataEvent,
    PrincipalId,
    ProjectId,
    ResourceRequest,
    RevisionId,
    RunId,
    RunState,
    WorkloadKind,
    can_transition,
    new_id,
    priority_for,
    utcnow,
)
from dataenginex.foundation.projects import FrozenModel
from dataenginex.runtime.state import ControlStore, Transaction

__all__ = ["ClaimedWork", "DurableQueue", "QueueError", "QueueItemState"]


class QueueError(RuntimeError):
    """Queue operation that could not be completed."""


class QueueItemState:
    """Queue row states.

    Plain constants rather than an enum: these never leave the queue module or
    reach the domain model, and the schema stores them as text.
    """

    READY = "ready"
    LEASED = "leased"
    DONE = "done"
    ABANDONED = "abandoned"


class ClaimedWork(FrozenModel):
    """One unit of work handed to a worker, with its lease.

    ``commit_token`` fences the result: the worker presents it when committing,
    and it stops matching once the lease is reclaimed.
    """

    queue_item_id: str
    run_id: RunId
    attempt_id: AttemptId
    project_id: ProjectId
    revision_id: RevisionId
    workload_kind: WorkloadKind
    attempt_number: int
    lease_id: str
    lease_expires_at: datetime
    commit_token: str
    retry_count: int


class DurableQueue:
    """Enqueue, claim, heartbeat, complete, and reclaim (§7.6)."""

    def __init__(
        self,
        store: ControlStore,
        *,
        lease_duration: timedelta = timedelta(minutes=5),
    ) -> None:
        self.store = store
        self.lease_duration = lease_duration

    # --- workers ------------------------------------------------------------

    def register_worker(
        self,
        worker_id: str,
        *,
        pool: str = "batch",
        hostname: str = "",
        pid: int | None = None,
        capabilities: tuple[str, ...] = (),
    ) -> None:
        """Register a worker, or mark an existing one alive again.

        A worker must exist before it can hold a lease — the ``leases`` foreign
        key enforces that, which is deliberate: an unregistered worker holding
        work would be invisible to the recovery sweep that reclaims leases from
        dead processes (§14.5).

        Idempotent, because a restarting worker reuses its ID.
        """
        now = utcnow()
        with self.store.transaction() as tx:
            tx.execute(
                "INSERT INTO workers (worker_id, pool, hostname, pid, state, "
                "started_at, last_heartbeat_at) VALUES (?,?,?,?,?,?,?) "
                "ON CONFLICT(worker_id) DO UPDATE SET state = excluded.state, "
                "last_heartbeat_at = excluded.last_heartbeat_at",
                (
                    worker_id,
                    pool,
                    hostname,
                    pid,
                    "alive",
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
            for capability in capabilities:
                tx.execute(
                    "INSERT OR IGNORE INTO worker_capabilities (worker_id, "
                    "capability) VALUES (?,?)",
                    (worker_id, capability),
                )

    # --- enqueue ------------------------------------------------------------

    def enqueue(
        self,
        run_id: RunId,
        *,
        project_id: ProjectId,
        revision_id: RevisionId,
        workload_kind: WorkloadKind = WorkloadKind.BATCH,
        priority: int | None = None,
        not_before: datetime | None = None,
        idempotency_key: str | None = None,
        retry_count: int = 0,
        resource_request: ResourceRequest | None = None,
    ) -> str:
        """Queue a run for execution.

        The run must be in a state that can legally reach ``QUEUED`` (§7.4),
        which in practice means ``PLANNING`` — policy evaluation and any
        required approval happen *before* work reaches the queue. A freshly
        ``REQUESTED`` run is rejected here on purpose: enqueueing it directly
        would be the policy bypass §9.3 exists to prevent. Cancelled and
        terminal runs are rejected for the same structural reason.
        """
        row = self.store.query_one("SELECT state FROM runs WHERE run_id = ?", (run_id,))
        if row is None:
            raise QueueError(f"run {run_id} does not exist")

        current = RunState(row["state"])
        if current is not RunState.QUEUED and not can_transition(current, RunState.QUEUED):
            raise QueueError(f"run {run_id} is {current.value} and cannot be queued (§7.4)")

        # Kind decides priority unless the caller insists. Defaulting to a flat
        # number regardless of kind is what puts an interactive preview behind a
        # batch backlog (§7.3, §7.5).
        effective_priority = priority_for(workload_kind) if priority is None else priority

        queue_item_id = new_id("qi")
        now = utcnow()
        with self.store.transaction() as tx:
            tx.execute(
                "INSERT INTO queue_items (queue_item_id, run_id, project_id, "
                "revision_id, workload_kind, priority, not_before, retry_count, "
                "idempotency_key, state, created_at, resource_request_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    queue_item_id,
                    run_id,
                    project_id,
                    revision_id,
                    workload_kind.value,
                    effective_priority,
                    (not_before or now).isoformat(),
                    retry_count,
                    idempotency_key,
                    QueueItemState.READY,
                    now.isoformat(),
                    # What this item asks for, recorded at enqueue time. The
                    # column has existed since the first migration and nothing
                    # wrote it, so admission control had nothing to admit
                    # against — every item looked like it wanted the default.
                    (resource_request or ResourceRequest()).model_dump_json(),
                ),
            )
            if current is not RunState.QUEUED:
                tx.execute(
                    "UPDATE runs SET state = ? WHERE run_id = ?",
                    (RunState.QUEUED.value, run_id),
                )
            tx.emit_metadata(
                MetadataEvent(
                    envelope=EventEnvelope(
                        producer="queue",
                        project_id=project_id,
                        revision_id=revision_id,
                    ),
                    event_type="RunQueued",
                    subject_id=run_id,
                    subject_type="run",
                    payload={"priority": priority, "retry_count": retry_count},
                )
            )
        return queue_item_id

    # --- claim --------------------------------------------------------------

    def claim(
        self,
        worker_id: str,
        *,
        kinds: tuple[WorkloadKind, ...] = (),
        project_id: ProjectId | None = None,
        now: datetime | None = None,
    ) -> ClaimedWork | None:
        """Atomically claim the highest-priority ready item, or return None.

        Ordering is priority first, then ``not_before``, then insertion order.
        Selection and marking share one transaction, so two workers cannot claim
        the same row.

        ``project_id`` restricts the claim to one project. The scheduler passes
        it to enforce fairness: without it, priority ordering alone would keep
        handing work to whichever project queued the most urgent item, which is
        exactly the starvation §7.5 fairness exists to prevent.
        """
        moment = now or utcnow()
        if self.store.query_one("SELECT 1 FROM workers WHERE worker_id = ?", (worker_id,)) is None:
            raise QueueError(
                f"worker {worker_id!r} is not registered; call register_worker "
                "first so leases can be reclaimed if it dies (§14.5)"
            )

        filters = ""
        params: list[object] = [QueueItemState.READY, moment.isoformat()]
        if kinds:
            placeholders = ",".join("?" for _ in kinds)
            filters += f" AND workload_kind IN ({placeholders})"
            params.extend(k.value for k in kinds)
        if project_id is not None:
            filters += " AND project_id = ?"
            params.append(project_id)

        with self.store.transaction() as tx:
            row = tx.execute(
                "SELECT * FROM queue_items WHERE state = ? AND not_before <= ?"
                f"{filters} ORDER BY priority ASC, not_before ASC, created_at ASC "
                "LIMIT 1",
                params,
            ).fetchone()
            if row is None:
                return None

            # The guard that makes the claim safe: if another worker took this
            # row between the SELECT and here, zero rows update and we bail.
            expires = moment + self.lease_duration
            cursor = tx.execute(
                "UPDATE queue_items SET state = ?, leased_by = ?, "
                "lease_expires_at = ? WHERE queue_item_id = ? AND state = ?",
                (
                    QueueItemState.LEASED,
                    worker_id,
                    expires.isoformat(),
                    row["queue_item_id"],
                    QueueItemState.READY,
                ),
            )
            if cursor.rowcount != 1:
                return None

            run_id = RunId(row["run_id"])
            attempt_number = _next_attempt_number(tx, run_id)
            attempt_id = AttemptId(new_id("att"))
            commit_token = new_id("ctk")
            principal = _run_principal(tx, run_id)

            tx.execute(
                "INSERT INTO attempts (attempt_id, run_id, project_id, revision_id, "
                "attempt_number, state, principal_id, worker_id, started_at, "
                "last_heartbeat_at, commit_token) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    attempt_id,
                    run_id,
                    row["project_id"],
                    row["revision_id"],
                    attempt_number,
                    AttemptState.LEASED.value,
                    principal,
                    worker_id,
                    moment.isoformat(),
                    moment.isoformat(),
                    commit_token,
                ),
            )
            tx.execute(
                "UPDATE queue_items SET attempt_id = ? WHERE queue_item_id = ?",
                (attempt_id, row["queue_item_id"]),
            )
            tx.execute(
                "UPDATE runs SET state = ?, attempt_count = attempt_count + 1, "
                "started_at = COALESCE(started_at, ?) WHERE run_id = ?",
                (RunState.LEASED.value, moment.isoformat(), run_id),
            )

            lease_id = new_id("lse")
            tx.execute(
                "INSERT INTO leases (lease_id, queue_item_id, attempt_id, worker_id, "
                "acquired_at, expires_at) VALUES (?,?,?,?,?,?)",
                (
                    lease_id,
                    row["queue_item_id"],
                    attempt_id,
                    worker_id,
                    moment.isoformat(),
                    expires.isoformat(),
                ),
            )
            tx.execute(
                "INSERT INTO heartbeats (attempt_id, worker_id, beat_at) VALUES (?,?,?)",
                (attempt_id, worker_id, moment.isoformat()),
            )

            return ClaimedWork(
                queue_item_id=row["queue_item_id"],
                run_id=run_id,
                attempt_id=attempt_id,
                project_id=ProjectId(row["project_id"]),
                revision_id=RevisionId(row["revision_id"]),
                workload_kind=WorkloadKind(row["workload_kind"]),
                attempt_number=attempt_number,
                lease_id=lease_id,
                lease_expires_at=expires,
                commit_token=commit_token,
                retry_count=int(row["retry_count"]),
            )

    # --- heartbeat ----------------------------------------------------------

    def heartbeat(self, attempt_id: AttemptId, worker_id: str, now: datetime | None = None) -> bool:
        """Extend a lease. False means the lease is no longer this worker's.

        A False here means the worker was reclaimed while it was working; it
        must stop, because a newer attempt may already be running.
        """
        moment = now or utcnow()
        expires = moment + self.lease_duration
        with self.store.transaction() as tx:
            cursor = tx.execute(
                "UPDATE leases SET expires_at = ? WHERE attempt_id = ? "
                "AND worker_id = ? AND released_at IS NULL AND lost = 0",
                (expires.isoformat(), attempt_id, worker_id),
            )
            if cursor.rowcount != 1:
                return False
            tx.execute(
                "UPDATE queue_items SET lease_expires_at = ? WHERE attempt_id = ?",
                (expires.isoformat(), attempt_id),
            )
            tx.execute(
                "UPDATE attempts SET last_heartbeat_at = ?, state = ? "
                "WHERE attempt_id = ? AND state IN (?, ?)",
                (
                    moment.isoformat(),
                    AttemptState.RUNNING.value,
                    attempt_id,
                    AttemptState.LEASED.value,
                    AttemptState.RUNNING.value,
                ),
            )
            tx.execute(
                "UPDATE heartbeats SET beat_at = ? WHERE attempt_id = ?",
                (moment.isoformat(), attempt_id),
            )
            tx.execute(
                "UPDATE runs SET state = ? WHERE run_id = "
                "(SELECT run_id FROM attempts WHERE attempt_id = ?) AND state = ?",
                (RunState.RUNNING.value, attempt_id, RunState.LEASED.value),
            )
        return True

    # --- completion ---------------------------------------------------------

    def complete(
        self,
        attempt_id: AttemptId,
        commit_token: str,
        *,
        succeeded: bool,
        error: str | None = None,
        error_class: str | None = None,
        now: datetime | None = None,
    ) -> bool:
        """Finish an attempt, if its commit token is still valid (§14.3).

        False means the token was fenced — a late completion from a reclaimed
        worker. Accepting it would let stale output overwrite a newer attempt's
        result.
        """
        moment = now or utcnow()
        row = self.store.query_one(
            "SELECT run_id, commit_token FROM attempts WHERE attempt_id = ?",
            (attempt_id,),
        )
        if row is None:
            raise QueueError(f"attempt {attempt_id} does not exist")
        if row["commit_token"] != commit_token:
            return False

        run_id = RunId(row["run_id"])
        attempt_state = AttemptState.SUCCEEDED if succeeded else AttemptState.FAILED
        run_state = RunState.COMPLETED if succeeded else RunState.FAILED

        with self.store.transaction() as tx:
            tx.execute(
                "UPDATE attempts SET state = ?, completed_at = ?, error = ?, "
                "error_class = ? WHERE attempt_id = ?",
                (
                    attempt_state.value,
                    moment.isoformat(),
                    error,
                    error_class,
                    attempt_id,
                ),
            )
            tx.execute(
                "UPDATE queue_items SET state = ?, leased_by = NULL, "
                "lease_expires_at = NULL WHERE attempt_id = ?",
                (QueueItemState.DONE, attempt_id),
            )
            tx.execute(
                "UPDATE leases SET released_at = ? WHERE attempt_id = ?",
                (moment.isoformat(), attempt_id),
            )
            # Runs pass through COMMITTING on the way to COMPLETED (§7.4).
            if succeeded:
                tx.execute(
                    "UPDATE runs SET state = ? WHERE run_id = ?",
                    (RunState.COMMITTING.value, run_id),
                )
            tx.execute(
                "UPDATE runs SET state = ?, completed_at = ?, error = ? WHERE run_id = ?",
                (run_state.value, moment.isoformat(), error, run_id),
            )
            tx.emit_metadata(
                MetadataEvent(
                    envelope=EventEnvelope(producer="queue"),
                    event_type="RunCompleted" if succeeded else "RunFailed",
                    subject_id=run_id,
                    subject_type="run",
                    payload={"attempt_id": attempt_id, "error": error},
                )
            )
        return True

    # --- reclaim ------------------------------------------------------------

    def reclaim_expired(self, now: datetime | None = None) -> tuple[AttemptId, ...]:
        """Reclaim attempts whose leases expired (§7.6).

        The expired attempt becomes ``LOST``, not ``FAILED``: the worker may
        still be alive and the outcome is unknown. Clearing its commit token
        fences any late completion — the mechanism that stops "the previous
        worker's late completion" from overwriting a new attempt.

        Returns the reclaimed attempt IDs. Requeueing is the scheduler's call,
        since it owns the retry policy.
        """
        moment = now or utcnow()
        expired = self.store.query(
            "SELECT queue_item_id, attempt_id FROM queue_items "
            "WHERE state = ? AND lease_expires_at IS NOT NULL "
            "AND lease_expires_at <= ?",
            (QueueItemState.LEASED, moment.isoformat()),
        )
        if not expired:
            return ()

        reclaimed: list[AttemptId] = []
        with self.store.transaction() as tx:
            for row in expired:
                attempt_id = AttemptId(row["attempt_id"])
                reclaimed.append(attempt_id)
                tx.execute(
                    "UPDATE attempts SET state = ?, commit_token = NULL, error = ? "
                    "WHERE attempt_id = ?",
                    (
                        AttemptState.LOST.value,
                        "lease expired; worker stopped heartbeating",
                        attempt_id,
                    ),
                )
                tx.execute(
                    "UPDATE leases SET lost = 1, released_at = ? WHERE attempt_id = ?",
                    (moment.isoformat(), attempt_id),
                )
                tx.execute(
                    "UPDATE queue_items SET state = ?, leased_by = NULL, "
                    "lease_expires_at = NULL, attempt_id = NULL "
                    "WHERE queue_item_id = ?",
                    (QueueItemState.ABANDONED, row["queue_item_id"]),
                )
                tx.emit_metadata(
                    MetadataEvent(
                        envelope=EventEnvelope(producer="queue"),
                        event_type="AttemptLost",
                        subject_id=attempt_id,
                        subject_type="attempt",
                        payload={"reason": "lease_expired"},
                    )
                )
        return tuple(reclaimed)

    def requeue(
        self,
        run_id: RunId,
        *,
        project_id: ProjectId,
        revision_id: RevisionId,
        retry_count: int,
        backoff: timedelta = timedelta(seconds=5),
        workload_kind: WorkloadKind = WorkloadKind.BATCH,
        priority: int = 100,
    ) -> str:
        """Queue a retry after a lost or failed attempt.

        ``not_before`` carries the backoff so a failing workload does not spin
        the queue at full speed.
        """
        with self.store.transaction() as tx:
            tx.execute(
                "UPDATE runs SET state = ? WHERE run_id = ?",
                (RunState.QUEUED.value, run_id),
            )
        return self.enqueue(
            run_id,
            project_id=project_id,
            revision_id=revision_id,
            workload_kind=workload_kind,
            priority=priority,
            not_before=utcnow() + backoff,
            retry_count=retry_count,
        )

    def cancel(self, run_id: RunId, now: datetime | None = None) -> bool:
        """Cancel a queued or running run (§14.7).

        A run already in a terminal state is left alone — cancelling a finished
        run would rewrite its outcome.
        """
        moment = now or utcnow()
        row = self.store.query_one("SELECT state FROM runs WHERE run_id = ?", (run_id,))
        if row is None:
            raise QueueError(f"run {run_id} does not exist")

        current = RunState(row["state"])
        if not can_transition(current, RunState.CANCELLED):
            return False

        with self.store.transaction() as tx:
            tx.execute(
                "UPDATE runs SET state = ?, completed_at = ? WHERE run_id = ?",
                (RunState.CANCELLED.value, moment.isoformat(), run_id),
            )
            tx.execute(
                "UPDATE queue_items SET state = ?, leased_by = NULL, "
                "lease_expires_at = NULL WHERE run_id = ? AND state IN (?, ?)",
                (
                    QueueItemState.ABANDONED,
                    run_id,
                    QueueItemState.READY,
                    QueueItemState.LEASED,
                ),
            )
            # Fence any in-flight attempt so a late finish cannot resurrect it.
            tx.execute(
                "UPDATE attempts SET state = ?, commit_token = NULL, completed_at = ? "
                "WHERE run_id = ? AND state IN (?, ?)",
                (
                    AttemptState.CANCELLED.value,
                    moment.isoformat(),
                    run_id,
                    AttemptState.LEASED.value,
                    AttemptState.RUNNING.value,
                ),
            )
        return True

    # --- introspection ------------------------------------------------------

    def depth(self, project_id: ProjectId | None = None) -> int:
        """Ready items, optionally scoped to one project."""
        if project_id is None:
            row = self.store.query_one(
                "SELECT COUNT(*) AS n FROM queue_items WHERE state = ?",
                (QueueItemState.READY,),
            )
        else:
            row = self.store.query_one(
                "SELECT COUNT(*) AS n FROM queue_items WHERE state = ? AND project_id = ?",
                (QueueItemState.READY, project_id),
            )
        return int(row["n"]) if row else 0


def _next_attempt_number(tx: Transaction, run_id: RunId) -> int:
    row = tx.execute(
        "SELECT COALESCE(MAX(attempt_number), 0) + 1 AS n FROM attempts WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    return int(row["n"])


def _run_principal(tx: Transaction, run_id: RunId) -> PrincipalId:
    row = tx.execute("SELECT requested_by FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    return PrincipalId(row["requested_by"]) if row else PrincipalId("unknown")
