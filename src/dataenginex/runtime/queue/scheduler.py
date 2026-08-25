"""Scheduler: fairness, concurrency caps, and admission control (§7.5).

One scheduler per installation. It does not perform work — it decides what may
run and hands leases to workers. Keeping placement and execution separate is
what lets the control plane stay responsive while a worker saturates a core
(ADR-0004).

The policies, in the order they apply:

1. **Reserved capacity** for continuous workloads. A batch backlog must never
   starve a stream, so a slice of capacity is held back for streams and
   services only.
2. **Weighted fairness across projects.** Least-recently-served ordering rather
   than raw priority, so one busy project cannot monopolise the installation.
3. **Concurrency caps** per project and per workspace.
4. **Resource admission.** A workload is not dispatched unless its request fits
   the remaining CPU and memory budget.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta

from pydantic import ValidationError

from dataenginex.foundation import (
    ProjectId,
    ResourceRequest,
    RevisionId,
    RunId,
    RunState,
    WorkloadKind,
    utcnow,
)
from dataenginex.foundation.projects import FrozenModel
from dataenginex.runtime.queue.queue import ClaimedWork, DurableQueue, QueueItemState
from dataenginex.runtime.state import ControlStore

__all__ = ["Capacity", "Scheduler", "SchedulerPolicy"]


class Capacity(FrozenModel):
    """What the installation has available right now."""

    cpu_cores: float = 4.0
    memory_mb: int = 4096
    max_concurrent: int = 4


class SchedulerPolicy(FrozenModel):
    """Limits the scheduler enforces (§7.5)."""

    max_concurrent_per_project: int = 2
    max_concurrent_per_workspace: int = 8
    # Fraction of capacity reserved for streams and services. Batch work cannot
    # touch it, which is what stops a training backlog starving a live stream.
    continuous_reservation: float = 0.25
    # Quiet hours as UTC hour numbers. Batch work is deferred during these.
    quiet_hours: tuple[int, ...] = ()


class Scheduler:
    """Decides which queued work may be dispatched, and to whom."""

    def __init__(
        self,
        store: ControlStore,
        queue: DurableQueue,
        *,
        capacity: Capacity | None = None,
        policy: SchedulerPolicy | None = None,
    ) -> None:
        self.store = store
        self.queue = queue
        self.capacity = capacity or Capacity()
        self.policy = policy or SchedulerPolicy()
        # Least-recently-served ordering for project fairness. In-memory: a
        # scheduling hint, not state worth persisting across restarts.
        self._last_served: dict[ProjectId, datetime] = {}

    # --- admission ----------------------------------------------------------

    def running_count(self, project_id: ProjectId | None = None) -> int:
        """Runs currently leased or running."""
        states = (RunState.LEASED.value, RunState.RUNNING.value)
        if project_id is None:
            row = self.store.query_one(
                "SELECT COUNT(*) AS n FROM runs WHERE state IN (?, ?)", states
            )
        else:
            row = self.store.query_one(
                "SELECT COUNT(*) AS n FROM runs WHERE state IN (?, ?) AND project_id = ?",
                (*states, project_id),
            )
        return int(row["n"]) if row else 0

    def committed_resources(self) -> tuple[float, int]:
        """CPU and memory committed to in-flight attempts.

        Planned rather than observed: admission decides before work starts, and
        the plan is the only number available at that moment.
        """
        rows = self.store.query(
            "SELECT planned_resources_json FROM attempts WHERE state IN ('leased', 'running')"
        )
        cpu = 0.0
        memory = 0
        for row in rows:
            request = _parse_request(row["planned_resources_json"])
            cpu += request.cpu_cores
            memory += request.memory_mb
        return cpu, memory

    def can_admit(self, request: ResourceRequest, kind: WorkloadKind = WorkloadKind.BATCH) -> bool:
        """Whether a request fits the remaining budget (§7.5).

        Batch work is measured against capacity *minus* the continuous
        reservation; streams and services may use the whole pool.
        """
        cpu_used, memory_used = self.committed_resources()
        available_cpu = self.capacity.cpu_cores
        available_memory = float(self.capacity.memory_mb)

        if kind not in (WorkloadKind.SPARK_STREAM, WorkloadKind.SERVICE):
            reserve = self.policy.continuous_reservation
            available_cpu *= 1 - reserve
            available_memory *= 1 - reserve

        return (
            cpu_used + request.cpu_cores <= available_cpu
            and memory_used + request.memory_mb <= available_memory
        )

    def within_concurrency_limits(self, project_id: ProjectId) -> bool:
        if self.running_count() >= self.capacity.max_concurrent:
            return False
        if self.running_count(project_id) >= self.policy.max_concurrent_per_project:
            return False

        row = self.store.query_one(
            "SELECT workspace_id FROM projects WHERE project_id = ?", (project_id,)
        )
        if row is None:
            return True
        workspace_running = self.store.query_one(
            "SELECT COUNT(*) AS n FROM runs r JOIN projects p "
            "ON p.project_id = r.project_id WHERE p.workspace_id = ? "
            "AND r.state IN (?, ?)",
            (row["workspace_id"], RunState.LEASED.value, RunState.RUNNING.value),
        )
        count = int(workspace_running["n"]) if workspace_running else 0
        return count < self.policy.max_concurrent_per_workspace

    def in_quiet_hours(self, now: datetime | None = None) -> bool:
        if not self.policy.quiet_hours:
            return False
        return (now or utcnow()).hour in self.policy.quiet_hours

    # --- dispatch -----------------------------------------------------------

    def next_project(self, now: datetime | None = None) -> ProjectId | None:
        """Pick the project to serve next (weighted fairness).

        Least-recently-served wins. A project with queued work that has never
        been served sorts first, so a new project is not stuck behind an
        established one with a deep backlog.
        """
        moment = now or utcnow()
        rows = self.store.query(
            "SELECT DISTINCT project_id FROM queue_items WHERE state = ? AND not_before <= ?",
            (QueueItemState.READY, moment.isoformat()),
        )
        candidates = [ProjectId(r["project_id"]) for r in rows]
        if not candidates:
            return None

        epoch = datetime.min.replace(tzinfo=moment.tzinfo)
        return min(candidates, key=lambda p: self._last_served.get(p, epoch))

    def dispatch(
        self,
        worker_id: str,
        *,
        kinds: tuple[WorkloadKind, ...] = (),
        now: datetime | None = None,
    ) -> ClaimedWork | None:
        """Claim work for a worker, honouring every §7.5 policy.

        Returns None when nothing may run — an empty queue and a fully
        committed installation are deliberately indistinguishable to the
        worker, which simply asks again later.
        """
        moment = now or utcnow()

        # Interactive work is claimed before the fairness rotation runs, across
        # all projects at once (§7.3, §7.7 "Interactive" pool). Someone is
        # waiting on a SQL preview; making it wait its turn behind another
        # project's batch backlog is the starvation line 1431 calls out. The
        # pool is bounded elsewhere — short timeouts and a small resource
        # ceiling — so it cannot itself become the thing that starves batch.
        if not kinds or WorkloadKind.INTERACTIVE in kinds:
            interactive = self.queue.claim(
                worker_id, kinds=(WorkloadKind.INTERACTIVE,), now=moment
            )
            if interactive is not None:
                self._last_served.setdefault(interactive.project_id, moment)
                return interactive

        project_id = self.next_project(moment)
        if project_id is None:
            return None

        if not self.within_concurrency_limits(project_id):
            return None

        allowed = kinds
        if self.in_quiet_hours(moment):
            # Quiet hours defer batch work only; a stream must keep consuming
            # and a person at a keyboard still deserves an answer.
            allowed = tuple(
                k
                for k in (kinds or tuple(WorkloadKind))
                if k in (WorkloadKind.SPARK_STREAM, WorkloadKind.SERVICE, WorkloadKind.INTERACTIVE)
            )
            if not allowed:
                return None

        # Scoped to the chosen project: an unscoped claim would follow global
        # priority and undo the fairness decision just made.
        claimed = self.queue.claim(worker_id, kinds=allowed, project_id=project_id, now=moment)
        if claimed is None:
            return None

        self._last_served[claimed.project_id] = moment
        return claimed

    # --- maintenance --------------------------------------------------------

    def reclaim_and_retry(
        self,
        *,
        max_retries: int = 3,
        backoff_base: timedelta = timedelta(seconds=5),
        now: datetime | None = None,
    ) -> tuple[int, int]:
        """Reclaim expired leases and requeue what may still be retried.

        Returns ``(reclaimed, requeued)``. A lost attempt past its retry budget
        fails the run rather than looping — an infinitely retried workload is
        indistinguishable from a stuck one.
        """
        reclaimed = self.queue.reclaim_expired(now)
        requeued = 0

        for attempt_id in reclaimed:
            row = self.store.query_one(
                "SELECT a.run_id, a.project_id, a.revision_id, a.attempt_number, "
                "r.kind FROM attempts a JOIN runs r ON r.run_id = a.run_id "
                "WHERE a.attempt_id = ?",
                (attempt_id,),
            )
            if row is None:
                continue

            attempt_number = int(row["attempt_number"])
            if attempt_number > max_retries:
                with self.store.transaction() as tx:
                    tx.execute(
                        "UPDATE runs SET state = ?, completed_at = ?, error = ? WHERE run_id = ?",
                        (
                            RunState.FAILED.value,
                            (now or utcnow()).isoformat(),
                            f"exhausted {max_retries} retries after lost leases",
                            row["run_id"],
                        ),
                    )
                continue

            # Exponential backoff, so a systematically failing workload backs
            # off instead of hammering the queue.
            self.queue.requeue(
                RunId(row["run_id"]),
                project_id=ProjectId(row["project_id"]),
                revision_id=RevisionId(row["revision_id"]),
                retry_count=attempt_number,
                backoff=backoff_base * (2 ** (attempt_number - 1)),
                workload_kind=WorkloadKind(row["kind"]),
            )
            requeued += 1

        return len(reclaimed), requeued

    def queue_depth_by_project(self) -> dict[ProjectId, int]:
        rows = self.store.query(
            "SELECT project_id, COUNT(*) AS n FROM queue_items WHERE state = ? GROUP BY project_id",
            (QueueItemState.READY,),
        )
        depths: dict[ProjectId, int] = defaultdict(int)
        for row in rows:
            depths[ProjectId(row["project_id"])] = int(row["n"])
        return dict(depths)


def _parse_request(raw: str) -> ResourceRequest:
    """Read a planned request, tolerating rows that predate a field."""
    try:
        data = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return ResourceRequest()
    if not isinstance(data, dict) or not data:
        return ResourceRequest()
    try:
        return ResourceRequest.model_validate(data)
    except ValidationError:
        return ResourceRequest()
