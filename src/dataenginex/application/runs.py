"""Run lifecycle service (§7.4, §14.7).

Requesting a run is a command with four obligations, in this order: recognise a
replay, pin a revision, get a policy decision, then create the run and enqueue
it. The order is the design. Authorizing after creating would leave denied runs
in the table; enqueueing before the state machine would let a client put work in
the queue that never passed the gate.

This service exists so that ordering lives in one place. It previously lived in
``EmbeddedGateway``, which meant the HTTP API and the CLI would each have had to
re-implement it — and one of them would have got it subtly wrong.

**Nothing here executes anything.** The service records intent and hands work to
the durable queue; workers pick it up. That is §17 Phase 1's exit criterion — *no
workload runs in the Studio process* — expressed as code rather than as a rule
people are asked to remember.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from dataenginex.application.services import ApplicationError, NotFoundError, Service
from dataenginex.domains.security import GovernanceService
from dataenginex.domains.security.governance import ApprovalRequired
from dataenginex.foundation import (
    AuthorizationRequest,
    FrozenModel,
    InteractiveRequest,
    InteractiveResult,
    PolicyDecision,
    PrincipalId,
    ProjectId,
    RevisionId,
    RiskLevel,
    RunId,
    RunState,
    WorkloadKind,
    can_transition,
    new_id,
    utcnow,
)
from dataenginex.runtime.queue import DurableQueue
from dataenginex.runtime.state import ControlStore

__all__ = ["PolicyDenied", "RunAccepted", "RunService", "RunView"]


class PolicyDenied(ApplicationError):
    """Policy refused the request.

    Carries the decision so the caller can cite a decision id. "Denied" without
    one is unauditable — nobody can later reconstruct which rule fired.
    """

    def __init__(self, decision: PolicyDecision) -> None:
        super().__init__(decision.reason)
        self.decision = decision


class RunView(FrozenModel):
    """The client-facing view of a run."""

    run_id: RunId
    project_id: ProjectId
    revision_id: RevisionId
    workload_name: str
    kind: WorkloadKind
    state: RunState
    attempt_count: int = 0
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None


class RunAccepted(FrozenModel):
    """A run was authorized and queued.

    ``replayed`` distinguishes "created" from "already existed" so a client
    retrying after a timeout does not double-count. ``decision_id`` ties the run
    to the policy evaluation that permitted it.
    """

    run_id: RunId
    replayed: bool = False
    decision_id: str | None = None


class RunService(Service):
    """Requesting, observing, and cancelling runs (§7.4)."""

    def __init__(
        self,
        store: ControlStore,
        *,
        governance: GovernanceService | None = None,
        queue: DurableQueue | None = None,
    ) -> None:
        super().__init__(store)
        self.governance = governance or GovernanceService(store)
        self.queue = queue or DurableQueue(store)

    # --- commands -----------------------------------------------------------

    def request_run(
        self,
        project_id: ProjectId,
        workload: str,
        *,
        principal_id: PrincipalId,
        revision_id: RevisionId | None = None,
        idempotency_key: str | None = None,
        trigger_type: str = "manual",
    ) -> RunAccepted:
        """Authorize a run and put it on the durable queue (§7.4, §13.4).

        ``trigger_type`` is recorded rather than assumed. Every run used to be
        stamped "manual", which made a scheduled fire indistinguishable from a
        person pressing the button in the audit trail.
        """
        # Replay check first: a retry must cost nothing and must not
        # re-authorize, or a denied-then-permitted race can produce two runs.
        if idempotency_key:
            existing = self.store.query_one(
                "SELECT run_id FROM runs WHERE project_id = ? AND idempotency_key = ?",
                (project_id, idempotency_key),
            )
            if existing is not None:
                return RunAccepted(run_id=RunId(existing["run_id"]), replayed=True)

        revision = revision_id or self.active_revision(project_id)
        decision = self._authorize(principal_id, project_id, revision, f"run:{workload}")
        kind = self._workload_kind(revision, workload)

        run_id = RunId(new_id("run"))
        with self.store.transaction() as tx:
            tx.execute(
                "INSERT INTO runs (run_id, project_id, revision_id, workload_name, kind, "
                "state, trigger_type, requested_by, created_at, attempt_count, "
                "idempotency_key) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    project_id,
                    revision,
                    workload,
                    kind.value,
                    RunState.REQUESTED.value,
                    trigger_type,
                    principal_id,
                    utcnow().isoformat(),
                    0,
                    idempotency_key,
                ),
            )

        # Through the §7.4 machine, not around it. Every transition is validated,
        # so a state nobody designed cannot be reached by an INSERT.
        self._advance(run_id, RunState.REQUESTED, RunState.AWAITING_POLICY)
        self._advance(run_id, RunState.AWAITING_POLICY, RunState.PLANNING)
        self._advance(run_id, RunState.PLANNING, RunState.QUEUED)

        self.queue.enqueue(
            run_id,
            project_id=project_id,
            revision_id=revision,
            workload_kind=kind,
        )
        return RunAccepted(run_id=run_id, decision_id=decision.decision_id)

    def request_interactive(
        self,
        project_id: ProjectId,
        request: InteractiveRequest,
        *,
        principal_id: PrincipalId,
        revision_id: RevisionId | None = None,
    ) -> RunAccepted:
        """Queue work the user composed just now (§7.3).

        Everything ``request_run`` does — pin a revision, authorize, drive the
        state machine, enqueue — applies here too. The only difference is where
        the operations come from: the request carries them, because there is no
        declared workload to look up.

        Deliberately *not* a shortcut around the queue. Running a SQL preview
        inline would be faster, and would reintroduce exactly what §17 Phase 1
        forbids: a workload executing in the Studio process, outside the lease,
        the capability token, and the resource ceiling.

        No idempotency key. Each press of "run" is a new question, and replaying
        the previous answer would show a stale preview of edited SQL.
        """
        revision = revision_id or self.active_revision(project_id)
        # Authorized as its own action, so policy can permit reading a project
        # interactively while still denying, say, a training run.
        decision = self._authorize(
            principal_id, project_id, revision, f"interactive:{request.label}"
        )

        run_id = RunId(new_id("run"))
        with self.store.transaction() as tx:
            tx.execute(
                "INSERT INTO runs (run_id, project_id, revision_id, workload_name, kind, "
                "state, trigger_type, requested_by, created_at, attempt_count, "
                "ad_hoc_plan_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    project_id,
                    revision,
                    request.label,
                    WorkloadKind.INTERACTIVE.value,
                    RunState.REQUESTED.value,
                    "manual",
                    principal_id,
                    utcnow().isoformat(),
                    0,
                    request.model_dump_json(),
                ),
            )

        self._advance(run_id, RunState.REQUESTED, RunState.AWAITING_POLICY)
        self._advance(run_id, RunState.AWAITING_POLICY, RunState.PLANNING)
        self._advance(run_id, RunState.PLANNING, RunState.QUEUED)

        self.queue.enqueue(
            run_id,
            project_id=project_id,
            revision_id=revision,
            workload_kind=WorkloadKind.INTERACTIVE,
            resource_request=request.resource_request(),
        )
        return RunAccepted(run_id=run_id, decision_id=decision.decision_id)

    def interactive_result(self, run_id: RunId) -> InteractiveResult | None:
        """The stored result of an interactive run, if it is still valid.

        ``None`` covers three genuinely different situations — not produced yet,
        never produced, and expired — because a caller polling for a result
        treats all three the same way: keep waiting, or give up based on the
        run's state. The run's own state distinguishes them when it matters.
        """
        row = self.store.query_one(
            "SELECT * FROM interactive_results WHERE run_id = ?", (run_id,)
        )
        if row is None:
            return None

        result = InteractiveResult(
            run_id=RunId(row["run_id"]),
            project_id=ProjectId(row["project_id"]),
            payload=json.loads(row["payload_json"]),
            row_count=int(row["row_count"]),
            truncated=bool(row["truncated"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            expires_at=datetime.fromisoformat(row["expires_at"]),
        )
        # Checked on read, not left to a sweeper. A result served after its
        # expiry is a stale preview presented as current, and no cleanup
        # schedule can be relied upon to have run.
        if result.expires_at <= utcnow():
            return None
        return result

    def _workload_kind(self, revision_id: RevisionId, workload: str) -> WorkloadKind:
        """The kind the revision declares, defaulting to batch.

        Reading it matters for admission: the scheduler holds back a slice of
        capacity for continuous work (§7.5), and enqueueing a stream as batch
        charges it against the batch pool — the very reservation meant to
        protect it from a batch backlog.
        """
        row = self.store.query_one(
            "SELECT kind FROM workload_definitions WHERE revision_id = ? AND name = ?",
            (revision_id, workload),
        )
        if row is None:
            return WorkloadKind.BATCH
        try:
            return WorkloadKind(row["kind"])
        except ValueError:
            return WorkloadKind.BATCH

    def cancel_run(self, run_id: RunId) -> RunView:
        """Request cancellation (§14.7).

        A run that is committing is not cancellable. Interrupting between
        publishing an output and registering it is what leaves an artifact that
        nothing points at — the §14.3 commit protocol exists to close exactly
        that window, and cancellation must not reopen it.
        """
        row = self.store.query_one("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        if row is None:
            raise NotFoundError(f"no run {run_id}")

        current = RunState(row["state"])
        if not can_transition(current, RunState.CANCELLED):
            raise ApplicationError(f"a run in state {current.value} cannot be cancelled")

        self.queue.cancel(run_id)
        self._advance(run_id, current, RunState.CANCELLED)
        return self.get_run(run_id)

    # --- queries ------------------------------------------------------------

    def get_run(self, run_id: RunId) -> RunView:
        row = self.require_row(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,), subject=f"no run {run_id}"
        )
        return _row_to_run(row)

    def list_runs(
        self,
        project_id: ProjectId | None = None,
        *,
        state: RunState | None = None,
        workload: str | None = None,
        cursor: str | None = None,
        limit: int = 50,
    ) -> tuple[list[RunView], str | None]:
        """Runs newest first, cursor-paginated (§13.8).

        Returns ``(items, next_cursor)``. The cursor is the last seen
        ``created_at``; offsets would skip or repeat rows as new runs arrive
        mid-pagination, which for a run list is the normal case.
        """
        clauses = ["1 = 1"]
        params: list[Any] = []

        if project_id is not None:
            clauses.append("project_id = ?")
            params.append(project_id)
        if state is not None:
            clauses.append("state = ?")
            params.append(state.value)
        if workload is not None:
            clauses.append("workload_name = ?")
            params.append(workload)
        if cursor:
            clauses.append("created_at < ?")
            params.append(cursor)

        # One extra row answers "is there another page?" without a count query.
        params.append(limit + 1)
        rows = self.store.query(
            f"SELECT * FROM runs WHERE {' AND '.join(clauses)} "  # noqa: S608 - literals
            "ORDER BY created_at DESC LIMIT ?",
            params,
        )

        has_more = len(rows) > limit
        page = rows[:limit]
        next_cursor = page[-1]["created_at"] if has_more and page else None
        return [_row_to_run(row) for row in page], next_cursor

    # --- internals ----------------------------------------------------------

    def _authorize(
        self,
        principal_id: PrincipalId,
        project_id: ProjectId,
        revision_id: RevisionId,
        action: str,
    ) -> PolicyDecision:
        request = AuthorizationRequest(
            principal_id=principal_id,
            action=action,
            project_id=project_id,
            revision_id=revision_id,
            risk_level=RiskLevel.CREATE_LOCAL_ARTIFACT,
        )
        try:
            decision = self.governance.authorize(request)
        except ApprovalRequired:
            # Deliberately re-raised untouched. The caller needs the approval id
            # to show the user what to approve; flattening it into a generic
            # denial would turn a solvable state into a dead end.
            raise

        if not decision.allowed:
            raise PolicyDenied(decision)
        return decision

    def _advance(self, run_id: RunId, current: RunState, target: RunState) -> None:
        if not can_transition(current, target):
            raise ApplicationError(f"a run cannot move from {current.value} to {target.value}")
        with self.store.transaction() as tx:
            # Guarded on the current state so two concurrent advances cannot
            # both succeed — the UPDATE simply matches nothing for the loser.
            tx.execute(
                "UPDATE runs SET state = ? WHERE run_id = ? AND state = ?",
                (target.value, run_id, current.value),
            )


def _row_to_run(row: Any) -> RunView:
    data = dict(row)
    return RunView(
        run_id=RunId(data["run_id"]),
        project_id=ProjectId(data["project_id"]),
        revision_id=RevisionId(data["revision_id"]),
        workload_name=data["workload_name"],
        kind=WorkloadKind(data["kind"]),
        state=RunState(data["state"]),
        attempt_count=int(data["attempt_count"]),
        created_at=datetime.fromisoformat(data["created_at"]),
        started_at=datetime.fromisoformat(data["started_at"]) if data["started_at"] else None,
        completed_at=(
            datetime.fromisoformat(data["completed_at"]) if data["completed_at"] else None
        ),
        error=data["error"],
    )
