"""Workload definitions, runs, and execution attempts (§4.9-4.10).

The run/attempt split is the point of this module. A *run* is the logical
occurrence of a workload; an *attempt* is one physical try. Modelling them as
one record means a retry overwrites the failed attempt's history — the
diagnostic evidence disappears exactly when it is most needed.

Pipelines, streams, and services are specialized workload definitions here, not
unrelated execution engines with their own state handling.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import Field

from dataenginex.foundation.ids import (
    ArtifactId,
    AttemptId,
    PolicyDecisionId,
    PrincipalId,
    ProjectId,
    RevisionId,
    RunId,
    new_id,
)
from dataenginex.foundation.operations import Operation, ResourceRequest
from dataenginex.foundation.projects import FrozenModel, utcnow

__all__ = [
    "AttemptState",
    "ExecutionAttempt",
    "InteractiveRequest",
    "InteractiveResult",
    "ObservedResources",
    "RunState",
    "ServiceState",
    "TriggerType",
    "WorkloadDefinition",
    "WorkloadKind",
    "WorkloadRun",
    "can_service_transition",
    "can_transition",
    "is_service_terminal",
    "is_terminal",
]


class WorkloadKind(StrEnum):
    """How a workload runs (§4.9)."""

    INTERACTIVE = "interactive"
    BATCH = "batch"
    SPARK_STREAM = "spark_stream"
    SPARK_PIPELINE = "spark_pipeline"
    EXTERNAL_ACTION = "external_action"
    SERVICE = "service"


# Queue priority by kind. Lower dispatches sooner (§7.3).
_KIND_PRIORITY: dict[str, int] = {
    "interactive": 10,
    "service": 50,
    "spark_stream": 60,
    "spark_pipeline": 80,
    "external_action": 90,
    "batch": 100,
}


def priority_for(kind: WorkloadKind) -> int:
    """Default queue priority for a workload kind.

    Callers may override per run, but the default has to encode the ordering —
    a flat priority for every kind means a five-second SQL preview waits behind
    an hour of queued training, which is the starvation §7.5 exists to prevent.
    """
    return _KIND_PRIORITY.get(kind.value, 100)


class InteractiveRequest(FrozenModel):
    """Work a user asked for directly, composed on the spot (§7.3).

    A SQL preview or a schema inspection has no declared workload — the user
    typed it a second ago. So the operations travel with the request rather than
    being looked up, and the run carries them.

    What does *not* change is the revision. An interactive run pins one like any
    other (§17 Phase 1), so a preview can only reach resources the project
    declared. Ad hoc means "not declared in advance", never "unbounded".

    The ceilings are part of the type because §7.3 names them: short timeout,
    small resource ceiling. A preview that can run for an hour and return a
    million rows is a batch job wearing a preview's clothes, and it will hold a
    worker slot the next person's preview needs.
    """

    operations: tuple[Operation, ...]
    # Short by default. §7.3 wants interactive work to fail fast rather than
    # occupy the pool; a user watching a spinner has already given up by 30s.
    timeout_seconds: int = Field(default=30, gt=0, le=300)
    # Caps what a handler may return. Enforced in the handler, not just here,
    # because the memory cost is paid where the rows are materialized.
    max_rows: int = Field(default=1000, gt=0, le=10_000)
    # Free-form label for logs and the run list ("sql_preview", "schema").
    label: str = "interactive"

    def resource_request(self) -> ResourceRequest:
        """The ceiling this request runs under.

        Deliberately smaller than a batch default: interactive work is sized to
        be admitted immediately, not to be efficient.
        """
        return ResourceRequest(
            cpu_cores=1.0, memory_mb=512, timeout_seconds=self.timeout_seconds
        )


class InteractiveResult(FrozenModel):
    """What an interactive run produced, and how long it stays available.

    Ephemeral by design (§7.3: "Results may be ephemeral until explicitly
    saved"). This is not an artifact — nothing here is content-addressed or
    retained, and a caller that wants to keep a preview must save it as
    something else.
    """

    run_id: RunId
    project_id: ProjectId
    payload: dict[str, Any]
    row_count: int = Field(default=0, ge=0)
    # True when the handler stopped at ``max_rows``. Surfaced rather than
    # inferred from a row count that happens to equal the cap: "exactly 1000
    # rows" and "the first 1000 of more" are different answers, and a UI that
    # cannot tell them apart will show a total that is quietly wrong.
    truncated: bool = False
    created_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime


class TriggerType(StrEnum):
    MANUAL = "manual"
    SCHEDULE = "schedule"
    EVENT = "event"
    DEPENDENCY = "dependency"
    API = "api"


class RunState(StrEnum):
    """Lifecycle of a logical run (§7.4).

    The intermediate states are not ceremony. Each one is a point where a run
    can legitimately sit while something outside it decides: policy evaluation,
    a human approval, planning, an idle queue, a worker holding a lease, or the
    commit protocol writing outputs. Collapsing them loses the ability to say
    *why* a run is not progressing.
    """

    REQUESTED = "requested"
    AWAITING_POLICY = "awaiting_policy"
    AWAITING_APPROVAL = "awaiting_approval"
    PLANNING = "planning"
    QUEUED = "queued"
    LEASED = "leased"
    RUNNING = "running"
    COMMITTING = "committing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class ServiceState(StrEnum):
    """Extra states for stream and service workloads (§7.4).

    Long-lived workloads are not simply running or finished — they degrade,
    restart, and pause while remaining the same logical run.
    """

    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    RESTARTING = "restarting"
    PAUSED = "paused"
    STOPPED = "stopped"


class AttemptState(StrEnum):
    """Lifecycle of one physical attempt.

    ``LOST`` is distinct from ``FAILED``: the worker stopped heartbeating and
    the outcome is genuinely unknown. Recording it as failure would assert
    something the control plane cannot know, and the commit protocol (§14.3)
    still has to defend against a lost attempt finishing late.
    """

    PENDING = "pending"
    LEASED = "leased"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    LOST = "lost"


_TERMINAL_RUN_STATES = frozenset(
    {RunState.COMPLETED, RunState.FAILED, RunState.CANCELLED, RunState.TIMED_OUT}
)

# Every non-terminal state can be cancelled or time out, so those are folded in
# below rather than repeated on each row.
_INTERRUPTIBLE = frozenset({RunState.CANCELLED, RunState.TIMED_OUT})

# Allowed run transitions (§7.4). Anything absent is rejected — no component
# writes run state directly, which is what keeps a run from resurrecting after
# a terminal outcome or skipping the policy gate.
_RUN_TRANSITIONS: dict[RunState, frozenset[RunState]] = {
    # Policy is evaluated before anything is planned or queued: a run that will
    # be denied must not consume planning or scheduler capacity first.
    RunState.REQUESTED: frozenset({RunState.AWAITING_POLICY}) | _INTERRUPTIBLE,
    RunState.AWAITING_POLICY: frozenset(
        # Approval is optional — a permit goes straight to planning, a denial
        # fails the run outright.
        {RunState.AWAITING_APPROVAL, RunState.PLANNING, RunState.FAILED}
    )
    | _INTERRUPTIBLE,
    RunState.AWAITING_APPROVAL: frozenset({RunState.PLANNING, RunState.FAILED})
    | _INTERRUPTIBLE,
    RunState.PLANNING: frozenset({RunState.QUEUED, RunState.FAILED}) | _INTERRUPTIBLE,
    RunState.QUEUED: frozenset({RunState.LEASED}) | _INTERRUPTIBLE,
    # A lease can expire before work starts, which returns the run to the queue.
    RunState.LEASED: frozenset({RunState.RUNNING, RunState.QUEUED, RunState.FAILED})
    | _INTERRUPTIBLE,
    # Back to QUEUED when an attempt fails and the retry policy allows another.
    RunState.RUNNING: frozenset(
        {RunState.COMMITTING, RunState.QUEUED, RunState.FAILED}
    )
    | _INTERRUPTIBLE,
    # Commit is the last point a run can fail: outputs may not validate. It is
    # deliberately not cancellable — interrupting a half-written commit is what
    # the protocol in §14.3 exists to prevent.
    RunState.COMMITTING: frozenset({RunState.COMPLETED, RunState.FAILED}),
    RunState.COMPLETED: frozenset(),
    RunState.FAILED: frozenset(),
    RunState.CANCELLED: frozenset(),
    RunState.TIMED_OUT: frozenset(),
}


# Allowed service transitions (§7.4). A long-lived workload has a second,
# smaller lifecycle running inside ``RunState.RUNNING``: the run does not end
# when the service degrades or restarts, because it is still the same logical
# run with the same identity, cursor, and history.
#
# The shape worth noticing is that DEGRADED is neither terminal nor a failure.
# A stream whose source is briefly unreachable is degraded, not broken, and
# collapsing that into FAILED would either restart work that did not need
# restarting or hide a real outage behind a retry.
_SERVICE_TRANSITIONS: dict[ServiceState, frozenset[ServiceState]] = {
    ServiceState.STARTING: frozenset(
        {ServiceState.HEALTHY, ServiceState.DEGRADED, ServiceState.STOPPED}
    ),
    ServiceState.HEALTHY: frozenset(
        {
            ServiceState.DEGRADED,
            ServiceState.RESTARTING,
            ServiceState.PAUSED,
            ServiceState.STOPPED,
        }
    ),
    # Recovers on its own, or is restarted, or gives up. All three happen.
    ServiceState.DEGRADED: frozenset(
        {
            ServiceState.HEALTHY,
            ServiceState.RESTARTING,
            ServiceState.PAUSED,
            ServiceState.STOPPED,
        }
    ),
    # A restart goes back through STARTING rather than straight to HEALTHY:
    # "restarted" and "confirmed working again" are different claims, and only
    # a health check can make the second one.
    ServiceState.RESTARTING: frozenset({ServiceState.STARTING, ServiceState.STOPPED}),
    # Paused is deliberate and reversible — backpressure, a maintenance window,
    # a human holding it. Resuming re-enters STARTING for the same reason.
    ServiceState.PAUSED: frozenset({ServiceState.STARTING, ServiceState.STOPPED}),
    # The only terminal service state. A stopped service that should run again
    # is a new run, so its cursor and history stay attributable.
    ServiceState.STOPPED: frozenset(),
}


def is_terminal(state: RunState) -> bool:
    return state in _TERMINAL_RUN_STATES


def can_transition(current: RunState, target: RunState) -> bool:
    """Whether a run may move between two states (§7.4)."""
    return target in _RUN_TRANSITIONS[current]


def is_service_terminal(state: ServiceState) -> bool:
    return state is ServiceState.STOPPED


def can_service_transition(current: ServiceState, target: ServiceState) -> bool:
    """Whether a stream or service may move between two states (§7.4)."""
    return target in _SERVICE_TRANSITIONS[current]


class ObservedResources(FrozenModel):
    """What an attempt actually consumed, for estimator feedback (§15.5)."""

    cpu_seconds: float | None = Field(default=None, ge=0)
    peak_memory_mb: int | None = Field(default=None, ge=0)
    disk_written_mb: int | None = Field(default=None, ge=0)
    egress_bytes: int | None = Field(default=None, ge=0)
    duration_seconds: float | None = Field(default=None, ge=0)


class WorkloadDefinition(FrozenModel):
    """A named composition of operations with declared runtime behavior (§4.9)."""

    name: str
    project_id: ProjectId
    revision_id: RevisionId
    kind: WorkloadKind = WorkloadKind.BATCH
    operations: tuple[Operation, ...] = ()
    depends_on: tuple[str, ...] = ()
    schedule: str | None = None
    max_retries: int = Field(default=3, ge=0)
    priority: int = Field(default=100, ge=0)
    resource_request: ResourceRequest = Field(default_factory=ResourceRequest)
    # Streams and services are long-lived; batch and interactive are not.
    continuous: bool = False


class WorkloadRun(FrozenModel):
    """The logical occurrence of a workload (§4.10).

    Pins exactly one revision (invariant 2) — the definition cannot shift under
    a run that is already in flight.
    """

    run_id: RunId = Field(default_factory=lambda: RunId(new_id("run")))
    project_id: ProjectId
    revision_id: RevisionId
    workload_name: str
    kind: WorkloadKind = WorkloadKind.BATCH
    state: RunState = RunState.REQUESTED
    trigger: TriggerType = TriggerType.MANUAL
    requested_by: PrincipalId
    created_at: datetime = Field(default_factory=utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    attempt_count: int = Field(default=0, ge=0)
    # Deduplicates client retries of the *submission* (§13.4), which is a
    # different concern from retrying execution.
    idempotency_key: str | None = None
    error: str | None = None


class ExecutionAttempt(FrozenModel):
    """One physical attempt to perform a run (§4.10).

    Every field the spec lists is here because post-mortems need them together:
    which worker, under which delegated capability, against which revision,
    with which policy decisions, producing which artifacts.
    """

    attempt_id: AttemptId = Field(default_factory=lambda: AttemptId(new_id("att")))
    run_id: RunId
    project_id: ProjectId
    revision_id: RevisionId
    attempt_number: int = Field(default=1, ge=1)
    state: AttemptState = AttemptState.PENDING
    principal_id: PrincipalId
    capability_token_id: str | None = None
    worker_id: str | None = None
    environment_id: str | None = None
    planned_resources: ResourceRequest = Field(default_factory=ResourceRequest)
    observed_resources: ObservedResources = Field(default_factory=ObservedResources)
    input_artifact_ids: tuple[ArtifactId, ...] = ()
    output_artifact_ids: tuple[ArtifactId, ...] = ()
    policy_decision_ids: tuple[PolicyDecisionId, ...] = ()
    started_at: datetime | None = None
    last_heartbeat_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    error_class: str | None = None
    checkpoint_ref: str | None = None
    trace_id: str | None = None
    # v0.7: commit token for output commit protocol (§14.3)
    commit_token: str | None = None
