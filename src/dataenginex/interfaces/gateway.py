"""The gateway contract: one entry point for every client (§13.2-13.5).

Studio, the CLI, the SDK, and the HTTP API all speak through this protocol.
That is what stops each of them from growing its own private access to the
control store — the current Studio reaches into ``DexEngine`` directly, which is
why a Studio feature can bypass policy simply by not calling the checking path.

Three structural choices:

**Commands and queries are separated (§13.3).** A command changes state, returns
an identifier, and is audited. A query never changes state and is not. Mixing
them produces "read" endpoints that quietly mutate, which makes both caching and
auditing unsound.

**Commands carry idempotency keys (§13.4).** A client that retries after a
timeout must not create a second run. The key is what makes retry safe, and it
is on the envelope rather than optional per-method because "which methods are
safe to retry?" is a question no caller should have to answer.

**Errors have stable codes (§13.5).** ``E_POLICY_DENIED`` means the same thing
across every transport and every version, so a client can branch on it. Message
text is for humans and may change freely.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import Field

from dataenginex.application import (
    AuditEventView,
    DecisionView,
    PolicyView,
    RevisionSummary,
    ScheduleView,
    WorkloadSummary,
)
from dataenginex.foundation import (
    FrozenModel,
    InteractiveRequest,
    InteractiveResult,
    LineageEdge,
    PrincipalId,
    ProjectId,
    Resource,
    ResourceType,
    RevisionId,
    RunId,
    RunState,
    WorkloadKind,
    new_id,
    utcnow,
)

__all__ = [
    "Command",
    "CommandResult",
    "CursorPage",
    "DexGateway",
    "ErrorCode",
    "GatewayError",
    "ProjectSummary",
    "Query",
    "RunSummary",
]


class ErrorCode(StrEnum):
    """Stable, transport-independent error codes (§13.5).

    Values are contract. Renaming one is a breaking change even though it looks
    like a refactor, because clients branch on these strings.
    """

    NOT_FOUND = "E_NOT_FOUND"
    INVALID_REQUEST = "E_INVALID_REQUEST"
    POLICY_DENIED = "E_POLICY_DENIED"
    APPROVAL_REQUIRED = "E_APPROVAL_REQUIRED"
    REVISION_NOT_PUBLISHED = "E_REVISION_NOT_PUBLISHED"
    VALIDATION_FAILED = "E_VALIDATION_FAILED"
    CONFLICT = "E_CONFLICT"
    CAPACITY_EXHAUSTED = "E_CAPACITY_EXHAUSTED"
    NOT_AUTHORIZED = "E_NOT_AUTHORIZED"
    INTERNAL = "E_INTERNAL"


class GatewayError(Exception):
    """A gateway operation failed with a stable code (§13.5).

    ``details`` carries structured context a client can act on — the pending
    approval id, the failing validation issues — rather than forcing it to parse
    the message.
    """

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Wire representation, identical across transports."""
        return {"code": self.code.value, "message": self.message, "details": self.details}


class Command(FrozenModel):
    """A state-changing request (§13.3).

    ``idempotency_key`` is what makes a retry safe: the control store has a
    partial unique index on ``(project_id, idempotency_key)``, so a replayed
    command returns the original result instead of creating a second run.
    """

    command_id: str = Field(default_factory=lambda: new_id("cmd"))
    principal_id: PrincipalId
    project_id: ProjectId | None = None
    idempotency_key: str | None = None
    correlation_id: str | None = None
    issued_at: datetime = Field(default_factory=utcnow)


class Query(FrozenModel):
    """A read-only request (§13.3).

    Deliberately has no idempotency key: a query changes nothing, so replaying
    it is free by construction rather than by bookkeeping.
    """

    principal_id: PrincipalId
    project_id: ProjectId | None = None
    cursor: str | None = None
    limit: int = Field(default=50, ge=1, le=500)


class CommandResult(FrozenModel):
    """What a command returns.

    ``replayed`` tells a client its retry was recognised rather than duplicated —
    without it, a caller cannot distinguish "created" from "already existed" and
    may double-count.
    """

    command_id: str
    accepted: bool = True
    subject_id: str | None = None
    replayed: bool = False
    message: str = ""


class CursorPage[T](FrozenModel):
    """One page of results with an opaque cursor (§13.8).

    Cursor-based rather than offset-based: offsets skip or repeat rows when the
    underlying set changes between requests, which for a run list is routine.
    """

    items: tuple[T, ...] = ()
    next_cursor: str | None = None
    has_more: bool = False


class RunSummary(FrozenModel):
    """The client-facing view of a run.

    A projection, not the internal record: clients get what they need to display
    and act, without depending on control-plane columns that may change.
    """

    run_id: RunId
    project_id: ProjectId
    revision_id: RevisionId
    workload_name: str
    kind: WorkloadKind
    state: RunState
    attempt_count: int = 0
    created_at: datetime = Field(default_factory=utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.state in (
            RunState.COMPLETED,
            RunState.FAILED,
            RunState.CANCELLED,
            RunState.TIMED_OUT,
        )


class ProjectSummary(FrozenModel):
    """A project and the revision it is currently serving.

    Studio's header, its project switcher, and every "is this published?" badge
    read this one shape. It carries the active revision inline because the
    alternative — a project call followed by a revision call on every page —
    doubles the round trips for a question that is always asked together.
    """

    project_id: ProjectId
    name: str
    workspace_id: str
    active_revision_id: RevisionId | None = None
    content_hash: str | None = None
    created_at: datetime = Field(default_factory=utcnow)

    @property
    def is_published(self) -> bool:
        """Nothing can run against a project without a published revision."""
        return self.active_revision_id is not None


@runtime_checkable
class DexGateway(Protocol):
    """Everything a client may do (§13.2).

    Implemented by ``EmbeddedGateway`` for in-process use (Lite mode, the CLI,
    Studio when co-located) and by ``RemoteGateway`` over HTTP for the server
    profile. A client written against this protocol runs unchanged in both,
    which is the point: Studio must not know whether the runtime is local.
    """

    # --- commands -----------------------------------------------------------

    def open_project(self, command: Command, *, source: str) -> CommandResult:
        """Register the project directory at *source* and publish it (§6.1, §6.3).

        The command a UI issues when the user picks a folder. Unlike
        :meth:`publish_revision` it does not require the project to exist yet —
        registering it is the point — so ``command.project_id`` is ignored and
        the new id comes back as the result's subject.

        A manifest that does not compile still registers the project, with no
        published revision. The user needs somewhere to see why it failed, and
        that somewhere is the project's own page.
        """
        ...

    def publish_revision(self, command: Command, *, source: str) -> CommandResult:
        """Compile and publish a project revision (§6.3)."""
        ...

    def start_run(
        self, command: Command, *, workload: str, revision_id: RevisionId | None = None
    ) -> CommandResult:
        """Request a workload run (§7.4)."""
        ...

    def run_interactive(self, command: Command, *, request: InteractiveRequest) -> CommandResult:
        """Queue a low-latency, user-composed workload (§7.3).

        Returns as soon as the run is accepted, not when it finishes. A gateway
        call that blocked until a worker picked the work up would hold a request
        thread for the length of someone else's queue.
        """
        ...

    def cancel_run(self, command: Command, *, run_id: RunId) -> CommandResult:
        """Request cancellation of an in-flight run (§14.7)."""
        ...

    def decide_approval(
        self, command: Command, *, approval_id: str, granted: bool, reason: str = ""
    ) -> CommandResult:
        """Grant or deny a pending approval (§4.13)."""
        ...

    def rollback_revision(self, command: Command, *, revision_id: RevisionId) -> CommandResult:
        """Re-point a project at an earlier published revision (§6.3)."""
        ...

    def create_schedule(
        self, command: Command, *, workload: str, cron: str, timezone: str = "UTC"
    ) -> CommandResult:
        """Attach a cron schedule to a workload (§7.5)."""
        ...

    def set_schedule_enabled(
        self, command: Command, *, schedule_id: str, enabled: bool
    ) -> CommandResult:
        """Pause or resume a schedule without deleting it (§7.5)."""
        ...

    def delete_schedule(self, command: Command, *, schedule_id: str) -> CommandResult:
        """Remove a schedule."""
        ...

    def tick_schedules(self, command: Command, *, limit: int = 50) -> CommandResult:
        """Fire every schedule that is due, returning how many ran (§7.5).

        A command rather than a query because it changes state: firing advances
        ``next_fire_at`` and creates runs. The control-plane daemon calls this on
        a loop; exposing it here is what lets ``dex runtime serve`` drive cron
        without a second private path into the store.
        """
        ...

    # --- queries ------------------------------------------------------------

    def get_project(self, query: Query) -> ProjectSummary:
        """The project and the revision it is serving."""
        ...

    def list_projects(self, query: Query) -> CursorPage[ProjectSummary]:
        """Projects in the caller's workspace."""
        ...

    def get_revision(
        self, query: Query, *, revision_id: RevisionId | None = None
    ) -> RevisionSummary:
        """One revision; the active one when ``revision_id`` is omitted (§6.3)."""
        ...

    def list_revisions(self, query: Query) -> CursorPage[RevisionSummary]:
        """Revision history, newest first — the rollback menu (§6.3)."""
        ...

    def list_resources(
        self, query: Query, *, resource_type: ResourceType | None = None
    ) -> CursorPage[Resource]:
        """Declared resources of the active revision (§4.6).

        Typed by ``ResourceType`` rather than a filter string: an untyped filter
        cannot be validated at this boundary, and every caller would rebuild the
        same query fragment.
        """
        ...

    def get_resource(self, query: Query, *, name: str) -> Resource:
        """One resource by name, scoped to the project's active revision."""
        ...

    def list_workloads(self, query: Query) -> CursorPage[WorkloadSummary]:
        """Workloads of the active revision, each with its latest run state."""
        ...

    def get_workload(self, query: Query, *, name: str) -> WorkloadSummary:
        """One workload definition from the active revision (§4.9)."""
        ...

    def get_workload_definition(self, query: Query, *, name: str) -> dict[str, Any]:
        """The compiled definition as declared.

        Plain data on purpose: this is what the manifest said, and a client
        rendering it should not pin against IR types that are explicitly
        unstable before v1.
        """
        ...

    def list_schedules(self, query: Query) -> CursorPage[ScheduleView]:
        """Cron schedules declared for this project (§7.5)."""
        ...

    def get_interactive_result(
        self, query: Query, *, run_id: RunId
    ) -> InteractiveResult | None:
        """What an interactive run produced, or ``None`` if not yet or no longer.

        A query, not a wait: the caller polls, or watches the run's state. A
        blocking read here would put the client's patience in the gateway,
        where it cannot be cancelled.
        """
        ...

    def get_run(self, query: Query, *, run_id: RunId) -> RunSummary:
        """One run's current state."""
        ...

    def list_runs(
        self, query: Query, *, state: RunState | None = None, workload: str | None = None
    ) -> CursorPage[RunSummary]:
        """Runs for a project, newest first, optionally for one workload."""
        ...

    def list_lineage(self, query: Query, *, node: str | None = None) -> CursorPage[LineageEdge]:
        """Provenance edges for this project, oldest first (§8.5).

        *node* narrows to one resource or artifact and returns edges in both
        directions, which is the question a lineage page asks: what did this
        come from, and what came from it.
        """
        ...

    def list_approvals(self, query: Query) -> CursorPage[dict[str, Any]]:
        """Approvals awaiting a decision."""
        ...

    def list_policies(self, query: Query) -> CursorPage[PolicyView]:
        """Policies in force for this project, highest priority first (§9.3).

        The live policy set the engine evaluates — not manifest configuration.
        A rule returned here is a rule that runs.
        """
        ...

    def list_decisions(
        self, query: Query, *, denied_only: bool = False
    ) -> CursorPage[DecisionView]:
        """Recorded authorization decisions, newest first (§4.12)."""
        ...

    def list_audit_events(
        self, query: Query, *, action: str | None = None
    ) -> CursorPage[AuditEventView]:
        """The audit trail, newest first (§4.15)."""
        ...
