"""In-process gateway implementation (§13.2, §5.4).

A transport, not a place logic lives. Every method here does three things:
translate the request, delegate to an application service, and map failures onto
the stable error codes clients branch on (§13.5).

That division is the point. This class previously wrote SQL against the control
store, which worked — and meant the HTTP API and the CLI would each have had to
re-implement the same ordering rules, with one of them eventually getting the
authorize-then-create sequence subtly wrong. §5.4 is explicit: *"EmbeddedGateway
invokes local application services."*

The remote gateway speaks the same protocol over HTTP against the same services,
so a client written against ``DexGateway`` cannot tell which one it holds.
"""

from __future__ import annotations

from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from dataenginex.application import (
    ApplicationError,
    AuditEventView,
    DecisionView,
    GovernanceQueryService,
    NotFoundError,
    PolicyDenied,
    PolicyView,
    ProjectService,
    PublishRejected,
    ResourceService,
    RevisionSummary,
    RunService,
    ScheduleService,
    ScheduleView,
    WorkloadService,
    WorkloadSummary,
)
from dataenginex.domains.governance.lineage import LineageService
from dataenginex.domains.security import GovernanceService
from dataenginex.domains.security.governance import ApprovalRequired, GovernanceError
from dataenginex.foundation import (
    InteractiveRequest,
    InteractiveResult,
    LineageEdge,
    ProjectId,
    Resource,
    ResourceQuery,
    ResourceType,
    RevisionId,
    RunId,
    RunState,
)
from dataenginex.interfaces.gateway import (
    Command,
    CommandResult,
    CursorPage,
    ErrorCode,
    GatewayError,
    ProjectSummary,
    Query,
    RunSummary,
)
from dataenginex.runtime.queue import DurableQueue
from dataenginex.runtime.state import ControlStore

__all__ = ["EmbeddedGateway"]


class EmbeddedGateway:
    """Runs commands and queries in the calling process (§13.2).

    Lite mode, the CLI, and a co-located Studio all use this.
    """

    def __init__(
        self,
        store: ControlStore,
        *,
        governance: GovernanceService | None = None,
        queue: DurableQueue | None = None,
    ) -> None:
        self.store = store
        self.governance = governance or GovernanceService(store)
        self.queue = queue or DurableQueue(store)
        self.projects = ProjectService(store)
        self.runs = RunService(store, governance=self.governance, queue=self.queue)
        self.resources = ResourceService(store)
        self.workloads = WorkloadService(store)
        self.lineage = LineageService(store)
        self.governance_queries = GovernanceQueryService(store)
        self.schedules = ScheduleService(store, runs=self.runs)

    # --- commands -----------------------------------------------------------

    def open_project(self, command: Command, *, source: str) -> CommandResult:
        """Register a project directory and publish it (§6.1, §6.3).

        Registration and publication are one command because they are one user
        action: picking a folder. Splitting them would leave a caller able to
        register a project and forget to publish it, which is a project with no
        resources and no workloads — indistinguishable from an empty one.
        """
        root = Path(source)
        root = root.parent if root.is_file() else root

        project_id = self.projects.ensure_project(_manifest_name(root))
        try:
            revision = self.projects.publish(
                project_id, root, principal_id=command.principal_id
            )
        except PublishRejected as exc:
            # Registered, unpublished, and said so. Raising would leave the UI
            # with an error and no project to attach it to.
            return CommandResult(
                command_id=command.command_id,
                subject_id=project_id,
                message=f"opened {root.name} without a published revision: {exc}",
            )

        return CommandResult(
            command_id=command.command_id,
            subject_id=project_id,
            message=f"opened {root.name} at revision {revision.content_hash}",
        )

    def publish_revision(self, command: Command, *, source: str) -> CommandResult:
        """Compile and publish a project revision (§6.3).

        A rejected draft comes back as ``E_VALIDATION_FAILED`` carrying every
        issue, so a UI can point at the offending manifest location instead of
        showing the user a sentence.
        """
        project_id = self._require_project(command)
        # A manifest path is accepted as well as a directory, because that is
        # what a caller has in hand — ``open_project`` already normalises the
        # same way, and requiring the two to differ is a trap.
        root = Path(source)
        root = root.parent if root.is_file() else root
        try:
            revision = self.projects.publish(
                project_id, root, principal_id=command.principal_id
            )
        except PublishRejected as exc:
            raise GatewayError(
                ErrorCode.VALIDATION_FAILED,
                str(exc),
                details={"issues": [i.model_dump(mode="json") for i in exc.report.issues]},
            ) from exc
        except NotFoundError as exc:
            raise GatewayError(ErrorCode.NOT_FOUND, str(exc)) from exc

        return CommandResult(
            command_id=command.command_id,
            subject_id=revision.revision_id,
            message=f"published revision {revision.content_hash}",
        )

    def start_run(
        self,
        command: Command,
        *,
        workload: str,
        revision_id: RevisionId | None = None,
    ) -> CommandResult:
        """Authorize and enqueue a workload run (§7.4, §13.4)."""
        project_id = self._require_project(command)
        try:
            accepted = self.runs.request_run(
                project_id,
                workload,
                principal_id=command.principal_id,
                revision_id=revision_id,
                idempotency_key=command.idempotency_key,
            )
        except ApprovalRequired as exc:
            # The approval id is the actionable part. Flattening this into a
            # generic denial turns a solvable state into a dead end.
            raise GatewayError(
                ErrorCode.APPROVAL_REQUIRED,
                exc.approval.action_summary,
                details={"approval_id": exc.approval.approval_id},
            ) from exc
        except PolicyDenied as exc:
            raise GatewayError(
                ErrorCode.POLICY_DENIED,
                str(exc),
                details={"decision_id": exc.decision.decision_id},
            ) from exc
        except NotFoundError as exc:
            # No published revision is its own code: the fix is "publish", not
            # "check your project id".
            code = (
                ErrorCode.REVISION_NOT_PUBLISHED
                if "published revision" in str(exc)
                else ErrorCode.NOT_FOUND
            )
            raise GatewayError(code, str(exc)) from exc

        return CommandResult(
            command_id=command.command_id,
            subject_id=accepted.run_id,
            replayed=accepted.replayed,
            message=(
                "returned the existing run for this idempotency key"
                if accepted.replayed
                else f"queued run for workload {workload!r}"
            ),
        )

    def run_interactive(self, command: Command, *, request: InteractiveRequest) -> CommandResult:
        """Queue a low-latency, user-composed workload (§7.3)."""
        project_id = self._require_project(command)
        try:
            accepted = self.runs.request_interactive(
                project_id, request, principal_id=command.principal_id
            )
        except PolicyDenied as exc:
            raise GatewayError(
                ErrorCode.POLICY_DENIED,
                str(exc),
                details={"decision_id": exc.decision.decision_id},
            ) from exc
        except NotFoundError as exc:
            code = (
                ErrorCode.REVISION_NOT_PUBLISHED
                if "published" in str(exc)
                else ErrorCode.NOT_FOUND
            )
            raise GatewayError(code, str(exc)) from exc

        return CommandResult(
            command_id=command.command_id,
            subject_id=accepted.run_id,
            message=f"{request.label} queued",
        )

    def get_interactive_result(
        self, query: Query, *, run_id: RunId
    ) -> InteractiveResult | None:
        """What an interactive run produced, or ``None`` if not yet or no longer."""
        result = self.runs.interactive_result(run_id)
        # Scoped like every other read. Without this check a run id from
        # another project would return that project's data to whoever guessed
        # the id — the cross-project leak invariant 6 forbids.
        if result is not None and result.project_id != query.project_id:
            return None
        return result

    def cancel_run(self, command: Command, *, run_id: RunId) -> CommandResult:
        """Request cancellation of an in-flight run (§14.7)."""
        try:
            self.runs.cancel_run(run_id)
        except NotFoundError as exc:
            raise GatewayError(ErrorCode.NOT_FOUND, str(exc)) from exc
        except ApplicationError as exc:
            raise GatewayError(ErrorCode.CONFLICT, str(exc)) from exc

        return CommandResult(
            command_id=command.command_id,
            subject_id=run_id,
            message="cancellation requested",
        )

    def decide_approval(
        self, command: Command, *, approval_id: str, granted: bool, reason: str = ""
    ) -> CommandResult:
        try:
            approval = self.governance.decide_approval(
                approval_id,
                approver_id=command.principal_id,
                granted=granted,
                reason=reason,
            )
        except GovernanceError as exc:
            raise GatewayError(ErrorCode.CONFLICT, str(exc)) from exc

        return CommandResult(
            command_id=command.command_id,
            subject_id=approval_id,
            message=f"approval {approval.state.value}",
        )

    def rollback_revision(self, command: Command, *, revision_id: RevisionId) -> CommandResult:
        """Re-point a project at an earlier published revision (§6.3).

        A command rather than a query even though it writes only one column:
        it changes what every subsequent run executes, which is exactly the kind
        of change that has to be audited.
        """
        project_id = self._require_project(command)
        try:
            revision = self.projects.rollback(project_id, revision_id)
        except NotFoundError as exc:
            raise GatewayError(ErrorCode.NOT_FOUND, str(exc)) from exc
        except ApplicationError as exc:
            # Rolling onto an unpublished draft is refused, not a missing row.
            raise GatewayError(ErrorCode.CONFLICT, str(exc)) from exc

        return CommandResult(
            command_id=command.command_id,
            subject_id=revision.revision_id,
            message=f"activated revision {revision.content_hash}",
        )

    # --- schedules (§7.5) ---------------------------------------------------

    def create_schedule(
        self, command: Command, *, workload: str, cron: str, timezone: str = "UTC"
    ) -> CommandResult:
        """Attach a cron schedule to a workload of the published revision."""
        project_id = self._require_project(command)
        try:
            schedule = self.schedules.create(
                project_id, workload, cron=cron, timezone=timezone
            )
        except NotFoundError as exc:
            raise GatewayError(ErrorCode.NOT_FOUND, str(exc)) from exc
        except ApplicationError as exc:
            # A bad cron expression is the caller's input, not a missing subject.
            raise GatewayError(ErrorCode.INVALID_REQUEST, str(exc)) from exc
        return CommandResult(
            command_id=command.command_id,
            subject_id=schedule.schedule_id,
            message=f"scheduled {workload!r} at {cron!r} ({timezone})",
        )

    def set_schedule_enabled(
        self, command: Command, *, schedule_id: str, enabled: bool
    ) -> CommandResult:
        try:
            self.schedules.set_enabled(schedule_id, enabled=enabled)
        except NotFoundError as exc:
            raise GatewayError(ErrorCode.NOT_FOUND, str(exc)) from exc
        return CommandResult(
            command_id=command.command_id,
            subject_id=schedule_id,
            message="enabled" if enabled else "paused",
        )

    def delete_schedule(self, command: Command, *, schedule_id: str) -> CommandResult:
        self.schedules.delete(schedule_id)
        return CommandResult(
            command_id=command.command_id, subject_id=schedule_id, message="deleted"
        )

    def tick_schedules(self, command: Command, *, limit: int = 50) -> CommandResult:
        """Fire whatever is due. Driven by the control-plane daemon (§7.5).

        Reports the count rather than the runs: a tick that fires forty
        schedules should not return forty payloads to a loop that discards them.
        Failures are counted too — a schedule whose run was denied still fired,
        and hiding that would make a broken workload look like an idle one.
        """
        fired = self.schedules.tick(limit=limit)
        failed = sum(1 for item in fired if item.error)
        return CommandResult(
            command_id=command.command_id,
            message=f"fired {len(fired)} schedule(s), {failed} refused",
        )

    # --- queries ------------------------------------------------------------

    def list_schedules(self, query: Query) -> CursorPage[ScheduleView]:
        project_id = self._require_project_query(query)
        return CursorPage(items=tuple(self.schedules.list_for_project(project_id)))

    def get_run(self, query: Query, *, run_id: RunId) -> RunSummary:
        try:
            return _to_summary(self.runs.get_run(run_id))
        except NotFoundError as exc:
            raise GatewayError(ErrorCode.NOT_FOUND, str(exc)) from exc

    def list_runs(
        self, query: Query, *, state: RunState | None = None, workload: str | None = None
    ) -> CursorPage[RunSummary]:
        items, next_cursor = self.runs.list_runs(
            query.project_id,
            state=state,
            workload=workload,
            cursor=query.cursor,
            limit=query.limit,
        )
        return CursorPage(
            items=tuple(_to_summary(run) for run in items),
            next_cursor=next_cursor,
            has_more=next_cursor is not None,
        )

    def list_lineage(self, query: Query, *, node: str | None = None) -> CursorPage[LineageEdge]:
        """Provenance edges for this project (§8.5).

        Scoped inside the query rather than filtered afterwards: node ids are
        plain names like ``orders``, so an unscoped read would match another
        project's table of the same name (invariant 6).
        """
        project_id = self._require_project_query(query)
        if node:
            edges = self.lineage.edges_for(node, project_id=project_id)
        else:
            edges = self.lineage.project_edges(project_id, limit=query.limit + 1)
        return CursorPage(
            items=tuple(edges[: query.limit]),
            has_more=len(edges) > query.limit,
        )

    def list_approvals(self, query: Query) -> CursorPage[dict[str, Any]]:
        approvals = self.governance.pending_approvals(query.project_id)
        items = tuple(
            {
                "approval_id": a.approval_id,
                "project_id": a.project_id,
                "action_summary": a.action_summary,
                "risk_level": int(a.risk_level),
                "requested_at": a.requested_at.isoformat(),
                "expires_at": a.expires_at.isoformat() if a.expires_at else None,
            }
            for a in approvals[: query.limit]
        )
        return CursorPage(items=items, has_more=len(approvals) > query.limit)

    def list_policies(self, query: Query) -> CursorPage[PolicyView]:
        policies = self.governance_queries.list_policies(query.project_id)
        return CursorPage(
            items=tuple(policies[: query.limit]),
            has_more=len(policies) > query.limit,
        )

    def list_decisions(
        self, query: Query, *, denied_only: bool = False
    ) -> CursorPage[DecisionView]:
        read = (
            self.governance_queries.list_denials
            if denied_only
            else self.governance_queries.list_decisions
        )
        decisions = read(query.project_id, limit=query.limit + 1)
        return CursorPage(
            items=tuple(decisions[: query.limit]),
            has_more=len(decisions) > query.limit,
        )

    def list_audit_events(
        self, query: Query, *, action: str | None = None
    ) -> CursorPage[AuditEventView]:
        events = self.governance_queries.list_audit_events(
            query.project_id, action=action, limit=query.limit + 1
        )
        return CursorPage(
            items=tuple(events[: query.limit]),
            has_more=len(events) > query.limit,
        )

    def get_project(self, query: Query) -> ProjectSummary:
        try:
            view = self.projects.get_project(self._require_project_query(query))
        except NotFoundError as exc:
            raise GatewayError(ErrorCode.NOT_FOUND, str(exc)) from exc
        return _to_project(view)

    def list_projects(self, query: Query) -> CursorPage[ProjectSummary]:
        """Every project, newest first.

        Shaped as a page even though Lite returns them all: the server profile
        adds a real cursor without any client changing.
        """
        views = self.projects.list_projects(limit=query.limit + 1)
        return CursorPage(
            items=tuple(_to_project(v) for v in views[: query.limit]),
            has_more=len(views) > query.limit,
        )

    def get_revision(
        self, query: Query, *, revision_id: RevisionId | None = None
    ) -> RevisionSummary:
        try:
            if revision_id is not None:
                return self.projects.get_revision(revision_id)
            return self.projects.active_revision_summary(self._require_project_query(query))
        except NotFoundError as exc:
            # "No active revision" is a publish problem, not a bad id.
            code = (
                ErrorCode.REVISION_NOT_PUBLISHED
                if "active revision" in str(exc)
                else ErrorCode.NOT_FOUND
            )
            raise GatewayError(code, str(exc)) from exc

    def list_revisions(self, query: Query) -> CursorPage[RevisionSummary]:
        revisions = self.projects.list_revisions(
            self._require_project_query(query), limit=query.limit + 1
        )
        has_more = len(revisions) > query.limit
        return CursorPage(items=tuple(revisions[: query.limit]), has_more=has_more)

    def list_resources(
        self, query: Query, *, resource_type: ResourceType | None = None
    ) -> CursorPage[Resource]:
        project_id = self._require_project_query(query)
        resources = self.resources.search(
            ResourceQuery(project_id=project_id, resource_type=resource_type),
            limit=query.limit + 1,
        )
        has_more = len(resources) > query.limit
        return CursorPage(items=tuple(resources[: query.limit]), has_more=has_more)

    def get_resource(self, query: Query, *, name: str) -> Resource:
        try:
            return self.resources.get_by_name(self._require_project_query(query), name)
        except NotFoundError as exc:
            raise GatewayError(ErrorCode.NOT_FOUND, str(exc)) from exc

    def list_workloads(self, query: Query) -> CursorPage[WorkloadSummary]:
        try:
            workloads = self.workloads.list_workloads(self._require_project_query(query))
        except NotFoundError as exc:
            raise GatewayError(ErrorCode.REVISION_NOT_PUBLISHED, str(exc)) from exc
        return CursorPage(
            items=tuple(workloads[: query.limit]),
            has_more=len(workloads) > query.limit,
        )

    def get_workload(self, query: Query, *, name: str) -> WorkloadSummary:
        try:
            return self.workloads.get_workload(self._require_project_query(query), name)
        except NotFoundError as exc:
            raise GatewayError(ErrorCode.NOT_FOUND, str(exc)) from exc

    def get_workload_definition(self, query: Query, *, name: str) -> dict[str, Any]:
        try:
            return self.workloads.definition(self._require_project_query(query), name)
        except NotFoundError as exc:
            raise GatewayError(ErrorCode.NOT_FOUND, str(exc)) from exc

    # --- internals ----------------------------------------------------------

    def _require_project(self, command: Command) -> ProjectId:
        if command.project_id is None:
            raise GatewayError(ErrorCode.INVALID_REQUEST, "this command requires a project")
        return command.project_id

    def _require_project_query(self, query: Query) -> ProjectId:
        """Same rule for reads.

        A project-scoped query with no project is a client bug, and answering it
        across every project would leak one workspace's resources into another's
        page.
        """
        if query.project_id is None:
            raise GatewayError(ErrorCode.INVALID_REQUEST, "this query requires a project")
        return query.project_id


def _manifest_name(root: Path) -> str:
    """The manifest's declared name, falling back to the directory's.

    Read directly rather than compiled, because a manifest that fails to compile
    still has to resolve to *some* project — and the directory name is what the
    user calls it either way. Naming by manifest is what makes re-opening the
    same folder find the same project instead of forking one per path spelling.
    """
    with suppress(Exception):
        loaded = yaml.safe_load((root / "dex.yaml").read_text())
        declared = (loaded or {}).get("metadata", {}).get("name")
        if declared:
            return str(declared)
    return root.name


def _to_project(view: Any) -> ProjectSummary:
    """Project the application view onto the wire type.

    ``created_at`` crosses the application boundary as an ISO string because
    that is how SQLite stores it; the published contract uses a real datetime so
    clients are not each writing their own parser.
    """
    return ProjectSummary(
        project_id=view.project_id,
        name=view.name,
        workspace_id=view.workspace_id,
        active_revision_id=view.active_revision_id,
        content_hash=view.content_hash,
        created_at=datetime.fromisoformat(view.created_at),
    )


def _to_summary(run: Any) -> RunSummary:
    """Project the application view onto the wire type.

    Two types rather than one because they answer to different masters: the
    application view follows the domain, the summary is a published contract
    that clients pin against.
    """
    return RunSummary(
        run_id=run.run_id,
        project_id=run.project_id,
        revision_id=run.revision_id,
        workload_name=run.workload_name,
        kind=run.kind,
        state=run.state,
        attempt_count=run.attempt_count,
        created_at=run.created_at,
        started_at=run.started_at,
        completed_at=run.completed_at,
        error=run.error,
    )
