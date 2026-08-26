"""Resource and workload query services (§4.6, §4.9).

These carry the bulk of the read traffic — listing datasets, tables, models,
prompts, and the workloads that produce them. Studio alone has roughly 250 read
call sites against this material.

Every read is **revision-scoped**. A resource is not "the users table"; it is
"the users table as declared by revision X". Without that scoping a UI showing a
table's schema cannot say which definition it is showing, and two users looking
at the same page during a publish see different things with no way to tell.

Filtering is typed (``ResourceQuery``) rather than a filter string. An untyped
filter is a SQL-injection surface that cannot be validated at the gateway
boundary, and it pushes query construction out to every caller.
"""

from __future__ import annotations

import json
from typing import Any

from dataenginex.application.services import Service
from dataenginex.foundation import (
    Classification,
    FrozenModel,
    LifecycleState,
    ProjectId,
    Resource,
    ResourceId,
    ResourceQuery,
    ResourceType,
    RevisionId,
    WorkloadKind,
)

__all__ = ["ResourceService", "WorkloadService", "WorkloadSummary"]


class WorkloadSummary(FrozenModel):
    """The client-facing view of a workload definition.

    ``last_run_state`` is folded in because every list view needs it and
    fetching it per row is the N+1 that makes a pipelines page slow.
    """

    workload_id: str
    project_id: ProjectId
    revision_id: RevisionId
    name: str
    kind: WorkloadKind
    continuous: bool = False
    created_at: str = ""
    last_run_state: str | None = None
    last_run_at: str | None = None


class ResourceService(Service):
    """Typed reads over declared resources (§4.6)."""

    def get(self, resource_id: ResourceId) -> Resource:
        row = self.require_row(
            "SELECT * FROM resources WHERE resource_id = ?",
            (resource_id,),
            subject=f"no resource {resource_id}",
        )
        return _row_to_resource(row)

    def get_by_name(self, project_id: ProjectId, name: str) -> Resource:
        """Resources are unique by name within a project.

        Name lookup exists because that is what a manifest and a URL both carry;
        forcing callers to resolve an opaque id first would mean every page load
        does two round trips to answer one question.

        Scoped to the active revision. Every revision redeclares its resources,
        so an unscoped lookup by name matches one row per revision and would
        answer with whichever the database happened to return first.
        """
        revision_id = self.active_revision(project_id)
        row = self.require_row(
            "SELECT * FROM resources WHERE project_id = ? AND revision_id = ? AND name = ?",
            (project_id, revision_id, name),
            subject=f"no resource {name!r} in the active revision of {project_id}",
        )
        return _row_to_resource(row)

    def search(self, query: ResourceQuery, *, limit: int = 200) -> list[Resource]:
        """Typed search (§13.6 ``ResourceRepository.search``).

        Clauses are literal strings chosen by field presence; user input only
        ever reaches the database as a bound parameter.

        Scoped to the active revision when a project is named, for the same
        reason as :meth:`get_by_name`: without it a project with ten revisions
        lists every table ten times.
        """
        clauses = ["1 = 1"]
        params: list[Any] = []

        if query.project_id is not None:
            clauses.append("project_id = ?")
            params.append(query.project_id)
            clauses.append("revision_id = ?")
            params.append(self.active_revision(query.project_id))
        if query.resource_type is not None:
            clauses.append("resource_type = ?")
            params.append(query.resource_type.value)

        params.append(limit)
        rows = self.store.query(
            f"SELECT * FROM resources WHERE {' AND '.join(clauses)} "  # noqa: S608 - literals
            "ORDER BY name LIMIT ?",
            params,
        )
        return [_row_to_resource(row) for row in rows]

    def list_by_type(
        self, project_id: ProjectId, resource_type: ResourceType, *, limit: int = 200
    ) -> list[Resource]:
        """The common case: every table, every model, every prompt."""
        return self.search(
            ResourceQuery(project_id=project_id, resource_type=resource_type), limit=limit
        )


class WorkloadService(Service):
    """Reads over workload definitions and their run history (§4.9)."""

    def list_workloads(self, project_id: ProjectId) -> list[WorkloadSummary]:
        """Workloads of the active revision, each with its latest run state.

        Scoped to the active revision rather than all history: a list showing
        workloads from superseded revisions would offer the user a run button
        for a definition that is no longer current.
        """
        revision_id = self.active_revision(project_id)

        # One correlated subquery beats a second round trip per row. The run
        # table is indexed on (project_id, created_at), so this stays cheap.
        rows = self.store.query(
            "SELECT w.*, "
            "  (SELECT state FROM runs r WHERE r.project_id = w.project_id "
            "     AND r.workload_name = w.name "
            "   ORDER BY r.created_at DESC LIMIT 1) AS last_run_state, "
            "  (SELECT created_at FROM runs r WHERE r.project_id = w.project_id "
            "     AND r.workload_name = w.name "
            "   ORDER BY r.created_at DESC LIMIT 1) AS last_run_at "
            "FROM workload_definitions w "
            "WHERE w.project_id = ? AND w.revision_id = ? ORDER BY w.name",
            (project_id, revision_id),
        )
        return [_row_to_workload(row) for row in rows]

    def get_workload(self, project_id: ProjectId, name: str) -> WorkloadSummary:
        revision_id = self.active_revision(project_id)
        row = self.require_row(
            "SELECT * FROM workload_definitions "
            "WHERE project_id = ? AND revision_id = ? AND name = ?",
            (project_id, revision_id, name),
            subject=f"no workload {name!r} in the active revision of {project_id}",
        )
        return _row_to_workload(row)

    def definition(self, project_id: ProjectId, name: str) -> dict[str, Any]:
        """The compiled definition as declared.

        Returned as plain data rather than a parsed model: this is what the
        manifest said, and a caller rendering it should not have to keep up with
        internal IR types that are explicitly unstable before v1.
        """
        revision_id = self.active_revision(project_id)
        row = self.require_row(
            "SELECT definition_json FROM workload_definitions "
            "WHERE project_id = ? AND revision_id = ? AND name = ?",
            (project_id, revision_id, name),
            subject=f"no workload {name!r} in the active revision of {project_id}",
        )
        parsed: dict[str, Any] = json.loads(row["definition_json"])
        return parsed


def _row_to_resource(row: Any) -> Resource:
    data = dict(row)
    facets = json.loads(data["facets_json"] or "{}")
    return Resource(
        resource_id=ResourceId(data["resource_id"]),
        project_id=ProjectId(data["project_id"]),
        revision_id=RevisionId(data["revision_id"]),
        resource_type=ResourceType(data["resource_type"]),
        name=data["name"],
        description=data["description"],
        labels=json.loads(data["labels_json"] or "{}"),
        owner=data["owner"],
        classification=Classification(data["classification"]),
        lifecycle_state=LifecycleState(data["lifecycle_state"]),
        version=data["version"],
        snapshot_ref=data["snapshot_ref"],
        # Facets are stored as one JSON blob and unpacked by key. A resource
        # that has none is normal — identity does not depend on description.
        data=facets.get("data"),
        model=facets.get("model"),
        prompt=facets.get("prompt"),
        # The declaration's own config, uninterpreted. Dropping it on the way
        # out would leave every reader with a resource it cannot connect to.
        # Stringified because a manifest writes numbers and booleans too, and
        # the field is typed for display rather than for re-parsing.
        provider_facets={k: str(v) for k, v in (facets.get("provider") or {}).items()},
        connector=str(facets.get("connector") or ""),
    )


def _row_to_workload(row: Any) -> WorkloadSummary:
    data = dict(row)
    return WorkloadSummary(
        workload_id=data["workload_id"],
        project_id=ProjectId(data["project_id"]),
        revision_id=RevisionId(data["revision_id"]),
        name=data["name"],
        kind=WorkloadKind(data["kind"]),
        continuous=bool(data["continuous"]),
        created_at=data["created_at"],
        last_run_state=data.get("last_run_state"),
        last_run_at=data.get("last_run_at"),
    )
