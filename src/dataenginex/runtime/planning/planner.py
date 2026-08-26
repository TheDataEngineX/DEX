"""Turn a claimed attempt into something a backend can run (§7.8, §6.8).

The missing link between the control plane and execution. A worker claims an
attempt and knows three things: the run, the revision, and the attempt id. A
backend needs the resolved operations, a resource request, and a capability
token scoped to exactly this attempt. This module converts one into the other.

Two properties matter more than the mechanics.

**The plan comes from the published revision, not from live config.** Every run
pins a revision (§17 Phase 1). Reading the workload from
``workload_definitions`` — written when the revision was compiled — means an
edit to ``dex.yaml`` after a run was queued cannot change what that run does.

**The capability token is minted per attempt and expires.** §9.4 wants a grant
bound to the exact project, revision, run, and operation scope, valid for the
attempt's deadline and no longer. A long-lived or broadly-scoped token would
make the envelope decorative — a worker could do more than its attempt
justified, which is the escalation invariant 8 exists to prevent.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import timedelta
from typing import Any

from dataenginex.foundation import (
    AttemptId,
    CapabilityToken,
    ExecutionContext,
    ExecutionPlan,
    InteractiveRequest,
    Operation,
    PrincipalId,
    ProjectId,
    ResourceRequest,
    RevisionId,
    RunId,
    utcnow,
)
from dataenginex.runtime.state import ControlStore

__all__ = ["PlanningError", "build_context", "build_plan", "plan_attempt"]


class PlanningError(RuntimeError):
    """The attempt could not be turned into a runnable plan.

    Distinct from an operation failing: this means execution never legitimately
    started, so the attempt is recorded as failed rather than retried forever
    against a revision that cannot produce a plan.
    """


def plan_attempt(
    store: ControlStore, attempt_id: AttemptId
) -> tuple[ExecutionPlan, ExecutionContext]:
    """Resolve everything a backend needs to run one claimed attempt."""
    row = store.query_one(
        "SELECT a.attempt_id, a.run_id, a.project_id, a.revision_id, a.principal_id, "
        "r.workload_name, r.kind, r.ad_hoc_plan_json "
        "FROM attempts a JOIN runs r ON r.run_id = a.run_id "
        "WHERE a.attempt_id = ?",
        (attempt_id,),
    )
    if row is None:
        raise PlanningError(f"attempt {attempt_id} does not exist")

    project_id = ProjectId(row["project_id"])
    revision_id = RevisionId(row["revision_id"])
    workload_name = str(row["workload_name"])

    compiled = _compiled_workload(store, row, revision_id, workload_name)
    plan = build_plan(
        attempt_id=AttemptId(row["attempt_id"]),
        project_id=project_id,
        revision_id=revision_id,
        compiled=compiled,
        resources=_resources_for(store, project_id, revision_id),
    )
    context = build_context(
        attempt_id=plan.attempt_id,
        principal_id=PrincipalId(row["principal_id"]),
        project_id=project_id,
        revision_id=revision_id,
        run_id=RunId(row["run_id"]),
        plan=plan,
    )
    return plan, context


def _compiled_workload(
    store: ControlStore,
    row: Any,
    revision_id: RevisionId,
    workload_name: str,
) -> dict[str, Any]:
    """The operations this attempt should run, from wherever they were declared.

    Two sources, checked in this order:

    **The run's own ad-hoc plan**, for interactive work the user composed a
    moment ago (§7.3). It has no entry in ``workload_definitions`` and never
    should — that table is the revision's declared workload set, and a SQL
    preview is not one of them.

    **The compiled revision**, for everything else.

    The revision is pinned either way, so an interactive run is no less scoped
    than a batch one: it can only reach resources the revision declared.
    """
    # ``.keys()`` is required, not stylistic: ``in`` on a ``sqlite3.Row``
    # iterates the row's *values*, so the membership test would silently be
    # False and every interactive plan would fall through to the workload
    # lookup it does not have. Ruff's SIM118 fix is wrong for this type.
    ad_hoc = row["ad_hoc_plan_json"] if "ad_hoc_plan_json" in row.keys() else None  # noqa: SIM118
    if ad_hoc:
        try:
            request = InteractiveRequest.model_validate_json(ad_hoc)
        except Exception as exc:  # noqa: BLE001 — surfaced as a planning failure
            raise PlanningError(f"interactive request is unreadable: {exc}") from exc
        return {
            "name": request.label,
            "operations": [op.model_dump(mode="json") for op in request.operations],
            "resource_request": request.resource_request().model_dump(mode="json"),
            # Carried on the plan so the handler enforces the same cap the
            # request declared. A ceiling the executor cannot see is a ceiling
            # that only documents an intention.
            "parameters": {"max_rows": str(request.max_rows)},
        }

    definition = store.query_one(
        "SELECT definition_json FROM workload_definitions WHERE revision_id = ? AND name = ?",
        (revision_id, workload_name),
    )
    if definition is None:
        # The revision genuinely has no such workload. Running *something* here
        # — a default, an empty plan — would commit a result for work nobody
        # declared, so refusing is the only honest option.
        raise PlanningError(
            f"revision {revision_id} declares no workload {workload_name!r}; "
            "it was renamed or removed after this run was queued"
        )
    parsed: dict[str, Any] = json.loads(definition["definition_json"])
    return parsed


def build_plan(
    *,
    attempt_id: AttemptId,
    project_id: ProjectId,
    revision_id: RevisionId,
    compiled: dict[str, Any],
    resources: Mapping[str, Mapping[str, str]] | None = None,
) -> ExecutionPlan:
    """Rebuild the compiled workload's operations into a runnable plan.

    Operations are validated back into :class:`Operation` rather than passed as
    dicts: the row was written by a previous version of the compiler, and a
    field that no longer parses should fail here — loudly, before execution —
    instead of surfacing as an attribute error inside a handler.

    *resources* resolves the names operations bind to. Only the ones this
    workload actually names are carried: a plan holding every resource in the
    project would hand a handler connection settings for data its operations
    never declared they touch.
    """
    raw_operations = compiled.get("operations") or []
    if not raw_operations:
        raise PlanningError(
            f"workload {compiled.get('name')!r} declares no operations; nothing to execute"
        )

    try:
        operations = tuple(Operation.model_validate(item) for item in raw_operations)
    except Exception as exc:  # noqa: BLE001 — surfaced as a planning failure
        raise PlanningError(f"compiled operations are unreadable: {exc}") from exc

    request = ResourceRequest.model_validate(compiled.get("resource_request") or {})
    return ExecutionPlan(
        attempt_id=attempt_id,
        project_id=project_id,
        revision_id=revision_id,
        operations=operations,
        resource_request=request,
        inputs=_bound_resources(operations, resources or {}),
        # The workload's own name rides along so what runs can say what it was.
        # Lineage records it per edge, and recovering it afterwards would mean
        # joining back through the run — the walk the derivation edges exist to
        # avoid. Written last so a workload's parameters cannot shadow it.
        parameters={
            **(compiled.get("parameters") or {}),
            "workload": str(compiled.get("name", "")),
        },
    )


def _bound_resources(
    operations: tuple[Operation, ...], resources: Mapping[str, Mapping[str, str]]
) -> dict[str, str]:
    """Serialised config for every resource the operations name.

    A missing name is left out rather than defaulted. The handler will refuse
    for want of a resource, which names the real problem — a workload bound to
    something the revision does not declare — instead of quietly reading from
    somewhere nobody asked for.
    """
    named = {
        name
        for operation in operations
        for name in (*operation.bound_inputs, *operation.bound_outputs)
    }
    return {name: json.dumps(dict(resources[name])) for name in sorted(named) if name in resources}


def _resources_for(
    store: ControlStore, project_id: ProjectId, revision_id: RevisionId
) -> dict[str, dict[str, str]]:
    """Every resource the revision declared, keyed by name.

    Read from the revision the run pinned, never the active one: a publish
    between queueing and execution must not change where a run reads from.
    """
    rows = store.query(
        "SELECT name, facets_json FROM resources WHERE project_id = ? AND revision_id = ?",
        (project_id, revision_id),
    )
    resolved: dict[str, dict[str, str]] = {}
    for row in rows:
        facets = json.loads(row["facets_json"] or "{}")
        config = dict(facets.get("provider") or {})
        # The connector kind travels with the config so a handler can open one
        # without a second lookup it has no scoped way to make. Read from the
        # facet rather than ``resource_type``: that column holds the catalogue
        # classification, and using it here is what put ``csv`` in it.
        config["type"] = str(facets.get("connector") or "")
        resolved[row["name"]] = config
    return resolved


def build_context(
    *,
    attempt_id: AttemptId,
    principal_id: PrincipalId,
    project_id: ProjectId,
    revision_id: RevisionId,
    run_id: RunId,
    plan: ExecutionPlan,
) -> ExecutionContext:
    """Mint the narrow envelope this attempt runs inside (§7.8, §9.4).

    The token's lifetime is the workload's own timeout, not a fixed window: a
    grant that outlives the work it authorized is a grant available for
    something else.
    """
    deadline = utcnow() + timedelta(seconds=plan.resource_request.timeout_seconds)

    # Actions derive from what the operations declare they need, so a plan of
    # pure reads cannot receive a token that permits writing.
    actions = sorted({cap for op in plan.operations for cap in op.required_capabilities})

    return ExecutionContext(
        attempt_id=attempt_id,
        capability=CapabilityToken(
            principal_id=principal_id,
            project_id=project_id,
            revision_id=revision_id,
            run_id=run_id,
            actions=tuple(actions),
            # Scoped to this project's resources. A bare "*" would let a token
            # minted for one project touch another's data (invariant 6).
            resource_scope=(f"{project_id}/*",),
            expires_at=deadline,
        ),
        # Per-attempt namespaces, so a retry cannot see or overwrite a previous
        # attempt's partial output before the commit protocol decides (§14.3).
        artifact_namespace=f"{project_id}/{run_id}/{attempt_id}",
        checkpoint_namespace=f"{project_id}/{run_id}",
        deadline=deadline,
        correlation_id=str(run_id),
    )
