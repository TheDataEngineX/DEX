"""Workflow coordinator (§7.11).

Cross-domain orchestration: coordinates Spark pipelines with ML, AI,
approvals, and external actions. Records every step and policy transition.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from dataenginex.foundation.ids import ProjectId, RevisionId, RunId
from dataenginex.runtime.state import ControlStore

__all__ = ["WorkflowCoordinator", "StepState", "WorkflowStep"]


class StepState(StrEnum):
    """Lifecycle of a workflow step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING_APPROVAL = "waiting_approval"
    SKIPPED = "skipped"


class WorkflowStep:
    """A single step in a workflow."""

    def __init__(
        self,
        name: str,
        step_type: str,
        ref: str | None = None,
        depends_on: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.step_type = step_type
        self.ref = ref
        self.depends_on = depends_on or []
        self.config = config or {}
        self.state = StepState.PENDING
        self.result: Any = None
        self.error: str | None = None


class WorkflowCoordinator:
    """Multi-step workflow execution (§7.11).

    Coordinates: pipeline → quality → model → AI → approval → action.
    Records every step and policy transition.
    """

    def __init__(self, store: ControlStore) -> None:
        self.store = store

    def execute_workflow(
        self,
        run_id: RunId,
        project_id: ProjectId,
        revision_id: RevisionId,
        steps: list[WorkflowStep],
    ) -> dict[str, StepState]:
        """Execute a workflow's steps in dependency order.

        Returns a mapping of step name to final state.
        """
        results: dict[str, StepState] = {}
        completed: set[str] = set()

        for step in steps:
            # Check dependencies
            deps_met = all(d in completed for d in step.depends_on)
            if not deps_met:
                step.state = StepState.SKIPPED
                results[step.name] = StepState.SKIPPED
                continue

            # Execute the step
            step.state = StepState.RUNNING
            self._record_step_start(run_id, step)

            try:
                self._execute_step(step, project_id, revision_id)
                step.state = StepState.COMPLETED
                completed.add(step.name)
                results[step.name] = StepState.COMPLETED
                self._record_step_complete(run_id, step)
            except Exception as e:
                step.state = StepState.FAILED
                step.error = str(e)
                results[step.name] = StepState.FAILED
                self._record_step_failure(run_id, step)

        return results

    def _execute_step(
        self,
        step: WorkflowStep,
        project_id: ProjectId,
        revision_id: RevisionId,
    ) -> None:
        """Execute a single workflow step based on its type."""
        if step.step_type == "approval":
            # Mark as waiting for approval
            step.state = StepState.WAITING_APPROVAL
            return

        # For other step types, delegate to the appropriate handler
        # The actual execution is handled by the runtime workers
        pass

    def _record_step_start(self, run_id: RunId, step: WorkflowStep) -> None:
        """Record step execution start for audit/lineage."""
        self.store.query_one(
            "INSERT INTO workflow_steps "
            "(run_id, step_name, step_type, state, started_at) "
            "VALUES (?, ?, ?, ?, datetime('now'))",
            (run_id, step.name, step.step_type, step.state.value),
        )

    def _record_step_complete(self, run_id: RunId, step: WorkflowStep) -> None:
        """Record step completion."""
        self.store.query_one(
            "UPDATE workflow_steps SET state = ?, completed_at = datetime('now') "
            "WHERE run_id = ? AND step_name = ?",
            (step.state.value, run_id, step.name),
        )

    def _record_step_failure(self, run_id: RunId, step: WorkflowStep) -> None:
        """Record step failure."""
        self.store.query_one(
            "UPDATE workflow_steps SET state = ?, error = ?, completed_at = datetime('now') "
            "WHERE run_id = ? AND step_name = ?",
            (step.state.value, step.error, run_id, step.name),
        )
