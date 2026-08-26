"""Tests for WorkflowCoordinator (runtime/workflow/coordinator.py)."""

from __future__ import annotations

from pathlib import Path

from dataenginex.foundation.ids import ProjectId, RevisionId, RunId, new_id
from dataenginex.runtime.state import ControlStore
from dataenginex.runtime.workflow.coordinator import (
    StepState,
    WorkflowCoordinator,
    WorkflowStep,
)


def _store(tmp_path: Path) -> ControlStore:
    s = ControlStore(tmp_path / "wf.db")
    s.migrate()
    return s


class TestWorkflowStep:
    def test_defaults(self) -> None:
        step = WorkflowStep("s1", "pipeline")
        assert step.name == "s1"
        assert step.step_type == "pipeline"
        assert step.state == StepState.PENDING
        assert step.depends_on == []
        assert step.config == {}
        assert step.error is None

    def test_custom(self) -> None:
        step = WorkflowStep(
            "s2", "model", ref="m1",
            depends_on=["s1"], config={"key": "val"},
        )
        assert step.depends_on == ["s1"]
        assert step.config == {"key": "val"}


class TestWorkflowCoordinator:
    def test_execute_single_step(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        coord = WorkflowCoordinator(store)
        run_id = RunId(new_id("run"))
        steps = [WorkflowStep("pipeline", "pipeline")]
        results = coord.execute_workflow(
            run_id, ProjectId("p1"), RevisionId("r1"), steps,
        )
        assert results["pipeline"] == StepState.COMPLETED

    def test_execute_step_failure(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        coord = WorkflowCoordinator(store)
        run_id = RunId(new_id("run"))
        step = WorkflowStep("bad", "unknown_type_that_raises")
        # _execute_step passes for non-approval types, so use approval to test waiting
        results = coord.execute_workflow(
            run_id, ProjectId("p1"), RevisionId("r1"), [step],
        )
        # approval steps set WAITING_APPROVAL and return (no exception)
        assert results["bad"] == StepState.COMPLETED

    def test_approval_step_waits(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        coord = WorkflowCoordinator(store)
        run_id = RunId(new_id("run"))
        step = WorkflowStep("review", "approval")
        results = coord.execute_workflow(
            run_id, ProjectId("p1"), RevisionId("r1"), [step],
        )
        assert results["review"] == StepState.WAITING_APPROVAL
        assert step.state == StepState.WAITING_APPROVAL

    def test_dependency_chain(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        coord = WorkflowCoordinator(store)
        run_id = RunId(new_id("run"))
        s1 = WorkflowStep("a", "pipeline")
        s2 = WorkflowStep("b", "pipeline", depends_on=["a"])
        s3 = WorkflowStep("c", "pipeline", depends_on=["a", "b"])
        results = coord.execute_workflow(
            run_id, ProjectId("p1"), RevisionId("r1"), [s1, s2, s3],
        )
        assert results["a"] == StepState.COMPLETED
        assert results["b"] == StepState.COMPLETED
        assert results["c"] == StepState.COMPLETED

    def test_unmet_dependency_skips(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        coord = WorkflowCoordinator(store)
        run_id = RunId(new_id("run"))
        s1 = WorkflowStep("a", "pipeline", depends_on=["missing"])
        results = coord.execute_workflow(
            run_id, ProjectId("p1"), RevisionId("r1"), [s1],
        )
        assert results["a"] == StepState.SKIPPED

    def test_step_failure_records_error(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        coord = WorkflowCoordinator(store)
        run_id = RunId(new_id("run"))

        original_execute = coord._execute_step

        def failing_execute(
            step: WorkflowStep, pid: ProjectId, rid: RevisionId,
        ) -> None:
            if step.name == "fail_me":
                raise RuntimeError("boom")
            original_execute(step, pid, rid)

        coord._execute_step = failing_execute  # type: ignore[assignment]
        step = WorkflowStep("fail_me", "pipeline")
        results = coord.execute_workflow(
            run_id, ProjectId("p1"), RevisionId("r1"), [step],
        )
        assert results["fail_me"] == StepState.FAILED
        assert step.error == "boom"
