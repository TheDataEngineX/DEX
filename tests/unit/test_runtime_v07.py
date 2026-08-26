"""Tests for v0.7 Runtime modules: workflow step states."""


from dataenginex.runtime.workflow.coordinator import StepState, WorkflowStep


class TestStepState:
    def test_step_states_are_distinct(self) -> None:
        assert StepState.PENDING != StepState.RUNNING
        assert StepState.RUNNING != StepState.COMPLETED
        assert StepState.COMPLETED != StepState.FAILED
        assert StepState.FAILED != StepState.WAITING_APPROVAL
        assert StepState.WAITING_APPROVAL != StepState.SKIPPED

    def test_step_state_values(self) -> None:
        assert StepState.PENDING.value == "pending"
        assert StepState.RUNNING.value == "running"
        assert StepState.COMPLETED.value == "completed"
        assert StepState.FAILED.value == "failed"
        assert StepState.WAITING_APPROVAL.value == "waiting_approval"
        assert StepState.SKIPPED.value == "skipped"


class TestWorkflowStep:
    def test_create_step(self) -> None:
        step = WorkflowStep(
            name="extract",
            step_type="pipeline",
            ref="pipeline:123",
        )
        assert step.name == "extract"
        assert step.step_type == "pipeline"
        assert step.state == StepState.PENDING

    def test_step_with_dependencies(self) -> None:
        step = WorkflowStep(
            name="load",
            step_type="pipeline",
            depends_on=["extract", "transform"],
        )
        assert "extract" in step.depends_on
        assert "transform" in step.depends_on

    def test_step_with_config(self) -> None:
        step = WorkflowStep(
            name="train",
            step_type="ml",
            config={"algorithm": "random_forest"},
        )
        assert step.config["algorithm"] == "random_forest"

    def test_step_initial_state(self) -> None:
        step = WorkflowStep(name="test", step_type="pipeline")
        assert step.state == StepState.PENDING
        assert step.result is None
        assert step.error is None
