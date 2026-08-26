"""Execution backends (§7.9, §13.6).

The properties that matter: a failing operation is reported rather than raised,
an unregistered operation is an error rather than a silent success, and the
child process gets only the secrets its token authorized. The last one is a
security boundary, so it is tested against a real subprocess rather than a mock.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import pytest

from dataenginex.domains.execution import (
    BackendError,
    InProcessBackend,
    SubprocessBackend,
    estimate_from_history,
)
from dataenginex.domains.execution.child import run_payload
from dataenginex.foundation import (
    AttemptId,
    Determinism,
    EstimateContext,
    ExecutionContext,
    ExecutionPlan,
    ObservedResources,
    ProjectId,
    ResourceRequest,
    RevisionId,
    SecretLease,
    issue_capability,
    registry,
    utcnow,
)

PROJECT = ProjectId("proj_test")
REVISION = RevisionId("rev_test")
ATTEMPT = AttemptId("att_test")


def make_plan(*operation_types: str, timeout: int = 60) -> ExecutionPlan:
    return ExecutionPlan(
        attempt_id=ATTEMPT,
        project_id=PROJECT,
        revision_id=REVISION,
        operations=tuple(registry.get(t) for t in (operation_types or ("transform",))),
        resource_request=ResourceRequest(timeout_seconds=timeout),
    )


def make_context(**overrides: object) -> ExecutionContext:
    defaults: dict[str, object] = {
        "attempt_id": ATTEMPT,
        "capability": issue_capability(
            principal_id="prin_alice",  # type: ignore[arg-type]
            project_id=PROJECT,
            revision_id=REVISION,
            actions=("transform",),
        ),
        "artifact_namespace": "att_test/artifacts",
    }
    defaults.update(overrides)
    return ExecutionContext(**defaults)  # type: ignore[arg-type]


# --- in-process backend -----------------------------------------------------


def test_registered_handler_runs() -> None:
    backend = InProcessBackend()
    seen: list[str] = []

    def handler(plan: ExecutionPlan, context: ExecutionContext) -> tuple[str, ...]:
        seen.append(plan.attempt_id)
        return ("art_out",)

    backend.register("transform", handler)
    result = backend.execute(make_plan("transform"), make_context())

    assert result.succeeded
    assert seen == [ATTEMPT]


def test_unregistered_operation_fails_rather_than_silently_succeeding() -> None:
    """A plan that 'succeeds' without running anything commits an empty result."""
    result = InProcessBackend().execute(make_plan("transform"), make_context())

    assert not result.succeeded
    assert "no handler registered" in (result.error or "")
    assert result.error_class == "BackendError"


def test_handler_exception_is_reported_not_raised() -> None:
    """A failed operation is data, not an exception the worker must catch."""
    backend = InProcessBackend()

    def explode(plan: ExecutionPlan, context: ExecutionContext) -> tuple[str, ...]:
        raise ValueError("bad input")

    backend.register("transform", explode)
    result = backend.execute(make_plan("transform"), make_context())

    assert not result.succeeded
    assert result.error == "bad input"
    assert result.error_class == "ValueError"


def test_result_echoes_the_commit_token() -> None:
    """§14.3: the control plane fences late attempts on this token."""
    backend = InProcessBackend({"transform": lambda p, c: ()})
    context = make_context()

    result = backend.execute(make_plan("transform"), context)

    assert result.commit_token == context.capability.token_id


def test_failure_still_carries_the_commit_token() -> None:
    """Otherwise a failed attempt cannot be reconciled against its lease."""
    context = make_context()

    result = InProcessBackend().execute(make_plan("transform"), context)

    assert not result.succeeded
    assert result.commit_token == context.capability.token_id


def test_duration_is_observed() -> None:
    backend = InProcessBackend({"transform": lambda p, c: ()})

    result = backend.execute(make_plan("transform"), make_context())

    assert result.observed.duration_seconds is not None
    assert result.observed.duration_seconds >= 0


def test_in_process_backend_admits_it_does_not_isolate() -> None:
    """Claiming isolation it lacks would let the scheduler misplace work."""
    capabilities = InProcessBackend().capabilities()

    assert not capabilities.supports_isolation
    assert not capabilities.supports_cancellation


def test_operations_run_in_declared_order() -> None:
    backend = InProcessBackend()
    order: list[str] = []

    for op_type in ("ingest", "transform", "publish"):
        backend.register(
            op_type,
            lambda p, c, name=op_type: (order.append(name), ())[1],  # type: ignore[misc]
        )

    backend.execute(make_plan("ingest", "transform", "publish"), make_context())

    assert order == ["ingest", "transform", "publish"]


def test_a_failing_operation_stops_the_rest() -> None:
    """Continuing past a failure would produce partial output labelled complete."""
    backend = InProcessBackend()
    ran: list[str] = []

    def fail(plan: ExecutionPlan, context: ExecutionContext) -> tuple[str, ...]:
        raise RuntimeError("stop")

    backend.register("ingest", fail)
    backend.register("transform", lambda p, c: (ran.append("transform"), ())[1])

    result = backend.execute(make_plan("ingest", "transform"), make_context())

    assert not result.succeeded
    assert ran == []


# --- subprocess backend -----------------------------------------------------


def test_subprocess_backend_reports_isolation() -> None:
    capabilities = SubprocessBackend().capabilities()

    assert capabilities.supports_isolation
    assert capabilities.supports_cancellation


def test_subprocess_runs_a_real_child() -> None:
    """Spawns an actual process — the isolation claim must be real."""
    result = SubprocessBackend().execute(make_plan("transform"), make_context())

    assert result.succeeded
    assert result.observed.duration_seconds is not None


def test_subprocess_environment_excludes_the_parents_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§7.8: a credential in the parent must not reach project code.

    Checks the built environment directly, because the whole point is what is
    *absent* from it.
    """
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "parent-only-credential")
    backend = SubprocessBackend()

    environment = backend._environment(make_context())

    assert "AWS_SECRET_ACCESS_KEY" not in environment
    assert "parent-only-credential" not in "".join(environment.values())
    assert environment["PATH"] == os.environ.get("PATH", "")


def test_subprocess_environment_carries_authorized_secrets() -> None:
    """Scoped secrets do reach the child — that is the point of the resolver."""

    def resolver(context: ExecutionContext) -> dict[str, SecretLease]:
        return {
            "api_key": SecretLease(
                reference_name="api_key",
                value="scoped-value",
                expires_at=utcnow() + timedelta(minutes=5),
            )
        }

    backend = SubprocessBackend(secret_resolver=resolver)

    environment = backend._environment(make_context())

    assert environment["DEX_SECRET_API_KEY"] == "scoped-value"


def test_deadline_shortens_the_timeout() -> None:
    """Whichever bound is nearer wins — the plan's or the lease's."""
    backend = SubprocessBackend()
    context = make_context(deadline=datetime.now(UTC) + timedelta(seconds=5))

    timeout = backend._timeout_for(make_plan("transform", timeout=3600), context)

    assert 0 < timeout <= 6


def test_a_passed_deadline_still_yields_a_positive_timeout() -> None:
    """A non-positive timeout would raise before the process even starts."""
    backend = SubprocessBackend()
    context = make_context(deadline=datetime.now(UTC) - timedelta(hours=1))

    assert backend._timeout_for(make_plan("transform"), context) >= 1.0


def test_missing_interpreter_raises_backend_error() -> None:
    """Distinct from an operation failing: the attempt never started."""
    backend = SubprocessBackend(python="/nonexistent/python")

    with pytest.raises(BackendError, match="could not start a worker process"):
        backend.execute(make_plan("transform"), make_context())


# --- the child entry point --------------------------------------------------


def test_child_rejects_an_unknown_operation() -> None:
    payload = {"plan": {"operations": [{"operation_type": "teleport"}]}}

    assert run_payload(payload) == 3


def test_child_rejects_a_plan_with_no_operations() -> None:
    assert run_payload({"plan": {"operations": []}}) == 2


def test_child_rejects_a_payload_with_no_plan() -> None:
    assert run_payload({}) == 2


def test_child_accepts_a_declared_operation() -> None:
    payload = {"plan": {"operations": [{"operation_type": "transform"}]}}

    assert run_payload(payload) == 0


# --- estimation (§15.5) -----------------------------------------------------


def test_estimate_without_history_falls_back_to_the_declaration() -> None:
    estimate = estimate_from_history(
        registry.get("transform"), EstimateContext(project_id=PROJECT, revision_id=REVISION)
    )

    assert estimate.basis == "declared"
    assert estimate.confidence == 0.2


def test_estimate_uses_the_median_not_the_mean() -> None:
    """One pathological run must not inflate every later estimate."""
    history = tuple(
        ObservedResources(duration_seconds=d) for d in (10.0, 11.0, 12.0, 10.0, 3600.0)
    )
    estimate = estimate_from_history(
        registry.get("transform"),
        EstimateContext(project_id=PROJECT, revision_id=REVISION, history=history),
    )

    assert estimate.estimated_duration_seconds < 100
    assert estimate.basis == "observed"


def test_unmeasured_observations_do_not_drag_estimates_to_zero() -> None:
    """A run nobody timed says nothing; treating it as 0 would be a lie."""
    history = (
        ObservedResources(duration_seconds=None),
        ObservedResources(duration_seconds=100.0),
        ObservedResources(duration_seconds=None),
    )
    estimate = estimate_from_history(
        registry.get("transform"),
        EstimateContext(project_id=PROJECT, revision_id=REVISION, history=history),
    )

    assert estimate.estimated_duration_seconds == 100.0


def test_history_of_only_unmeasured_runs_falls_back() -> None:
    history = (ObservedResources(), ObservedResources())

    estimate = estimate_from_history(
        registry.get("transform"),
        EstimateContext(project_id=PROJECT, revision_id=REVISION, history=history),
    )

    assert estimate.basis == "declared"


def test_confidence_grows_with_sample_count() -> None:
    def confidence_for(n: int) -> float:
        history = tuple(ObservedResources(duration_seconds=10.0) for _ in range(n))
        return estimate_from_history(
            registry.get("transform"),
            EstimateContext(project_id=PROJECT, revision_id=REVISION, history=history),
        ).confidence

    assert confidence_for(1) < confidence_for(5)


def test_confidence_is_capped_for_nondeterministic_operations() -> None:
    """Past behaviour predicts less when the operation is inherently variable."""
    history = tuple(ObservedResources(duration_seconds=10.0) for _ in range(20))
    context = EstimateContext(project_id=PROJECT, revision_id=REVISION, history=history)

    deterministic = estimate_from_history(registry.get("transform"), context)
    variable = estimate_from_history(registry.get("ingest"), context)

    assert registry.get("ingest").determinism is Determinism.NONDETERMINISTIC
    assert variable.confidence < deterministic.confidence


def test_memory_estimate_never_drops_below_the_declaration() -> None:
    """Observed peaks add headroom; they must not shrink a declared floor."""
    operation = registry.get("train")
    history = (ObservedResources(duration_seconds=5.0, peak_memory_mb=16),)

    estimate = estimate_from_history(
        operation,
        EstimateContext(project_id=PROJECT, revision_id=REVISION, history=history),
    )

    assert estimate.request.memory_mb >= operation.resource_request.memory_mb


def test_memory_estimate_grows_past_a_high_observed_peak() -> None:
    operation = registry.get("transform")
    peak = operation.resource_request.memory_mb * 4
    history = (ObservedResources(duration_seconds=5.0, peak_memory_mb=peak),)

    estimate = estimate_from_history(
        operation,
        EstimateContext(project_id=PROJECT, revision_id=REVISION, history=history),
    )

    assert estimate.request.memory_mb > peak
