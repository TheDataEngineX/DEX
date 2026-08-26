"""Execution backends: where operations actually run (§7.9, §13.6).

Two implementations behind one protocol.

``InProcessBackend`` runs an operation in the calling process. It is the fast
path for pure and read-only work where isolation buys nothing, and it is what
tests use.

``SubprocessBackend`` runs the operation in a child process. It exists because
the control plane must survive a workload that segfaults, leaks memory, or
blocks forever — none of which an in-process call can be protected from. The
child gets a scrubbed environment carrying only the secrets its capability token
authorized (§7.8), so a leaked variable in the parent cannot reach project code.

Both report their capabilities honestly. A backend that claims isolation it does
not provide would let the scheduler place work that silently runs unisolated,
which is worse than refusing the placement.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime

from dataenginex.foundation import (
    BackendCapabilities,
    Determinism,
    EstimateContext,
    ExecutionContext,
    ExecutionPlan,
    ExecutionResult,
    ObservedResources,
    Operation,
    ResourceEstimate,
    SecretLease,
)

__all__ = [
    "BackendError",
    "InProcessBackend",
    "OperationHandler",
    "SubprocessBackend",
    "estimate_from_history",
]

# An operation implementation: given the plan and its resolved parameters,
# return the logical names of the artifacts it produced.
OperationHandler = Callable[[ExecutionPlan, ExecutionContext], tuple[str, ...]]


class BackendError(RuntimeError):
    """The backend could not run the plan.

    Distinct from an operation *failing*: a failure is reported in
    ``ExecutionResult.succeeded``, whereas this means the attempt never got far
    enough to produce one.
    """


def estimate_from_history(
    operation: Operation, context: EstimateContext
) -> ResourceEstimate:
    """Predict cost from prior observations, falling back to the declaration.

    Uses the median of recent runs rather than the mean: one pathological run
    should not permanently inflate every subsequent estimate. Confidence rises
    with sample count, and stays low for nondeterministic operations whose cost
    genuinely varies run to run.
    """
    # Unmeasured observations are dropped rather than treated as zero: a run
    # nobody timed says nothing about duration, and averaging it in as 0 would
    # drag every estimate toward "instant".
    durations = sorted(
        o.duration_seconds for o in context.history if o.duration_seconds is not None
    )
    if not durations:
        return ResourceEstimate(
            request=operation.resource_request,
            estimated_duration_seconds=float(operation.resource_request.timeout_seconds) / 10,
            confidence=0.2,
            basis="declared",
        )

    median = durations[len(durations) // 2]
    peak_memory = max(
        (o.peak_memory_mb for o in context.history if o.peak_memory_mb is not None),
        default=0,
    )

    # More samples means more confidence, capped: past behaviour never fully
    # predicts the next input size.
    confidence = min(0.9, 0.3 + 0.1 * len(durations))
    if operation.determinism is Determinism.NONDETERMINISTIC:
        confidence = min(confidence, 0.5)

    request = operation.resource_request
    if peak_memory > 0:
        # Headroom over observed peak, never below what was declared.
        request = request.model_copy(
            update={"memory_mb": max(request.memory_mb, int(peak_memory * 1.25))}
        )

    return ResourceEstimate(
        request=request,
        estimated_duration_seconds=median,
        confidence=confidence,
        basis="observed",
    )


class InProcessBackend:
    """Runs operations in the calling process (§13.6).

    Handlers are registered by operation type. An unregistered type is an error
    rather than a silent no-op — a plan that "succeeds" without running anything
    is the worst possible outcome, because it commits an empty result.
    """

    def __init__(self, handlers: Mapping[str, OperationHandler] | None = None) -> None:
        self._handlers: dict[str, OperationHandler] = dict(handlers or {})

    def register(self, operation_type: str, handler: OperationHandler) -> None:
        self._handlers[operation_type] = handler

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name="in_process",
            # Honest: nothing here isolates anything.
            supports_isolation=False,
            supports_gpu=False,
            # No way to interrupt a Python call that ignores signals.
            supports_cancellation=False,
            supports_checkpointing=False,
            max_concurrent=1,
        )

    def estimate(self, operation: Operation, context: EstimateContext) -> ResourceEstimate:
        return estimate_from_history(operation, context)

    def execute(self, plan: ExecutionPlan, context: ExecutionContext) -> ExecutionResult:
        started = time.monotonic()
        produced: list[str] = []

        try:
            for operation in plan.operations:
                handler = self._handlers.get(operation.operation_type)
                if handler is None:
                    raise BackendError(
                        f"no handler registered for operation {operation.operation_type!r}"
                    )
                produced.extend(handler(plan, context))
        except Exception as exc:
            return ExecutionResult(
                attempt_id=plan.attempt_id,
                succeeded=False,
                observed=ObservedResources(duration_seconds=time.monotonic() - started),
                commit_token=context.capability.token_id,
                error=str(exc),
                error_class=type(exc).__name__,
            )

        return ExecutionResult(
            attempt_id=plan.attempt_id,
            succeeded=True,
            observed=ObservedResources(duration_seconds=time.monotonic() - started),
            # Echoed back so the control plane can fence a late attempt (§14.3).
            commit_token=context.capability.token_id,
        )


class SubprocessBackend:
    """Runs operations in a child process (§7.9).

    The child is launched with ``-m dataenginex.domains.execution.child`` and
    receives its plan on stdin as JSON. Nothing is inherited implicitly: the
    environment is rebuilt from scratch, so a credential in the parent's
    environment cannot leak into project code that was never granted it.
    """

    def __init__(
        self,
        *,
        python: str | None = None,
        secret_resolver: Callable[[ExecutionContext], Mapping[str, SecretLease]] | None = None,
        max_concurrent: int = 4,
    ) -> None:
        self._python = python or sys.executable
        self._secret_resolver = secret_resolver
        self._max_concurrent = max_concurrent

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name="subprocess",
            supports_isolation=True,
            supports_gpu=False,
            # A child can be signalled, then killed if it ignores the signal.
            supports_cancellation=True,
            supports_checkpointing=False,
            max_concurrent=self._max_concurrent,
        )

    def estimate(self, operation: Operation, context: EstimateContext) -> ResourceEstimate:
        return estimate_from_history(operation, context)

    def execute(self, plan: ExecutionPlan, context: ExecutionContext) -> ExecutionResult:
        started = time.monotonic()
        timeout = self._timeout_for(plan, context)

        try:
            completed = subprocess.run(  # noqa: S603 - argv is built here, never shell
                [self._python, "-m", "dataenginex.domains.execution.child"],
                input=self._payload(plan, context),
                env=self._environment(context),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return self._failure(
                plan, context, started, f"exceeded its {timeout:.0f}s deadline", "Timeout"
            )
        except OSError as exc:
            raise BackendError(f"could not start a worker process: {exc}") from exc

        duration = time.monotonic() - started
        if completed.returncode != 0:
            return self._failure(
                plan,
                context,
                started,
                completed.stderr.strip()[-2000:] or f"exited with code {completed.returncode}",
                "SubprocessFailed",
            )

        return ExecutionResult(
            attempt_id=plan.attempt_id,
            succeeded=True,
            observed=ObservedResources(duration_seconds=duration),
            commit_token=context.capability.token_id,
        )

    # --- helpers ------------------------------------------------------------

    def _timeout_for(self, plan: ExecutionPlan, context: ExecutionContext) -> float:
        """Deadline in seconds, honouring whichever bound is nearer."""
        declared = float(plan.resource_request.timeout_seconds)
        if context.deadline is None:
            return declared
        remaining = (context.deadline - datetime.now(UTC)).total_seconds()
        # A non-positive remainder still gets a moment to fail cleanly rather
        # than raising before the process is even started.
        return max(1.0, min(declared, remaining))

    def _payload(self, plan: ExecutionPlan, context: ExecutionContext) -> str:
        return json.dumps(
            {
                "plan": plan.model_dump(mode="json"),
                "attempt_id": context.attempt_id,
                "artifact_namespace": context.artifact_namespace,
                "checkpoint_namespace": context.checkpoint_namespace,
            }
        )

    def _environment(self, context: ExecutionContext) -> dict[str, str]:
        """Build the child's environment from nothing (§7.8).

        Only PATH and the interpreter essentials are carried over, plus the
        secrets this attempt's token authorized. Inheriting ``os.environ``
        wholesale would hand project code every credential the control plane
        happens to hold.
        """
        environment = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "PYTHONHASHSEED": "0",
            "DEX_ATTEMPT_ID": context.attempt_id,
            "DEX_ARTIFACT_NAMESPACE": context.artifact_namespace,
        }

        if self._secret_resolver is not None:
            for name, lease in self._secret_resolver(context).items():
                environment[f"DEX_SECRET_{name.upper()}"] = lease.value

        return environment

    def _failure(
        self,
        plan: ExecutionPlan,
        context: ExecutionContext,
        started: float,
        message: str,
        error_class: str,
    ) -> ExecutionResult:
        return ExecutionResult(
            attempt_id=plan.attempt_id,
            succeeded=False,
            observed=ObservedResources(duration_seconds=time.monotonic() - started),
            commit_token=context.capability.token_id,
            error=message,
            error_class=error_class,
        )
