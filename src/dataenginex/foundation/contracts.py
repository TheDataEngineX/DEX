"""Provider contracts (§13.6).

These Protocols are the seam that makes the §5.5 dependency rule enforceable.
Foundation declares what it needs; ``providers/`` implement it with whatever
external libraries they require; ``bootstrap/`` is the only place the two meet.

They are ``Protocol`` rather than ABCs deliberately: a provider should not have
to import and subclass a foundation base class to satisfy a contract, which
would invert the dependency this layering exists to keep pointing inward.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, BinaryIO, Protocol, runtime_checkable

from pydantic import Field

from dataenginex.foundation.artifacts import ArtifactDescriptor, ArtifactReference
from dataenginex.foundation.identity import CapabilityToken, SecretReference
from dataenginex.foundation.ids import AttemptId, ProjectId, ResourceId, RevisionId
from dataenginex.foundation.operations import (
    Operation,
    ResourceEstimate,
    ResourceRequest,
)
from dataenginex.foundation.policy import AuthorizationRequest, PolicyDecision
from dataenginex.foundation.projects import FrozenModel, utcnow
from dataenginex.foundation.resources import Resource, ResourceQuery
from dataenginex.foundation.workloads import ObservedResources

__all__ = [
    "ArtifactStore",
    "BackendCapabilities",
    "EstimateContext",
    "ExecutionBackend",
    "ExecutionContext",
    "ExecutionPlan",
    "ExecutionResult",
    "PolicyEngine",
    "ResourceRepository",
    "SecretLease",
    "SecretProvider",
]


class BackendCapabilities(FrozenModel):
    """What an execution backend can do.

    Lets the scheduler refuse to place work a backend cannot honor — asking for
    isolated execution on a backend that runs in-process should fail at
    admission, not silently run unisolated.
    """

    name: str
    supports_isolation: bool = False
    supports_gpu: bool = False
    supports_cancellation: bool = True
    supports_checkpointing: bool = False
    max_concurrent: int = Field(default=1, ge=1)


class EstimateContext(FrozenModel):
    """Inputs available when predicting an operation's cost (§15.5)."""

    project_id: ProjectId
    revision_id: RevisionId
    input_size_bytes: int | None = Field(default=None, ge=0)
    input_row_count: int | None = Field(default=None, ge=0)
    # Prior observations for this operation type, newest last.
    history: tuple[ObservedResources, ...] = ()


class ExecutionPlan(FrozenModel):
    """A fully resolved, ready-to-run unit of work.

    Produced by the compiler and planner, consumed by a backend. Everything the
    backend needs is here — it never reaches back into the control plane, which
    is what keeps the control/execution split honest (ADR-0004).
    """

    attempt_id: AttemptId
    project_id: ProjectId
    revision_id: RevisionId
    operations: tuple[Operation, ...]
    resource_request: ResourceRequest = Field(default_factory=ResourceRequest)
    # Resolved paths/URIs keyed by the operation's declared input names.
    inputs: dict[str, str] = Field(default_factory=dict)
    parameters: dict[str, str] = Field(default_factory=dict)
    environment_id: str | None = None


class ExecutionContext(FrozenModel):
    """The narrow envelope a worker executes inside (§7.8).

    Carries a scoped capability token, secret *references* only, and dedicated
    artifact/checkpoint namespaces. Invariant 8: a worker receives only the
    permissions and secret references its assigned attempt requires.
    """

    attempt_id: AttemptId
    capability: CapabilityToken
    secret_refs: tuple[SecretReference, ...] = ()
    artifact_namespace: str
    checkpoint_namespace: str | None = None
    deadline: datetime | None = None
    correlation_id: str | None = None


class ExecutionResult(FrozenModel):
    """What a backend reports back.

    ``commit_token`` is the anti-clobber mechanism from §14.3: the control plane
    issues it per attempt and rejects a commit carrying a stale one, so a
    late-finishing lost attempt cannot overwrite a newer successful result.
    """

    attempt_id: AttemptId
    succeeded: bool
    output_artifacts: tuple[ArtifactReference, ...] = ()
    observed: ObservedResources = Field(default_factory=ObservedResources)
    commit_token: str | None = None
    error: str | None = None
    error_class: str | None = None
    completed_at: datetime = Field(default_factory=utcnow)


class SecretLease(FrozenModel):
    """A time-boxed secret resolution (§9.6).

    Holds the value in memory for the duration of one operation. It is a lease,
    not a fetch, because the expiry is what stops a resolved secret from
    outliving the attempt that justified it.
    """

    reference_name: str
    value: str
    expires_at: datetime

    def __repr__(self) -> str:
        # Never let a secret reach a log line, traceback, or REPL echo.
        return f"SecretLease(reference_name={self.reference_name!r}, value=***)"

    def __str__(self) -> str:
        return self.__repr__()


@runtime_checkable
class ResourceRepository(Protocol):
    """Persistence for resources (§13.6)."""

    def register(self, resource: Resource) -> Resource: ...

    def get(self, resource_id: ResourceId) -> Resource: ...

    def search(self, query: ResourceQuery) -> tuple[Resource, ...]: ...


@runtime_checkable
class ArtifactStore(Protocol):
    """Content-addressed byte storage (§13.6)."""

    def put(self, descriptor: ArtifactDescriptor, content: BinaryIO) -> ArtifactReference: ...

    def open(self, reference: ArtifactReference) -> BinaryIO: ...


@runtime_checkable
class ExecutionBackend(Protocol):
    """Where work actually runs (§13.6).

    Subprocess today; containers, Wasm, remote, and Kubernetes later without
    touching callers.
    """

    def capabilities(self) -> BackendCapabilities: ...

    def estimate(self, operation: Operation, context: EstimateContext) -> ResourceEstimate: ...

    def execute(self, plan: ExecutionPlan, context: ExecutionContext) -> ExecutionResult: ...


@runtime_checkable
class PolicyEngine(Protocol):
    """Authorization decisions (§13.6)."""

    def evaluate(self, request: AuthorizationRequest) -> PolicyDecision: ...


@runtime_checkable
class SecretProvider(Protocol):
    """Just-in-time secret resolution (§13.6).

    Takes the capability token as a required argument, not ambient state — a
    provider cannot resolve a secret without the caller proving it was
    authorized for this attempt.
    """

    def resolve(self, reference: SecretReference, capability: CapabilityToken) -> SecretLease: ...


# --- domain persistence ports -------------------------------------------------
# The catalog and model registry each need durable storage, but neither should
# know which store provides it. Structural typing is what makes that work: the
# existing persistence layer already satisfies these shapes, so nothing has to
# declare conformance, and replacing it is a wiring change.
#
# Rows are ``Any`` deliberately. These ports describe *which operations exist*,
# not the storage layout — pinning a concrete row type here would drag the
# schema into the foundation and defeat the separation.


@runtime_checkable
class CatalogRepository(Protocol):
    """Persistence for dataset catalog entries (§8.2)."""

    def register_catalog(self, entry: Any) -> Any: ...

    def get_catalog(self, name: str) -> Any | None: ...

    def search_catalog(
        self, *, layer: str | None = ..., name_contains: str | None = ...
    ) -> list[Any]: ...

    def all_catalog(self) -> list[Any]: ...

    def delete_catalog(self, name: str) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class ModelRepository(Protocol):
    """Persistence for registered model versions (§4.7, §8.2)."""

    def register_model(self, artifact: Any) -> Any: ...

    def get_model(self, name: str, version: str) -> Any | None: ...

    def get_latest_model(self, name: str) -> Any | None: ...

    def get_production_model(self, name: str) -> Any | None: ...

    def list_model_names(self) -> list[str]: ...

    def list_model_versions(self, name: str) -> list[str]: ...

    def promote_model(self, name: str, version: str, stage: str) -> Any: ...

    def close(self) -> None: ...
