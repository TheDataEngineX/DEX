"""The built-in operation catalogue (§4.7, §5.3).

Every operation the core ships, declared in full: I/O contracts, side-effect
class, determinism, idempotency strategy, required capabilities, risk level, and
a resource request. The compiler validates against these declarations and the
runtime reads them to decide retry safety, so a wrong entry here is a real bug
rather than stale documentation.

Why declarations rather than inspecting implementations: the scheduler must know
an operation's cost and safety *before* running it, and the policy engine must
know its risk level before authorizing it. Neither can call the function to find
out. Declaring up front is also what lets a plugin supply an operation the core
has never seen (§10.6) on equal terms with a built-in one.

The tricky entries are worth reading:

* ``ingest`` is ``EXTERNAL_READ`` and non-deterministic — it observes a source
  that may have changed since the last run, so its output is not reproducible
  from its inputs alone.
* ``notify`` and ``export`` are ``EXTERNAL_WRITE`` with ``IdempotencyStrategy.
  NONE``: sending twice sends two messages. They are deliberately *not*
  retry-safe, which forces the runtime to require an idempotency key or an
  approval instead of quietly replaying them.
* ``delete`` is ``DESTRUCTIVE`` at risk level 5, so it always needs human
  approval (§9.5).

This lives in the foundation because a declaration is data, not behaviour. It
started in ``domains/`` and had to move: the compiler in ``runtime/`` reads the
catalogue, while ``domains/`` reads the control store in ``runtime/``, so the
two packages depended on each other and neither could be loaded alone.
"""

from __future__ import annotations

from collections.abc import Iterator

from dataenginex.foundation import (
    Determinism,
    IdempotencyStrategy,
    IOContract,
    Operation,
    ResourceRequest,
    RiskLevel,
    SideEffectClass,
)

__all__ = ["BUILTIN_OPERATIONS", "OperationRegistry", "UnknownOperationError", "registry"]


class UnknownOperationError(KeyError):
    """An operation type nobody declared.

    Raised at compile time rather than execution time — discovering an unknown
    operation halfway through a run means partial side effects with no plan for
    the rest.
    """


def _op(
    operation_type: str,
    *,
    inputs: tuple[IOContract, ...] = (),
    outputs: tuple[IOContract, ...] = (),
    side_effect: SideEffectClass = SideEffectClass.READ,
    determinism: Determinism = Determinism.DETERMINISTIC,
    idempotency: IdempotencyStrategy = IdempotencyStrategy.NATURAL,
    capabilities: tuple[str, ...] = (),
    risk: RiskLevel = RiskLevel.READ_PROJECT_DATA,
    request: ResourceRequest | None = None,
    implementation: str | None = None,
) -> Operation:
    return Operation(
        operation_type=operation_type,
        inputs=inputs,
        outputs=outputs,
        side_effect_class=side_effect,
        determinism=determinism,
        idempotency=idempotency,
        required_capabilities=capabilities,
        risk_level=risk,
        resource_request=request or ResourceRequest(),
        implementation_ref=implementation,
    )


_DATASET = "dataset"
_MODEL = "model"


BUILTIN_OPERATIONS: tuple[Operation, ...] = (
    # --- interactive (§7.3) -------------------------------------------------
    #
    # What the SQL console, the schema panel, and the catalog page ask for.
    # Declared here like anything else, because they are workloads: they run on
    # a worker, under a capability token, against a pinned revision.
    #
    # All three are reads with a short timeout and a small ceiling — §7.3's
    # "high priority, short timeout, small resource ceiling". A preview that
    # could write would turn the SQL box into an editing surface for data the
    # project never declared it could change.
    _op(
        "sql_preview",
        outputs=(IOContract(name="rows", resource_type="preview", required=False),),
        side_effect=SideEffectClass.READ,
        # The same query over changing data gives different rows.
        determinism=Determinism.NONDETERMINISTIC,
        capabilities=("data.batch",),
        risk=RiskLevel.READ_PROJECT_DATA,
        request=ResourceRequest(memory_mb=512, timeout_seconds=30),
        implementation="dataenginex.domains.execution.handlers",
    ),
    _op(
        "schema_inspect",
        inputs=(IOContract(name="resource", resource_type=_DATASET),),
        outputs=(IOContract(name="schema", resource_type="preview", required=False),),
        side_effect=SideEffectClass.EXTERNAL_READ,
        determinism=Determinism.NONDETERMINISTIC,
        capabilities=("data.batch",),
        request=ResourceRequest(memory_mb=512, timeout_seconds=30),
        implementation="dataenginex.domains.execution.handlers",
    ),
    _op(
        "table_stats",
        inputs=(IOContract(name="table", resource_type=_DATASET),),
        outputs=(IOContract(name="stats", resource_type="preview", required=False),),
        side_effect=SideEffectClass.READ,
        determinism=Determinism.NONDETERMINISTIC,
        capabilities=("data.batch",),
        request=ResourceRequest(memory_mb=512, timeout_seconds=30),
        implementation="dataenginex.domains.execution.handlers",
    ),
    _op(
        "lakehouse_inventory",
        outputs=(IOContract(name="inventory", resource_type="preview", required=False),),
        side_effect=SideEffectClass.READ,
        determinism=Determinism.NONDETERMINISTIC,
        capabilities=("data.batch",),
        # Counts rows across every table in a layer, so it gets more headroom
        # and more time than a single-table preview.
        request=ResourceRequest(memory_mb=512, timeout_seconds=60),
        implementation="dataenginex.domains.execution.handlers",
    ),
    # --- data (§5.3 domains/data) ------------------------------------------
    _op(
        "ingest",
        outputs=(IOContract(name="output", resource_type=_DATASET),),
        side_effect=SideEffectClass.EXTERNAL_READ,
        # Reads a source that may have changed; the same inputs can yield
        # different bytes, so provenance records the result rather than
        # promising recomputation.
        determinism=Determinism.NONDETERMINISTIC,
        idempotency=IdempotencyStrategy.KEYED,
        capabilities=("data.batch",),
        request=ResourceRequest(memory_mb=1024, timeout_seconds=1800),
        implementation="dataenginex.providers.connectors",
    ),
    _op(
        "transform",
        inputs=(IOContract(name="input", resource_type=_DATASET),),
        outputs=(IOContract(name="output", resource_type=_DATASET),),
        side_effect=SideEffectClass.LOCAL_WRITE,
        idempotency=IdempotencyStrategy.NATURAL,
        capabilities=("data.batch",),
        risk=RiskLevel.CREATE_LOCAL_ARTIFACT,
        request=ResourceRequest(memory_mb=2048),
        implementation="dataenginex.domains.analytics.transforms",
    ),
    _op(
        "validate",
        inputs=(IOContract(name="input", resource_type=_DATASET),),
        outputs=(
            IOContract(name="report", resource_type="quality_report", required=False),
        ),
        side_effect=SideEffectClass.READ,
        capabilities=("data.batch",),
        implementation="dataenginex.domains.analytics.quality",
    ),
    _op(
        "publish",
        inputs=(IOContract(name="input", resource_type=_DATASET),),
        outputs=(IOContract(name="artifact", resource_type="artifact"),),
        side_effect=SideEffectClass.LOCAL_WRITE,
        # The commit protocol (§14.3) makes republication of identical content
        # a no-op, so a retry is safe.
        idempotency=IdempotencyStrategy.TRANSACTIONAL,
        risk=RiskLevel.CREATE_LOCAL_ARTIFACT,
    ),
    # --- ml (§5.3 domains/ml) ----------------------------------------------
    _op(
        "train",
        inputs=(IOContract(name="training_data", resource_type=_DATASET),),
        outputs=(IOContract(name="model", resource_type=_MODEL),),
        side_effect=SideEffectClass.LOCAL_WRITE,
        # Seeded rather than deterministic: reproducible only when the seed is
        # pinned, which the training config is responsible for.
        determinism=Determinism.SEEDED,
        idempotency=IdempotencyStrategy.NATURAL,
        capabilities=("ml.training",),
        risk=RiskLevel.CREATE_LOCAL_ARTIFACT,
        request=ResourceRequest(cpu_cores=2.0, memory_mb=4096, timeout_seconds=7200),
        implementation="dataenginex.domains.ml.training",
    ),
    _op(
        "evaluate",
        inputs=(
            IOContract(name="model", resource_type=_MODEL),
            IOContract(name="evaluation_data", resource_type=_DATASET),
        ),
        outputs=(IOContract(name="metrics", resource_type="metrics"),),
        side_effect=SideEffectClass.READ,
        capabilities=("ml.training",),
        request=ResourceRequest(memory_mb=2048),
        implementation="dataenginex.domains.ml.metrics",
    ),
    # --- ai (§5.3 domains/ai) ----------------------------------------------
    _op(
        "embed",
        inputs=(IOContract(name="input", resource_type=_DATASET),),
        outputs=(IOContract(name="embeddings", resource_type="embeddings"),),
        side_effect=SideEffectClass.LOCAL_WRITE,
        determinism=Determinism.SEEDED,
        capabilities=("ai.local",),
        risk=RiskLevel.CREATE_LOCAL_ARTIFACT,
        request=ResourceRequest(memory_mb=2048, timeout_seconds=1800),
        implementation="dataenginex.providers.vector.vectorstore",
    ),
    _op(
        "retrieve",
        inputs=(IOContract(name="query", resource_type="query"),),
        outputs=(IOContract(name="documents", resource_type="documents"),),
        side_effect=SideEffectClass.READ,
        capabilities=("ai.local",),
        implementation="dataenginex.domains.ai.retrieval",
    ),
    _op(
        "infer",
        inputs=(IOContract(name="prompt", resource_type="prompt"),),
        outputs=(IOContract(name="completion", resource_type="completion"),),
        # EXTERNAL_READ even for a local model: the boundary that matters is
        # whether the operation leaves the process, and a hosted model call
        # must be visible to egress policy.
        side_effect=SideEffectClass.EXTERNAL_READ,
        determinism=Determinism.NONDETERMINISTIC,
        idempotency=IdempotencyStrategy.NONE,
        capabilities=("ai.remote",),
        request=ResourceRequest(timeout_seconds=300),
        implementation="dataenginex.domains.ai.llm",
    ),
    # --- external effects ---------------------------------------------------
    _op(
        "notify",
        inputs=(IOContract(name="message", resource_type="message"),),
        side_effect=SideEffectClass.EXTERNAL_WRITE,
        determinism=Determinism.NONDETERMINISTIC,
        # Sending twice sends two messages. Declaring NONE is what stops the
        # runtime from silently replaying it after a lost lease.
        idempotency=IdempotencyStrategy.NONE,
        risk=RiskLevel.TRANSMIT_EXTERNAL,
        request=ResourceRequest(timeout_seconds=120),
    ),
    _op(
        "export",
        inputs=(IOContract(name="input", resource_type=_DATASET),),
        side_effect=SideEffectClass.EXTERNAL_WRITE,
        determinism=Determinism.NONDETERMINISTIC,
        idempotency=IdempotencyStrategy.NONE,
        risk=RiskLevel.TRANSMIT_EXTERNAL,
        request=ResourceRequest(timeout_seconds=1800),
    ),
    _op(
        "delete",
        inputs=(IOContract(name="target", resource_type="resource"),),
        side_effect=SideEffectClass.DESTRUCTIVE,
        idempotency=IdempotencyStrategy.NATURAL,
        # Level 5: always requires human approval (§9.5).
        risk=RiskLevel.CONSEQUENTIAL,
    ),
)


class OperationRegistry:
    """Resolves operation types to their declarations.

    Plugin operations register alongside built-ins so the compiler and runtime
    treat them identically. Registration refuses to shadow an existing type: a
    plugin silently redefining ``delete`` with a lower risk level would be a
    policy bypass dressed as an extension point.
    """

    def __init__(self, operations: tuple[Operation, ...] = BUILTIN_OPERATIONS) -> None:
        self._operations: dict[str, Operation] = {op.operation_type: op for op in operations}

    def register(self, operation: Operation, *, source: str = "plugin") -> None:
        existing = self._operations.get(operation.operation_type)
        if existing is not None:
            raise ValueError(
                f"{source} cannot redefine operation {operation.operation_type!r}; "
                "it is already declared"
            )
        self._operations[operation.operation_type] = operation

    def get(self, operation_type: str) -> Operation:
        try:
            return self._operations[operation_type]
        except KeyError as exc:
            raise UnknownOperationError(
                f"unknown operation type {operation_type!r}; declare it in a plugin manifest"
            ) from exc

    def has(self, operation_type: str) -> bool:
        return operation_type in self._operations

    def requires_egress_policy(self, operation_type: str) -> bool:
        """Whether this operation must pass destination policy (invariant 7)."""
        operation = self.get(operation_type)
        return operation.side_effect_class in (
            SideEffectClass.EXTERNAL_READ,
            SideEffectClass.EXTERNAL_WRITE,
        )

    def __contains__(self, operation_type: object) -> bool:
        return isinstance(operation_type, str) and operation_type in self._operations

    def __iter__(self) -> Iterator[Operation]:
        return iter(self._operations.values())

    def __len__(self) -> int:
        return len(self._operations)


# The default registry. A process-wide instance is appropriate here because
# operation declarations are immutable and identical across projects; per-project
# variation belongs in the manifest, not in a mutable global.
registry = OperationRegistry()
