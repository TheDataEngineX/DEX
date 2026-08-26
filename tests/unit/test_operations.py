"""The built-in operation catalogue (§4.7).

These declarations drive retry safety, admission control, and policy risk. A
wrong entry is a live bug: declaring ``notify`` retry-safe would let the runtime
send a message twice after a lost lease, and declaring ``delete`` low-risk would
skip the human approval §9.5 requires.
"""

from __future__ import annotations

import pytest

from dataenginex.foundation import (
    BUILTIN_OPERATIONS,
    Determinism,
    IdempotencyStrategy,
    Operation,
    OperationRegistry,
    RiskLevel,
    SideEffectClass,
    UnknownOperationError,
    registry,
)

# The twelve types §4.7 names.
DOCUMENTED_TYPES = {
    "ingest", "transform", "validate", "train", "evaluate", "embed",
    "retrieve", "infer", "publish", "notify", "export", "delete",
}

# What §7.3's interactive workloads ask for: "SQL preview, schema inspection".
# Kept as a separate set so the §4.7 list above stays checkable against the
# document — one merged set would let a future addition quietly erode the claim
# that every documented operation is declared.
INTERACTIVE_TYPES = {
    "sql_preview",
    "schema_inspect",
    "table_stats",
    "lakehouse_inventory",
}

EXPECTED_TYPES = DOCUMENTED_TYPES | INTERACTIVE_TYPES


def test_every_documented_operation_is_declared() -> None:
    declared = {op.operation_type for op in BUILTIN_OPERATIONS}
    assert declared >= DOCUMENTED_TYPES
    assert declared == EXPECTED_TYPES


def test_interactive_operations_only_read() -> None:
    """§7.3's interactive work must not be able to change project data.

    A preview that could write would make the SQL console an editing surface,
    which is neither what the user asked for nor something policy reviewed.
    """
    for operation in BUILTIN_OPERATIONS:
        if operation.operation_type in INTERACTIVE_TYPES:
            assert operation.side_effect_class in (
                SideEffectClass.READ,
                SideEffectClass.EXTERNAL_READ,
            ), operation.operation_type
            assert operation.risk_level <= RiskLevel.READ_PROJECT_DATA
            # Short, per §7.3. A preview holding a worker for an hour is a
            # batch job that skipped admission control.
            assert operation.resource_request.timeout_seconds <= 60


@pytest.mark.parametrize("operation", BUILTIN_OPERATIONS, ids=lambda o: o.operation_type)
def test_every_operation_declares_its_contract(operation: Operation) -> None:
    """§18.8 conformance: no operation may leave these unspecified."""
    assert operation.side_effect_class in SideEffectClass
    assert operation.determinism in Determinism
    assert operation.idempotency in IdempotencyStrategy
    assert operation.risk_level in RiskLevel
    assert operation.resource_request.cpu_cores > 0
    assert operation.resource_request.timeout_seconds > 0


@pytest.mark.parametrize("operation", BUILTIN_OPERATIONS, ids=lambda o: o.operation_type)
def test_operations_producing_output_declare_it(operation: Operation) -> None:
    """A declared output contract is what lets the planner wire a graph."""
    if operation.side_effect_class is SideEffectClass.LOCAL_WRITE:
        assert operation.outputs, f"{operation.operation_type} writes but declares no output"


# --- retry safety (ADR-0007) ------------------------------------------------


@pytest.mark.parametrize("op_type", ["notify", "export"])
def test_external_writes_are_not_retry_safe(op_type: str) -> None:
    """Sending twice sends two messages — the runtime must not replay these."""
    operation = registry.get(op_type)

    assert operation.idempotency is IdempotencyStrategy.NONE
    assert not operation.retry_safe


def test_infer_is_not_retry_safe_despite_only_reading() -> None:
    """The bug the old inference had: side-effect class alone is not enough.

    A model call is billable and non-deterministic, so replaying it is not free
    even though it writes nothing locally.
    """
    operation = registry.get("infer")

    assert operation.side_effect_class is SideEffectClass.EXTERNAL_READ
    assert not operation.retry_safe


def test_delete_is_retry_safe_despite_being_destructive() -> None:
    """The other direction of the same bug.

    Deleting an already-deleted resource is a no-op, so a retry after a lost
    lease is harmless — the risk level, not the retry logic, is what forces the
    human approval.
    """
    operation = registry.get("delete")

    assert operation.side_effect_class is SideEffectClass.DESTRUCTIVE
    assert operation.idempotency is IdempotencyStrategy.NATURAL
    assert operation.requires_approval


def test_transform_is_retry_safe() -> None:
    """The ordinary case: rewriting its own output is harmless."""
    assert registry.get("transform").retry_safe


# --- risk levels (§9.5) -----------------------------------------------------


def test_delete_requires_human_approval() -> None:
    assert registry.get("delete").risk_level is RiskLevel.CONSEQUENTIAL
    assert registry.get("delete").requires_approval


@pytest.mark.parametrize("op_type", ["notify", "export"])
def test_transmitting_operations_are_level_three(op_type: str) -> None:
    """Level 3 means the action needs explicit configuration to run."""
    assert registry.get(op_type).risk_level is RiskLevel.TRANSMIT_EXTERNAL


def test_reads_do_not_require_approval() -> None:
    for op_type in ("validate", "evaluate", "retrieve"):
        assert not registry.get(op_type).requires_approval


# --- egress (invariant 7) ---------------------------------------------------


@pytest.mark.parametrize("op_type", ["ingest", "infer", "notify", "export"])
def test_externally_acting_operations_need_egress_policy(op_type: str) -> None:
    assert registry.requires_egress_policy(op_type)


@pytest.mark.parametrize("op_type", ["transform", "validate", "train", "publish"])
def test_local_operations_do_not_need_egress_policy(op_type: str) -> None:
    assert not registry.requires_egress_policy(op_type)


def test_infer_counts_as_egress_even_for_a_local_model() -> None:
    """The boundary is leaving the process, not leaving the machine."""
    assert registry.requires_egress_policy("infer")


# --- determinism ------------------------------------------------------------


def test_ingest_is_nondeterministic() -> None:
    """It observes a source that may have changed since the last run."""
    assert registry.get("ingest").determinism is Determinism.NONDETERMINISTIC


def test_training_is_seeded_not_deterministic() -> None:
    """Reproducible only when the seed is pinned — an honest declaration."""
    assert registry.get("train").determinism is Determinism.SEEDED


def test_transform_is_deterministic() -> None:
    assert registry.get("transform").determinism is Determinism.DETERMINISTIC


# --- the registry -----------------------------------------------------------


def test_unknown_operation_raises_rather_than_defaulting() -> None:
    """Discovering this mid-run means partial side effects with no plan."""
    with pytest.raises(UnknownOperationError, match="unknown operation type"):
        registry.get("teleport")


def test_membership_checks_do_not_raise() -> None:
    assert "ingest" in registry
    assert "teleport" not in registry
    assert registry.has("ingest")
    assert not registry.has("teleport")


def test_registry_is_iterable_and_sized() -> None:
    fresh = OperationRegistry()

    assert len(fresh) == len(EXPECTED_TYPES)
    assert {op.operation_type for op in fresh} == EXPECTED_TYPES


def test_a_plugin_can_add_an_operation() -> None:
    fresh = OperationRegistry()
    fresh.register(
        Operation(operation_type="scrape", side_effect_class=SideEffectClass.EXTERNAL_READ)
    )

    assert fresh.has("scrape")
    assert fresh.requires_egress_policy("scrape")


def test_a_plugin_cannot_redefine_a_builtin() -> None:
    """Silently lowering delete's risk level would be a policy bypass."""
    fresh = OperationRegistry()

    with pytest.raises(ValueError, match="cannot redefine"):
        fresh.register(
            Operation(
                operation_type="delete",
                side_effect_class=SideEffectClass.READ,
                risk_level=RiskLevel.READ_METADATA,
            )
        )


def test_registering_twice_is_refused() -> None:
    fresh = OperationRegistry()
    operation = Operation(operation_type="scrape")
    fresh.register(operation)

    with pytest.raises(ValueError, match="cannot redefine"):
        fresh.register(operation)


def test_registries_are_independent() -> None:
    """A plugin registered for one project must not leak into another."""
    first = OperationRegistry()
    second = OperationRegistry()
    first.register(Operation(operation_type="scrape"))

    assert first.has("scrape")
    assert not second.has("scrape")
