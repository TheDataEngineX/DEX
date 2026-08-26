"""Operation: a typed description of work (§4.7).

An operation declares what it does before it does it — I/O contracts,
side-effect class, determinism, idempotency strategy, required capabilities, and
how to estimate its cost. The scheduler, policy engine, and retry logic all read
these declarations rather than guessing from an operation's name.

That declaration-first design is what makes safe retry possible: the runtime
retries at-least-once (ADR-0007) and needs to know, per operation, whether a
second execution is harmless.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum

from pydantic import Field

from dataenginex.foundation.projects import FrozenModel

__all__ = [
    "Determinism",
    "IOContract",
    "IdempotencyStrategy",
    "Operation",
    "ResourceEstimate",
    "ResourceRequest",
    "RiskLevel",
    "SideEffectClass",
]


class SideEffectClass(StrEnum):
    """What an operation changes (§4.7).

    Drives retry safety and policy: ``EXTERNAL_WRITE`` cannot be retried
    blindly and always requires destination policy evaluation (invariant 7).
    """

    PURE = "pure"
    READ = "read"
    LOCAL_WRITE = "local_write"
    EXTERNAL_READ = "external_read"
    EXTERNAL_WRITE = "external_write"
    DESTRUCTIVE = "destructive"


class Determinism(StrEnum):
    """Whether identical inputs yield identical outputs.

    ``NONDETERMINISTIC`` covers model inference with sampling and anything
    reading wall-clock or live external state — such outputs cannot be
    reproduced from inputs alone, so provenance must record the observed result
    rather than promise recomputation.
    """

    DETERMINISTIC = "deterministic"
    SEEDED = "seeded"
    NONDETERMINISTIC = "nondeterministic"


class IdempotencyStrategy(StrEnum):
    """How a repeated execution is made safe (§4.7).

    ``NONE`` is a declaration that repetition is unsafe, which forces the
    runtime to require an idempotency key or human approval rather than
    silently retrying.
    """

    NONE = "none"
    NATURAL = "natural"
    KEYED = "keyed"
    TRANSACTIONAL = "transactional"
    DEDUPLICATED = "deduplicated"


class RiskLevel(IntEnum):
    """Action-risk levels 0-5 (§9.5).

    Integer-valued because policy compares them: default policy requires
    explicit configuration at ``TRANSMIT_EXTERNAL`` and human approval at or
    above ``MODIFY_EXTERNAL``.
    """

    READ_METADATA = 0
    READ_PROJECT_DATA = 1
    CREATE_LOCAL_ARTIFACT = 2
    TRANSMIT_EXTERNAL = 3
    MODIFY_EXTERNAL = 4
    CONSEQUENTIAL = 5


class IOContract(FrozenModel):
    """Declared shape of one input or output.

    ``schema_ref`` points at a registered JSON Schema rather than embedding it,
    so contracts stay comparable across revisions by reference.
    """

    name: str
    resource_type: str
    schema_ref: str | None = None
    required: bool = True
    description: str = ""


class ResourceRequest(FrozenModel):
    """Resources an operation asks for, used for admission control (§7.5)."""

    cpu_cores: float = Field(default=1.0, gt=0)
    memory_mb: int = Field(default=512, gt=0)
    disk_mb: int = Field(default=0, ge=0)
    gpu_count: int = Field(default=0, ge=0)
    timeout_seconds: int = Field(default=3600, gt=0)


class ResourceEstimate(FrozenModel):
    """Predicted cost with explicit uncertainty (§15.5).

    The confidence band is part of the estimate, not an afterthought: the
    scheduler must distinguish a firm 2-second estimate from a wild guess when
    deciding whether to admit work.
    """

    request: ResourceRequest
    estimated_duration_seconds: float = Field(ge=0)
    # Multiplicative uncertainty band, e.g. 0.5 => within a factor of 2.
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    basis: str = "static"


class Operation(FrozenModel):
    """A typed, versioned unit of work (§4.7).

    Examples: ingest, transform, validate, train, evaluate, embed, retrieve,
    infer, publish, notify, export, delete.
    """

    operation_type: str
    version: str = "1"
    # What the project called this step, unique within its workload. The type
    # says what kind of work it is; this says which one it is, so a workload
    # with two ``transform`` steps stays distinguishable in logs and lineage.
    name: str = ""
    inputs: tuple[IOContract, ...] = ()
    outputs: tuple[IOContract, ...] = ()
    # The declaration bound onto the catalogue entry: which resources this step
    # reads and writes, and the settings a handler needs. Without these an
    # operation describes a *category* of work and a handler has nothing to act
    # on — it could know it must ingest, but not from where.
    bound_inputs: tuple[str, ...] = ()
    bound_outputs: tuple[str, ...] = ()
    parameters: dict[str, str] = Field(default_factory=dict)
    # References, never bodies. §6.7 forbids inline programs in configuration.
    sql_file: str | None = None
    script_file: str | None = None
    side_effect_class: SideEffectClass = SideEffectClass.READ
    determinism: Determinism = Determinism.DETERMINISTIC
    idempotency: IdempotencyStrategy = IdempotencyStrategy.NATURAL
    required_capabilities: tuple[str, ...] = ()
    risk_level: RiskLevel = RiskLevel.READ_PROJECT_DATA
    resource_request: ResourceRequest = Field(default_factory=ResourceRequest)
    # Dotted path to the callable implementing this operation, resolved by the
    # runtime registry. Foundation never imports it — that would drag providers
    # into this layer and break the §5.5 dependency rule.
    implementation_ref: str | None = None

    @property
    def retry_safe(self) -> bool:
        """Whether the runtime may retry without extra ceremony.

        A retry is safe when repetition is declared harmless. Everything that
        touches the outside world with no idempotency mechanism must not be
        replayed automatically.
        """
        if self.idempotency is IdempotencyStrategy.NONE:
            return False
        return self.side_effect_class not in (
            SideEffectClass.EXTERNAL_WRITE,
            SideEffectClass.DESTRUCTIVE,
        )

    @property
    def requires_approval(self) -> bool:
        """Levels 4-5 need human approval by default (§9.5)."""
        return self.risk_level >= RiskLevel.MODIFY_EXTERNAL
