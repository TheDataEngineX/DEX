"""Public project manifest schema, ``dex/v1alpha1`` (§6.4, §6.7).

This format is public and versioned; the compiled IR downstream is internal and
may change (§6.8). The split matters — users write against this, and it is the
thing we owe compatibility to once the schema leaves alpha.

§6.7 draws a hard line around what configuration *may not* contain: secret
values, large inline programs, arbitrary shell commands, unbounded templating,
hidden network destinations, and mutable runtime state. Several of those are
enforced here as validators rather than left to review, because each one is a
path by which a project file becomes an execution vector.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import Field, field_validator

from dataenginex.foundation import FrozenModel, WorkloadKind

__all__ = [
    "SCHEMA_VERSION",
    "SECRET_PATTERNS",
    "CapabilitySpec",
    "EnvironmentSpec",
    "LimitsSpec",
    "NetworkAllowRule",
    "NetworkPolicyDeclaration",
    "OperationDeclaration",
    "PolicyDeclaration",
    "ProjectManifest",
    "ProjectMetadata",
    "ProjectSpec",
    "ProvidersSpec",
    "ResourceDeclaration",
    "RetrySpec",
    "WorkloadDeclaration",
]

SCHEMA_VERSION = "dex/v1alpha1"

# Values that look like embedded credentials. §6.7 forbids secret values in
# configuration outright, and invariant 5 forbids them reaching a revision — so
# they are rejected at parse time rather than caught later in review.
SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^-{5}BEGIN [A-Z ]*PRIVATE KEY-{5}"),
    re.compile(r"^(sk|pk)-[A-Za-z0-9]{16,}$"),
    re.compile(r"^gh[pousr]_[A-Za-z0-9]{20,}$"),
    re.compile(r"^AKIA[0-9A-Z]{16}$"),
    re.compile(r"^xox[baprs]-[A-Za-z0-9-]{10,}$"),
)

# Reference to a secret, e.g. "${secret:gmail_token}". Permitted anywhere a
# value is expected, because it names a secret without carrying one.
SECRET_REF = re.compile(r"^\$\{secret:[a-zA-Z0-9_.-]+\}$")

Name = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,62}$")]


def _reject_secret_values(value: str, field: str) -> str:
    """Fail a manifest that inlines something shaped like a credential."""
    if SECRET_REF.match(value):
        return value
    for pattern in SECRET_PATTERNS:
        if pattern.match(value.strip()):
            raise ValueError(
                f"{field} looks like an inline secret. Use ${{secret:name}} "
                "and store the value in the secret provider (§9.6)."
            )
    return value


class ProjectMetadata(FrozenModel):
    """Identity block (§6.4)."""

    name: Name
    display_name: str = ""
    description: str = ""
    labels: dict[str, str] = Field(default_factory=dict)


class CapabilitySpec(FrozenModel):
    """Capabilities a project needs (§6.4).

    Declared up front so the compiler can fail a project whose required
    capabilities are not installed, instead of failing at the first run that
    reaches the missing feature.
    """

    required: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()


class ProvidersSpec(FrozenModel):
    """Which provider implementation backs each concern (§6.4)."""

    metadata: str = "embedded"
    artifacts: str = "local"
    analytical: str = "duckdb"
    vector: str | None = None
    model: str | None = None


class LimitsSpec(FrozenModel):
    """Resource ceiling for the whole project (§6.4).

    Sizes stay human strings ("6GiB") because the manifest is meant to be read
    and edited by hand; the compiler parses them into bytes.
    """

    cpu: float = Field(default=2.0, gt=0)
    memory: str = "4GiB"
    working_storage: str = "20GiB"


class EnvironmentSpec(FrozenModel):
    """Dependency environment reference (§6.9)."""

    lockfile: str = "dex.lock"
    python_version: str | None = None
    plugins: tuple[str, ...] = ()


class ResourceDeclaration(FrozenModel):
    """One declared resource (§6.7).

    ``config`` holds provider-specific settings the core does not interpret.
    It is validated for inline secrets, since that is where a connection string
    would otherwise be pasted.
    """

    name: Name
    type: str
    description: str = ""
    labels: dict[str, str] = Field(default_factory=dict)
    classification: str = "internal"
    config: dict[str, str] = Field(default_factory=dict)

    @field_validator("config")
    @classmethod
    def _no_inline_secrets(cls, value: dict[str, str]) -> dict[str, str]:
        for key, item in value.items():
            _reject_secret_values(item, f"resource config '{key}'")
        return value


class OperationDeclaration(FrozenModel):
    """One step inside a workload (§6.7).

    ``sql_file`` and ``script_file`` are *references*, not inline bodies. §6.7
    forbids large inline programs: they defeat diffing, review, and the
    content-addressed hashing revisions depend on.
    """

    type: str
    name: str = ""
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    sql_file: str | None = None
    script_file: str | None = None
    parameters: dict[str, str] = Field(default_factory=dict)

    @field_validator("parameters")
    @classmethod
    def _no_inline_secrets(cls, value: dict[str, str]) -> dict[str, str]:
        for key, item in value.items():
            _reject_secret_values(item, f"parameter '{key}'")
        return value


class RetrySpec(FrozenModel):
    max_attempts: int = Field(default=3, ge=0, le=100)
    backoff_seconds: float = Field(default=5.0, ge=0)


class WorkloadDeclaration(FrozenModel):
    """A workload and its DAG position (§6.7)."""

    name: Name
    kind: WorkloadKind = WorkloadKind.BATCH
    operations: tuple[OperationDeclaration, ...] = ()
    depends_on: tuple[str, ...] = ()
    schedule: str | None = None
    retry: RetrySpec = Field(default_factory=RetrySpec)
    priority: int = Field(default=100, ge=0, le=1000)
    enabled: bool = True


class PolicyDeclaration(FrozenModel):
    """A governance rule declared in the project (§6.7)."""

    name: Name
    effect: str = "deny"
    actions: tuple[str, ...] = ()
    resources: tuple[str, ...] = ()
    classifications: tuple[str, ...] = ()
    destinations: tuple[str, ...] = ()
    description: str = ""


class NetworkAllowRule(FrozenModel):
    """One permitted egress destination (§9.7)."""

    host: str
    operations: tuple[str, ...] = ()
    purpose: str = ""


class NetworkPolicyDeclaration(FrozenModel):
    """Egress policy. Default deny (§9.7)."""

    default: Literal["deny", "allow"] = "deny"
    allow: tuple[NetworkAllowRule, ...] = ()


DeploymentProfile = Literal["lite", "quickstart", "home-server", "distributed", "cloud"]
"""The five deployment profiles (§11.2–11.6).

Constrained rather than free text. A profile decides which components exist —
whether the control store is SQLite or PostgreSQL, whether workers are local
subprocesses or a separate pool — so a typo'd profile is a project that
compiles and then asks for a deployment nobody built.
"""


class ProjectSpec(FrozenModel):
    """The ``spec`` block (§6.4)."""

    # Lite is the default because §11.1 says one installation manages many
    # projects: a project asks for the smallest thing that runs it, and the
    # installation decides what it actually gets.
    profile: DeploymentProfile = "lite"
    imports: tuple[str, ...] = ()
    capabilities: CapabilitySpec = Field(default_factory=CapabilitySpec)
    providers: ProvidersSpec = Field(default_factory=ProvidersSpec)
    limits: LimitsSpec = Field(default_factory=LimitsSpec)
    environment: EnvironmentSpec = Field(default_factory=EnvironmentSpec)
    resources: tuple[ResourceDeclaration, ...] = ()
    workloads: tuple[WorkloadDeclaration, ...] = ()
    policies: tuple[PolicyDeclaration, ...] = ()
    network: NetworkPolicyDeclaration = Field(default_factory=NetworkPolicyDeclaration)


class ProjectManifest(FrozenModel):
    """A ``kind: Project`` document (§6.4).

    Every revision records the exact ``apiVersion`` it was written against
    (§13.7), so a stored revision stays interpretable after the schema moves on.
    """

    model_config = FrozenModel.model_config | {"populate_by_name": True}

    api_version: str = Field(default=SCHEMA_VERSION, alias="apiVersion")
    kind: Literal["Project"] = "Project"
    metadata: ProjectMetadata
    spec: ProjectSpec = Field(default_factory=ProjectSpec)

    @field_validator("api_version")
    @classmethod
    def _known_version(cls, value: str) -> str:
        if value != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported apiVersion {value!r}; this build understands {SCHEMA_VERSION!r}"
            )
        return value
