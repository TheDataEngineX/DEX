"""Resource types with composable facets (§4.6).

A resource is anything identifiable, describable, versionable, governable, and
relatable: datasets, models, prompts, vector indexes, pipelines, connections.

The spec is explicit that this is "composable metadata rather than a single god
object". Facets are therefore separate models attached by name, not a widening
pile of optional columns — a DuckDB table and a prompt template share identity,
ownership, and classification while sharing nothing else.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field

from dataenginex.foundation.ids import (
    PrincipalId,
    ProjectId,
    ResourceId,
    RevisionId,
    new_id,
)
from dataenginex.foundation.projects import FrozenModel, utcnow

__all__ = [
    "Classification",
    "DataFacet",
    "LifecycleState",
    "ModelFacet",
    "PromptFacet",
    "Resource",
    "ResourceQuery",
    "ResourceType",
    "SensitivityFacet",
]


class ResourceType(StrEnum):
    """Kinds of managed resource (§4.6).

    Deliberately closed. An open string field would make policy rules and
    lineage queries unwritable — you cannot authorize "all model resources" if
    anything can call itself a model.
    """

    DATASET = "dataset"
    TABLE = "table"
    DOCUMENT_COLLECTION = "document_collection"
    EVENT_STREAM = "event_stream"
    MODEL = "model"
    PROMPT = "prompt"
    ASSISTANT = "assistant"
    VECTOR_INDEX = "vector_index"
    PIPELINE = "pipeline"
    METRIC = "metric"
    DASHBOARD = "dashboard"
    REPORT = "report"
    CONNECTION = "connection"
    SECRET_REFERENCE = "secret_reference"
    EXTERNAL_ENDPOINT = "external_endpoint"


class Classification(StrEnum):
    """Data classification (§9.8). Small and ordered by restriction."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class LifecycleState(StrEnum):
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    DELETED = "deleted"


class SensitivityFacet(FrozenModel):
    """Privacy facets (§9.8).

    Flags categories present in the data without claiming legal compliance —
    the spec is careful about that distinction and so is this type.
    """

    personal_data: bool = False
    credentials: bool = False
    financial_data: bool = False
    health_data: bool = False
    customer_data: bool = False
    model_inputs: bool = False
    # Free-form extension point for categories the core does not name.
    other: tuple[str, ...] = ()


class DataFacet(FrozenModel):
    """Tabular or file-backed data specifics."""

    schema_digest: str | None = None
    row_count: int | None = Field(default=None, ge=0)
    size_bytes: int | None = Field(default=None, ge=0)
    partition_keys: tuple[str, ...] = ()
    format: str | None = None


class ModelFacet(FrozenModel):
    """Trained-model specifics."""

    framework: str | None = None
    task: str | None = None
    target: str | None = None
    features: tuple[str, ...] = ()
    metrics: dict[str, float] = Field(default_factory=dict)


class PromptFacet(FrozenModel):
    """Prompt/assistant specifics."""

    template_digest: str | None = None
    model_hint: str | None = None
    variables: tuple[str, ...] = ()


class Resource(FrozenModel):
    """A governed, versioned, project-scoped concept (§4.6).

    Scoped to a project *and* the revision that declared it, so a resource's
    definition can be traced back to exact source without consulting mutable
    project state.
    """

    resource_id: ResourceId = Field(default_factory=lambda: ResourceId(new_id("res")))
    project_id: ProjectId
    revision_id: RevisionId
    resource_type: ResourceType
    name: str
    description: str = ""
    labels: dict[str, str] = Field(default_factory=dict)
    owner: PrincipalId | None = None
    classification: Classification = Classification.INTERNAL
    sensitivity: SensitivityFacet = Field(default_factory=SensitivityFacet)
    lifecycle_state: LifecycleState = LifecycleState.ACTIVE
    version: str | None = None
    snapshot_ref: str | None = None
    created_at: datetime = Field(default_factory=utcnow)

    # Typed facets. Exactly which apply depends on resource_type; none are
    # required, because a resource is identifiable before it is described.
    data: DataFacet | None = None
    model: ModelFacet | None = None
    prompt: PromptFacet | None = None
    # Provider-specific extras that the core deliberately does not interpret.
    provider_facets: dict[str, str] = Field(default_factory=dict)
    # Which connector reads or writes this — ``csv``, ``kafka``, ``postgres``.
    # Kept apart from ``resource_type``: that is the catalogue classification and
    # is a closed enum, while connectors are open and arrive with plugins.
    connector: str = ""


class ResourceQuery(FrozenModel):
    """Typed search over resources (§13.6 ``ResourceRepository.search``).

    Typed rather than a free-form filter string: an untyped filter is a
    SQL-injection surface and cannot be validated at the gateway boundary.
    """

    project_id: ProjectId | None = None
    resource_type: ResourceType | None = None
    name_prefix: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    classification: Classification | None = None
    lifecycle_state: LifecycleState | None = None
    limit: int = Field(default=100, ge=1, le=1000)
    cursor: str | None = None
