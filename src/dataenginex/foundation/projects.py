"""Installation, workspace, project, and revision types (§4.2-4.5).

The hierarchy is Installation -> Workspace -> Project -> ProjectRevision.
Project identity is stable; project *definitions* are not. That split is the
reason revisions exist as separate immutable records rather than as mutable
fields on ``Project``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from dataenginex.foundation.ids import (
    InstallationId,
    PrincipalId,
    ProjectId,
    RevisionId,
    WorkspaceId,
    new_id,
)

__all__ = [
    "FrozenModel",
    "Installation",
    "Project",
    "ProjectRevision",
    "PublicationStatus",
    "RevisionFile",
    "ValidationIssue",
    "ValidationReport",
    "ValidationSeverity",
    "utcnow",
]


def utcnow() -> datetime:
    """Timezone-aware UTC now.

    Naive datetimes compare and serialize inconsistently once they cross a
    process or storage boundary, so every timestamp in the model is aware.
    """
    return datetime.now(UTC)


class FrozenModel(BaseModel):
    """Immutable, strictly-validated base for all foundation types.

    ``extra="forbid"`` matters more than it looks: revision content is hashed,
    and silently-accepted unknown fields would let two structurally different
    manifests produce the same canonical form.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_assignment=True)


class PublicationStatus(StrEnum):
    """Lifecycle of a revision (§4.5)."""

    DRAFT = "draft"
    PUBLISHED = "published"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"


class ValidationSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationIssue(FrozenModel):
    """One finding from the project compiler (§6.8)."""

    severity: ValidationSeverity
    code: str
    message: str
    # Dotted path into the manifest, e.g. "workloads.clean_users.transforms[1]".
    location: str | None = None


class ValidationReport(FrozenModel):
    """Compiler output attached to a revision (§4.5).

    A revision may only be published when this reports no errors — publication
    fails closed. The current engine discards validation errors and runs anyway;
    that is the defect this type exists to make impossible.
    """

    issues: tuple[ValidationIssue, ...] = ()
    validated_at: datetime = Field(default_factory=utcnow)

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if i.severity is ValidationSeverity.ERROR)

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        """Issues worth showing that do not block publication.

        Paired with :attr:`errors` so a caller can surface both without
        filtering by severity itself — a hand-rolled filter at each call site is
        how a warning ends up silently dropped.
        """
        return tuple(i for i in self.issues if i.severity is ValidationSeverity.WARNING)

    @property
    def ok(self) -> bool:
        return not self.errors


class RevisionFile(FrozenModel):
    """One file captured in a revision bundle, addressed by content."""

    path: str
    digest: str
    size_bytes: int = Field(ge=0)
    media_type: str = "application/octet-stream"
    plugin_versions: dict[str, str] = Field(default_factory=dict)
    provider_requirements: tuple[str, ...] = ()


class Installation(FrozenModel):
    """One operational control plane and its providers (§4.2).

    Explicitly not a container for user business data — it holds system settings,
    the provider registry, global limits, and trust roots only.
    """

    installation_id: InstallationId = Field(default_factory=lambda: InstallationId(new_id("inst")))
    name: str
    created_at: datetime = Field(default_factory=utcnow)
    settings: dict[str, str] = Field(default_factory=dict)
    trust_roots: tuple[str, ...] = ()


class Project(FrozenModel):
    """Stable identity and lifecycle container for one user purpose (§4.4).

    ``active_revision_id`` is a pointer, flipped atomically on publish. Rollback
    re-points it at an earlier revision rather than mutating history.
    """

    project_id: ProjectId = Field(default_factory=lambda: ProjectId(new_id("proj")))
    workspace_id: WorkspaceId
    name: str
    description: str = ""
    created_at: datetime = Field(default_factory=utcnow)
    # None only during creation or after archival (invariant 1, §4.16).
    active_revision_id: RevisionId | None = None
    archived: bool = False


class ProjectRevision(FrozenModel):
    """Immutable validated snapshot of a project's files (§4.5).

    Every execution pins exactly one revision, which is what makes a run
    reproducible: the manifest, the dependency lock, and the declared
    capabilities are all fixed at publish time and cannot drift underneath a
    running workload.
    """

    revision_id: RevisionId = Field(default_factory=lambda: RevisionId(new_id("rev")))
    project_id: ProjectId
    parent_revision_id: RevisionId | None = None
    # SHA-256 over the canonical revision package.
    content_hash: str
    created_by: PrincipalId
    created_at: datetime = Field(default_factory=utcnow)
    manifest_schema_version: str = "dex/v0.7"
    dependency_lock_hash: str | None = None
    capability_requirements: tuple[str, ...] = ()
    # v0.7: Spark pipeline references in this revision
    spark_pipeline_refs: tuple[str, ...] = ()
    # v0.7: Required API version range for compatibility
    required_api_range: str | None = None
    # v0.7: Required Spark version range when Spark workloads exist
    required_spark_range: str | None = None
    validation_report: ValidationReport = Field(default_factory=ValidationReport)
    status: PublicationStatus = PublicationStatus.DRAFT
    files: tuple[RevisionFile, ...] = ()

    @property
    def publishable(self) -> bool:
        return self.status is PublicationStatus.DRAFT and self.validation_report.ok
