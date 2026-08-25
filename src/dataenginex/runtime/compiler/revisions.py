"""Draft, publish, and rollback of immutable project revisions (§6.3).

The rules that make this safe:

- A revision is written once and never mutated. Editing a project produces a
  new draft; publishing produces a new revision.
- Publication is atomic. The pointer flip and the status changes happen in one
  transaction, so no reader ever sees two published revisions or none.
- Rollback re-points at an earlier revision. It does not delete, rewrite, or
  resurrect history — the rolled-back revision keeps its record.
- Runs already in flight keep executing their pinned revision (§6.3 item 9).
  Nothing here touches ``runs``.
- Publishing fails closed: a compile with errors produces no published
  revision at all.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Final

from dataenginex.foundation import (
    AuditEvent,
    AuditEventType,
    EventEnvelope,
    LifecycleState,
    MetadataEvent,
    PrincipalId,
    ProjectId,
    ProjectRevision,
    PublicationStatus,
    ResourceType,
    RevisionFile,
    RevisionId,
    ValidationReport,
    digest_bytes,
    new_id,
    utcnow,
)
from dataenginex.runtime.compiler.compiler import CompiledProject, compile_project
from dataenginex.runtime.state import ControlStore

__all__ = ["PublicationError", "RevisionService"]


class PublicationError(RuntimeError):
    """A revision could not be published.

    Carries the validation report so a caller can show the user every problem
    rather than a single message.
    """

    def __init__(self, message: str, report: ValidationReport | None = None) -> None:
        super().__init__(message)
        self.report = report or ValidationReport()


# Which connector kinds name something the catalogue models as a stream, a
# collection, or an endpoint rather than a plain dataset. Everything unlisted is
# a DATASET: the manifest's ``type`` selects a connector, and inferring a richer
# classification from it would be a guess presented as a fact.
_RESOURCE_TYPE_BY_CONNECTOR: Final[dict[str, ResourceType]] = {
    "kafka": ResourceType.EVENT_STREAM,
    "rabbitmq": ResourceType.EVENT_STREAM,
    "elasticsearch": ResourceType.DOCUMENT_COLLECTION,
    "qdrant": ResourceType.VECTOR_INDEX,
    "http": ResourceType.EXTERNAL_ENDPOINT,
    "graphql": ResourceType.EXTERNAL_ENDPOINT,
    "rest": ResourceType.EXTERNAL_ENDPOINT,
}


def _resource_type(connector: str) -> ResourceType:
    """The catalogue type a declared connector implies."""
    return _RESOURCE_TYPE_BY_CONNECTOR.get(connector.lower(), ResourceType.DATASET)


class RevisionService:
    """Creates drafts, publishes revisions, and moves the active pointer."""

    def __init__(self, store: ControlStore) -> None:
        self.store = store

    # --- draft --------------------------------------------------------------

    def create_draft(
        self,
        project_id: ProjectId,
        root: Path,
        principal_id: PrincipalId,
    ) -> tuple[ProjectRevision, CompiledProject]:
        """Compile a project directory into a draft revision.

        A draft is stored even when compilation fails, so the user can see the
        validation report attached to something concrete. It simply cannot be
        published — :meth:`publish` checks the report again.
        """
        compiled = compile_project(root)
        parent = self._active_revision_id(project_id)

        revision = ProjectRevision(
            revision_id=RevisionId(new_id("rev")),
            project_id=project_id,
            parent_revision_id=parent,
            content_hash=compiled.content_hash or "sha256:invalid",
            created_by=principal_id,
            manifest_schema_version=compiled.manifest.api_version,
            dependency_lock_hash=compiled.dependency_lock_hash,
            capability_requirements=compiled.required_capabilities,
            validation_report=compiled.report,
            status=PublicationStatus.DRAFT,
            files=self._collect_files(root, compiled),
        )

        with self.store.transaction() as tx:
            tx.execute(
                "INSERT INTO project_revisions (revision_id, project_id, "
                "parent_revision_id, content_hash, created_by, created_at, "
                "manifest_schema_version, dependency_lock_hash, "
                "capability_requirements_json, validation_report_json, status) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    revision.revision_id,
                    revision.project_id,
                    revision.parent_revision_id,
                    revision.content_hash,
                    revision.created_by,
                    revision.created_at.isoformat(),
                    revision.manifest_schema_version,
                    revision.dependency_lock_hash,
                    json.dumps(list(revision.capability_requirements)),
                    revision.validation_report.model_dump_json(),
                    revision.status.value,
                ),
            )
            for file in revision.files:
                tx.execute(
                    "INSERT INTO revision_files (revision_id, path, digest, "
                    "size_bytes, media_type) VALUES (?,?,?,?,?)",
                    (
                        revision.revision_id,
                        file.path,
                        file.digest,
                        file.size_bytes,
                        file.media_type,
                    ),
                )
            # The compiled workloads, without which a published revision is a
            # manifest nobody can run. Five call sites read this table — run
            # admission, scheduling, and the workload views — and until now
            # nothing wrote it, so every lookup silently fell back to a default:
            # every workload looked like a BATCH one with no operations.
            for workload in compiled.workloads:
                tx.execute(
                    "INSERT INTO workload_definitions (workload_id, project_id, "
                    "revision_id, name, kind, definition_json, continuous, created_at) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (
                        new_id("wl"),
                        project_id,
                        revision.revision_id,
                        workload.name,
                        workload.kind.value,
                        workload.model_dump_json(),
                        int(workload.continuous),
                        revision.created_at.isoformat(),
                    ),
                )
            # The declared resources, for the same reason. ``provider_facets``
            # carries the declaration's ``config`` verbatim — the core does not
            # interpret it (§4.6), but a handler cannot ingest from a source
            # whose path it was never told.
            for resource in compiled.resources:
                tx.execute(
                    "INSERT INTO resources (resource_id, project_id, revision_id, "
                    "resource_type, name, description, labels_json, classification, "
                    "lifecycle_state, created_at, facets_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        new_id("res"),
                        project_id,
                        revision.revision_id,
                        _resource_type(resource.type).value,
                        resource.name,
                        resource.description,
                        json.dumps(resource.labels),
                        resource.classification,
                        LifecycleState.ACTIVE.value,
                        revision.created_at.isoformat(),
                        # The declared ``type`` is the *connector* kind — ``csv``,
                        # ``postgres`` — not a ``ResourceType``. It was written to
                        # that column anyway, so reading a published resource
                        # raised "'csv' is not a valid ResourceType". It belongs
                        # beside the settings it selects.
                        json.dumps(
                            {"provider": dict(resource.config), "connector": resource.type}
                        ),
                    ),
                )

        return revision, compiled

    # --- publish ------------------------------------------------------------

    def publish(self, revision_id: RevisionId, principal_id: PrincipalId) -> ProjectRevision:
        """Publish a draft and make it active, atomically (§6.3).

        Everything happens in one transaction: the new revision becomes
        ``published``, the previous active one becomes ``superseded``, the
        project pointer moves, and the event is queued. A crash at any point
        leaves the previous revision active and untouched.
        """
        row = self.store.query_one(
            "SELECT * FROM project_revisions WHERE revision_id = ?", (revision_id,)
        )
        if row is None:
            raise PublicationError(f"revision {revision_id} does not exist")

        if row["status"] != PublicationStatus.DRAFT.value:
            raise PublicationError(
                f"revision {revision_id} has status {row['status']}, only drafts can be published"
            )


        report = ValidationReport.model_validate_json(row["validation_report_json"])
        if not report.ok:
            # Fail closed. This is the check the superseded engine skipped.
            raise PublicationError(
                f"revision {revision_id} has {len(report.errors)} validation "
                "error(s) and cannot be published",
                report,
            )

        project_id = ProjectId(row["project_id"])
        previous = self._active_revision_id(project_id)

        with self.store.transaction() as tx:
            if previous is not None:
                tx.execute(
                    "UPDATE project_revisions SET status = ? WHERE revision_id = ?",
                    (PublicationStatus.SUPERSEDED.value, previous),
                )
            tx.execute(
                "UPDATE project_revisions SET status = ? WHERE revision_id = ?",
                (PublicationStatus.PUBLISHED.value, revision_id),
            )
            # The atomic pointer flip (§6.3).
            tx.execute(
                "UPDATE projects SET active_revision_id = ? WHERE project_id = ?",
                (revision_id, project_id),
            )
            tx.emit_metadata(
                MetadataEvent(
                    envelope=EventEnvelope(
                        producer="revision-service",
                        project_id=project_id,
                        revision_id=revision_id,
                        principal_id=principal_id,
                        occurred_at=utcnow(),
                    ),
                    event_type="ProjectRevisionPublished",
                    subject_id=revision_id,
                    subject_type="project_revision",
                    payload={
                        "content_hash": row["content_hash"],
                        "parent_revision_id": row["parent_revision_id"],
                        "superseded": previous,
                    },
                )
            )

        return self.get(revision_id)

    # --- rollback -----------------------------------------------------------

    def rollback(
        self,
        project_id: ProjectId,
        target_revision_id: RevisionId,
        principal_id: PrincipalId,
    ) -> ProjectRevision:
        """Re-point the project at an earlier revision (§6.3 item 10).

        History is not mutated: the revision being rolled away from stays in the
        table with its own record. Only the pointer and the two statuses move.

        Recorded as an audit event, not just metadata — reverting the definition
        of a project is a security-relevant administrative action.
        """
        row = self.store.query_one(
            "SELECT * FROM project_revisions WHERE revision_id = ? AND project_id = ?",
            (target_revision_id, project_id),
        )
        if row is None:
            raise PublicationError(
                f"revision {target_revision_id} does not belong to project {project_id}"
            )
        if row["status"] == PublicationStatus.DRAFT.value:
            raise PublicationError(
                "cannot roll back to a draft; only previously published revisions "
                "are valid rollback targets"
            )

        current = self._active_revision_id(project_id)
        if current == target_revision_id:
            raise PublicationError(f"revision {target_revision_id} is already active")

        with self.store.transaction() as tx:
            if current is not None:
                tx.execute(
                    "UPDATE project_revisions SET status = ? WHERE revision_id = ?",
                    (PublicationStatus.SUPERSEDED.value, current),
                )
            tx.execute(
                "UPDATE project_revisions SET status = ? WHERE revision_id = ?",
                (PublicationStatus.PUBLISHED.value, target_revision_id),
            )
            tx.execute(
                "UPDATE projects SET active_revision_id = ? WHERE project_id = ?",
                (target_revision_id, project_id),
            )
            tx.emit_audit(
                AuditEvent(
                    envelope=EventEnvelope(
                        producer="revision-service",
                        project_id=project_id,
                        revision_id=target_revision_id,
                        principal_id=principal_id,
                    ),
                    event_type=AuditEventType.POLICY_CHANGE,
                    action="project.rollback",
                    outcome="succeeded",
                    target_id=target_revision_id,
                    target_type="project_revision",
                    detail={"from": current or "", "to": target_revision_id},
                )
            )

        return self.get(target_revision_id)

    # --- reads --------------------------------------------------------------

    def get(self, revision_id: RevisionId) -> ProjectRevision:
        row = self.store.query_one(
            "SELECT * FROM project_revisions WHERE revision_id = ?", (revision_id,)
        )
        if row is None:
            raise PublicationError(f"revision {revision_id} does not exist")

        files = self.store.query(
            "SELECT path, digest, size_bytes, media_type FROM revision_files "
            "WHERE revision_id = ? ORDER BY path",
            (revision_id,),
        )
        return ProjectRevision(
            revision_id=RevisionId(row["revision_id"]),
            project_id=ProjectId(row["project_id"]),
            parent_revision_id=(
                RevisionId(row["parent_revision_id"]) if row["parent_revision_id"] else None
            ),
            content_hash=row["content_hash"],
            created_by=PrincipalId(row["created_by"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            manifest_schema_version=row["manifest_schema_version"],
            dependency_lock_hash=row["dependency_lock_hash"],
            capability_requirements=tuple(_string_list(row["capability_requirements_json"])),
            validation_report=ValidationReport.model_validate_json(row["validation_report_json"]),
            status=PublicationStatus(row["status"]),
            files=tuple(
                RevisionFile(
                    path=f["path"],
                    digest=f["digest"],
                    size_bytes=int(f["size_bytes"]),
                    media_type=f["media_type"],
                )
                for f in files
            ),
        )

    def history(self, project_id: ProjectId) -> tuple[ProjectRevision, ...]:
        """Every revision of a project, newest first."""
        rows = self.store.query(
            "SELECT revision_id FROM project_revisions WHERE project_id = ? "
            "ORDER BY created_at DESC",
            (project_id,),
        )
        return tuple(self.get(RevisionId(r["revision_id"])) for r in rows)

    def active(self, project_id: ProjectId) -> ProjectRevision | None:
        revision_id = self._active_revision_id(project_id)
        return self.get(revision_id) if revision_id else None

    # --- internals ----------------------------------------------------------

    def _active_revision_id(self, project_id: ProjectId) -> RevisionId | None:
        row = self.store.query_one(
            "SELECT active_revision_id FROM projects WHERE project_id = ?",
            (project_id,),
        )
        if row is None or row["active_revision_id"] is None:
            return None
        return RevisionId(row["active_revision_id"])

    def _collect_files(self, root: Path, compiled: CompiledProject) -> tuple[RevisionFile, ...]:
        """Content-address every source file the compile consumed.

        Only files the compiler actually read are captured. A revision must
        describe exactly what it was built from — sweeping in the whole
        directory would hash editor backups and caches into project identity.
        """
        files: list[RevisionFile] = []
        for source in compiled.source_files:
            path = Path(source)
            if not path.is_file():
                continue
            data = path.read_bytes()
            try:
                relative = path.relative_to(root)
            except ValueError:
                relative = path
            files.append(
                RevisionFile(
                    path=str(relative),
                    digest=str(digest_bytes(data)),
                    size_bytes=len(data),
                    media_type=(
                        "application/yaml"
                        if path.suffix in (".yaml", ".yml")
                        else "application/octet-stream"
                    ),
                )
            )
        return tuple(sorted(files, key=lambda f: f.path))


def _string_list(value: str) -> list[str]:
    parsed = json.loads(value)
    return [str(item) for item in parsed] if isinstance(parsed, list) else []
