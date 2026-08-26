"""Project and revision services (§6.3, §6.8).

The draft/publish cycle lives here. It is the single most important behavioural
difference between the old design and this one, so it is worth stating plainly:

**Configuration is immutable.** The old engine edited ``dex.yaml`` in place and
left a ``.yaml.bak`` beside it. That makes "what did this run actually execute?"
unanswerable the moment someone saves twice, and it makes rollback a file-copy
with no record. Here a change is a *new revision*: compiled, validated, and
published, with the active pointer flipped atomically. Rollback re-points at an
earlier revision without mutating history.

**Publishing can fail, and that is a feature.** A draft that does not compile is
refused with its validation report intact, so a caller can show the user exactly
which stage rejected it. The old path discarded ``validate_config``'s errors and
published anyway.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dataenginex.application.services import ApplicationError, NotFoundError, Service
from dataenginex.foundation import (
    FrozenModel,
    PrincipalId,
    ProjectId,
    PublicationStatus,
    RevisionId,
    ValidationReport,
    ValidationSeverity,
    new_id,
    utcnow,
)
from dataenginex.runtime.compiler import ProjectCompiler
from dataenginex.runtime.compiler.revisions import RevisionService

__all__ = ["ProjectService", "ProjectView", "PublishRejected", "RevisionSummary"]


class ProjectView(FrozenModel):
    """A project and its active revision, as clients see it.

    ``active_revision_id`` being ``None`` is the meaningful state: a project
    exists as soon as it is created, but nothing can run against it until a
    revision is published (§6.3).
    """

    project_id: ProjectId
    workspace_id: str
    name: str
    active_revision_id: RevisionId | None = None
    content_hash: str | None = None
    created_at: str = ""


class PublishRejected(ApplicationError):
    """A draft did not compile cleanly and was not published.

    Carries the report rather than a message so the caller can render each issue
    against the manifest location that produced it. A publish failure that
    reduces to a string forces the UI to re-parse prose.
    """

    def __init__(self, report: ValidationReport) -> None:
        errors = [i for i in report.issues if i.severity is ValidationSeverity.ERROR]
        super().__init__(f"{len(errors)} error(s) blocked publication")
        self.report = report


class RevisionSummary(FrozenModel):
    """The client-facing view of a revision.

    A projection, not the stored row: clients need identity, status, and
    provenance, and should not couple to control-plane columns that may change.
    """

    revision_id: RevisionId
    project_id: ProjectId
    content_hash: str
    status: PublicationStatus
    created_by: PrincipalId
    created_at: str
    dependency_lock_hash: str | None = None
    is_active: bool = False


class ProjectService(Service):
    """Projects and their immutable revisions (§6.3)."""

    # --- queries ------------------------------------------------------------

    def get_project(self, project_id: ProjectId) -> ProjectView:
        """A project with the revision it is currently serving.

        The join is here rather than in two calls because every caller that
        wants the project also wants to know whether it is publishable, and
        splitting it makes that two round trips on every page load.
        """
        row = self.require_row(
            "SELECT p.*, r.content_hash FROM projects p "
            "LEFT JOIN project_revisions r ON r.revision_id = p.active_revision_id "
            "WHERE p.project_id = ?",
            (project_id,),
            subject=f"no project {project_id}",
        )
        return _row_to_project(row)

    def list_projects(self, *, limit: int = 200) -> list[ProjectView]:
        """Every project, newest first.

        Unfiltered by workspace for now: Lite has exactly one (§11.3). The
        argument belongs on this method when the server profile makes a second
        one reachable, which is a signature change, not a rewrite.
        """
        rows = self.store.query(
            "SELECT p.*, r.content_hash FROM projects p "
            "LEFT JOIN project_revisions r ON r.revision_id = p.active_revision_id "
            "ORDER BY p.created_at DESC LIMIT ?",
            (limit,),
        )
        return [_row_to_project(row) for row in rows]

    def get_revision(self, revision_id: RevisionId) -> RevisionSummary:
        row = self.require_row(
            "SELECT r.*, p.active_revision_id FROM project_revisions r "
            "JOIN projects p ON p.project_id = r.project_id WHERE r.revision_id = ?",
            (revision_id,),
            subject=f"no revision {revision_id}",
        )
        return _row_to_revision(row)

    def active_revision_summary(self, project_id: ProjectId) -> RevisionSummary:
        """The revision a run would pin right now."""
        return self.get_revision(self.active_revision(project_id))

    def list_revisions(self, project_id: ProjectId, *, limit: int = 50) -> list[RevisionSummary]:
        """Revision history, newest first.

        This is what makes rollback a real operation rather than a restore from
        backup: every previously published definition is still addressable.
        """
        rows = self.store.query(
            "SELECT r.*, p.active_revision_id FROM project_revisions r "
            "JOIN projects p ON p.project_id = r.project_id "
            "WHERE r.project_id = ? ORDER BY r.created_at DESC LIMIT ?",
            (project_id, limit),
        )
        return [_row_to_revision(row) for row in rows]

    # --- commands -----------------------------------------------------------

    def ensure_project(self, name: str, *, description: str = "") -> ProjectId:
        """Find or create the project called *name*, with its enclosing scopes.

        Nothing created installations, workspaces, or projects outside test
        fixtures, so ``publish`` — which requires the row to exist — could never
        succeed on a real install. A project has to exist before it can have a
        revision, and creating it was the step with no owner.

        Idempotent by name: a Studio restart, or a second process opening the
        same store, must land on the same project rather than forking a parallel
        one that shares the manifest but not the run history.

        The default installation and workspace are created on demand. §4.2–4.4
        makes them real scopes, but a single-user install has exactly one of
        each, and making the user name them before they can open a project is
        ceremony that answers no question.
        """
        existing = self.store.query_one("SELECT project_id FROM projects WHERE name = ?", (name,))
        if existing is not None:
            return ProjectId(existing["project_id"])

        now = utcnow().isoformat()
        with self.store.transaction() as tx:
            installation = tx.execute(
                "SELECT installation_id FROM installations LIMIT 1"
            ).fetchone()
            installation_id = installation["installation_id"] if installation else new_id("inst")
            if installation is None:
                tx.execute(
                    "INSERT INTO installations (installation_id, name, created_at) "
                    "VALUES (?, 'default', ?)",
                    (installation_id, now),
                )

            workspace = tx.execute(
                "SELECT workspace_id FROM workspaces WHERE installation_id = ? LIMIT 1",
                (installation_id,),
            ).fetchone()
            workspace_id = workspace["workspace_id"] if workspace else new_id("ws")
            if workspace is None:
                tx.execute(
                    "INSERT INTO workspaces (workspace_id, installation_id, name, created_at) "
                    "VALUES (?, ?, 'default', ?)",
                    (workspace_id, installation_id, now),
                )

            project_id = ProjectId(new_id("proj"))
            tx.execute(
                "INSERT INTO projects (project_id, workspace_id, name, description, created_at) "
                "VALUES (?,?,?,?,?)",
                (project_id, workspace_id, name, description, now),
            )
        return project_id

    def publish(
        self,
        project_id: ProjectId,
        source_root: Path,
        *,
        principal_id: PrincipalId,
    ) -> RevisionSummary:
        """Compile a project directory and publish it as the active revision.

        Delegates to :class:`RevisionService`, which is the one implementation
        of §6.3. This method used to write ``project_revisions`` itself and stop
        there — no ``resources``, no ``workload_definitions`` — so a revision
        published through the gateway compiled cleanly and then answered every
        "what does this project contain?" query with nothing. Two publish paths
        that disagreed about what publishing means is how that survived: the
        complete one had only tests as callers.

        Compilation happens *before* anything is written. A draft that fails
        validation leaves no published revision and the previous one still
        serving, so a bad edit cannot take a working project down.
        """
        if self.store.query_one(
            "SELECT project_id FROM projects WHERE project_id = ?", (project_id,)
        ) is None:
            raise NotFoundError(f"no project {project_id}")

        result = ProjectCompiler(source_root).compile()
        if not result.ok:
            raise PublishRejected(result.report)

        # Checked before drafting, not after: ``project_revisions`` is unique on
        # (project_id, content_hash), so drafting identical content again would
        # raise on the insert rather than reach a duplicate check below. A
        # repeated publish — a Studio restart, a second process opening the same
        # project — has to be idempotent.
        existing = self.store.query_one(
            "SELECT revision_id FROM project_revisions "
            "WHERE project_id = ? AND content_hash = ?",
            (project_id, result.content_hash),
        )
        if existing is not None:
            self._activate(project_id, RevisionId(existing["revision_id"]))
            return self.get_revision(RevisionId(existing["revision_id"]))

        revisions = RevisionService(self.store)
        draft, _ = revisions.create_draft(project_id, source_root, principal_id)
        revisions.publish(draft.revision_id, principal_id)
        return self.get_revision(draft.revision_id)

    def rollback(self, project_id: ProjectId, revision_id: RevisionId) -> RevisionSummary:
        """Re-point the project at an earlier revision (§6.3).

        History is not mutated and the abandoned revision is not deleted —
        rolling forward again must remain possible, and an incident review needs
        to see the revision that caused it.
        """
        row = self.store.query_one(
            "SELECT status FROM project_revisions WHERE revision_id = ? AND project_id = ?",
            (revision_id, project_id),
        )
        if row is None:
            raise NotFoundError(f"revision {revision_id} does not belong to {project_id}")
        # Superseded counts: it is what every revision becomes once a newer one
        # is published, so the revision a user rolls *back* to is superseded by
        # definition. Requiring ``published`` made rollback impossible — the
        # only revision with that status is the active one.
        #
        # Draft does not count. Rolling onto one would put content into
        # production by a path that never finished the compiler.
        if row["status"] not in (
            PublicationStatus.PUBLISHED.value,
            PublicationStatus.SUPERSEDED.value,
        ):
            raise ApplicationError(
                f"revision {revision_id} is {row['status']}, only a published or "
                "superseded revision can be activated"
            )

        self._activate(project_id, revision_id)
        return self.get_revision(revision_id)

    def _activate(self, project_id: ProjectId, revision_id: RevisionId) -> None:
        """Point the project at *revision_id* and make the statuses agree.

        Moving the pointer alone leaves the store describing two truths: the
        newly active revision still marked ``superseded``, and the one it
        replaced still marked ``published``. Both updates go in one transaction
        with the pointer, so no reader sees a project whose active revision
        claims to be superseded.
        """
        with self.store.transaction() as tx:
            tx.execute(
                "UPDATE project_revisions SET status = ? WHERE project_id = ? "
                "AND status = ? AND revision_id != ?",
                (
                    PublicationStatus.SUPERSEDED.value,
                    project_id,
                    PublicationStatus.PUBLISHED.value,
                    revision_id,
                ),
            )
            tx.execute(
                "UPDATE project_revisions SET status = ? WHERE revision_id = ?",
                (PublicationStatus.PUBLISHED.value, revision_id),
            )
            tx.execute(
                "UPDATE projects SET active_revision_id = ? WHERE project_id = ?",
                (revision_id, project_id),
            )


def _row_to_project(row: Any) -> ProjectView:
    data = dict(row)
    active = data.get("active_revision_id")
    return ProjectView(
        project_id=ProjectId(data["project_id"]),
        workspace_id=data["workspace_id"],
        name=data["name"],
        active_revision_id=RevisionId(active) if active else None,
        content_hash=data.get("content_hash"),
        created_at=data["created_at"],
    )


def _row_to_revision(row: Any) -> RevisionSummary:
    data = dict(row)
    return RevisionSummary(
        revision_id=RevisionId(data["revision_id"]),
        project_id=ProjectId(data["project_id"]),
        content_hash=data["content_hash"],
        status=PublicationStatus(data["status"]),
        created_by=PrincipalId(data["created_by"]),
        created_at=data["created_at"],
        dependency_lock_hash=data["dependency_lock_hash"],
        is_active=data.get("active_revision_id") == data["revision_id"],
    )
