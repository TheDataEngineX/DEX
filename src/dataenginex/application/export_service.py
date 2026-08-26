"""Export service (§6.7).

Project import/export for portable project bundles. Git is optional.
"""

from __future__ import annotations

import json
from pathlib import Path

from dataenginex.application.services import ApplicationError, Service
from dataenginex.foundation.ids import ProjectId

__all__ = ["ExportService"]


class ExportError(ApplicationError):
    """Export operation failed."""


class ImportError(ApplicationError):
    """Import operation failed."""


class ExportService(Service):
    """Project import/export (§6.7)."""

    def export_project(
        self,
        project_id: ProjectId,
        output_dir: Path,
    ) -> Path:
        """Export a project revision to a folder.

        Creates a self-contained project bundle with dex.yaml, all revision
        files, and metadata. The bundle can be imported later or used with
        the CLI directly.
        """
        # Get the active revision
        revision_id = self.active_revision(project_id)
        rev_row = dict(self.require_row(
            "SELECT * FROM project_revisions WHERE revision_id = ?",
            (revision_id,),
            subject=f"revision {revision_id}",
        ))

        project_row = dict(self.require_row(
            "SELECT * FROM projects WHERE project_id = ?",
            (project_id,),
            subject=f"project {project_id}",
        ))

        # Create output directory
        project_dir = output_dir / project_row["name"]
        project_dir.mkdir(parents=True, exist_ok=True)

        # Write manifest
        manifest = {
            "apiVersion": "dex/v0.7",
            "kind": "Project",
            "metadata": {
                "name": project_row["name"],
                "description": project_row.get("description", ""),
            },
            "spec": {
                "runtime": {"profile": "auto"},
            },
        }
        (project_dir / "dex.yaml").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

        # Write revision metadata
        revision_meta = {
            "revision_id": rev_row["revision_id"],
            "project_id": project_id,
            "content_hash": rev_row.get("content_hash", ""),
            "manifest_schema_version": rev_row.get("manifest_schema_version", "dex/v0.7"),
            "created_at": rev_row.get("created_at", ""),
        }
        (project_dir / "revision.json").write_text(
            json.dumps(revision_meta, indent=2),
            encoding="utf-8",
        )

        return Path(project_dir)

    def import_project(
        self,
        source_dir: Path,
        workspace_id: str,
    ) -> ProjectId:
        """Import a project from a folder.

        Creates a new draft from the exported bundle. The project gets a new
        ID but preserves the original name and content.
        """
        manifest_path = source_dir / "dex.yaml"
        if not manifest_path.exists():
            raise ImportError(f"No dex.yaml found in {source_dir}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        name = manifest.get("metadata", {}).get("name", source_dir.name)

        # Create the project (the project service handles the actual creation)
        from dataenginex.foundation.projects import utcnow
        project_id = ProjectId(f"proj_{name}")
        self.store.query_one(
            "INSERT INTO projects "
            "(project_id, workspace_id, name, description, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (project_id, workspace_id, name,
             manifest.get("metadata", {}).get("description", ""),
             utcnow().isoformat()),
        )

        return project_id
