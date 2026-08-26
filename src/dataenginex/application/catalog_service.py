"""Catalog service — DEX Resource Catalog (§8.2).

Manages cross-domain resources: datasets, models, prompts, pipelines,
dashboards, connections, and other governed resources.
"""

from __future__ import annotations

from typing import Any

from dataenginex.application.services import Service
from dataenginex.foundation.ids import ProjectId, ResourceId
from dataenginex.foundation.resources import LifecycleState

__all__ = ["CatalogService", "CatalogEntry"]


class CatalogEntry:
    """Read-only projection of a catalog resource."""

    def __init__(self, row: Any) -> None:
        d = dict(row)
        self.resource_id = d["resource_id"]
        self.name = d["name"]
        self.resource_type = d["resource_type"]
        self.project_id = d["project_id"]
        self.classification = d.get("classification", "internal")
        self.lifecycle_state = d.get("lifecycle_state", "active")
        self.description = d.get("description", "")


class CatalogService(Service):
    """DEX Resource Catalog operations (§8.2)."""

    def register_resource(
        self,
        *,
        name: str,
        resource_type: str,
        project_id: ProjectId,
        classification: str = "internal",
        provider: str | None = None,
        description: str = "",
        schema_info: dict[str, Any] | None = None,
    ) -> ResourceId:
        """Register a new resource in the catalog."""
        from dataenginex.foundation.projects import utcnow
        resource_id = ResourceId(f"res_{name}")
        self.store.query_one(
            "INSERT INTO resources "
            "(resource_id, project_id, revision_id, resource_type, name, description, "
            "classification, lifecycle_state, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (resource_id, project_id, "draft", resource_type, name, description,
             classification, LifecycleState.ACTIVE.value, utcnow().isoformat()),
        )
        return resource_id

    def get_resource(self, resource_id: ResourceId) -> CatalogEntry:
        """Fetch a resource by ID."""
        row = self.require_row(
            "SELECT * FROM resources WHERE resource_id = ?",
            (resource_id,),
            subject=f"resource {resource_id}",
        )
        return CatalogEntry(row)

    def search_resources(
        self,
        *,
        project_id: ProjectId | None = None,
        resource_type: str | None = None,
        classification: str | None = None,
        limit: int = 50,
    ) -> list[CatalogEntry]:
        """Search resources with optional filters."""
        conditions = []
        params: list[Any] = []
        if project_id:
            conditions.append("project_id = ?")
            params.append(project_id)
        if resource_type:
            conditions.append("resource_type = ?")
            params.append(resource_type)
        if classification:
            conditions.append("classification = ?")
            params.append(classification)

        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        sql = f"SELECT * FROM resources {where} ORDER BY name LIMIT ?"
        params.append(limit)
        rows = self.store.query(sql, tuple(params))
        return [CatalogEntry(dict(r)) for r in rows]

    def list_resources(self, project_id: ProjectId) -> list[CatalogEntry]:
        """List all resources for a project."""
        return self.search_resources(project_id=project_id)

    def update_resource(
        self,
        resource_id: ResourceId,
        *,
        classification: str | None = None,
        lifecycle_state: str | None = None,
        description: str | None = None,
    ) -> None:
        """Update mutable resource metadata."""
        updates = []
        params: list[Any] = []
        if classification is not None:
            updates.append("classification = ?")
            params.append(classification)
        if lifecycle_state is not None:
            updates.append("lifecycle_state = ?")
            params.append(lifecycle_state)
        if description is not None:
            updates.append("description = ?")
            params.append(description)
        if not updates:
            return
        params.append(resource_id)
        sql = f"UPDATE resources SET {', '.join(updates)} WHERE resource_id = ?"
        self.store.query_one(sql, tuple(params))

    def delete_resource(self, resource_id: ResourceId) -> None:
        """Soft-delete a resource (sets lifecycle to deleted)."""
        self.update_resource(resource_id, lifecycle_state=LifecycleState.DELETED.value)
