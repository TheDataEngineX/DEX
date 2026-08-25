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

    def __init__(self, row: dict[str, Any]) -> None:
        self.resource_id = row["resource_id"]
        self.name = row["name"]
        self.resource_type = row["resource_type"]
        self.project_id = row["project_id"]
        self.classification = row.get("classification", "internal")
        self.lifecycle_state = row.get("lifecycle_state", "active")
        self.provider = row.get("provider")
        self.description = row.get("description", "")


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
        resource_id = ResourceId(f"res_{name}")
        self.store.query_one(
            "INSERT INTO catalog_entries "
            "(resource_id, name, resource_type, project_id, classification, "
            "provider, description, lifecycle_state) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (resource_id, name, resource_type, project_id, classification,
             provider, description, LifecycleState.ACTIVE.value),
        )
        return resource_id

    def get_resource(self, resource_id: ResourceId) -> CatalogEntry:
        """Fetch a resource by ID."""
        row = self.require_row(
            "SELECT * FROM catalog_entries WHERE resource_id = ?",
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
        sql = f"SELECT * FROM catalog_entries {where} ORDER BY name LIMIT ?"
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
        sql = f"UPDATE catalog_entries SET {', '.join(updates)} WHERE resource_id = ?"
        self.store.query_one(sql, tuple(params))

    def delete_resource(self, resource_id: ResourceId) -> None:
        """Soft-delete a resource (sets lifecycle to deleted)."""
        self.update_resource(resource_id, lifecycle_state=LifecycleState.DELETED.value)
