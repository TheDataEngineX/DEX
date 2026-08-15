"""OpenLineage projection for Spark (§20.11).

Integrates with OpenLineage for data lineage tracking.
Provides Spark-specific lineage event generation.
"""

from __future__ import annotations

from datetime import UTC
from typing import Any

import structlog

from dataenginex.foundation.ids import ProjectId

logger = structlog.get_logger()

__all__ = ["OpenLineageProjection"]

try:
    import httpx

    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False


class OpenLineageProjection:
    """OpenLineage integration for Spark lineage tracking (§20.11).

    Provides:
    - Run event generation
    - Dataset lineage tracking
    - Job dependency mapping
    - Facet generation
    """

    def __init__(
        self,
        project_id: ProjectId,
        openlineage_url: str = "http://localhost:5001",
        api_key: str | None = None,
    ) -> None:
        self.project_id = project_id
        self.openlineage_url = openlineage_url.rstrip("/")
        self.api_key = api_key
        self._client: Any | None = None

    def connect(self) -> None:
        """Initialize OpenLineage client."""
        if _HTTPX_AVAILABLE:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            self._client = httpx.Client(
                base_url=self.openlineage_url,
                headers=headers,
                timeout=30.0,
            )
            logger.info(
                "openlineage client connected",
                url=self.openlineage_url,
            )

    def disconnect(self) -> None:
        """Close OpenLineage client."""
        if self._client:
            self._client.close()
            self._client = None

    def create_run_event(
        self,
        run_id: str,
        job_name: str,
        event_type: str = "START",
        inputs: list[dict] | None = None,
        outputs: list[dict] | None = None,
    ) -> dict:
        """Create an OpenLineage run event.

        Args:
            run_id: Run identifier
            job_name: Job name
            event_type: Event type (START, COMPLETE, FAIL)
            inputs: Input datasets
            outputs: Output datasets

        Returns:
            OpenLineage event
        """
        event = {
            "eventType": event_type,
            "eventTime": self._get_timestamp(),
            "run": {
                "runId": run_id,
                "facets": {
                    "processing_engine": {
                        "name": "spark",
                        "version": "4.2.0",
                    },
                },
            },
            "job": {
                "namespace": f"dex-{self.project_id}",
                "name": job_name,
                "facets": {},
            },
            "inputs": inputs or [],
            "outputs": outputs or [],
            "producer": "dex-dataenginex",
            "schemaURL": "https://openlineage.io/spec/1-0-5/OpenLineage.json",
        }

        # Send event
        self._send_event(event)

        return event

    def create_dataset(
        self,
        name: str,
        namespace: str | None = None,
        dataset_type: str = "table",
        schema: list[dict] | None = None,
        facets: dict | None = None,
    ) -> dict:
        """Create an OpenLineage dataset descriptor.

        Args:
            name: Dataset name
            namespace: Dataset namespace
            dataset_type: Dataset type (table, file, etc.)
            schema: Dataset schema
            facets: Additional facets

        Returns:
            OpenLineage dataset
        """
        return {
            "namespace": namespace or f"dex-{self.project_id}",
            "name": name,
            "facets": {
                "datasetType": dataset_type,
                **(facets or {}),
            },
        }

    def create_input_dataset(
        self,
        name: str,
        source_type: str = "delta",
        path: str | None = None,
    ) -> dict:
        """Create an input dataset.

        Args:
            name: Dataset name
            source_type: Source type (delta, parquet, csv, etc.)
            path: Optional path for file-based datasets

        Returns:
            Input dataset
        """
        facets = {"sourceType": source_type}
        if path:
            facets["path"] = path

        return self.create_dataset(name, facets=facets)

    def create_output_dataset(
        self,
        name: str,
        sink_type: str = "delta",
        path: str | None = None,
    ) -> dict:
        """Create an output dataset.

        Args:
            name: Dataset name
            sink_type: Sink type (delta, parquet, etc.)
            path: Optional path for file-based datasets

        Returns:
            Output dataset
        """
        facets = {"sinkType": sink_type}
        if path:
            facets["path"] = path

        return self.create_dataset(name, facets=facets)

    def create_schema_facet(self, fields: list[dict]) -> dict:
        """Create a schema facet.

        Args:
            fields: List of field definitions

        Returns:
            Schema facet
        """
        return {
            "fields": [
                {
                    "name": f["name"],
                    "type": f.get("type", "string"),
                    "description": f.get("description", ""),
                }
                for f in fields
            ]
        }

    def create_lineage_facet(self, inputs: list[str], outputs: list[str]) -> dict:
        """Create a lineage facet showing input-output relationships.

        Args:
            inputs: Input dataset names
            outputs: Output dataset names

        Returns:
            Lineage facet
        """
        return {
            "inputs": inputs,
            "outputs": outputs,
        }

    def _send_event(self, event: dict) -> None:
        """Send event to OpenLineage API.

        Args:
            event: OpenLineage event
        """
        if self._client is None:
            logger.warning("openlineage client not connected, event not sent")
            return

        try:
            response = self._client.post("/api/v1/lineage", json=event)
            response.raise_for_status()
            logger.info(
                "openlineage event sent",
                event_type=event.get("eventType"),
                job_name=event.get("job", {}).get("name"),
            )
        except Exception as exc:
            logger.error("failed to send openlineage event", error=str(exc))

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime

        return datetime.now(UTC).isoformat()

    def get_lineage(
        self,
        dataset_name: str | None = None,
        job_name: str | None = None,
        depth: int = 3,
    ) -> dict:
        """Query lineage from OpenLineage.

        Args:
            dataset_name: Dataset name to query
            job_name: Job name to query
            depth: Lineage depth

        Returns:
            Lineage data
        """
        if self._client is None:
            return {"error": "OpenLineage client not connected"}

        try:
            params = {"depth": depth}
            if dataset_name:
                params["dataset"] = dataset_name
            if job_name:
                params["job"] = job_name

            response = self._client.get("/api/v1/lineage", params=params)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            logger.error("failed to get lineage", error=str(exc))
            return {"error": str(exc)}
