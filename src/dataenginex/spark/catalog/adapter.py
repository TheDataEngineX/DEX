"""Spark catalog adapter (§20.5).

Maps Spark catalog semantics to DEX Resource Catalog entries.
Supports Hive Metastore and Unity Catalog.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import structlog

from dataenginex.spark.catalog.identifiers import SparkCatalogIdentifier

logger = structlog.get_logger()

__all__ = ["SparkCatalogAdapter"]


class SparkCatalogAdapter:
    """Maps Spark catalog to DEX resource entries (§20.5).

    Provides unified catalog access across:
    - Hive Metastore
    - Unity Catalog
    - File-based catalog (local development)
    """

    def __init__(
        self,
        catalog_type: str = "hive",
        connection_config: dict[str, Any] | None = None,
    ) -> None:
        self.catalog_type = catalog_type
        self.connection_config = connection_config or {}
        self._client: Any | None = None
        # Set eagerly (not just in _connect_file_catalog) so list_databases/list_tables
        # don't raise AttributeError if called before connect().
        self._lakehouse_root = Path(self.connection_config.get("lakehouse_path", ".dex/lakehouse"))

    def connect(self) -> None:
        """Establish connection to catalog service."""
        if self.catalog_type == "hive":
            self._connect_hive_metastore()
        elif self.catalog_type == "unity":
            self._connect_unity_catalog()
        elif self.catalog_type == "file":
            self._connect_file_catalog()
        else:
            msg = f"Unsupported catalog type: {self.catalog_type}"
            raise ValueError(msg)

    def _connect_hive_metastore(self) -> None:
        """Connect to Hive Metastore via Thrift."""
        try:
            from pyhive import hive

            host = self.connection_config.get("host", "localhost")
            port = self.connection_config.get("port", 9083)

            self._client = hive.connect(host=host, port=port)
            logger.info("connected to hive metastore", host=host, port=port)
        except ImportError:
            logger.warning("pyhive not available, using mock catalog")
            self._client = None
        except Exception as exc:
            logger.error("hive metastore connection failed", error=str(exc))
            raise

    def _connect_unity_catalog(self) -> None:
        """Connect to Unity Catalog via REST API."""
        # Unity Catalog integration would go here
        logger.info("unity catalog integration not yet implemented")
        self._client = None

    def _connect_file_catalog(self) -> None:
        """Connect to file-based catalog — scans .dex/lakehouse/ for Delta tables."""
        lakehouse_path = self.connection_config.get("lakehouse_path", ".dex/lakehouse")
        self._lakehouse_root = Path(lakehouse_path)
        self._client = None  # no network client for file mode — real lookups go via _lakehouse_root
        logger.info("using file-based catalog", lakehouse_path=str(self._lakehouse_root))

    def disconnect(self) -> None:
        """Disconnect from catalog service."""
        if self._client is not None:
            with contextlib.suppress(Exception):
                self._client.close()
            self._client = None

    def spark_table_to_resource(self, identifier: SparkCatalogIdentifier) -> dict[str, Any]:
        """Convert a Spark table identifier to a DEX resource entry.

        Args:
            identifier: Spark catalog identifier

        Returns:
            DEX resource entry dict
        """
        # Get table metadata from catalog
        metadata = self._get_table_metadata(identifier)

        resource = {
            "name": identifier.table,
            "resource_type": "table",
            "spark_identifier": identifier.to_spark_sql(),
            "provider": f"spark.catalog.{self.catalog_type}",
            "catalog": identifier.catalog,
            "namespace": identifier.namespace,
            "table": identifier.table,
        }

        # Add metadata if available
        if metadata:
            resource.update(
                {
                    "columns": metadata.get("columns", []),
                    "partition_columns": metadata.get("partition_columns", []),
                    "table_type": metadata.get("table_type", "MANAGED"),
                    "location": metadata.get("location", ""),
                    "format": metadata.get("format", "parquet"),
                    "owner": metadata.get("owner", ""),
                    "created_at": metadata.get("created_at", ""),
                    "last_accessed": metadata.get("last_accessed", ""),
                }
            )

        return resource

    def list_databases(self, catalog: str | None = None) -> list[dict[str, Any]]:
        """List all databases in a catalog.

        Args:
            catalog: Optional catalog name (uses default if not specified)

        Returns:
            List of database dicts
        """
        if self._client:
            return self._list_databases_from_hive(catalog)

        if self.catalog_type != "file" or not self._lakehouse_root.exists():
            return []
        return [
            {"name": layer_dir.name, "description": f"{layer_dir.name} layer tables"}
            for layer_dir in sorted(self._lakehouse_root.iterdir())
            if layer_dir.is_dir()
        ]

    def list_tables(self, database: str = "default") -> list[dict[str, Any]]:
        """List all tables in a database.

        Args:
            database: Database name

        Returns:
            List of table dicts
        """
        if self._client:
            return self._list_tables_from_hive(database)

        if self.catalog_type != "file":
            return []
        layer_dir = self._lakehouse_root / database
        if not layer_dir.exists():
            return []
        tables = []
        for table_dir in sorted(layer_dir.iterdir()):
            if table_dir.is_dir() and (table_dir / "_delta_log").exists():
                tables.append({"name": table_dir.name, "type": "MANAGED", "format": "delta"})
        return tables

    def get_table_schema(self, identifier: SparkCatalogIdentifier) -> dict[str, Any]:
        """Get table schema.

        Args:
            identifier: Table identifier

        Returns:
            Table schema with columns and types
        """
        if self._client:
            return self._get_table_schema_from_hive(identifier)

        if self.catalog_type != "file":
            return {"columns": [], "partition_columns": []}
        table_dir = self._lakehouse_root / identifier.namespace / identifier.table
        if not (table_dir / "_delta_log").exists():
            return {"columns": [], "partition_columns": []}
        from deltalake import DeltaTable

        dt = DeltaTable(str(table_dir))
        schema = dt.schema()
        return {
            "columns": [
                {"name": f.name, "type": f.type.type, "nullable": f.nullable} for f in schema.fields
            ],
            "partition_columns": dt.metadata().partition_columns,
        }

    def _get_table_metadata(self, identifier: SparkCatalogIdentifier) -> dict[str, Any]:
        """Get table metadata from catalog.

        Args:
            identifier: Table identifier

        Returns:
            Table metadata dict
        """
        if self._client:
            return self._get_table_metadata_from_hive(identifier)

        if self.catalog_type != "file":
            return {}
        table_dir = self._lakehouse_root / identifier.namespace / identifier.table
        if not (table_dir / "_delta_log").exists():
            return {}
        from datetime import UTC, datetime

        from deltalake import DeltaTable

        dt = DeltaTable(str(table_dir))
        md = dt.metadata()
        created_at = ""
        if md.created_time:
            created_at = datetime.fromtimestamp(md.created_time / 1000, tz=UTC).isoformat()
        return {
            "table_type": "MANAGED",
            "format": "delta",
            "location": str(table_dir),
            "created_at": created_at,
        }

    def _list_databases_from_hive(self, catalog: str | None) -> list[dict[str, Any]]:
        """List databases from Hive Metastore."""
        if not self._client:
            return []

        try:
            cursor = self._client.cursor()
            cursor.execute("SHOW DATABASES")
            databases = []
            for row in cursor.fetchall():
                databases.append({"name": row[0], "description": ""})
            return databases
        except Exception as exc:
            logger.error("failed to list databases", error=str(exc))
            return []

    def _list_tables_from_hive(self, database: str) -> list[dict[str, Any]]:
        """List tables from Hive Metastore."""
        if not self._client:
            return []

        try:
            cursor = self._client.cursor()
            cursor.execute(f"USE {database}")
            cursor.execute("SHOW TABLES")
            tables = []
            for row in cursor.fetchall():
                tables.append({"name": row[0], "type": "MANAGED", "format": "delta"})
            return tables
        except Exception as exc:
            logger.error("failed to list tables", error=str(exc))
            return []

    def _get_table_schema_from_hive(self, identifier: SparkCatalogIdentifier) -> dict[str, Any]:
        """Get table schema from Hive Metastore."""
        if not self._client:
            return {"columns": [], "partition_columns": []}

        try:
            cursor = self._client.cursor()
            cursor.execute(f"DESCRIBE {identifier.to_hive_sql()}")
            columns = []
            for row in cursor.fetchall():
                if row[0] and not row[0].startswith("#"):
                    columns.append(
                        {
                            "name": row[0],
                            "type": row[1],
                            "nullable": "YES" in (row[2] if len(row) > 2 else "YES"),
                        }
                    )
            return {"columns": columns, "partition_columns": []}
        except Exception as exc:
            logger.error("failed to get table schema", error=str(exc))
            return {"columns": [], "partition_columns": []}

    def _get_table_metadata_from_hive(self, identifier: SparkCatalogIdentifier) -> dict[str, Any]:
        """Get table metadata from Hive Metastore."""
        if not self._client:
            return {}

        try:
            cursor = self._client.cursor()
            cursor.execute(f"DESCRIBE FORMATTED {identifier.to_hive_sql()}")
            metadata = {}
            for row in cursor.fetchall():
                if row[0] and row[1]:
                    key = row[0].strip().lower().replace(" ", "_")
                    metadata[key] = row[1].strip()
            return metadata
        except Exception as exc:
            logger.error("failed to get table metadata", error=str(exc))
            return {}
