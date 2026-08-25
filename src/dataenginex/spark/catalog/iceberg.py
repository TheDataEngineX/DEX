"""Iceberg catalog adapter (§20.5).

Wraps PyIceberg for read/write operations.
Complements SparkCatalogAdapter which handles Delta/Hive discovery.

Ponytail: only the read/write/scan paths. Schema evolution,
partitioning, and time-travel can be added when needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()

__all__ = ["IcebergAdapter"]


class IcebergAdapter:
    """Read/write Iceberg tables via PyIceberg."""

    def __init__(
        self, warehouse: str = ".dex/lakehouse", connection_config: dict[str, str] | None = None
    ) -> None:
        self.warehouse = Path(warehouse)
        self.connection_config = connection_config or {}
        self._catalog: Any | None = None

    def connect(self) -> None:
        """Initialize the catalog."""
        try:
            from pyiceberg.catalog import load_catalog

            catalog_type = self.connection_config.get("type", "hive")
            self._catalog = load_catalog(
                "default",
                **{
                    "type": catalog_type,
                    "warehouse": str(self.warehouse),
                },
            )
            logger.info("iceberg catalog connected", warehouse=str(self.warehouse))
        except ImportError:
            logger.warning("pyiceberg not installed")
            self._catalog = None

    def disconnect(self) -> None:
        self._catalog = None

    def table_exists(self, namespace: str, table_name: str) -> bool:
        """Check if a table exists."""
        if not self._catalog:
            return False
        try:
            identifier = (namespace, table_name)
            self._catalog.load_table(identifier)
            return True
        except Exception:
            return False

    def read_table(self, namespace: str, table_name: str) -> Any:
        """Scan an Iceberg table and return a PyArrow table."""
        if not self._catalog:
            msg = "Iceberg catalog not connected"
            raise RuntimeError(msg)
        identifier = (namespace, table_name)
        table = self._catalog.load_table(identifier)
        return table.scan().to_arrow()

    def write_table(
        self,
        namespace: str,
        table_name: str,
        arrow_table: Any,
        mode: str = "append",
    ) -> None:
        """Write a PyArrow table to Iceberg.

        Args:
            namespace: Database/namespace.
            table_name: Table name.
            arrow_table: PyArrow table to write.
            mode: 'append' or 'overwrite'.
        """
        if not self._catalog:
            msg = "Iceberg catalog not connected"
            raise RuntimeError(msg)

        identifier = (namespace, table_name)
        try:
            table = self._catalog.load_table(identifier)
        except Exception:
            # Create namespace if needed, then create table
            if not self._catalog.namespace_exists(namespace):
                self._catalog.create_namespace(namespace)
            table = self._catalog.create_table(identifier, schema=arrow_table.schema)

        if mode == "overwrite":
            table.overwrite(arrow_table)
        else:
            table.append(arrow_table)

        logger.info(
            "iceberg write complete",
            table=f"{namespace}.{table_name}",
            rows=len(arrow_table),
            mode=mode,
        )

    def list_tables(self, namespace: str) -> list[str]:
        """List all tables in a namespace."""
        if not self._catalog:
            return []
        try:
            tables = self._catalog.list_tables(namespace)
            return [t[1] if isinstance(t, tuple) else str(t) for t in tables]
        except Exception:
            return []

    def drop_table(self, namespace: str, table_name: str) -> None:
        """Drop a table."""
        if not self._catalog:
            msg = "Iceberg catalog not connected"
            raise RuntimeError(msg)
        identifier = (namespace, table_name)
        self._catalog.drop_table(identifier)
        logger.info("iceberg table dropped", table=f"{namespace}.{table_name}")
