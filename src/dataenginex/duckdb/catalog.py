"""DuckDB catalog adapter (§20.8).

Maps DuckDB schema to DEX Resource Catalog.
"""

from __future__ import annotations

from typing import Any

__all__ = ["DuckDBCatalog"]


class DuckDBCatalog:
    """DuckDB catalog adapter (§20.8)."""

    def __init__(self) -> None:
        self._schemas: dict[str, list[str]] = {}

    def register_schema(self, schema_name: str, tables: list[str]) -> None:
        self._schemas[schema_name] = tables

    def list_schemas(self) -> list[str]:
        return list(self._schemas.keys())

    def list_tables(self, schema_name: str) -> list[str]:
        return self._schemas.get(schema_name, [])

    def table_to_resource(self, schema: str, table: str) -> dict[str, Any]:
        return {
            "name": table,
            "resource_type": "table",
            "duckdb_identifier": f"{schema}.{table}",
            "provider": "duckdb",
        }
