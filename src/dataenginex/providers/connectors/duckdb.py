"""DuckDB connector — default data engine.

DuckDB reads CSV, Parquet, JSON natively.
This connector wraps DuckDB as a data source/sink for pipelines.
"""

from __future__ import annotations

from typing import Any

import duckdb
import structlog

from dataenginex.foundation.plugin_contracts import BaseConnector
from dataenginex.providers.connectors import connector_registry
from dataenginex.providers.connectors._utils import NOT_CONNECTED, rows_to_dicts

logger = structlog.get_logger()


@connector_registry.decorator("duckdb", is_default=True)
class DuckDBConnector(BaseConnector):
    """DuckDB-backed connector.

    Args:
        database: Path to DuckDB file. ":memory:" for in-memory.
    """

    def __init__(self, database: str = ":memory:", **kwargs: Any) -> None:
        self._database = database
        self._conn: duckdb.DuckDBPyConnection | None = None

    def connect(self) -> None:
        if self._conn is not None:
            return
        self._conn = duckdb.connect(self._database)
        logger.debug("duckdb connected", database=self._database)

    def disconnect(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.debug("duckdb disconnected", database=self._database)

    def read(self, *, table: str, default: Any = None, **kwargs: Any) -> list[dict[str, Any]]:
        if self._conn is None:
            raise RuntimeError(NOT_CONNECTED)
        try:
            result = self._conn.execute(f"SELECT * FROM {table}")  # noqa: S608
            return rows_to_dicts(result)
        except duckdb.CatalogException:
            if default is not None:
                return list(default)
            raise

    def write(self, data: Any, *, table: str, **kwargs: Any) -> None:
        if self._conn is None:
            raise RuntimeError(NOT_CONNECTED)
        import pyarrow as pa

        if isinstance(data, list):
            if len(data) == 0:
                return
            tbl = pa.Table.from_pylist(data)
        elif isinstance(data, pa.Table):
            tbl = data
        else:
            msg = f"Unsupported data type: {type(data)}"
            raise TypeError(msg)

        # Register to avoid frame-scan (not reliable on all connection types).
        # CREATE OR REPLACE overwrites existing data; CREATE TABLE IF NOT EXISTS was a silent no-op.
        self._conn.register("_dex_write_tmp", tbl)
        try:
            self._conn.execute(  # noqa: S608
                f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM _dex_write_tmp"
            )
        finally:
            self._conn.unregister("_dex_write_tmp")
        logger.info("data written", table=table, rows=len(tbl))

    def health_check(self) -> bool:
        if self._conn is None:
            return False
        try:
            self._conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    def execute(self, sql: str) -> list[dict[str, Any]]:
        """Execute raw SQL and return results as list of dicts."""
        if self._conn is None:
            raise RuntimeError(NOT_CONNECTED)
        result = self._conn.execute(sql)
        return rows_to_dicts(result)

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """Direct access to the DuckDB connection for advanced use."""
        if self._conn is None:
            raise RuntimeError(NOT_CONNECTED)
        return self._conn
