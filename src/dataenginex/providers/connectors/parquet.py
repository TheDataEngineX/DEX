"""Parquet file connector — reads Parquet files via DuckDB.

Supports single-file and directory glob patterns.  IMDB-style files with
null markers and tab-delimited TSV.gz are handled by DuckDB natively
through ``read_parquet``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import structlog

from dataenginex.foundation.plugin_contracts import BaseConnector
from dataenginex.providers.connectors import connector_registry
from dataenginex.providers.connectors._utils import NOT_CONNECTED, rows_to_dicts

logger = structlog.get_logger()


@connector_registry.decorator("parquet")
class ParquetConnector(BaseConnector):
    """Parquet connector backed by DuckDB ``read_parquet``.

    Args:
        path: Path to a ``.parquet`` file or a directory / glob pattern.
        default_file: Fallback filename when *path* is a directory.
    """

    def __init__(
        self,
        path: str = ".",
        default_file: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._path = Path(path)
        self._default_file = default_file
        self._conn: duckdb.DuckDBPyConnection | None = None

    def connect(self) -> None:
        self._conn = duckdb.connect(":memory:")
        logger.debug("parquet connector ready", path=str(self._path))

    def disconnect(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _resolve_path(self, table: str | None) -> Path:
        """Return the concrete file path to read."""
        if self._path.is_file():
            return self._path
        # directory mode — look up by table/file name
        filename = table or self._default_file
        if filename is None:
            msg = "No table/file specified and path is a directory"
            raise ValueError(msg)
        candidate = self._path / filename
        if candidate.exists():
            return candidate
        with_ext = self._path / f"{filename}.parquet"
        if with_ext.exists():
            return with_ext
        msg = f"Parquet file not found: {candidate} or {with_ext}"
        raise FileNotFoundError(msg)

    def read(
        self,
        *,
        table: str | None = None,
        default: Any = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        if self._conn is None:
            raise RuntimeError(NOT_CONNECTED)
        try:
            filepath = self._resolve_path(table)
        except FileNotFoundError:
            if default is not None:
                return list(default)
            raise
        safe = str(filepath).replace("'", "''")
        result = self._conn.execute(f"SELECT * FROM read_parquet('{safe}')")
        dicts = rows_to_dicts(result)
        logger.info("parquet read", path=safe, rows=len(dicts))
        return dicts

    def write(self, data: Any, *, table: str = "output.parquet", **kwargs: Any) -> None:
        if self._conn is None:
            raise RuntimeError(NOT_CONNECTED)

        filepath = self._path / table if self._path.is_dir() else self._path
        if isinstance(data, list):
            tbl = pa.Table.from_pylist(data)
        elif isinstance(data, pa.Table):
            tbl = data
        else:
            msg = f"Unsupported data type: {type(data)}"
            raise TypeError(msg)
        tmp = Path(str(filepath) + ".tmp")
        pq.write_table(tbl, str(tmp))
        os.replace(tmp, filepath)
        logger.info("parquet written", path=str(filepath), rows=len(tbl))

    def health_check(self) -> bool:
        return self._conn is not None and self._path.exists()
