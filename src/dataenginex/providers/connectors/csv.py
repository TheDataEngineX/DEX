"""CSV file connector — reads/writes CSV files via DuckDB.

Uses DuckDB's native CSV reader for performance (columnar scan,
parallel reads, auto-type detection).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.csv as pcsv
import structlog

from dataenginex.foundation.plugin_contracts import BaseConnector
from dataenginex.providers.connectors import connector_registry
from dataenginex.providers.connectors._utils import NOT_CONNECTED, rows_to_dicts

logger = structlog.get_logger()


@connector_registry.decorator("csv")
class CsvConnector(BaseConnector):
    """CSV connector backed by DuckDB CSV reader.

    Args:
        path: Directory containing CSV files.
        default_file: Default file to read (for conformance test).
    """

    def __init__(self, path: str = ".", default_file: str | None = None, **kwargs: Any) -> None:
        self._path = Path(path)
        self._default_file = default_file
        self._conn: duckdb.DuckDBPyConnection | None = None

    def connect(self) -> None:
        self._conn = duckdb.connect(":memory:")
        logger.debug("csv connector ready", path=str(self._path))

    def disconnect(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def read(
        self,
        *,
        table: str | None = None,
        default: Any = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        if self._conn is None:
            raise RuntimeError(NOT_CONNECTED)

        # If path itself is a file, use it directly; otherwise treat as directory
        if self._path.is_file():
            filepath = self._path
        else:
            filename = table or self._default_file
            if filename is None:
                msg = "No table/file specified"
                raise ValueError(msg)
            filepath = self._path / filename
            if not filepath.exists():
                filepath = self._path / f"{filename}.csv"

        if not filepath.exists():
            if default is not None:
                return list(default)
            msg = f"CSV file not found: {filepath}"
            raise FileNotFoundError(msg)

        safe_path = str(filepath).replace("'", "''")
        result = self._conn.execute(f"SELECT * FROM read_csv_auto('{safe_path}')")
        return rows_to_dicts(result)

    def write(self, data: Any, *, table: str = "output.csv", **kwargs: Any) -> None:
        if self._conn is None:
            raise RuntimeError(NOT_CONNECTED)

        filepath = self._path / table
        if isinstance(data, list):
            tbl = pa.Table.from_pylist(data)
        elif isinstance(data, pa.Table):
            tbl = data
        else:
            msg = f"Unsupported data type: {type(data)}"
            raise TypeError(msg)

        pcsv.write_csv(tbl, filepath)
        logger.info("csv written", path=str(filepath), rows=len(tbl))

    def health_check(self) -> bool:
        return self._conn is not None and self._path.exists()
