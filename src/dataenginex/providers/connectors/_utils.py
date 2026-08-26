"""Shared utilities for data connectors."""

from __future__ import annotations

from typing import Any

NOT_CONNECTED: str = "Not connected — call connect() first"


def rows_to_dicts(cursor: Any) -> list[dict[str, Any]]:
    """Convert a DBAPI cursor result to a list of dicts.

    Uses ``cursor.description`` for column names and ``cursor.fetchall()``
    for row data.  Compatible with DuckDB and any DBAPI-2 cursor.
    """
    columns: list[str] = [desc[0] for desc in cursor.description]
    return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
