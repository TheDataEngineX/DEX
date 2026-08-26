"""Generic, reusable mechanism for exposing lakehouse tables over GraphQL.

Given a table-reading backend and a mapping of
``{graphql_type_name: table_name}``, :func:`build_schema` builds read-only
Strawberry GraphQL types plus a ``Query`` root exposing each table as a list
field with optional filtering by an id/key column.

This module is deliberately generic — it knows nothing about "movies",
"credits", or any other project-specific concept. A project (e.g. moviedex)
builds its *actual* schema by calling :func:`build_schema` with its own
table names — see ``dex-studio/examples/movie-dex/plugins/graphql_schema.py``.

Mutations are out of scope — reads only.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

import duckdb
import strawberry

__all__ = ["GoldTable", "build_schema"]


# DuckDB DESCRIBE column_type -> Python/GraphQL scalar. Anything unmapped
# (VARCHAR, DATE, TIMESTAMP, JSON, ...) falls back to str — good enough for
# a read-only display layer; add a real scalar here if a project needs one.
_DUCKDB_TYPE_MAP: dict[str, type] = {
    "BIGINT": int,
    "INTEGER": int,
    "SMALLINT": int,
    "TINYINT": int,
    "HUGEINT": int,
    "UBIGINT": int,
    "UINTEGER": int,
    "DOUBLE": float,
    "FLOAT": float,
    "REAL": float,
    "DECIMAL": float,
    "BOOLEAN": bool,
}


def _py_type(duckdb_type: str) -> type:
    base = duckdb_type.split("(")[0].upper()
    return _DUCKDB_TYPE_MAP.get(base, str)


@dataclasses.dataclass(frozen=True)
class GoldTable:
    """Config for one GraphQL list field.

    ``id_column`` defaults to the table's first schema column when omitted.
    ``layer`` overrides ``build_schema()``'s default layer per-table (e.g. a
    project mixing gold and silver tables under one schema).
    """

    table: str
    id_column: str | None = None
    layer: str | None = None


def _table_path(engine: Any, table: str, layer: str) -> Path:
    layer_path = Path(engine.project_dir) / ".dex" / "lakehouse" / layer
    parquet_path = layer_path / f"{table}.parquet"
    return parquet_path if parquet_path.exists() else layer_path / table


def _query_rows(path: Path, id_column: str, id_value: str | None) -> list[dict[str, Any]]:
    with duckdb.connect(":memory:") as conn:
        if path.is_dir() and (path / "_delta_log").exists():
            from dataenginex.providers.object_store.storage import DeltaStorage

            scan = DeltaStorage(base_path=str(path.parent)).parquet_scan_sql(path.name)
            sql = f"SELECT * FROM {scan}"
        else:
            safe_path = str(path).replace("'", "''")
            sql = f"SELECT * FROM read_parquet('{safe_path}')"
        params: list[Any] = []
        if id_value is not None:
            safe_id_column = id_column.replace('"', '""')
            sql += f' WHERE CAST("{safe_id_column}" AS VARCHAR) = ?'
            params.append(id_value)
        cursor = conn.execute(sql, params)
        columns = [d[0] for d in cursor.description] if cursor.description else []
        return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def build_schema(
    engine: Any,
    tables: dict[str, str | GoldTable],
    layer: str = "gold",
) -> strawberry.Schema:
    """Build a read-only Strawberry schema over lakehouse tables.

    Args:
        engine: A DexEngine (or anything with ``.project_dir`` and
            ``.warehouse_table_schema(table, layer) -> list[dict]``).
        tables: Maps a GraphQL type/field name to either a table name (str,
            using the default ``layer``) or a :class:`GoldTable` for
            per-table overrides.
        layer: Default lakehouse layer (``"gold"``, ``"silver"``, ...) for
            entries given as a plain table name.

    Returns:
        A ``strawberry.Schema`` with one list field per configured table.
        Tables with no materialized schema yet (pipeline hasn't run) are
        skipped rather than failing the whole schema build.
    """
    query_namespace: dict[str, Any] = {}
    query_annotations: dict[str, Any] = {}

    for type_name, cfg in tables.items():
        cfg = GoldTable(table=cfg) if isinstance(cfg, str) else cfg
        table_layer = cfg.layer or layer
        schema_cols = engine.warehouse_table_schema(cfg.table, table_layer) or []
        if not schema_cols:
            continue

        def _column_name(column: dict[str, Any]) -> str:
            return str(column.get("column_name", column.get("name", "")))

        def _column_type(column: dict[str, Any]) -> str:
            return str(column.get("column_type", column.get("dtype", "VARCHAR")))

        id_column = cfg.id_column or _column_name(schema_cols[0])
        row_fields = [(_column_name(c), _py_type(_column_type(c))) for c in schema_cols]
        row_type = strawberry.type(dataclasses.make_dataclass(type_name, row_fields))
        path = _table_path(engine, cfg.table, table_layer)
        # Query root field uses lowerCamel-ish convention (type "Movie" -> field "movie");
        # the type name itself (PascalCase, as given) is what shows up in the GraphQL schema.
        field_name = type_name[:1].lower() + type_name[1:] if type_name else type_name

        def _make_resolver(
            path: Path = path, id_column: str = id_column, row_type: Any = row_type
        ) -> Callable[..., list[Any]]:
            def resolver(id: str | None = None) -> list[row_type]:
                return [row_type(**row) for row in _query_rows(path, id_column, id)]

            return resolver

        query_annotations[field_name] = list[row_type]  # type: ignore[valid-type]
        query_namespace[field_name] = strawberry.field(resolver=_make_resolver())

    query_namespace["__annotations__"] = query_annotations
    query_cls = type("Query", (), query_namespace)
    return strawberry.Schema(query=strawberry.type(query_cls))
