"""DuckDB SQL-based transforms.

All transforms execute SQL against a DuckDB connection and return
the name of the output table. Each transform is registered in the transform_registry.
A workload chains them: input_table -> transform1 -> transform2 -> ...
"""

from __future__ import annotations

from typing import Any

import duckdb
import structlog

from dataenginex.domains.analytics.transforms import transform_registry
from dataenginex.foundation.plugin_contracts import BaseTransform

logger = structlog.get_logger()


@transform_registry.decorator("filter", is_default=True)
class FilterTransform(BaseTransform):
    """Filter rows using a SQL WHERE condition.

    Config: {type: filter, condition: "rating > 5.0"}
    """

    def __init__(self, condition: str, **kwargs: Any) -> None:
        self._condition = condition

    def apply(self, conn: duckdb.DuckDBPyConnection, input_table: str) -> str:
        output = f"{input_table}_filtered"
        sql = (
            f"CREATE OR REPLACE TABLE {output} AS "
            f"SELECT * FROM {input_table} WHERE {self._condition}"
        )
        conn.execute(sql)
        count_row = conn.execute(f"SELECT count(*) FROM {output}").fetchone()
        count = int(count_row[0]) if count_row else 0
        logger.info("filter applied", condition=self._condition, output_rows=count)
        return output

    def validate(self) -> list[str]:
        if not self._condition.strip():
            return ["filter condition is empty"]
        return []


@transform_registry.decorator("derive")
class DeriveTransform(BaseTransform):
    """Add a derived column using a SQL expression.

    Config: {type: derive, name: "rating_pct", expression: "rating / 10.0 * 100"}
    """

    def __init__(self, name: str, expression: str, **kwargs: Any) -> None:
        self._col_name = name
        self._expression = expression

    def apply(self, conn: duckdb.DuckDBPyConnection, input_table: str) -> str:
        output = f"{input_table}_derived"
        sql = (
            f"CREATE OR REPLACE TABLE {output} AS "
            f"SELECT *, ({self._expression}) AS {self._col_name} FROM {input_table}"
        )
        conn.execute(sql)
        logger.info("derive applied", column=self._col_name, expression=self._expression)
        return output

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self._col_name.strip():
            errors.append("derive column name is empty")
        if not self._expression.strip():
            errors.append("derive expression is empty")
        return errors


@transform_registry.decorator("cast")
class CastTransform(BaseTransform):
    """Cast columns to specified types.

    Config: {type: cast, columns: {rating: DOUBLE, year: INTEGER}}
    """

    def __init__(self, columns: dict[str, str], **kwargs: Any) -> None:
        self._columns = columns

    def apply(self, conn: duckdb.DuckDBPyConnection, input_table: str) -> str:
        output = f"{input_table}_cast"
        all_cols = [row[0] for row in conn.execute(f"DESCRIBE {input_table}").fetchall()]
        select_parts = []
        for col in all_cols:
            if col in self._columns:
                select_parts.append(f"CAST({col} AS {self._columns[col]}) AS {col}")
            else:
                select_parts.append(col)
        select_sql = ", ".join(select_parts)
        conn.execute(f"CREATE OR REPLACE TABLE {output} AS SELECT {select_sql} FROM {input_table}")
        logger.info("cast applied", columns=self._columns)
        return output

    def validate(self) -> list[str]:
        if not self._columns:
            return ["cast requires at least one column"]
        return []


@transform_registry.decorator("deduplicate")
class DeduplicateTransform(BaseTransform):
    """Remove duplicate rows based on key columns.

    Config: {type: deduplicate, key: [id]}
    """

    def __init__(self, key: str | list[str], **kwargs: Any) -> None:
        self._key = [key] if isinstance(key, str) else key

    def apply(self, conn: duckdb.DuckDBPyConnection, input_table: str) -> str:
        output = f"{input_table}_deduped"
        key_cols = ", ".join(self._key)
        conn.execute(f"""
            CREATE OR REPLACE TABLE {output} AS
            SELECT * FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY {key_cols} ORDER BY rowid) AS _rn
                FROM {input_table}
            ) WHERE _rn = 1
        """)
        conn.execute(f"ALTER TABLE {output} DROP COLUMN _rn")
        before_row = conn.execute(f"SELECT count(*) FROM {input_table}").fetchone()
        before = int(before_row[0]) if before_row else 0
        after_row = conn.execute(f"SELECT count(*) FROM {output}").fetchone()
        after = int(after_row[0]) if after_row else 0
        logger.info(
            "deduplicate applied",
            key=self._key,
            before=before,
            after=after,
            removed=before - after,
        )
        return output

    def validate(self) -> list[str]:
        if not self._key:
            return ["deduplicate requires at least one key column"]
        return []


@transform_registry.decorator("sql")
class SQLTransform(BaseTransform):
    """Arbitrary SQL transform — use ``_data`` to reference the current table.

    Config: {type: sql, sql: "SELECT * FROM _data WHERE x > 0"}
    """

    def __init__(self, sql: str, **kwargs: Any) -> None:
        self._sql = sql

    def apply(self, conn: duckdb.DuckDBPyConnection, input_table: str) -> str:
        output = f"{input_table}_sql"
        resolved = self._sql.replace("_data", input_table)
        conn.execute(f"CREATE OR REPLACE TABLE {output} AS ({resolved})")
        row = conn.execute(f"SELECT count(*) FROM {output}").fetchone()
        count = int(row[0]) if row else 0
        logger.info("sql transform applied", rows=count)
        return output

    def validate(self) -> list[str]:
        if not self._sql.strip():
            return ["sql transform requires a non-empty sql query"]
        return []


@transform_registry.decorator("rename")
class RenameTransform(BaseTransform):
    """Rename one or more columns.

    Config: {type: rename, mapping: {old_name: new_name, ...}}
    """

    def __init__(self, mapping: dict[str, str], **kwargs: Any) -> None:
        self._mapping = mapping

    def apply(self, conn: duckdb.DuckDBPyConnection, input_table: str) -> str:
        output = f"{input_table}_renamed"
        cols = [row[0] for row in conn.execute(f"DESCRIBE {input_table}").fetchall()]
        parts = [f'"{c}" AS "{self._mapping[c]}"' if c in self._mapping else f'"{c}"' for c in cols]
        conn.execute(
            f"CREATE OR REPLACE TABLE {output} AS SELECT {', '.join(parts)} FROM {input_table}"
        )
        logger.info("rename applied", mapping=self._mapping)
        return output

    def validate(self) -> list[str]:
        if not self._mapping:
            return ["rename requires a non-empty mapping"]
        return []


@transform_registry.decorator("drop_columns")
class DropColumnsTransform(BaseTransform):
    """Drop one or more columns by name.

    Config: {type: drop_columns, columns: [col1, col2]}
    """

    def __init__(self, columns: list[str], **kwargs: Any) -> None:
        self._drop = set(columns)

    def apply(self, conn: duckdb.DuckDBPyConnection, input_table: str) -> str:
        output = f"{input_table}_dropped"
        cols = [row[0] for row in conn.execute(f"DESCRIBE {input_table}").fetchall()]
        keep = [f'"{c}"' for c in cols if c not in self._drop]
        conn.execute(
            f"CREATE OR REPLACE TABLE {output} AS SELECT {', '.join(keep)} FROM {input_table}"
        )
        logger.info("drop_columns applied", dropped=list(self._drop))
        return output

    def validate(self) -> list[str]:
        if not self._drop:
            return ["drop_columns requires at least one column"]
        return []


@transform_registry.decorator("fill_null")
class FillNullTransform(BaseTransform):
    """Replace NULL values with per-column defaults.

    Config: {type: fill_null, defaults: {col: value, ...}}
    String defaults are quoted automatically; numeric/boolean values are used as-is.
    """

    def __init__(self, defaults: dict[str, Any], **kwargs: Any) -> None:
        self._defaults = defaults

    def apply(self, conn: duckdb.DuckDBPyConnection, input_table: str) -> str:
        output = f"{input_table}_filled"
        cols = [row[0] for row in conn.execute(f"DESCRIBE {input_table}").fetchall()]
        parts = []
        for col in cols:
            if col in self._defaults:
                raw = self._defaults[col]
                default_expr = f"'{raw}'" if isinstance(raw, str) else str(raw)
                parts.append(f'COALESCE("{col}", {default_expr}) AS "{col}"')
            else:
                parts.append(f'"{col}"')
        conn.execute(
            f"CREATE OR REPLACE TABLE {output} AS SELECT {', '.join(parts)} FROM {input_table}"
        )
        logger.info("fill_null applied", columns=list(self._defaults.keys()))
        return output

    def validate(self) -> list[str]:
        if not self._defaults:
            return ["fill_null requires at least one column default"]
        return []


@transform_registry.decorator("aggregate")
class AggregateTransform(BaseTransform):
    """GROUP BY aggregation — produces one row per group.

    Config:
      type: aggregate
      group_by: [col1, col2]
      agg_exprs:
        total_count: "COUNT(*)"
        avg_score: "AVG(score)"
    """

    def __init__(
        self,
        group_by: list[str],
        agg_exprs: dict[str, str],
        **kwargs: Any,
    ) -> None:
        self._group_by = group_by
        self._agg_exprs = agg_exprs

    def apply(self, conn: duckdb.DuckDBPyConnection, input_table: str) -> str:
        output = f"{input_table}_agg"
        group_cols = ", ".join(f'"{c}"' for c in self._group_by)
        agg_parts = [f"({expr}) AS {col}" for col, expr in self._agg_exprs.items()]
        select_cols = [f'"{c}"' for c in self._group_by] + agg_parts
        conn.execute(
            f"CREATE OR REPLACE TABLE {output} AS "
            f"SELECT {', '.join(select_cols)} FROM {input_table} GROUP BY {group_cols}"
        )
        row = conn.execute(f"SELECT count(*) FROM {output}").fetchone()
        logger.info("aggregate applied", groups=int(row[0]) if row else 0)
        return output

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self._group_by:
            errors.append("aggregate requires at least one group_by column")
        if not self._agg_exprs:
            errors.append("aggregate requires at least one agg_exprs entry")
        return errors


@transform_registry.decorator("window")
class WindowTransform(BaseTransform):
    """Add a window-function column to the current table.

    Config:
      type: window
      name: rank_in_decade
      expression: "RANK()"
      partition_by: [decade]
      order_by: "weighted_rating DESC"
    """

    def __init__(
        self,
        name: str,
        expression: str,
        partition_by: list[str] | None = None,
        order_by: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._name = name
        self._expression = expression
        self._partition_by = partition_by or []
        self._order_by = order_by or ""

    def apply(self, conn: duckdb.DuckDBPyConnection, input_table: str) -> str:
        output = f"{input_table}_win"
        over_parts: list[str] = []
        if self._partition_by:
            over_parts.append(f"PARTITION BY {', '.join(self._partition_by)}")
        if self._order_by:
            over_parts.append(f"ORDER BY {self._order_by}")
        over_clause = " ".join(over_parts)
        win_expr = f"({self._expression}) OVER ({over_clause}) AS {self._name}"
        conn.execute(f"CREATE OR REPLACE TABLE {output} AS SELECT *, {win_expr} FROM {input_table}")
        logger.info("window applied", column=self._name, expression=self._expression)
        return output

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self._name:
            errors.append("window requires a column name")
        if not self._expression:
            errors.append("window requires an expression")
        return errors


@transform_registry.decorator("explode")
class ExplodeTransform(BaseTransform):
    """Explode a LIST column into one row per element (array un-nesting).

    Accepts a dot-separated path to explode a LIST nested inside a STRUCT
    column (e.g. "credits.cast" for a STRUCT column `credits` with a LIST
    field `cast`), as well as a plain top-level LIST column (e.g. "genres").

    Rows whose array is empty or NULL are dropped by DuckDB's UNNEST — same
    default semantic as Spark's explode().

    The top-level column named by the first path segment is dropped from
    the output (its exploded field replaces it) — sibling fields of a
    struct are not preserved; this transform explodes only what you ask for.
    If you need to explode multiple fields from the same struct (e.g. both
    credits.cast and credits.crew), explode into separate downstream tables
    from independent copies of the source, or explode the outer struct
    first and re-derive each field — exploding the same struct column twice
    in one pipeline chain fails, since the first explode drops it entirely.

    Config: {type: explode, options: {column: "credits.cast", alias: "credit"}}
    """

    def __init__(self, column: str, alias: str | None = None, **kwargs: Any) -> None:
        self._path = column.split(".")
        self._top_level_column = self._path[0]
        self._alias = alias or self._path[-1]

    def apply(self, conn: duckdb.DuckDBPyConnection, input_table: str) -> str:
        output = f"{input_table}_exploded"
        all_cols = [row[0] for row in conn.execute(f"DESCRIBE {input_table}").fetchall()]
        keep = [c for c in all_cols if c != self._top_level_column]
        keep_sql = ", ".join(f'"{c}"' for c in keep)
        path_sql = ".".join(f'"{p}"' for p in self._path)
        path_description = conn.execute(
            f"DESCRIBE SELECT {path_sql} AS _value FROM {input_table}"
        ).fetchone()
        if path_description is None:
            raise ValueError(f"explode column {'.'.join(self._path)!r} was not found")
        path_type = str(path_description[1])
        if not (path_type.endswith("[]") or path_type.startswith("LIST(")):
            msg = f"explode column {'.'.join(self._path)!r} must be a LIST, got {path_type}"
            raise ValueError(msg)
        prefix = f"{keep_sql}, " if keep_sql else ""
        conn.execute(
            f"CREATE OR REPLACE TABLE {output} AS "
            f'SELECT {prefix}UNNEST({path_sql}) AS "{self._alias}" '
            f"FROM {input_table}"
        )
        count_row = conn.execute(f"SELECT count(*) FROM {output}").fetchone()
        count = int(count_row[0]) if count_row else 0
        logger.info("explode applied", column=".".join(self._path), output_rows=count)
        return output

    def validate(self) -> list[str]:
        if not self._path[0].strip():
            return ["explode requires a column name"]
        return []


@transform_registry.decorator("json_normalize")
class JsonNormalizeTransform(BaseTransform):
    """Flatten a DuckDB STRUCT column into regular columns."""

    def __init__(self, column: str, prefix: str = "", **kwargs: Any) -> None:
        self._column = column
        self._prefix = prefix

    def apply(self, conn: duckdb.DuckDBPyConnection, input_table: str) -> str:
        output = f"{input_table}_normalized"
        column = self._column.replace('"', '""')
        fields = [
            str(row[0])
            for row in conn.execute(
                f'DESCRIBE SELECT UNNEST("{column}") FROM {input_table}'
            ).fetchall()
        ]
        if not fields:
            msg = f"json_normalize column {self._column!r} has no fields"
            raise ValueError(msg)

        field_sql = ", ".join(
            f'"{column}"."{field.replace(chr(34), chr(34) * 2)}" '
            f'AS "{(self._prefix + field).replace(chr(34), chr(34) * 2)}"'
            for field in fields
        )
        conn.execute(
            f'CREATE OR REPLACE TABLE {output} AS SELECT * EXCLUDE ("{column}"), '
            f"{field_sql} FROM {input_table}"
        )
        return output

    def validate(self) -> list[str]:
        if not self._column.strip():
            return ["json_normalize requires a column name"]
        return []
