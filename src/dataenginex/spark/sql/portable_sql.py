"""Portable SQL adapter (§20.8).

Translates DEX portable SQL to Spark SQL or DuckDB dialect.

Ponytail: only translates the 15 most common divergences.
Full ANSI SQL is portable as-is. Add more when a real query breaks.
"""

from __future__ import annotations

import re
from enum import StrEnum

__all__ = ["Dialect", "PortableSQLAdapter"]

# ponytail: translation table as dicts, not AST parsing.
# Covers the 80/20 of real pipeline SQL. Upgrade to sqlglot when
# someone writes a query this can't handle.

_DUCKDB_TO_SPARK: list[tuple[str, str]] = [
    # Type aliases
    (r"\bINTEGER\b", "INT"),
    (r"\bVARCHAR\b", "STRING"),
    (r"\bBOOLEAN\b", "BOOLEAN"),  # same, but normalize casing
    (r"\bBIGINT\b", "BIGINT"),
    (r"\bDOUBLE\b", "DOUBLE"),
    # Date functions
    (r"\bCURRENT_DATE\b", "current_date()"),
    (r"\bDATE_PART\(([^,]+),\s*([^)]+)\)", r"extract(\1 from \2)"),
    (r"\bDATE_TRUNC\(([^,]+),\s*([^)]+)\)", r"date_trunc(\1, \2)"),
    (r"\bDATE_ADD\(([^,]+),\s*INTERVAL\s+(\d+)\s+DAY\)", r"date_add(\1, \2)"),
    (r"\bDATE_DIFF\(([^,]+),\s*([^,]+),\s*([^)]+)\)", r"datediff(\3, \2)"),
    # String functions
    (r"\bSTRING_SPLIT\(([^,]+),\s*([^)]+)\)", r"split(\1, \2)"),
    (r"\bSTRING_AGG\(([^,]+),\s*([^)]+)\)", r"concat_ws(\2, collect_list(\1))"),
    (r"\bCONCAT_WS\(", "concat_ws("),  # already portable, keep
    # Array functions
    (r"\bLIST_CONTAINS\(([^,]+),\s*([^)]+)\)", r"array_contains(\1, \2)"),
    (r"\bLIST_LEN\(([^)]+)\)", r"size(\1)"),
    (r"\bGENERATE_SERIES\(([^,]+),\s*([^)]+)\)", r"sequence(\1, \2)"),
    # JSON
    (r"\bJSON_EXTRACT\(([^,]+),\s*([^)]+)\)", r"get_json_object(\1, \2)"),
    (r"\bJSON_EXTRACT_PATH_TEXT\(([^,]+),\s*([^)]+)\)", r"get_json_object(\1, CONCAT('$.', \2))"),
]

_SPARK_TO_DUCKDB: list[tuple[str, str]] = [
    # Type aliases
    (r"\bINT\b(?!\w)", "INTEGER"),
    (r"\bSTRING\b", "VARCHAR"),
    # Date functions
    (r"\bcurrent_date\(\)", "CURRENT_DATE"),
    (r"\bdate_add\(([^,]+),\s*(\d+)\)", r"DATE_ADD(\1, INTERVAL \2 DAY)"),
    (r"\bdatediff\(([^,]+),\s*([^)]+)\)", r"DATE_DIFF(\1, \2, DAY)"),
    # String functions
    (r"\bsplit\(([^,]+),\s*([^)]+)\)", r"STRING_SPLIT(\1, \2)"),
    (r"\bconcat_ws\(([^,]+),\s*collect_list\(([^)]+)\)\)", r"STRING_AGG(\2, \1)"),
    # Array functions
    (r"\barray_contains\(([^,]+),\s*([^)]+)\)", r"LIST_CONTAINS(\1, \2)"),
    (r"\bsize\(([^)]+)\)", r"LIST_LEN(\1)"),
    (r"\bsequence\(([^,]+),\s*([^)]+)\)", r"GENERATE_SERIES(\1, \2)"),
    # JSON
    (r"\bget_json_object\(([^,]+),\s*([^)]+)\)", r"JSON_EXTRACT(\1, \2)"),
]


class Dialect(StrEnum):
    SPARK = "spark"
    DUCKDB = "duckdb"
    PORTABLE = "portable"


class PortableSQLAdapter:
    """Adapts portable SQL to target dialect (§20.8)."""

    def __init__(self, dialect: Dialect = Dialect.PORTABLE) -> None:
        self.dialect = dialect

    def translate(self, sql: str) -> str:
        """Translate portable SQL to target dialect.

        Portable SQL uses ANSI-standard syntax. Dialect-specific
        translations are applied only when necessary.
        """
        if self.dialect == Dialect.PORTABLE:
            return sql

        table = (
            _DUCKDB_TO_SPARK
            if self.dialect == Dialect.SPARK
            else _SPARK_TO_DUCKDB
        )
        result = sql
        for pattern, replacement in table:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        return result
