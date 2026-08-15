"""Portable SQL adapter (§20.8).

Translates DEX portable SQL to Spark SQL or DuckDB dialect.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["Dialect", "PortableSQLAdapter"]


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
        # ponytail: dialect translation stubbed — real translation
        # handles date functions, type casts, storage properties
        return sql
