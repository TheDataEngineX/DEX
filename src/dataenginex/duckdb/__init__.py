"""DuckDB portable SQL execution path (§20.8).

Provides local SQL execution using DuckDB for portable SQL validation,
development, and lightweight analytics without requiring Spark.
"""

from dataenginex.duckdb.catalog import DuckDBCatalog
from dataenginex.duckdb.engine import DuckDBEngine
from dataenginex.duckdb.portable_validator import PortableSQLValidator

__all__ = ["DuckDBEngine", "DuckDBCatalog", "PortableSQLValidator"]
