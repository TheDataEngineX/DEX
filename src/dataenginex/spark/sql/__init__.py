"""Portable SQL dialect support (§20.8).

DEX favors dialect-portable SQL with explicit dialect fallbacks where
Spark SQL or DuckDB specifics are required.
"""

from dataenginex.spark.sql.portable_sql import PortableSQLAdapter
from dataenginex.spark.sql.spark_sql_executor import SparkSQLExecutor

__all__ = ["PortableSQLAdapter", "SparkSQLExecutor"]
