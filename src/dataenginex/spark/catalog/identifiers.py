"""Spark catalog identifiers (§20.5).

Typed identifiers for Spark catalog.namespace.table references.
"""

from __future__ import annotations

from dataenginex.foundation.projects import FrozenModel

__all__ = ["SparkCatalogIdentifier"]


class SparkCatalogIdentifier(FrozenModel):
    """Spark catalog.namespace.table identifier (§20.5)."""

    catalog: str = "spark_catalog"
    namespace: str = "default"
    table: str

    def to_spark_sql(self) -> str:
        """Convert to fully qualified Spark SQL identifier."""
        return f"{self.catalog}.{self.namespace}.{self.table}"

    def to_hive_sql(self) -> str:
        """Convert to Hive SQL identifier (namespace.table)."""
        return f"{self.namespace}.{self.table}"

    @classmethod
    def from_spark_sql(cls, identifier: str) -> SparkCatalogIdentifier:
        """Parse a Spark SQL identifier."""
        parts = identifier.split(".")
        if len(parts) == 3:
            return cls(catalog=parts[0], namespace=parts[1], table=parts[2])
        elif len(parts) == 2:
            return cls(namespace=parts[0], table=parts[1])
        return cls(table=parts[0])
