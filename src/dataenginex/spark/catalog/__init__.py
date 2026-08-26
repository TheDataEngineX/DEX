"""Spark catalog integration (§20.5).

DEX Resource Catalog maps to Spark catalog identifiers for Spark-addressable
data objects. DEX must never force non-Spark resources into fake Spark tables.
"""

from dataenginex.spark.catalog.adapter import SparkCatalogAdapter
from dataenginex.spark.catalog.identifiers import SparkCatalogIdentifier

__all__ = ["SparkCatalogAdapter", "SparkCatalogIdentifier"]
