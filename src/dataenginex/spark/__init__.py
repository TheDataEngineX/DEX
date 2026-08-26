"""Spark integration module (§20).

First-class Spark integration converting Spark-native events and objects into
DEX resources, execution records, policy decisions, lineage, and resource metrics.

Spark types must NOT leak across the DEX platform (§20.11). Adapters convert
between DEX domain types and Spark-specific objects inside this module.
"""

from dataenginex.spark.connect import SparkConnectClient, SparkSessionRegistry

__all__ = [
    "SparkConnectClient",
    "SparkSessionRegistry",
]
