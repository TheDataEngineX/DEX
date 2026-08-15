"""Spark Data Source V2 integration (§20.6).

DEX prefers Spark DSV2 for data connectors rather than parallel abstractions.
DEX adds platform layer: trust, secrets, permissions, egress, lineage.
"""

from dataenginex.spark.datasource.capabilities import DataSourceCapabilities
from dataenginex.spark.datasource.registry import DataSourceRegistry

__all__ = ["DataSourceCapabilities", "DataSourceRegistry"]
