"""OpenLineage integration (§20.10).

DEX emits OpenLineage events for Spark runs without creating a parallel
lineage taxonomy. DEX policy and lifecycle sit above OpenLineage, not in
competition with it.
"""

from dataenginex.spark.lineage.openlineage_projection import OpenLineageProjection

__all__ = ["OpenLineageProjection"]
