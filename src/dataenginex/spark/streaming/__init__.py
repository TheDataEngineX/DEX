"""Spark Structured Streaming support (§20.7).

DEX provides coordination without owning execution semantics.
Streaming queries remain native Spark constructs.
"""

from __future__ import annotations

from dataenginex.spark.streaming.checkpoint_registry import CheckpointRegistry
from dataenginex.spark.streaming.health import StreamingHealthMonitor
from dataenginex.spark.streaming.query_manager import StreamingQueryManager

__all__ = ["CheckpointRegistry", "StreamingHealthMonitor", "StreamingQueryManager"]
