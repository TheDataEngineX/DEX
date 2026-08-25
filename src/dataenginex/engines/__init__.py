"""Pluggable execution engines for data pipelines.

Phase 1: BaseEngine ABC, EngineCapabilities, EngineConfig, registry.
Phase 2: DuckDBEngine (full implementation).
Phase 3: SparkEngine (full implementation).

Usage::

    from dataenginex.engines import engine_registry, EngineConfig

    # Engines auto-register on import (done below)
    engine = engine_registry.get("duckdb")
    config = EngineConfig(type="duckdb", path="./local.duckdb")
    engine.connect(config)
"""

# Auto-register engines on import
import contextlib

import dataenginex.engines.duckdb_engine  # noqa: F401

with contextlib.suppress(ImportError):
    import dataenginex.engines.spark_engine  # noqa: F401

from dataenginex.engines.base import (
    BaseEngine,
    EngineCapabilities,
    EngineConfig,
    LoadResult,
    MergeResult,
    QualityResult,
    SCD2Result,
)
from dataenginex.engines.registry import EngineRegistry, engine_registry

__all__ = [
    "BaseEngine",
    "EngineCapabilities",
    "EngineConfig",
    "EngineRegistry",
    "LoadResult",
    "MergeResult",
    "QualityResult",
    "SCD2Result",
    "engine_registry",
]
