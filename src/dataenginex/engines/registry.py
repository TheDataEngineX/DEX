"""Engine registry — lookup and manage execution backends.

Usage::

    from dataenginex.engines.registry import engine_registry

    # Register engines (done at import time by engine modules)
    engine_registry.register("duckdb", DuckDBEngine())
    engine_registry.register("spark", SparkEngine())

    # Lookup
    engine = engine_registry.get("duckdb")
"""

from __future__ import annotations

import structlog

from dataenginex.engines.base import BaseEngine

logger = structlog.get_logger()

__all__ = ["EngineRegistry", "engine_registry"]


class EngineRegistry:
    """Registry for execution engines."""

    def __init__(self) -> None:
        self._engines: dict[str, BaseEngine] = {}

    def register(self, name: str, engine: BaseEngine) -> None:
        """Register an engine.

        Raises:
            ValueError: If an engine with the same name is already registered.
        """
        if name in self._engines:
            msg = f"engine already registered: {name}"
            raise ValueError(msg)
        self._engines[name] = engine
        logger.info("registered engine", name=name, capabilities=engine.capabilities())

    def get(self, name: str) -> BaseEngine:
        """Look up an engine by name.

        Raises:
            KeyError: If engine not found.
        """
        if name not in self._engines:
            available = ", ".join(self._engines.keys())
            msg = f"engine '{name}' not found (available: {available})"
            raise KeyError(msg)
        return self._engines[name]

    def list_engines(self) -> list[str]:
        """Return all registered engine names."""
        return list(self._engines.keys())

    def is_registered(self, name: str) -> bool:
        """Check if an engine is registered."""
        return name in self._engines


# Global registry — engines register themselves at import time
engine_registry = EngineRegistry()
