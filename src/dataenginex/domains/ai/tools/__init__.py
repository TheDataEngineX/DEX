"""Tool registry for agent runtimes.

Tools are callables that agents can invoke during reasoning.
Each tool has a name, description, and parameter schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class ToolSpec:
    """Specification for an agent tool."""

    name: str
    description: str
    fn: Any  # Callable
    parameters: dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    """Registry for agent tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        """Register a tool."""
        self._tools[spec.name] = spec
        logger.debug("tool registered", name=spec.name)

    def get(self, name: str) -> ToolSpec:
        """Get a tool by name."""
        try:
            return self._tools[name]
        except KeyError:
            available = ", ".join(sorted(self._tools)) or "(none)"
            msg = f"Tool '{name}' not found. Available: {available}"
            raise KeyError(msg) from None

    def list(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def call(self, tool_name: str, **kwargs: Any) -> Any:
        """Call a tool by name."""
        spec = self.get(tool_name)
        return spec.fn(**kwargs)


tool_registry = ToolRegistry()
