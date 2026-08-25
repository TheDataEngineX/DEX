"""Agent runtime registry.

Built-in agent uses tool-calling loop with LLM providers.
LangGraph available via ``[agents]`` extra.
"""

from __future__ import annotations

from dataenginex.foundation.plugin_contracts import BaseAgentRuntime
from dataenginex.runtime.registry import BackendRegistry

agent_registry: BackendRegistry[BaseAgentRuntime] = BackendRegistry("agent")
