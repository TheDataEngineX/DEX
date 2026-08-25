"""Prometheus agent metrics placeholder — no actual Prometheus integration."""

from __future__ import annotations

# Metric name constants — keep agent_ prefix generic, project name was consolidated
AGENT_REQUESTS_TOTAL = "agent_requests_total"
AGENT_LATENCY_SECONDS = "agent_latency_seconds"
AGENT_ERRORS_TOTAL = "agent_errors_total"
AGENT_TOOL_CALLS_TOTAL = "agent_tool_calls_total"
AGENT_TOKENS_TOTAL = "agent_tokens_total"


class AgentMetrics:
    """Stub metrics collector — tracks counters in-memory."""

    def __init__(self) -> None:
        self._requests: dict[str, int] = {}
        self._latencies: dict[str, list[float]] = {}

    def increment_requests(self, agent_name: str) -> None:
        """Increment the request counter for an agent."""
        self._requests[agent_name] = self._requests.get(agent_name, 0) + 1

    def record_latency(self, agent_name: str, seconds: float) -> None:
        """Record a latency observation for an agent."""
        self._latencies.setdefault(agent_name, []).append(seconds)
