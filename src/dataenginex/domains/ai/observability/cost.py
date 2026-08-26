"""Token usage and cost tracking for LLM calls."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class TokenUsage(BaseModel):
    """Token usage for a single LLM call."""

    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    agent_name: str = ""


class CostTracker:
    """Tracks cumulative token usage and costs across agents."""

    def __init__(self) -> None:
        self._records: list[TokenUsage] = []

    def record(self, usage: TokenUsage) -> None:
        """Record a token usage entry."""
        self._records.append(usage)

    def total_cost(self, agent_name: str | None = None) -> float:
        """Get total cost in USD, optionally filtered by agent."""
        if agent_name is None:
            return sum(r.cost_usd for r in self._records)
        return sum(r.cost_usd for r in self._records if r.agent_name == agent_name)

    def summary(self) -> dict[str, Any]:
        """Get a summary of all token usage and costs."""
        total_in = sum(r.tokens_in for r in self._records)
        total_out = sum(r.tokens_out for r in self._records)
        return {
            "total_records": len(self._records),
            "total_tokens_in": total_in,
            "total_tokens_out": total_out,
            "total_cost_usd": self.total_cost(),
            "by_model": self._by_model(),
        }

    def _by_model(self) -> dict[str, dict[str, Any]]:
        """Break down costs by model."""
        result: dict[str, dict[str, Any]] = {}
        for r in self._records:
            if r.model not in result:
                result[r.model] = {"tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}
            result[r.model]["tokens_in"] += r.tokens_in
            result[r.model]["tokens_out"] += r.tokens_out
            result[r.model]["cost_usd"] += r.cost_usd
        return result
