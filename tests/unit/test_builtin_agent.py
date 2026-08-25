from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from dataenginex.domains.ai.agents.builtin import BuiltinAgentRuntime


@dataclass
class MockLLMResponse:
    text: str


class MockLLM:
    """Mock LLM that implements the chat() interface used by BuiltinAgentRuntime."""

    def chat(self, messages: list[Any]) -> MockLLMResponse:
        return MockLLMResponse(text="ANSWER: Hello world")


@pytest.mark.asyncio()
async def test_run_returns_dict() -> None:
    agent = BuiltinAgentRuntime(
        llm=MockLLM(),
        system_prompt="You are a test agent.",
        tools=None,
        max_iterations=5,
    )
    result = await agent.run("Hello")
    assert isinstance(result, dict)
    assert "response" in result
    assert "iterations" in result
    assert "tool_calls" in result
    assert isinstance(result["response"], str)
    assert isinstance(result["iterations"], int)
    assert isinstance(result["tool_calls"], int)
    assert result["response"] == "Hello world"
    assert result["iterations"] >= 1
    assert result["tool_calls"] == 0
