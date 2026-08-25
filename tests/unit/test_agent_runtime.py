"""Tests for the built-in agent runtime."""

from __future__ import annotations

import pytest

from dataenginex.domains.ai.agents.builtin import BuiltinAgentRuntime
from dataenginex.domains.ai.tools import ToolRegistry, ToolSpec


class TestBuiltinAgentRuntime:
    @pytest.mark.asyncio()
    async def test_run_without_llm_echoes(self) -> None:
        agent = BuiltinAgentRuntime(llm=None)
        result = await agent.run("hello")
        assert isinstance(result, dict)
        assert result["response"] == "hello"
        assert result["iterations"] >= 1
        assert result["tool_calls"] == 0

    @pytest.mark.asyncio()
    async def test_step_without_llm(self) -> None:
        agent = BuiltinAgentRuntime(llm=None)
        step = await agent.step("test message")
        assert step["done"] is True
        assert step["response"] == "test message"

    @pytest.mark.asyncio()
    async def test_history_tracking(self) -> None:
        agent = BuiltinAgentRuntime(llm=None)
        await agent.run("first")
        await agent.run("second")
        assert len(agent.history) == 4  # 2 user + 2 assistant

    @pytest.mark.asyncio()
    async def test_clear_history(self) -> None:
        agent = BuiltinAgentRuntime(llm=None)
        await agent.run("hello")
        agent.clear_history()
        assert agent.history == []

    @pytest.mark.asyncio()
    async def test_custom_tools(self) -> None:
        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                name="add",
                description="Add two numbers",
                fn=lambda a, b: a + b,
                parameters={"a": "int", "b": "int"},
            )
        )
        agent = BuiltinAgentRuntime(llm=None, tools=registry)
        assert "add" in agent._tools.list()


class TestToolRegistry:
    def test_register_and_call(self) -> None:
        registry = ToolRegistry()
        registry.register(ToolSpec(name="greet", description="Greet", fn=lambda name: f"Hi {name}"))
        result = registry.call("greet", name="World")
        assert result == "Hi World"

    def test_list_tools(self) -> None:
        registry = ToolRegistry()
        registry.register(ToolSpec(name="a", description="A", fn=lambda: None))
        registry.register(ToolSpec(name="b", description="B", fn=lambda: None))
        assert set(registry.list()) == {"a", "b"}

    def test_get_missing_tool_raises(self) -> None:
        registry = ToolRegistry()
        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent")
