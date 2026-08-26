"""Integration tests — AI routing, agents, tools, and multi-agent workflows."""

from __future__ import annotations

from typing import Any

import pytest

from dataenginex.domains.ai.memory.base import MemoryEntry, ShortTermMemory
from dataenginex.domains.ai.providers import BaseProvider
from dataenginex.domains.ai.runtime.executor import AgentConfig, AgentExecutor
from dataenginex.domains.ai.tools import ToolRegistry, ToolSpec
from dataenginex.domains.ai.tools.builtin import register_builtin_tools
from dataenginex.domains.ai.workflows.dag import AgentDAG
from dataenginex.providers.model.router import ModelRouter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class EchoProvider(BaseProvider):
    """Provider that echoes the prompt — no LLM needed."""

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return f"ANSWER: {prompt[:40]}"


class ToolCallThenAnswerProvider(BaseProvider):
    """Provider that issues one tool call then provides a final answer."""

    def __init__(self, tool_name: str = "echo", tool_args: dict[str, Any] | None = None) -> None:
        self._tool_name = tool_name
        self._tool_args = tool_args or {"message": "hello"}
        self._call_count = 0

    def generate(self, prompt: str, **kwargs: Any) -> str:
        self._call_count += 1
        if self._call_count == 1:
            import json

            return f"TOOL: {self._tool_name}\nARGS: {json.dumps(self._tool_args)}"
        return "ANSWER: done"


# ---------------------------------------------------------------------------
# ModelRouter
# ---------------------------------------------------------------------------


class TestModelRouter:
    def test_routes_simple_to_correct_provider(self) -> None:
        local = EchoProvider()
        mid = EchoProvider()
        top = EchoProvider()
        router = ModelRouter(
            providers={"ollama": local, "openai": mid, "anthropic": top},
        )
        assert router.route("task", "simple") is local

    def test_routes_moderate_to_correct_provider(self) -> None:
        mid = EchoProvider()
        router = ModelRouter(
            providers={"openai": mid},
            mapping={"moderate": "openai"},
        )
        assert router.route("task", "moderate") is mid

    def test_routes_complex_to_correct_provider(self) -> None:
        top = EchoProvider()
        router = ModelRouter(
            providers={"anthropic": top},
            mapping={"complex": "anthropic"},
        )
        assert router.route("task", "complex") is top

    def test_unknown_complexity_raises_value_error(self) -> None:
        router = ModelRouter(providers={"openai": EchoProvider()}, mapping={"moderate": "openai"})
        with pytest.raises(ValueError, match="Unknown complexity level"):
            router.route("task", "extreme")

    def test_missing_provider_key_raises_key_error(self) -> None:
        router = ModelRouter(
            providers={},
            mapping={"simple": "ollama"},
        )
        with pytest.raises(KeyError):
            router.route("task", "simple")

    def test_custom_mapping_overrides_defaults(self) -> None:
        fast = EchoProvider()
        router = ModelRouter(
            providers={"fast": fast},
            mapping={"simple": "fast", "moderate": "fast", "complex": "fast"},
        )
        for level in ("simple", "moderate", "complex"):
            assert router.route("task", level) is fast

    def test_router_executes_routed_provider(self) -> None:
        provider = EchoProvider()
        router = ModelRouter(
            providers={"openai": provider},
            mapping={"moderate": "openai"},
        )
        selected = router.route("summarise this text", "moderate")
        result = selected.generate("summarise this text")
        assert "summarise" in result


# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------


class TestBuiltinTools:
    def setup_method(self) -> None:
        register_builtin_tools()

    def test_echo_tool_returns_message(self) -> None:
        from dataenginex.domains.ai.tools import tool_registry

        result = tool_registry.call("echo", message="hello world")
        assert result == "hello world"

    def test_list_tools_returns_registered_names(self) -> None:
        from dataenginex.domains.ai.tools import tool_registry

        tools = tool_registry.call("list_tools")
        assert "echo" in tools
        assert "query" in tools
        assert "list_tools" in tools

    def test_query_tool_runs_sql(self) -> None:
        from dataenginex.domains.ai.tools import tool_registry

        rows = tool_registry.call("query", sql="SELECT 1 AS n")
        assert isinstance(rows, list)
        assert rows[0]["n"] == 1

    def test_tool_registry_list_and_call(self) -> None:
        registry = ToolRegistry()
        spec = ToolSpec(
            name="double",
            description="double an int",
            fn=lambda x: x * 2,
            parameters={"x": "int"},
        )
        registry.register(spec)
        assert "double" in registry.list()
        assert registry.call("double", x=5) == 10

    def test_tool_registry_unknown_tool_raises_key_error(self) -> None:
        registry = ToolRegistry()
        with pytest.raises(KeyError):
            registry.call("nonexistent")


# ---------------------------------------------------------------------------
# BuiltinAgentRuntime — no LLM (echo mode)
# ---------------------------------------------------------------------------


class TestBuiltinAgentRuntimeNoLLM:
    @pytest.mark.asyncio
    async def test_run_without_llm_echoes_message(self) -> None:
        from dataenginex.domains.ai.agents.builtin import BuiltinAgentRuntime

        agent = BuiltinAgentRuntime(llm=None)
        result = await agent.run("Hello, agent!")
        assert result["response"] == "Hello, agent!"
        assert result["iterations"] == 1
        assert result["tool_calls"] == 0

    @pytest.mark.asyncio
    async def test_history_is_tracked(self) -> None:
        from dataenginex.domains.ai.agents.builtin import BuiltinAgentRuntime

        agent = BuiltinAgentRuntime(llm=None)
        await agent.run("first message")
        assert any(e["role"] == "user" for e in agent.history)
        assert any(e["role"] == "assistant" for e in agent.history)

    @pytest.mark.asyncio
    async def test_clear_history(self) -> None:
        from dataenginex.domains.ai.agents.builtin import BuiltinAgentRuntime

        agent = BuiltinAgentRuntime(llm=None)
        await agent.run("message")
        agent.clear_history()
        assert agent.history == []

    @pytest.mark.asyncio
    async def test_multiple_messages_accumulate_history(self) -> None:
        from dataenginex.domains.ai.agents.builtin import BuiltinAgentRuntime

        agent = BuiltinAgentRuntime(llm=None)
        await agent.run("first")
        await agent.run("second")
        assert len(agent.history) >= 4  # user+assistant per turn


# ---------------------------------------------------------------------------
# BuiltinAgentRuntime — mock LLM
# ---------------------------------------------------------------------------


class _MockLLMResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class _MockLLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._idx = 0

    def chat(self, messages: Any) -> _MockLLMResponse:
        resp = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return _MockLLMResponse(resp)


class TestBuiltinAgentRuntimeWithMockLLM:
    @pytest.mark.asyncio
    async def test_final_answer_returned(self) -> None:
        from dataenginex.domains.ai.agents.builtin import BuiltinAgentRuntime

        llm = _MockLLM(["ANSWER: The answer is 42"])
        agent = BuiltinAgentRuntime(llm=llm)
        result = await agent.run("What is the answer?")
        assert "42" in result["response"]
        assert result["iterations"] == 1

    @pytest.mark.asyncio
    async def test_tool_call_increments_tool_calls(self) -> None:
        from dataenginex.domains.ai.agents.builtin import BuiltinAgentRuntime

        register_builtin_tools()
        llm = _MockLLM(
            [
                'TOOL: echo ARGS: {"message": "pong"}',
                "ANSWER: done",
            ]
        )
        agent = BuiltinAgentRuntime(llm=llm)
        result = await agent.run("ping")
        assert result["tool_calls"] >= 1

    @pytest.mark.asyncio
    async def test_max_iterations_respected(self) -> None:
        from dataenginex.domains.ai.agents.builtin import BuiltinAgentRuntime

        # LLM never returns ANSWER — always does tool calls
        register_builtin_tools()
        llm = _MockLLM(['TOOL: echo ARGS: {"message": "loop"}'])
        agent = BuiltinAgentRuntime(llm=llm, max_iterations=3)
        result = await agent.run("keep going")
        assert result["iterations"] == 3
        assert "reasoning limit" in result["response"]


# ---------------------------------------------------------------------------
# AgentExecutor (stateful executor with memory)
# ---------------------------------------------------------------------------


class TestAgentExecutor:
    def _make_executor(self, provider: BaseProvider) -> AgentExecutor:
        register_builtin_tools()
        from dataenginex.domains.ai.tools import tool_registry

        config = AgentConfig(name="test", model="mock", max_iterations=5)
        memory = ShortTermMemory()
        return AgentExecutor(
            config=config,
            tool_registry=tool_registry,
            memory=memory,
            provider=provider,
        )

    def test_direct_answer(self) -> None:
        executor = self._make_executor(EchoProvider())
        response = executor.run("What is 2+2?")
        assert response.content  # provider echoed something
        assert response.iterations >= 1

    def test_tool_call_then_answer(self) -> None:
        provider = ToolCallThenAnswerProvider(tool_name="echo", tool_args={"message": "hi"})
        executor = self._make_executor(provider)
        response = executor.run("test tool use")
        # Should have recorded a tool call
        assert len(response.tool_calls) >= 1
        assert response.tool_calls[0]["tool"] == "echo"

    def test_memory_populated_after_run(self) -> None:
        memory = ShortTermMemory()
        register_builtin_tools()
        from dataenginex.domains.ai.tools import tool_registry

        config = AgentConfig(name="test", model="mock", max_iterations=5)
        executor = AgentExecutor(
            config=config,
            tool_registry=tool_registry,
            memory=memory,
            provider=EchoProvider(),
        )
        executor.run("hello")
        entries = memory.recent(10)
        roles = {e.role for e in entries}
        assert "user" in roles
        assert "assistant" in roles

    def test_memory_search(self) -> None:
        memory = ShortTermMemory()
        memory.add(MemoryEntry(content="pipeline failed at step 3", role="system"))
        memory.add(MemoryEntry(content="retry all pipelines", role="user"))
        results = memory.search("pipeline")
        assert len(results) == 2

    def test_memory_max_entries_evicts_oldest(self) -> None:
        memory = ShortTermMemory(max_entries=3)
        for i in range(5):
            memory.add(MemoryEntry(content=f"entry-{i}", role="user"))
        recent = memory.recent(10)
        assert len(recent) == 3
        assert recent[-1].content == "entry-4"


# ---------------------------------------------------------------------------
# AgentDAG — multi-agent workflows
# ---------------------------------------------------------------------------


class TestAgentDAG:
    def test_single_node_workflow(self) -> None:
        from dataenginex.domains.ai.runtime.executor import AgentConfig

        dag = AgentDAG()
        dag.add_node("extract", AgentConfig(name="extract", model="mock"))
        provider = EchoProvider()
        results = dag.execute(providers={"extract": provider}, initial_input="raw data")
        assert "extract" in results
        assert results["extract"]  # non-empty output

    def test_two_node_chain_passes_output(self) -> None:
        from dataenginex.domains.ai.runtime.executor import AgentConfig

        outputs: list[str] = []

        class RecordingProvider(BaseProvider):
            def generate(self, prompt: str, **kwargs: Any) -> str:
                outputs.append(prompt)
                return f"processed: {prompt[:20]}"

        dag = AgentDAG()
        dag.add_node("step1", AgentConfig(name="step1", model="mock"))
        dag.add_node("step2", AgentConfig(name="step2", model="mock"))
        dag.add_edge("step1", "step2")

        provider = RecordingProvider()
        results = dag.execute(
            providers={"step1": provider, "step2": provider},
            initial_input="start",
        )
        assert "step1" in results
        assert "step2" in results
        # step2 received step1's output as input
        assert "processed" in outputs[-1]

    def test_three_node_sequential_chain(self) -> None:
        from dataenginex.domains.ai.runtime.executor import AgentConfig

        dag = AgentDAG()
        for name in ("a", "b", "c"):
            dag.add_node(name, AgentConfig(name=name, model="mock"))
        dag.add_edge("a", "b")
        dag.add_edge("b", "c")

        results = dag.execute(
            providers={"a": EchoProvider(), "b": EchoProvider(), "c": EchoProvider()},
            initial_input="input",
        )
        assert set(results.keys()) == {"a", "b", "c"}

    def test_cycle_detection_raises_value_error(self) -> None:
        from dataenginex.domains.ai.runtime.executor import AgentConfig

        dag = AgentDAG()
        dag.add_node("x", AgentConfig(name="x", model="mock"))
        dag.add_node("y", AgentConfig(name="y", model="mock"))
        dag.add_edge("x", "y")
        dag.add_edge("y", "x")  # cycle!

        with pytest.raises(ValueError, match="cycle"):
            dag.execute(initial_input="test")

    def test_validate_acyclic_dag_returns_true(self) -> None:
        from dataenginex.domains.ai.runtime.executor import AgentConfig

        dag = AgentDAG()
        dag.add_node("a", AgentConfig(name="a", model="mock"))
        dag.add_node("b", AgentConfig(name="b", model="mock"))
        dag.add_edge("a", "b")
        assert dag.validate() is True

    def test_validate_cyclic_dag_returns_false(self) -> None:
        from dataenginex.domains.ai.runtime.executor import AgentConfig

        dag = AgentDAG()
        dag.add_node("a", AgentConfig(name="a", model="mock"))
        dag.add_node("b", AgentConfig(name="b", model="mock"))
        dag.add_edge("a", "b")
        dag.add_edge("b", "a")
        assert dag.validate() is False

    def test_node_without_provider_gets_placeholder(self) -> None:
        from dataenginex.domains.ai.runtime.executor import AgentConfig

        dag = AgentDAG()
        dag.add_node("orphan", AgentConfig(name="orphan", model="mock"))
        results = dag.execute(providers={}, initial_input="data")
        assert "orphan" in results
        assert "no provider" in results["orphan"]

    def test_empty_dag_returns_empty_results(self) -> None:
        dag = AgentDAG()
        results = dag.execute(initial_input="nothing")
        assert results == {}
