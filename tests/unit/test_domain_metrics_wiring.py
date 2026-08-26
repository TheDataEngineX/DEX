"""Tests verifying domain metrics are emitted from production code paths."""

from __future__ import annotations

from prometheus_client import REGISTRY

from dataenginex.domains.ai.agents.builtin import BuiltinAgentRuntime
from dataenginex.domains.ai.tools import ToolRegistry, ToolSpec
from dataenginex.domains.ml.drift import DriftDetector  # noqa: F401 — ensures registration


def _sample_value(metric: str, labels: dict[str, str]) -> float:
    """Return the current value for a labeled sample from REGISTRY."""
    for family in REGISTRY.collect():
        for sample in family.samples:
            if sample.name == metric and sample.labels == labels:
                return float(sample.value)
    return 0.0


class TestAgentRuntimeMetrics:
    async def _run(self, agent: BuiltinAgentRuntime, message: str) -> None:
        await agent.run(message)

    def test_agent_iterations_observed_on_run(self) -> None:
        import asyncio

        agent = BuiltinAgentRuntime(llm=None, name="wiring-test")
        asyncio.run(agent.run("hello"))
        # Histogram bucket count: _count sample must exist for this agent
        observed = _sample_value("dex_ai_agent_iterations_count", {"agent": "wiring-test"})
        assert observed >= 1

    def test_tool_call_ok_counter_increments(self) -> None:
        import asyncio

        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                name="echo",
                description="echo",
                fn=lambda text: text,
                parameters={"text": {"type": "string"}},
            )
        )

        class _ToolOnceLLM:
            def __init__(self) -> None:
                self._count = 0

            def chat(self, messages):  # type: ignore[no-untyped-def]
                from dataenginex.domains.ai.llm import LLMResponse

                self._count += 1
                if self._count == 1:
                    return LLMResponse(text='TOOL: echo ARGS: {"text": "hi"}', model="m")
                return LLMResponse(text="ANSWER: done", model="m")

        before = _sample_value("dex_ai_tool_calls_total", {"tool": "echo", "status": "ok"})
        agent = BuiltinAgentRuntime(
            llm=_ToolOnceLLM(), tools=registry, name="tool-test", max_iterations=3
        )
        asyncio.run(agent.run("go"))
        after = _sample_value("dex_ai_tool_calls_total", {"tool": "echo", "status": "ok"})
        assert after >= before + 1
