"""Tests for new AI submodules: memory, observability, routing, runtime, workflows."""

from __future__ import annotations

import tempfile
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


class TestShortTermMemory:
    def test_add_and_recent(self) -> None:
        from dataenginex.domains.ai.memory.base import MemoryEntry, ShortTermMemory

        m = ShortTermMemory()
        m.add(MemoryEntry(content="hello", role="user"))
        m.add(MemoryEntry(content="world", role="assistant"))
        recent = m.recent(5)
        assert len(recent) == 2
        assert recent[-1].content == "world"

    def test_max_entries_eviction(self) -> None:
        from dataenginex.domains.ai.memory.base import MemoryEntry, ShortTermMemory

        m = ShortTermMemory(max_entries=2)
        for i in range(3):
            m.add(MemoryEntry(content=f"msg{i}", role="user"))
        assert len(m.recent(10)) == 2
        assert m.recent(10)[0].content == "msg1"

    def test_search(self) -> None:
        from dataenginex.domains.ai.memory.base import MemoryEntry, ShortTermMemory

        m = ShortTermMemory()
        m.add(MemoryEntry(content="pipeline failed", role="user"))
        m.add(MemoryEntry(content="all good", role="assistant"))
        results = m.search("pipeline")
        assert len(results) == 1
        assert "pipeline" in results[0].content

    def test_clear(self) -> None:
        from dataenginex.domains.ai.memory.base import MemoryEntry, ShortTermMemory

        m = ShortTermMemory()
        m.add(MemoryEntry(content="x", role="user"))
        m.clear()
        assert m.recent() == []

    def test_base_memory_is_abstract(self) -> None:
        from dataenginex.domains.ai.memory.base import BaseMemory

        with pytest.raises(TypeError):
            BaseMemory()  # type: ignore[abstract]


class TestLongTermMemory:
    def test_add_and_recent(self) -> None:
        from dataenginex.domains.ai.memory.base import MemoryEntry
        from dataenginex.domains.ai.memory.long_term import LongTermMemory

        m = LongTermMemory()
        m.add(MemoryEntry(content="fact about pipelines", role="system"))
        assert len(m.recent(5)) == 1

    def test_search_keyword(self) -> None:
        from dataenginex.domains.ai.memory.base import MemoryEntry
        from dataenginex.domains.ai.memory.long_term import LongTermMemory

        m = LongTermMemory()
        m.add(MemoryEntry(content="duckdb is fast for analytics", role="system"))
        m.add(MemoryEntry(content="use structlog for logging", role="system"))
        results = m.search("duckdb analytics")
        assert len(results) >= 1
        assert "duckdb" in results[0].content

    def test_clear(self) -> None:
        from dataenginex.domains.ai.memory.base import MemoryEntry
        from dataenginex.domains.ai.memory.long_term import LongTermMemory

        m = LongTermMemory()
        m.add(MemoryEntry(content="x", role="user"))
        m.clear()
        assert m.recent() == []

    def test_persist_and_load(self) -> None:
        from dataenginex.domains.ai.memory.base import MemoryEntry
        from dataenginex.domains.ai.memory.long_term import LongTermMemory

        m = LongTermMemory()
        m.add(MemoryEntry(content="fact one", role="system"))
        m.add(MemoryEntry(content="fact two", role="system"))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        m.persist(path)

        m2 = LongTermMemory()
        m2.load_from_file(path)
        assert len(m2.recent(10)) == 2
        assert m2.recent(10)[0].content == "fact one"
        assert m2.recent(10)[1].content == "fact two"

    def test_persist_overwrites(self) -> None:
        from dataenginex.domains.ai.memory.base import MemoryEntry
        from dataenginex.domains.ai.memory.long_term import LongTermMemory

        m = LongTermMemory()
        m.add(MemoryEntry(content="v1", role="user"))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        m.persist(path)
        m.add(MemoryEntry(content="v2", role="user"))
        m.persist(path)

        m2 = LongTermMemory()
        m2.load_from_file(path)
        assert len(m2.recent(10)) == 2

    def test_timestamp_auto_set(self) -> None:
        from dataenginex.domains.ai.memory.base import MemoryEntry
        from dataenginex.domains.ai.memory.long_term import LongTermMemory

        m = LongTermMemory()
        entry = MemoryEntry(content="x", role="user", timestamp=0.0)
        m.add(entry)
        assert m.recent()[0].timestamp > 0


class TestEpisodicMemory:
    def _episode(self, task: str, reward: float = 1.0) -> Any:
        import time

        from dataenginex.domains.ai.memory.episodic import Episode

        return Episode(task=task, steps=[], outcome="ok", reward=reward, timestamp=time.time())

    def test_add_and_recall(self) -> None:
        from dataenginex.domains.ai.memory.episodic import EpisodicMemory

        m = EpisodicMemory()
        m.add_episode(self._episode("run pipeline"))
        results = m.recall_similar("pipeline")
        assert len(results) == 1

    def test_recall_similar_ranked(self) -> None:
        from dataenginex.domains.ai.memory.episodic import EpisodicMemory

        m = EpisodicMemory()
        m.add_episode(self._episode("run data pipeline on duckdb"))
        m.add_episode(self._episode("train ML model"))
        m.add_episode(self._episode("run pipeline sync"))
        results = m.recall_similar("run pipeline", top_k=2)
        assert len(results) == 2
        assert "pipeline" in results[0].task

    def test_best_episodes(self) -> None:
        from dataenginex.domains.ai.memory.episodic import EpisodicMemory

        m = EpisodicMemory()
        m.add_episode(self._episode("low reward", reward=0.1))
        m.add_episode(self._episode("high reward", reward=0.9))
        best = m.best_episodes(top_k=1)
        assert best[0].reward == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


class TestAuditLog:
    def test_log_and_get_all(self) -> None:
        import time

        from dataenginex.domains.ai.observability.audit import AuditEntry, AuditLog

        ts = time.time()
        log = AuditLog()
        log.log(AuditEntry(agent_name="a1", action="search", input="q", output="r", timestamp=ts))
        log.log(AuditEntry(agent_name="a2", action="run", input="x", output="y", timestamp=ts))
        assert len(log.get_entries()) == 2

    def test_filter_by_agent(self) -> None:
        import time

        from dataenginex.domains.ai.observability.audit import AuditEntry, AuditLog

        ts = time.time()
        log = AuditLog()
        log.log(AuditEntry(agent_name="a1", action="x", input="i", output="o", timestamp=ts))
        log.log(AuditEntry(agent_name="a2", action="y", input="i", output="o", timestamp=ts))
        entries = log.get_entries(agent_name="a1")
        assert len(entries) == 1
        assert entries[0].agent_name == "a1"

    def test_limit(self) -> None:
        import time

        from dataenginex.domains.ai.observability.audit import AuditEntry, AuditLog

        ts = time.time()
        log = AuditLog()
        for i in range(5):
            e = AuditEntry(agent_name="a", action=f"act{i}", input="i", output="o", timestamp=ts)
            log.log(e)
        assert len(log.get_entries(limit=3)) == 3


class TestCostTracker:
    def test_record_and_total(self) -> None:
        from dataenginex.domains.ai.observability.cost import CostTracker, TokenUsage

        t = CostTracker()
        t.record(
            TokenUsage(model="m1", tokens_in=100, tokens_out=50, cost_usd=0.01, agent_name="a")
        )  # noqa: E501
        t.record(
            TokenUsage(model="m2", tokens_in=200, tokens_out=100, cost_usd=0.05, agent_name="b")
        )  # noqa: E501
        assert abs(t.total_cost() - 0.06) < 1e-9

    def test_filter_by_agent(self) -> None:
        from dataenginex.domains.ai.observability.cost import CostTracker, TokenUsage

        t = CostTracker()
        t.record(
            TokenUsage(model="m1", tokens_in=100, tokens_out=50, cost_usd=0.01, agent_name="a")
        )  # noqa: E501
        t.record(
            TokenUsage(model="m2", tokens_in=200, tokens_out=100, cost_usd=0.05, agent_name="b")
        )  # noqa: E501
        assert abs(t.total_cost(agent_name="a") - 0.01) < 1e-9

    def test_summary(self) -> None:
        from dataenginex.domains.ai.observability.cost import CostTracker, TokenUsage

        t = CostTracker()
        t.record(TokenUsage(model="sonnet", tokens_in=100, tokens_out=50, cost_usd=0.01))
        s = t.summary()
        assert s["total_records"] == 1
        assert s["total_tokens_in"] == 100
        assert s["total_tokens_out"] == 50
        assert "sonnet" in s["by_model"]


class TestAgentMetrics:
    def test_increment_requests(self) -> None:
        from dataenginex.domains.ai.observability.metrics import AgentMetrics

        m = AgentMetrics()
        m.increment_requests("agent1")
        m.increment_requests("agent1")
        assert m._requests["agent1"] == 2

    def test_record_latency(self) -> None:
        from dataenginex.domains.ai.observability.metrics import AgentMetrics

        m = AgentMetrics()
        m.record_latency("agent1", 0.5)
        m.record_latency("agent1", 1.2)
        assert len(m._latencies["agent1"]) == 2


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


class TestModelRouter:
    def _providers(self) -> dict[str, Any]:
        from dataenginex.domains.ai.providers import BaseProvider

        class DummyProvider(BaseProvider):
            def generate(self, prompt: str, **kwargs: Any) -> str:
                return "ok"

        return {
            "ollama": DummyProvider(),
            "openai": DummyProvider(),
            "anthropic": DummyProvider(),
        }

    def test_route_simple(self) -> None:
        from dataenginex.providers.model.router import ModelRouter

        router = ModelRouter(self._providers())
        p = router.route("do something simple", complexity="simple")
        assert p is not None

    def test_route_moderate(self) -> None:
        from dataenginex.providers.model.router import ModelRouter

        router = ModelRouter(self._providers())
        p = router.route("moderate task", complexity="moderate")
        assert p is not None

    def test_route_complex(self) -> None:
        from dataenginex.providers.model.router import ModelRouter

        router = ModelRouter(self._providers())
        p = router.route("hard task", complexity="complex")
        assert p is not None

    def test_unknown_complexity_raises(self) -> None:
        from dataenginex.providers.model.router import ModelRouter

        router = ModelRouter(self._providers())
        with pytest.raises(ValueError, match="Unknown complexity"):
            router.route("task", complexity="impossible")

    def test_missing_provider_raises(self) -> None:
        from dataenginex.providers.model.router import ModelRouter

        router = ModelRouter({})  # no providers registered
        with pytest.raises(KeyError):
            router.route("task", complexity="simple")

    def test_base_provider_is_abstract(self) -> None:
        from dataenginex.domains.ai.providers import BaseProvider

        with pytest.raises(TypeError):
            BaseProvider()  # type: ignore[abstract]


class TestRoutingAdapters:
    def test_anthropic_no_key_raises(self) -> None:
        from dataenginex.providers.model.anthropic import AnthropicProvider

        p = AnthropicProvider(api_key="")
        with patch.dict("os.environ", {}, clear=True):
            p.api_key = ""
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                p.generate("hi")

    def test_anthropic_generate_mocked(self) -> None:
        from dataenginex.providers.model.anthropic import AnthropicProvider

        p = AnthropicProvider(api_key="sk-ant-test")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"content": [{"text": "Hello from Claude"}]}
        mock_resp.raise_for_status = MagicMock()
        with patch("httpx.post", return_value=mock_resp):
            result = p.generate("Say hello")
        assert result == "Hello from Claude"

    def test_anthropic_connection_error(self) -> None:
        import httpx

        from dataenginex.providers.model.anthropic import AnthropicProvider

        p = AnthropicProvider(api_key="sk-ant-test")
        with (
            patch("httpx.post", side_effect=httpx.ConnectError("refused")),
            pytest.raises(ConnectionError, match="not reachable"),
        ):
            p.generate("hi")

    def test_openai_no_key_raises(self) -> None:
        from dataenginex.providers.model.openai import OpenAIProvider

        p = OpenAIProvider(api_key="")
        with patch.dict("os.environ", {}, clear=True):
            p.api_key = ""
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                p.generate("hi")

    def test_openai_generate_mocked(self) -> None:
        from dataenginex.providers.model.openai import OpenAIProvider

        p = OpenAIProvider(api_key="sk-test")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "Hello from GPT"}}]}
        mock_resp.raise_for_status = MagicMock()
        with patch("httpx.post", return_value=mock_resp):
            result = p.generate("Say hello")
        assert result == "Hello from GPT"

    def test_openai_connection_error(self) -> None:
        import httpx

        from dataenginex.providers.model.openai import OpenAIProvider

        p = OpenAIProvider(api_key="sk-test")
        with (
            patch("httpx.post", side_effect=httpx.ConnectError("refused")),
            pytest.raises(ConnectionError, match="not reachable"),
        ):
            p.generate("hi")

    def test_ollama_generate_mocked(self) -> None:
        from dataenginex.providers.model.ollama import OllamaProvider

        p = OllamaProvider(model="llama3")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "Paris"}
        mock_resp.raise_for_status = MagicMock()
        with patch("httpx.post", return_value=mock_resp):
            result = p.generate("Capital of France?")
        assert result == "Paris"

    def test_ollama_connection_error(self) -> None:
        import httpx

        from dataenginex.providers.model.ollama import OllamaProvider

        p = OllamaProvider(host="http://localhost:99999")
        with (
            patch("httpx.post", side_effect=httpx.ConnectError("refused")),
            pytest.raises(ConnectionError, match="not reachable"),
        ):
            p.generate("hi")


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


class TestCheckpointManager:
    def test_save_and_load(self) -> None:
        from dataenginex.domains.ai.runtime.checkpoint import Checkpoint, CheckpointManager

        mgr = CheckpointManager()
        cp = Checkpoint(agent_name="agent1", state={"step": 3}, timestamp=1.0, iteration=3)
        mgr.save(cp)
        loaded = mgr.load("agent1")
        assert loaded is not None
        assert loaded.state["step"] == 3

    def test_load_missing_returns_none(self) -> None:
        from dataenginex.domains.ai.runtime.checkpoint import CheckpointManager

        mgr = CheckpointManager()
        assert mgr.load("ghost") is None

    def test_save_overwrites(self) -> None:
        from dataenginex.domains.ai.runtime.checkpoint import Checkpoint, CheckpointManager

        mgr = CheckpointManager()
        mgr.save(Checkpoint(agent_name="a", state={"v": 1}, timestamp=1.0, iteration=1))
        mgr.save(Checkpoint(agent_name="a", state={"v": 2}, timestamp=2.0, iteration=2))
        assert mgr.load("a").state["v"] == 2  # type: ignore[union-attr]


def _make_provider(response: str) -> Any:
    """Return a mock BaseProvider that always returns *response*."""
    from dataenginex.domains.ai.providers import BaseProvider

    class _Mock(BaseProvider):
        def generate(self, prompt: str, **kwargs: Any) -> str:
            return response

    return _Mock()


class TestAgentExecutor:
    def test_run_returns_final_answer(self) -> None:
        from dataenginex.domains.ai.memory.base import ShortTermMemory
        from dataenginex.domains.ai.runtime.executor import AgentConfig, AgentExecutor
        from dataenginex.domains.ai.tools import ToolRegistry

        config = AgentConfig(name="test", model="mock")
        executor = AgentExecutor(
            config, ToolRegistry(), ShortTermMemory(), _make_provider("The answer is 42.")
        )
        resp = executor.run("What is 6×7?")
        assert resp.content == "The answer is 42."
        assert resp.iterations == 1

    def test_run_tool_call_then_final(self) -> None:
        from dataenginex.domains.ai.memory.base import ShortTermMemory
        from dataenginex.domains.ai.providers import BaseProvider
        from dataenginex.domains.ai.runtime.executor import AgentConfig, AgentExecutor
        from dataenginex.domains.ai.tools import ToolRegistry, ToolSpec

        registry = ToolRegistry()
        registry.register(ToolSpec(name="add", description="adds", fn=lambda a, b: a + b))

        responses = iter(['TOOL: add\nARGS: {"a": 1, "b": 2}', "FINAL: 3"])

        class SequentialProvider(BaseProvider):
            def generate(self, prompt: str, **kwargs: Any) -> str:
                return next(responses)

        config = AgentConfig(name="test", model="mock")
        executor = AgentExecutor(config, registry, ShortTermMemory(), SequentialProvider())
        resp = executor.run("What is 1+2?")
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0]["tool"] == "add"

    def test_step_returns_step_result(self) -> None:
        from dataenginex.domains.ai.memory.base import MemoryEntry, ShortTermMemory
        from dataenginex.domains.ai.runtime.executor import AgentConfig, AgentExecutor, StepResult
        from dataenginex.domains.ai.tools import ToolRegistry

        mem = ShortTermMemory()
        mem.add(MemoryEntry(content="What is Python?", role="user"))
        config = AgentConfig(name="test", model="mock")
        executor = AgentExecutor(
            config, ToolRegistry(), mem, _make_provider("Python is a language.")
        )
        result = executor.step()
        assert isinstance(result, StepResult)
        assert result.action == "respond"
        assert result.result == "Python is a language."

    def test_run_max_iterations(self) -> None:
        from dataenginex.domains.ai.memory.base import ShortTermMemory
        from dataenginex.domains.ai.runtime.executor import AgentConfig, AgentExecutor
        from dataenginex.domains.ai.tools import ToolRegistry

        # Provider always asks for a (non-existent) tool — will hit max_iterations
        config = AgentConfig(name="test", model="mock", max_iterations=2)
        executor = AgentExecutor(
            config,
            ToolRegistry(),
            ShortTermMemory(),
            _make_provider("TOOL: missing_tool\nARGS: {}"),
        )
        resp = executor.run("loop forever")
        assert "Max iterations" in resp.content
        assert resp.iterations == 2


class TestSandbox:
    @pytest.fixture(autouse=True)
    def _suppress_experimental_warning(self) -> Any:
        """Sandbox is marked experimental — silence the warning for tests
        that focus on behavior, not the warning itself."""
        import warnings

        from dataenginex.domains.ai.runtime.sandbox import ExperimentalFeatureWarning

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ExperimentalFeatureWarning)
            yield

    def test_experimental_warning_fires_on_first_use(self) -> None:
        """First instantiation in a fresh module state emits the warning."""
        import warnings

        import dataenginex.domains.ai.runtime.sandbox as sb_mod

        sb_mod._experimental_warned = False
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", sb_mod.ExperimentalFeatureWarning)
            sb_mod.Sandbox()
        assert any(issubclass(w.category, sb_mod.ExperimentalFeatureWarning) for w in caught)

    def test_run_python(self) -> None:
        from dataenginex.domains.ai.runtime.sandbox import Sandbox

        sb = Sandbox()
        result = sb.execute_code("print('hello sandbox')", language="python")
        assert result.exit_code == 0
        assert "hello sandbox" in result.output
        assert not result.timed_out

    def test_run_python_error(self) -> None:
        from dataenginex.domains.ai.runtime.sandbox import Sandbox

        sb = Sandbox()
        result = sb.execute_code("raise ValueError('oops')", language="python")
        assert result.exit_code != 0

    def test_run_bash(self) -> None:
        from dataenginex.domains.ai.runtime.sandbox import Sandbox

        sb = Sandbox()
        result = sb.execute_code("echo hello_bash", language="bash")
        assert result.exit_code == 0
        assert "hello_bash" in result.output

    def test_timeout(self) -> None:
        from dataenginex.domains.ai.runtime.sandbox import Sandbox, SandboxConfig

        cfg = SandboxConfig(timeout_s=1)
        sb = Sandbox(config=cfg)
        result = sb.execute_code("import time; time.sleep(10)", language="python")
        assert result.timed_out

    def test_unsupported_language_raises(self) -> None:
        from dataenginex.domains.ai.runtime.sandbox import Sandbox, UnsupportedLanguageError

        sb = Sandbox()
        with pytest.raises(UnsupportedLanguageError):
            sb.execute_code("console.log('hi')", language="javascript")

    def test_metadata_contains_language(self) -> None:
        from dataenginex.domains.ai.runtime.sandbox import Sandbox

        sb = Sandbox()
        result = sb.execute_code("print(1)", language="python")
        assert result.metadata.get("language") == "python"


# ---------------------------------------------------------------------------
# Workflows
# ---------------------------------------------------------------------------


class TestCondition:
    def test_eq(self) -> None:
        from dataenginex.domains.ai.workflows.conditions import Condition

        c = Condition("status", "eq", "ok")
        assert c.evaluate({"status": "ok"}) is True
        assert c.evaluate({"status": "fail"}) is False

    def test_ne(self) -> None:
        from dataenginex.domains.ai.workflows.conditions import Condition

        c = Condition("status", "ne", "ok")
        assert c.evaluate({"status": "fail"}) is True

    def test_gt(self) -> None:
        from dataenginex.domains.ai.workflows.conditions import Condition

        c = Condition("score", "gt", 0.5)
        assert c.evaluate({"score": 0.9}) is True
        assert c.evaluate({"score": 0.3}) is False

    def test_lt(self) -> None:
        from dataenginex.domains.ai.workflows.conditions import Condition

        c = Condition("score", "lt", 0.5)
        assert c.evaluate({"score": 0.1}) is True

    def test_contains(self) -> None:
        from dataenginex.domains.ai.workflows.conditions import Condition

        c = Condition("tags", "contains", "prod")
        assert c.evaluate({"tags": ["prod", "v2"]}) is True
        assert c.evaluate({"tags": ["dev"]}) is False

    def test_unknown_operator_raises(self) -> None:
        from dataenginex.domains.ai.workflows.conditions import Condition

        with pytest.raises(ValueError, match="Unknown operator"):
            Condition("x", "magic", "y")

    def test_missing_field_raises(self) -> None:
        from dataenginex.domains.ai.workflows.conditions import Condition

        c = Condition("missing_key", "eq", "x")
        with pytest.raises(KeyError):
            c.evaluate({"other": "value"})


class TestAgentDAG:
    def test_add_nodes_and_edges(self) -> None:
        from dataenginex.domains.ai.runtime.executor import AgentConfig
        from dataenginex.domains.ai.workflows.dag import AgentDAG

        dag = AgentDAG()
        dag.add_node("a", AgentConfig(name="a", model="mock"))
        dag.add_node("b", AgentConfig(name="b", model="mock"))
        dag.add_edge("a", "b")
        assert len(dag._nodes) == 2
        assert len(dag._edges) == 1

    def test_validate_no_cycle(self) -> None:
        from dataenginex.domains.ai.runtime.executor import AgentConfig
        from dataenginex.domains.ai.workflows.dag import AgentDAG

        dag = AgentDAG()
        dag.add_node("a", AgentConfig(name="a", model="mock"))
        dag.add_node("b", AgentConfig(name="b", model="mock"))
        dag.add_node("c", AgentConfig(name="c", model="mock"))
        dag.add_edge("a", "b")
        dag.add_edge("b", "c")
        assert dag.validate() is True

    def test_validate_detects_cycle(self) -> None:
        from dataenginex.domains.ai.runtime.executor import AgentConfig
        from dataenginex.domains.ai.workflows.dag import AgentDAG

        dag = AgentDAG()
        dag.add_node("a", AgentConfig(name="a", model="mock"))
        dag.add_node("b", AgentConfig(name="b", model="mock"))
        dag.add_edge("a", "b")
        dag.add_edge("b", "a")  # cycle
        assert dag.validate() is False

    def test_execute_with_providers(self) -> None:
        from dataenginex.domains.ai.runtime.executor import AgentConfig
        from dataenginex.domains.ai.workflows.dag import AgentDAG

        dag = AgentDAG()
        dag.add_node("step1", AgentConfig(name="step1", model="mock"))
        dag.add_node("step2", AgentConfig(name="step2", model="mock"))
        dag.add_edge("step1", "step2")

        providers = {
            "step1": _make_provider("output_from_step1"),
            "step2": _make_provider("output_from_step2"),
        }
        results = dag.execute(providers=providers, initial_input="start")
        assert results["step1"] == "output_from_step1"
        assert results["step2"] == "output_from_step2"

    def test_execute_cycle_raises(self) -> None:
        from dataenginex.domains.ai.runtime.executor import AgentConfig
        from dataenginex.domains.ai.workflows.dag import AgentDAG

        dag = AgentDAG()
        dag.add_node("a", AgentConfig(name="a", model="mock"))
        dag.add_node("b", AgentConfig(name="b", model="mock"))
        dag.add_edge("a", "b")
        dag.add_edge("b", "a")
        with pytest.raises(ValueError, match="cycle"):
            dag.execute()

    def test_execute_no_provider_placeholder(self) -> None:
        from dataenginex.domains.ai.runtime.executor import AgentConfig
        from dataenginex.domains.ai.workflows.dag import AgentDAG

        dag = AgentDAG()
        dag.add_node("orphan", AgentConfig(name="orphan", model="mock"))
        results = dag.execute(providers={})
        assert "no provider" in results["orphan"]


class TestApprovalGate:
    def test_approve_yes(self) -> None:
        from dataenginex.domains.ai.workflows.human_loop import ApprovalGate

        gate = ApprovalGate("Deploy to production?")
        with patch("builtins.input", return_value="y"):
            assert gate.request_approval({"env": "prod"}) is True

    def test_approve_no(self) -> None:
        from dataenginex.domains.ai.workflows.human_loop import ApprovalGate

        gate = ApprovalGate("Deploy to production?")
        with patch("builtins.input", return_value="n"):
            assert gate.request_approval({"env": "prod"}) is False

    def test_approve_eof_returns_false(self) -> None:
        from dataenginex.domains.ai.workflows.human_loop import ApprovalGate

        gate = ApprovalGate("Approve?")
        with patch("builtins.input", side_effect=EOFError):
            assert gate.request_approval({}) is False

    def test_description_and_timeout_stored(self) -> None:
        from dataenginex.domains.ai.workflows.human_loop import ApprovalGate

        gate = ApprovalGate("Review data", timeout_seconds=60)
        assert gate.description == "Review data"
        assert gate.timeout_seconds == 60
