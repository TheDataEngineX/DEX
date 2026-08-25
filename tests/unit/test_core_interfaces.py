"""Tests proving Base* ABCs enforce their contracts."""

from __future__ import annotations

from typing import Any

import pytest

from dataenginex.foundation.plugin_contracts import (
    BaseAgentRuntime,
    BaseConnector,
    BaseFeatureStore,
    BaseLLMProvider,
    BaseOrchestrator,
    BaseRetriever,
    BaseServingEngine,
    BaseTracker,
    BaseTransform,
    BaseVectorStore,
)

ALL_ABCS = [
    BaseConnector,
    BaseTransform,
    BaseTracker,
    BaseRetriever,
    BaseFeatureStore,
    BaseOrchestrator,
    BaseServingEngine,
    BaseAgentRuntime,
    BaseLLMProvider,
    BaseVectorStore,
]


class TestABCsCannotBeInstantiated:
    @pytest.mark.parametrize("abc_cls", ALL_ABCS, ids=lambda c: c.__name__)
    def test_cannot_instantiate(self, abc_cls: type) -> None:
        with pytest.raises(TypeError, match="abstract"):
            abc_cls()  # type: ignore[abstract]


class TestBaseConnectorContract:
    def test_minimal_implementation(self) -> None:
        class DummyConnector(BaseConnector):
            def connect(self) -> None:
                pass

            def disconnect(self) -> None:
                pass

            def read(self, **kwargs: Any) -> Any:
                return []

            def write(self, data: Any, **kwargs: Any) -> None:
                pass

            def health_check(self) -> bool:
                return True

        c = DummyConnector()
        assert c.health_check() is True


class TestBaseTransformContract:
    def test_minimal_implementation(self) -> None:
        class DummyTransform(BaseTransform):
            @property
            def name(self) -> str:
                return "dummy"

            def apply(self, data: Any) -> Any:
                return data

        t = DummyTransform()
        assert t.name == "dummy"
        assert t.apply(42) == 42


class TestBaseTrackerContract:
    def test_minimal_implementation(self) -> None:
        class DummyTracker(BaseTracker):
            def create_experiment(self, name: str) -> str:
                return "exp-1"

            def log_params(self, run_id: str, params: dict[str, Any]) -> None:
                pass

            def log_metrics(
                self, run_id: str, metrics: dict[str, float], step: int | None = None
            ) -> None:
                pass

            def start_run(self, experiment_id: str, run_name: str | None = None) -> str:
                return "run-1"

            def end_run(self, run_id: str, status: str = "FINISHED") -> None:
                pass

            def list_runs(self, experiment_id: str) -> list[dict[str, Any]]:
                return []

            def list_experiments(self) -> list[dict[str, Any]]:
                return []

        t = DummyTracker()
        assert t.create_experiment("test") == "exp-1"


class TestBaseRetrieverContract:
    def test_minimal_implementation(self) -> None:
        class DummyRetriever(BaseRetriever):
            def retrieve(self, query: str, top_k: int = 10, **kwargs: Any) -> list[dict[str, Any]]:
                return []

        r = DummyRetriever()
        assert r.retrieve("test") == []


class TestBaseLLMProviderContract:
    def test_minimal_implementation(self) -> None:
        class DummyLLM(BaseLLMProvider):
            async def generate(self, prompt: str, **kwargs: Any) -> str:
                return "hello"

        llm = DummyLLM()
        assert isinstance(llm, BaseLLMProvider)
