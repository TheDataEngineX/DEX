"""Tests for the built-in serving engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dataenginex.domains.ml.registry import ModelArtifact, ModelRegistry, ModelStage
from dataenginex.domains.ml.serving_engine.builtin import BuiltinServingEngine
from dataenginex.domains.ml.training import SklearnTrainer


class _DummyEstimator:
    """Minimal sklearn-API-compatible estimator for testing."""

    def __init__(self, value: int = 42) -> None:
        self.value = value
        self._fitted = False

    def fit(self, X: Any, y: Any) -> None:
        self._fitted = True

    def predict(self, X: Any) -> list[int]:
        return [self.value] * len(X)

    def score(self, X: Any, y: Any) -> float:
        return 0.9

    def get_params(self) -> dict[str, Any]:
        return {"value": self.value}

    def set_params(self, **params: Any) -> None:
        for k, v in params.items():
            setattr(self, k, v)


class TestBuiltinServingEngine:
    def _setup_model(self, tmp_path: Path) -> tuple[ModelRegistry, str]:
        """Train, save, and register a dummy model. Returns (registry, model_path)."""
        model_path = str(tmp_path / "model.pkl")
        trainer = SklearnTrainer("test-model", "1.0.0", _DummyEstimator(42))
        trainer.train([[1], [2]], [0, 1])
        trainer.save(model_path)

        registry = ModelRegistry(persist_path=str(tmp_path / "registry.json"))
        registry.register(
            ModelArtifact(
                name="test-model",
                version="1.0.0",
                artifact_path=model_path,
                stage=ModelStage.PRODUCTION,
            )
        )
        return registry, model_path

    def test_load_and_predict(self, tmp_path: Path) -> None:
        registry, _ = self._setup_model(tmp_path)
        engine = BuiltinServingEngine(
            model_registry=registry,
            extra_modules=frozenset({"tests"}),
        )
        engine.load_model("test-model")
        preds = engine.predict("test-model", [{"x": 1}, {"x": 2}])
        assert len(preds) == 2
        assert all(p == 42 for p in preds)

    def test_list_models(self, tmp_path: Path) -> None:
        registry, _ = self._setup_model(tmp_path)
        engine = BuiltinServingEngine(
            model_registry=registry,
            extra_modules=frozenset({"tests"}),
        )
        assert engine.list_models() == []
        engine.load_model("test-model")
        assert "test-model" in engine.list_models()

    def test_predict_without_load_raises(self, tmp_path: Path) -> None:
        registry = ModelRegistry(persist_path=str(tmp_path / "empty_registry.json"))
        engine = BuiltinServingEngine(
            model_registry=registry,
            extra_modules=frozenset({"tests"}),
        )
        with pytest.raises((RuntimeError, KeyError)):
            engine.predict("test-model", [{"x": 1}])

    def test_load_nonexistent_model(self, tmp_path: Path) -> None:
        registry = ModelRegistry()
        engine = BuiltinServingEngine(
            model_registry=registry,
            extra_modules=frozenset({"tests"}),
        )
        with pytest.raises(KeyError, match="not found"):
            engine.load_model("nonexistent")
