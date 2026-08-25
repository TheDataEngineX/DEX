"""Built-in model serving engine.

Wraps the existing ModelServer + ModelRegistry for BaseServingEngine compliance.
For production: use BentoML via ``[bentoml]`` extra.
"""

from __future__ import annotations

from typing import Any

import structlog

from dataenginex.domains.ml.registry import ModelRegistry
from dataenginex.domains.ml.serving import ModelServer, PredictionRequest
from dataenginex.domains.ml.serving_engine import serving_registry
from dataenginex.domains.ml.training import SklearnTrainer
from dataenginex.foundation.plugin_contracts import BaseServingEngine

logger = structlog.get_logger()


@serving_registry.decorator("builtin", is_default=True)
class BuiltinServingEngine(BaseServingEngine):
    """Built-in serving engine backed by ModelServer + ModelRegistry.

    Args:
        model_registry: ModelRegistry instance for model resolution.
        model_dir: Directory where model artifacts are stored.
    """

    def __init__(
        self,
        model_registry: ModelRegistry | None = None,
        model_dir: str = ".dex/models",
        extra_modules: frozenset[str] | None = None,
        **kwargs: Any,
    ) -> None:
        persist = f"{model_dir}/registry.json"
        self._registry = model_registry or ModelRegistry(persist_path=persist)
        self._model_dir = model_dir
        self._extra_modules = extra_modules or frozenset()
        self._server = ModelServer(self._registry)
        self._loaded: dict[str, SklearnTrainer] = {}

    def load_model(self, model_name: str, version: str | None = None) -> None:
        """Load a model for serving."""
        if version:
            artifact = self._registry.get(model_name, version)
        else:
            artifact = self._registry.get_production(model_name)
            if artifact is None:
                artifact = self._registry.get_latest(model_name)

        if artifact is None:
            msg = f"Model '{model_name}' not found"
            raise KeyError(msg)

        trainer = SklearnTrainer(model_name, artifact.version)
        trainer.load(artifact.artifact_path, extra_modules=self._extra_modules)
        self._loaded[model_name] = trainer
        self._server.load_model(model_name, artifact.version, trainer)
        logger.info(
            "model loaded for serving",
            model=model_name,
            version=artifact.version,
        )

    def predict(self, model_name: str, data: Any) -> Any:
        """Run inference on loaded model, auto-loading from registry if needed."""
        if model_name not in self._loaded:
            self.load_model(model_name)

        request = PredictionRequest(
            model_name=model_name,
            features=data if isinstance(data, list) else [data],
        )
        response = self._server.predict(request)
        return response.predictions

    def list_models(self) -> list[str]:
        """List currently loaded models."""
        return list(self._loaded.keys())
