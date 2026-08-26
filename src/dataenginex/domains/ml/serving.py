"""
Model serving — lightweight prediction interface.

``ModelServer`` wraps a trained model (any object with a ``predict`` method)
and provides a structured request/response contract for inference.
"""

from __future__ import annotations

import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog

logger = structlog.get_logger()
__all__ = [
    "ModelServer",
    "PredictionRequest",
    "PredictionResponse",
]

_MAX_LOADED_MODELS = int(os.getenv("DEX_MAX_LOADED_MODELS", "3"))


@dataclass
class PredictionRequest:
    """Input to the serving layer.

    Attributes:
        model_name: Name of the model to invoke.
        version: Model version (``None`` resolves to the production version).
        features: List of feature dicts — each dict is one sample.
        request_id: Caller-provided request ID for tracing.
    """

    model_name: str
    version: str | None = None  # None → use production model
    features: list[dict[str, Any]] = field(default_factory=list)
    request_id: str = ""


@dataclass
class PredictionResponse:
    """Output from the serving layer.

    Attributes:
        model_name: Name of the model that produced predictions.
        version: Version of the model used.
        predictions: List of prediction values.
        latency_ms: Inference latency in milliseconds.
        request_id: Echoed request ID for tracing.
        served_at: Timestamp of the prediction.
    """

    model_name: str
    version: str
    predictions: list[Any] = field(default_factory=list)
    latency_ms: float = 0.0
    request_id: str = ""
    served_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the prediction response to a plain dictionary."""
        return {
            "model_name": self.model_name,
            "version": self.version,
            "predictions": self.predictions,
            "latency_ms": round(self.latency_ms, 2),
            "request_id": self.request_id,
            "served_at": self.served_at.isoformat(),
        }


class ModelServer:
    """Registry-aware model server.

    Loads a model from the ``ModelRegistry`` and serves predictions via
    the ``predict`` method.

    Parameters
    ----------
    registry:
        A ``ModelRegistry`` instance (from ``dataenginex.domains.ml.registry``).
    max_loaded:
        Maximum number of models to keep loaded in memory (LRU eviction).
        Can be set via DEX_MAX_LOADED_MODELS env var (default: 3).
    """

    def __init__(self, registry: Any = None, max_loaded: int | None = None) -> None:
        self._registry = registry
        self._max_loaded = max_loaded if max_loaded is not None else _MAX_LOADED_MODELS
        self._loaded: OrderedDict[str, Any] = OrderedDict()  # "name:version" → model object

    def load_model(self, name: str, version: str, model: Any) -> None:
        """Register a model object for serving.

        Parameters
        ----------
        name:
            Model name matching registry entries.
        version:
            Model version.
        model:
            Any object with a ``predict(X)`` method.
        """
        key = f"{name}:{version}"
        # LRU: move to end if exists, else add
        if key in self._loaded:
            del self._loaded[key]
        elif len(self._loaded) >= self._max_loaded:
            evicted = self._loaded.popitem(last=False)
            logger.info("evicted model (LRU)", evicted=evicted[0])
        self._loaded[key] = model
        logger.info("loaded model for serving", key=key, loaded_count=len(self._loaded))

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Run inference for *request*."""
        start = time.perf_counter()

        version = request.version or self._resolve_production_version(request.model_name)
        key = f"{request.model_name}:{version}"

        model = self._loaded.get(key)
        if model is None:
            raise RuntimeError(
                f"Model {request.model_name} v{version} not loaded — call load_model() first"
            )

        # LRU: mark as recently used
        self._loaded.move_to_end(key)

        # Convert features to the format the model expects
        feature_values = self._features_to_array(request.features)
        raw_predictions = model.predict(feature_values)

        latency = (time.perf_counter() - start) * 1000

        predictions: list[Any]
        if hasattr(raw_predictions, "tolist"):
            predictions = raw_predictions.tolist()
        else:
            predictions = list(raw_predictions)

        return PredictionResponse(
            model_name=request.model_name,
            version=version,
            predictions=predictions,
            latency_ms=latency,
            request_id=request.request_id,
        )

    def list_loaded(self) -> list[str]:
        """Return keys of all loaded models."""
        return list(self._loaded.keys())

    # -- helpers -------------------------------------------------------------

    def _resolve_production_version(self, name: str) -> str:
        if self._registry is not None:
            prod = self._registry.get_production(name)
            if prod:
                return prod.version  # type: ignore[no-any-return]
        # Fallback: use latest loaded
        loaded_keys: list[str] = list(self._loaded.keys())
        for key in reversed(loaded_keys):
            if key.startswith(f"{name}:"):
                version: str = key.split(":")[1]
                return version
        raise RuntimeError(f"No version found for model {name!r}")

    @staticmethod
    def _features_to_array(features: list[dict[str, Any]]) -> list[list[Any]]:
        """Convert list-of-dicts to a 2D list suitable for sklearn ``predict``."""
        if not features:
            return []
        keys = list(features[0].keys())
        return [[row.get(k) for k in keys] for row in features]
