"""Base interfaces (ABCs) for all pluggable subsystems.

Every backend (built-in or extra) implements the corresponding ABC.
This ensures interchangeability and enables conformance testing.

The interfaces defined here are:
- BaseConnector — data source/sink
- BaseTransform — data transformation step
- BaseTracker — experiment tracking
- BaseRetriever — document/vector retrieval
- BaseFeatureStore — feature storage and serving
- BaseOrchestrator — pipeline orchestration/scheduling
- BaseServingEngine — model serving
- BaseAgentRuntime — agent execution loop
- BaseLLMProvider — LLM API wrapper
- BaseVectorStore — vector CRUD operations
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

# --- Data Layer ---


class BaseConnector(ABC):
    """Interface for data source/sink connectors."""

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the data source."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection."""

    @abstractmethod
    def read(self, *, table: str, **kwargs: Any) -> Any:
        """Read data from the source."""

    @abstractmethod
    def write(self, data: Any, *, table: str, **kwargs: Any) -> None:
        """Write data to the sink."""

    @abstractmethod
    def health_check(self) -> bool:
        """Return True if the connection is healthy."""


class BaseTransform(ABC):
    """Interface for data transformation steps.

    Transforms operate on DuckDB tables: given a connection and input table
    name, they produce an output table and return its name.
    """

    @property
    def name(self) -> str:
        """Human-readable transform name."""
        return type(self).__name__

    @abstractmethod
    def apply(self, conn: Any, input_table: str) -> str:
        """Apply transformation to *input_table* and return output table name."""

    def validate(self) -> list[str]:
        """Return validation errors (empty list if valid)."""
        return []


# --- ML Layer ---


class BaseTracker(ABC):
    """Interface for experiment tracking backends."""

    @abstractmethod
    def create_experiment(self, name: str) -> str:
        """Create an experiment, return its ID."""

    @abstractmethod
    def start_run(self, experiment_id: str, run_name: str | None = None) -> str:
        """Start a run within an experiment, return run ID."""

    @abstractmethod
    def end_run(self, run_id: str, status: str = "FINISHED") -> None:
        """End a run with given status."""

    @abstractmethod
    def log_params(self, run_id: str, params: dict[str, Any]) -> None:
        """Log parameters for a run."""

    @abstractmethod
    def log_metrics(
        self,
        run_id: str,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        """Log metrics for a run at optional step."""

    @abstractmethod
    def list_runs(self, experiment_id: str) -> list[dict[str, Any]]:
        """List all runs for an experiment."""

    @abstractmethod
    def list_experiments(self) -> list[dict[str, Any]]:
        """List all experiments. Returns list of dicts with at least 'id' and 'name'."""


class BaseFeatureStore(ABC):
    """Interface for feature storage and serving."""

    @abstractmethod
    def save_features(self, feature_group: str, data: Any, entity_key: str) -> None:
        """Persist features for a feature group."""

    @abstractmethod
    def get_features(self, feature_group: str, entity_ids: list[str]) -> Any:
        """Retrieve features by entity IDs."""

    @abstractmethod
    def list_feature_groups(self) -> list[str]:
        """List all registered feature groups."""

    def feature_group_info(self) -> list[dict[str, str]]:
        """Return metadata for all feature groups.

        Each dict has keys: ``name``, ``entity``, ``row_count``.
        Default implementation builds from ``list_feature_groups()``
        with empty metadata; backends may override for richer detail.
        """
        return [{"name": g, "entity": "", "row_count": ""} for g in self.list_feature_groups()]


class BaseServingEngine(ABC):
    """Interface for model serving backends."""

    @abstractmethod
    def load_model(self, model_name: str, version: str | None = None) -> None:
        """Load a model for serving."""

    @abstractmethod
    def predict(self, model_name: str, data: Any) -> Any:
        """Run inference on loaded model."""

    @abstractmethod
    def list_models(self) -> list[str]:
        """List currently loaded models."""


# --- Retrieval Layer ---


class BaseRetriever(ABC):
    """Interface for document/vector retrieval."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10, **kwargs: Any) -> list[dict[str, Any]]:
        """Retrieve top_k relevant documents for query."""


class BaseVectorStore(ABC):
    """Interface for vector CRUD operations."""

    @abstractmethod
    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add vectors with documents and optional metadata."""

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Search for nearest neighbors."""

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Delete vectors by ID."""

    @abstractmethod
    def count(self) -> int:
        """Return number of stored vectors."""


# --- Orchestration Layer ---


class BaseOrchestrator(ABC):
    """Interface for pipeline orchestration/scheduling."""

    @abstractmethod
    def schedule(self, pipeline_name: str, cron: str) -> None:
        """Schedule a pipeline with a cron expression."""

    @abstractmethod
    def trigger(self, pipeline_name: str) -> str:
        """Trigger an immediate run, return run ID."""

    @abstractmethod
    def status(self, run_id: str) -> dict[str, Any]:
        """Get status of a run."""

    @abstractmethod
    def cancel(self, run_id: str) -> None:
        """Cancel a running pipeline."""


# --- Agent Layer ---


class BaseAgentRuntime(ABC):
    """Interface for agent execution runtimes."""

    @abstractmethod
    async def run(self, message: str, **kwargs: Any) -> dict[str, Any]:
        """Execute agent with message and return response.

        Returns dict with 'response', 'iterations', 'tool_calls'.
        """

    @abstractmethod
    async def step(self, message: str, **kwargs: Any) -> dict[str, Any]:
        """Execute one reasoning step, return step details."""


class BaseLLMProvider(ABC):
    """Interface for LLM API wrappers."""

    @abstractmethod
    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from prompt."""
