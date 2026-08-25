"""Model registry — local model versioning and stage promotion.

Tracks model artifacts with staging lifecycle:
``development`` → ``staging`` → ``production`` → ``archived``.

Persistence goes through the ``ModelRepository`` port (§13.6) rather than a
named store, so what a model version *is* stays separate from where it is kept.

The backing store is SQLite rather than a JSON file, which is what makes
registration atomic: ``register()`` either lands or does not, with no partial
file after a crash, and ``promote()`` is a single UPDATE that concurrent
readers never observe half-applied.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import structlog

from dataenginex.foundation.contracts import ModelRepository

logger = structlog.get_logger()

__all__ = [
    "ModelArtifact",
    "ModelRegistry",
    "ModelStage",
    "VERSION_AUTO",
]

# Sentinel: pass as artifact.version to get an auto-assigned patch increment.
VERSION_AUTO = "auto"


@runtime_checkable
class _TrainerProtocol(Protocol):
    """Minimum interface required by :meth:`ModelRegistry.register_from_trainer`."""

    model_name: str
    version: str

    def save(self, path: str) -> str: ...


class ModelStage(StrEnum):
    """Model lifecycle stages."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


@dataclass
class ModelArtifact:
    """Registry entry for a model version.

    Attributes:
        name: Model name (e.g. ``"churn_classifier"``).
        version: Semantic version string.
        stage: Current lifecycle stage.
        artifact_path: File path to the serialised model.
        metrics: Training/evaluation metrics.
        parameters: Hyper-parameters used for training.
        description: Free-text description.
        created_at: When the artifact was registered.
        promoted_at: When the artifact was last promoted.
        tags: Arbitrary labels for filtering.
    """

    name: str
    version: str
    stage: ModelStage = ModelStage.DEVELOPMENT
    artifact_path: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    promoted_at: datetime | None = None
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["stage"] = self.stage.value
        d["created_at"] = self.created_at.isoformat()
        d["promoted_at"] = self.promoted_at.isoformat() if self.promoted_at else None
        return d


class ModelRegistry:
    """SQLite-backed model registry.

    Constructor signatures:
      ModelRegistry()                  — in-memory SQLite (tests, one-shot use)
      ModelRegistry(persist_path=path) — SQLite file at *path* (any extension)
      ModelRegistry(store=repository)  — share the caller's repository

    Args:
        persist_path: Path to a SQLite file for persistence.
        store: Existing repository to share (engine integration).
    """

    def __init__(
        self,
        persist_path: str | Path | None = None,
        *,
        store: ModelRepository | None = None,
    ) -> None:
        from dataenginex.runtime.default_store import open_default_store

        if store is not None:
            self._store: ModelRepository = store
            self._owns_store = False
        else:
            self._store = open_default_store(persist_path)
            self._owns_store = True

    # -------------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------------

    def _next_version(self, name: str) -> str:
        """Return the next patch version for *name* (``major.minor.patch+1``)."""
        versions = self._store.list_model_versions(name)
        if not versions:
            return "1.0.0"
        best = (0, 0, 0)
        for v in versions:
            parts = v.split(".")
            try:
                triple = (int(parts[0]), int(parts[1]), int(parts[2]))
            except (IndexError, ValueError):
                continue
            if triple > best:
                best = triple
        major, minor, patch = best
        return f"{major}.{minor}.{patch + 1}"

    def register(
        self,
        artifact: ModelArtifact,
        *,
        upsert: bool = False,
    ) -> ModelArtifact:
        """Register a new model version.

        Args:
            artifact: Set ``artifact.version = VERSION_AUTO`` to auto-assign
                the next patch version.
            upsert: When ``True``, silently bump the patch version on conflict
                instead of raising ``ValueError``.  Never mutates an existing
                entry — always creates a new one.
        """
        if artifact.version == VERSION_AUTO:
            artifact.version = self._next_version(artifact.name)
        elif self._store.get_model(artifact.name, artifact.version) is not None:
            if not upsert:
                raise ValueError(
                    f"Model {artifact.name!r} version {artifact.version} already registered"
                )
            artifact.version = self._next_version(artifact.name)

        self._store.register_model(self._to_store(artifact))
        logger.info(
            "registered model",
            name=artifact.name,
            version=artifact.version,
            stage=artifact.stage.value,
        )
        return artifact

    def register_from_trainer(
        self,
        trainer: _TrainerProtocol,
        artifact_path: str,
        *,
        stage: ModelStage = ModelStage.DEVELOPMENT,
        metrics: dict[str, float] | None = None,
        parameters: dict[str, Any] | None = None,
        description: str = "",
        tags: list[str] | None = None,
        upsert: bool = False,
    ) -> ModelArtifact:
        """Save *trainer*'s model and register it in one step.

        Calls ``trainer.save(artifact_path)`` first, then delegates to
        :meth:`register`.  On conflict with ``upsert=True``, the artifact file
        is saved but the registry entry gets a bumped patch version.
        """
        saved_path = trainer.save(artifact_path)
        artifact = ModelArtifact(
            name=trainer.model_name,
            version=trainer.version,
            stage=stage,
            artifact_path=saved_path,
            metrics=metrics or {},
            parameters=parameters or {},
            description=description,
            tags=tags or [],
        )
        return self.register(artifact, upsert=upsert)

    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------

    def get(self, name: str, version: str) -> ModelArtifact | None:
        row = self._store.get_model(name, version)
        return self._from_store(row) if row else None

    def get_latest(self, name: str) -> ModelArtifact | None:
        """Return the most recently registered version of *name*."""
        row = self._store.get_latest_model(name)
        return self._from_store(row) if row else None

    def get_production(self, name: str) -> ModelArtifact | None:
        row = self._store.get_production_model(name)
        return self._from_store(row) if row else None

    def list_models(self) -> list[str]:
        names: list[str] = self._store.list_model_names()
        return names

    def list_versions(self, name: str) -> list[str]:
        versions: list[str] = self._store.list_model_versions(name)
        return versions

    # -------------------------------------------------------------------------
    # Promotion
    # -------------------------------------------------------------------------

    def promote(self, name: str, version: str, target_stage: ModelStage) -> ModelArtifact:
        """Promote a model version to a new stage.

        Promoting to ``production`` automatically archives the previous
        production model in a single atomic SQL transaction — no in-memory
        object mutation, no race condition.
        """
        if self._store.get_model(name, version) is None:
            raise ValueError(f"Model {name!r} version {version} not found")
        result = self._store.promote_model(name, version, target_stage.value)
        logger.info("model promoted", name=name, version=version, stage=target_stage.value)
        return self._from_store(result)

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def close(self) -> None:
        if self._owns_store:
            self._store.close()

    # -------------------------------------------------------------------------
    # Type bridge: registry.ModelArtifact <-> store.ModelArtifact
    # -------------------------------------------------------------------------

    @staticmethod
    def _to_store(artifact: ModelArtifact) -> Any:
        from dataenginex.runtime.default_store import model_row

        return model_row(
            name=artifact.name,
            version=artifact.version,
            stage=artifact.stage.value,
            artifact_path=artifact.artifact_path,
            metrics=dict(artifact.metrics),
            parameters=dict(artifact.parameters),
            description=artifact.description,
            tags=list(artifact.tags),
            created_at=artifact.created_at,
            promoted_at=artifact.promoted_at,
        )

    @staticmethod
    def _from_store(row: Any) -> ModelArtifact:
        return ModelArtifact(
            name=row.name,
            version=row.version,
            stage=ModelStage(row.stage),
            artifact_path=row.artifact_path,
            metrics=dict(row.metrics),
            parameters=dict(row.parameters),
            description=row.description,
            tags=list(row.tags),
            created_at=row.created_at,
            promoted_at=row.promoted_at,
        )
