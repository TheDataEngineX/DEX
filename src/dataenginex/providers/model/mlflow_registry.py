"""MLflow-backed model registry.

Wraps MLflow's tracking and model registry APIs using the same
``ModelRegistry`` interface.  Falls back gracefully when the MLflow
server is unreachable — callers should catch ``MLflowRegistryError``
and switch to ``ModelRegistry`` (JSON-backed) if needed.

Alias mapping (MLflow 3.x alias-based API)
--------------------------------------------
DataEngineX ``ModelStage``  →  MLflow alias
DEVELOPMENT         →  (no alias)
STAGING             →  ``staging``
PRODUCTION          →  ``production``
ARCHIVED            →  ``archived``
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

import structlog

from dataenginex.domains.ml.registry import ModelArtifact, ModelStage

logger = structlog.get_logger()

__all__ = [
    "MLflowModelRegistry",
    "MLflowRegistryError",
]

# MLflow alias strings (MLflow 3.x removed stage-based management)
_ALIAS_MAP: dict[ModelStage, str] = {
    ModelStage.STAGING: "staging",
    ModelStage.PRODUCTION: "production",
    ModelStage.ARCHIVED: "archived",
}
_REVERSE_ALIAS_MAP: dict[str, ModelStage] = {v: k for k, v in _ALIAS_MAP.items()}

_DEFAULT_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


class MLflowRegistryError(RuntimeError):
    """Raised when the MLflow server is unreachable or returns an error."""


def _get_client(tracking_uri: str) -> Any:
    """Import MlflowClient lazily so the module loads without mlflow installed."""
    try:
        from mlflow.tracking import MlflowClient  # noqa: PLC0415

        return MlflowClient(tracking_uri=tracking_uri)
    except ImportError as exc:
        raise MLflowRegistryError("mlflow is not installed — pip install mlflow") from exc


class MLflowModelRegistry:
    """MLflow-backed model registry compatible with ``ModelRegistry``.

    Parameters
    ----------
    tracking_uri:
        MLflow tracking server URI.  Defaults to ``MLFLOW_TRACKING_URI``
        env var or ``http://localhost:5000``.
    """

    def __init__(self, tracking_uri: str = _DEFAULT_TRACKING_URI) -> None:
        self._tracking_uri = tracking_uri
        self._client = _get_client(tracking_uri)
        logger.info("mlflow registry connected", uri=tracking_uri)

    # -- registration --------------------------------------------------------

    def register(self, artifact: ModelArtifact) -> ModelArtifact:
        """Register a model version in MLflow and log its run metadata."""
        import mlflow  # noqa: PLC0415

        mlflow.set_tracking_uri(self._tracking_uri)

        try:
            with mlflow.start_run(run_name=f"{artifact.name}_v{artifact.version}") as run:
                mlflow.log_params(artifact.parameters)
                mlflow.log_metrics(artifact.metrics)
                mlflow.set_tags(
                    {
                        "dex.version": artifact.version,
                        "dex.description": artifact.description,
                        **{f"dex.tag.{t}": "true" for t in artifact.tags},
                    }
                )

                # Register the model URI (use artifact_path if it's a local path)
                model_uri = (
                    f"runs:/{run.info.run_id}/model"
                    if not artifact.artifact_path
                    else artifact.artifact_path
                )
                mv = mlflow.register_model(model_uri=model_uri, name=artifact.name)

            logger.info(
                "Registered %s v%s in MLflow (version=%s)",
                artifact.name,
                artifact.version,
                mv.version,
            )
        except Exception as exc:
            raise MLflowRegistryError(
                f"Failed to register {artifact.name!r} in MLflow: {exc}"
            ) from exc

        return artifact

    # -- queries -------------------------------------------------------------

    def get(self, name: str, version: str) -> ModelArtifact | None:
        """Fetch a model version from MLflow."""
        try:
            mv = self._client.get_model_version(name=name, version=version)
        except Exception:  # noqa: BLE001
            return None

        return self._mv_to_artifact(mv)

    def get_latest(self, name: str) -> ModelArtifact | None:
        """Return the highest version number for *name* regardless of stage."""
        try:
            versions = self._client.search_model_versions(f"name='{name}'")
        except Exception:  # noqa: BLE001
            return None

        if not versions:
            return None

        latest = max(versions, key=lambda v: int(v.version))
        return self._mv_to_artifact(latest)

    def get_production(self, name: str) -> ModelArtifact | None:
        """Return the model currently aliased as ``production``."""
        try:
            mv = self._client.get_model_version_by_alias(name, "production")
        except Exception:  # noqa: BLE001
            return None

        return self._mv_to_artifact(mv)

    def list_models(self) -> list[str]:
        """Return all registered model names."""
        try:
            registered = self._client.search_registered_models()
            return [m.name for m in registered]
        except Exception as exc:  # noqa: BLE001
            raise MLflowRegistryError(f"Failed to list models: {exc}") from exc

    def list_versions(self, name: str) -> list[str]:
        """Return all version strings for *name*."""
        try:
            versions = self._client.search_model_versions(f"name='{name}'")
            return [v.version for v in versions]
        except Exception as exc:  # noqa: BLE001
            raise MLflowRegistryError(f"Failed to list versions for {name!r}: {exc}") from exc

    # -- promotion -----------------------------------------------------------

    def promote(self, name: str, version: str, target_stage: ModelStage) -> ModelArtifact:
        """Transition a model version to the target stage via MLflow aliases."""
        try:
            if target_stage == ModelStage.DEVELOPMENT:
                # Remove any DataEngineX-managed aliases from this version
                mv = self._client.get_model_version(name, version)
                for alias in list(getattr(mv, "aliases", [])):
                    if alias in _REVERSE_ALIAS_MAP:
                        self._client.delete_registered_model_alias(name, alias)
            else:
                alias = _ALIAS_MAP[target_stage]
                self._client.set_registered_model_alias(name, alias, version)
        except Exception as exc:
            raise MLflowRegistryError(
                f"Failed to promote {name!r} v{version} to {target_stage}: {exc}"
            ) from exc

        mv = self._client.get_model_version(name, version)
        logger.info("model promoted", name=name, version=version, stage=str(target_stage))
        return self._mv_to_artifact(mv)

    # -- helpers -------------------------------------------------------------

    def _mv_to_artifact(self, mv: Any) -> ModelArtifact:
        """Convert an MLflow ModelVersion object to a ``ModelArtifact``."""
        aliases: list[str] = list(getattr(mv, "aliases", []))

        # Resolve stage from aliases: production > staging > archived > development
        stage = ModelStage.DEVELOPMENT
        for alias in aliases:
            candidate = _REVERSE_ALIAS_MAP.get(alias)
            if candidate == ModelStage.PRODUCTION:
                stage = ModelStage.PRODUCTION
                break
            if candidate == ModelStage.STAGING:
                stage = ModelStage.STAGING
            elif candidate == ModelStage.ARCHIVED and stage == ModelStage.DEVELOPMENT:
                stage = ModelStage.ARCHIVED

        creation_ts = getattr(mv, "creation_timestamp", None)
        created_at = (
            datetime.fromtimestamp(creation_ts / 1000, tz=UTC)
            if creation_ts
            else datetime.now(tz=UTC)
        )

        last_updated_ts = getattr(mv, "last_updated_timestamp", None)
        promoted_at = (
            datetime.fromtimestamp(last_updated_ts / 1000, tz=UTC) if last_updated_ts else None
        )

        return ModelArtifact(
            name=mv.name,
            version=mv.version,
            stage=stage,
            artifact_path=getattr(mv, "source", ""),
            description=getattr(mv, "description", "") or "",
            tags=[t.key for t in getattr(mv, "tags", [])],
            created_at=created_at,
            promoted_at=promoted_at,
        )
