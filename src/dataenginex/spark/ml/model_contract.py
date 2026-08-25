"""Model contract (§20.9).

Defines the contract for ML models tracked by DEX.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from dataenginex.foundation.projects import FrozenModel

__all__ = ["ModelContract"]


class ModelContract(FrozenModel):
    """Model contract for DEX-tracked ML models (§20.9)."""

    model_name: str
    version: str
    framework: str  # spark_mllib, mlflow, custom
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifact_path: str | None = None
    dataset_version: str | None = None
