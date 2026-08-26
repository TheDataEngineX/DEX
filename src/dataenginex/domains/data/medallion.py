"""Medallion Architecture — Bronze / Silver / Gold data layers.

Provides configurable medallion layer management with pluggable storage backends.
All dataset names are generic (``bronze``, ``silver``, ``gold``); domain-specific
naming should be configured by the application.

Classes:
    StorageFormat: Supported storage format enumeration.
    DataLayer: Medallion architecture layer enumeration.
    LayerConfiguration: Configuration dataclass for a single medallion layer.
    MedallionArchitecture: Manages three-layer medallion configs.
    StorageBackend: Abstract storage backend interface.
    LocalParquetStorage: Real local Parquet file storage (pyarrow).
    DataLineage: In-memory data lineage tracker.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()
__all__ = [
    "DataLayer",
    "DataLineage",
    "LayerConfiguration",
    "LocalParquetStorage",
    "MedallionArchitecture",
    "StorageBackend",
    "StorageFormat",
]


class StorageFormat(StrEnum):
    """Supported storage formats"""

    PARQUET = "parquet"
    DELTA = "delta"
    ICEBERG = "iceberg"
    BIGQUERY = "bigquery"


class DataLayer(StrEnum):
    """Medallion architecture layers"""

    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"


@dataclass
class LayerConfiguration:
    """Configuration for a medallion layer."""

    layer_name: str
    description: str
    purpose: str
    storage_format: StorageFormat
    local_path: str
    bigquery_dataset: str
    retention_days: int | None
    schema_validation: bool
    quality_threshold: float
    compression: str = "snappy"

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.quality_threshold < 0 or self.quality_threshold > 1:
            raise ValueError("quality_threshold must be between 0 and 1")


class MedallionArchitecture:
    """Manages the three-layer medallion architecture for DataEngineX."""

    # Bronze Layer Configuration
    BRONZE_CONFIG = LayerConfiguration(
        layer_name=DataLayer.BRONZE.value,
        description="Raw, unprocessed data from all sources",
        purpose="Preserve original data for historical reproducibility",
        storage_format=StorageFormat.PARQUET,
        local_path="data/bronze",
        bigquery_dataset="bronze",
        retention_days=90,
        schema_validation=False,
        quality_threshold=0.0,  # No quality threshold for raw data
        compression="snappy",
    )

    # Silver Layer Configuration
    SILVER_CONFIG = LayerConfiguration(
        layer_name=DataLayer.SILVER.value,
        description="Cleaned, deduplicated, validated data",
        purpose="High-quality data ready for analytics and ML",
        storage_format=StorageFormat.PARQUET,
        local_path="data/silver",
        bigquery_dataset="silver",
        retention_days=365,
        schema_validation=True,
        quality_threshold=0.75,  # >= 75% quality score
        compression="snappy",
    )

    # Gold Layer Configuration
    GOLD_CONFIG = LayerConfiguration(
        layer_name=DataLayer.GOLD.value,
        description="Enriched, aggregated data ready for ML and APIs",
        purpose="Serve AI models and customer-facing APIs",
        storage_format=StorageFormat.PARQUET,
        local_path="data/gold",
        bigquery_dataset="gold",
        retention_days=None,  # Indefinite retention
        schema_validation=True,
        quality_threshold=0.90,  # >= 90% quality score
        compression="snappy",
    )

    @classmethod
    def get_layer_config(cls, layer: DataLayer) -> LayerConfiguration | None:
        """Get configuration for a specific layer."""
        configs = {
            DataLayer.BRONZE: cls.BRONZE_CONFIG,
            DataLayer.SILVER: cls.SILVER_CONFIG,
            DataLayer.GOLD: cls.GOLD_CONFIG,
        }
        return configs.get(layer)

    @classmethod
    def get_all_layers(cls) -> list[LayerConfiguration]:
        """Get configurations for all layers in order."""
        return [cls.BRONZE_CONFIG, cls.SILVER_CONFIG, cls.GOLD_CONFIG]


class StorageBackend(ABC):
    """Abstract storage backend interface.

    All lakehouse storage implementations must subclass this and provide
    concrete ``write``, ``read``, ``delete``, ``list_objects``, and
    ``exists`` methods.  The interface accepts a ``StorageFormat`` hint
    so backends can choose serialisation.

    Built-in implementations:
        - ``LocalParquetStorage`` — local Parquet files (this module)
        - ``BigQueryStorage`` — Google BigQuery tables (this module)
        - ``JsonStorage`` — JSON files (``dataenginex.providers.object_store.storage``)
        - ``ParquetStorage`` — pyarrow-backed Parquet, from
          ``dataenginex.providers.object_store.storage``
        - ``S3Storage`` — AWS S3 object storage (``dataenginex.providers.object_store.storage``)
        - ``GCSStorage`` — Google Cloud Storage (``dataenginex.providers.object_store.storage``)
    """

    @abstractmethod
    def write(self, data: Any, path: str, format: StorageFormat) -> bool:
        """Write *data* to *path* in the given format.

        Returns ``True`` on success, ``False`` on failure.
        """
        ...

    @abstractmethod
    def read(self, path: str, format: StorageFormat) -> Any:
        """Read data from *path*.  Returns ``None`` on failure."""
        ...

    @abstractmethod
    def delete(self, path: str) -> bool:
        """Delete the resource at *path*.  Returns ``True`` on success."""
        ...

    @abstractmethod
    def list_objects(self, prefix: str = "") -> list[str]:
        """List object paths under *prefix*.

        Returns a list of relative paths.  Empty list on failure or when
        no objects match.
        """
        ...

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Return ``True`` if *path* exists in the backend."""
        ...


class LocalParquetStorage(StorageBackend):
    """Local Parquet file storage backed by pyarrow.

    Raises ``RuntimeError`` if pyarrow is not installed.
    """

    def __init__(self, base_path: str = "data") -> None:
        self.base_path = base_path
        logger.info("local parquet storage initialised", path=base_path)

    @staticmethod
    def _require_pyarrow() -> Any:  # noqa: ANN401
        """Import and return the ``pyarrow.parquet`` module.

        Raises:
            RuntimeError: If pyarrow is not installed.
        """
        try:
            import pyarrow.parquet as pq  # noqa: PLC0415

            return pq
        except ImportError as exc:
            msg = (
                "pyarrow is required for LocalParquetStorage. "
                "Install it with: uv pip install pyarrow"
            )
            raise RuntimeError(msg) from exc

    def write(
        self,
        data: Any,
        path: str,
        format: StorageFormat = StorageFormat.PARQUET,
    ) -> bool:
        """Write data to a local Parquet file.

        Args:
            data: List of dicts, or a pyarrow Table.
            path: Relative path from base_path.
            format: Must be ``PARQUET``.

        Returns:
            True on success.

        Raises:
            ValueError: If format is not PARQUET.
            RuntimeError: If pyarrow is not installed.
        """
        if format != StorageFormat.PARQUET:
            msg = f"LocalParquetStorage only supports PARQUET format, got {format}"
            raise ValueError(msg)

        pq = self._require_pyarrow()
        import pyarrow as pa  # noqa: PLC0415

        full_path = Path(self.base_path) / path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if isinstance(data, list):
                table = pa.Table.from_pylist(data)
            elif hasattr(data, "schema"):
                # Already a pyarrow Table
                table = data
            else:
                msg = f"Unsupported data type: {type(data).__name__}"
                raise TypeError(msg)

            pq.write_table(table, str(full_path), compression="snappy")
            logger.info("wrote rows", count=len(table), path=str(full_path))
            return True
        except (TypeError, pa.ArrowInvalid) as exc:
            logger.error("write failed", path=path, exc=str(exc))
            return False

    def read(
        self,
        path: str,
        format: StorageFormat = StorageFormat.PARQUET,
    ) -> list[dict[str, Any]] | None:
        """Read data from a local Parquet file.

        Returns:
            List of record dicts, or None if the file doesn't exist.
        """
        pq = self._require_pyarrow()
        full_path = Path(self.base_path) / path

        if not full_path.exists():
            logger.warning("file not found", path=str(full_path))
            return None

        try:
            table = pq.read_table(str(full_path))
            records: list[dict[str, Any]] = table.to_pylist()
            logger.info("read rows", count=len(records), path=str(full_path))
            return records
        except Exception as exc:
            logger.error("read failed", path=path, exc=str(exc))
            return None

    def delete(self, path: str) -> bool:
        """Delete a Parquet file from disk."""
        full_path = Path(self.base_path) / path
        if not full_path.exists():
            logger.warning("cannot delete — file not found", path=str(full_path))
            return False
        try:
            full_path.unlink()
            logger.info("deleted file", path=str(full_path))
            return True
        except OSError as exc:
            logger.error("delete failed", path=path, exc=str(exc))
            return False

    def list_objects(self, prefix: str = "") -> list[str]:
        """List files under *prefix* relative to *base_path*."""
        target = Path(self.base_path) / prefix
        if not target.exists():
            return []
        return [str(p.relative_to(self.base_path)) for p in target.rglob("*") if p.is_file()]

    def exists(self, path: str) -> bool:
        """Return ``True`` if *path* exists on disk."""
        return (Path(self.base_path) / path).exists()


class DataLineage:
    """Tracks data lineage through the medallion layers."""

    def __init__(self) -> None:
        self.lineage: dict[str, dict[str, Any]] = {}

    def record_bronze_ingestion(self, source: str, record_count: int, timestamp: str) -> str:
        """Record data entry into Bronze layer."""
        lineage_id = f"bronze_{source}_{timestamp}"
        self.lineage[lineage_id] = {
            "layer": "bronze",
            "source": source,
            "record_count": record_count,
            "timestamp": timestamp,
            "status": "raw",
        }
        return lineage_id

    def record_silver_transformation(
        self, lineage_id: str, processed_count: int, quality_score: float
    ) -> str:
        """Record data transformation in Silver layer."""
        silver_id = f"{lineage_id}_silver"
        self.lineage[silver_id] = {
            "layer": "silver",
            "parent": lineage_id,
            "processed_count": processed_count,
            "quality_score": quality_score,
            "status": "cleaned",
        }
        return silver_id

    def record_gold_enrichment(
        self, lineage_id: str, enriched_count: int, embedding_model: str
    ) -> str:
        """Record data enrichment in Gold layer."""
        gold_id = f"{lineage_id}_gold"
        self.lineage[gold_id] = {
            "layer": "gold",
            "parent": lineage_id,
            "enriched_count": enriched_count,
            "embedding_model": embedding_model,
            "status": "enriched",
        }
        return gold_id

    def get_lineage(self, lineage_id: str) -> dict[str, Any] | None:
        """Get lineage information for a record."""
        return self.lineage.get(lineage_id)
