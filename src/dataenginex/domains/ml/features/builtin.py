"""Built-in DuckDB-backed feature store.

Stores feature groups as DuckDB tables in a persistent database.
For production: use Feast via ``[feast]`` extra.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import duckdb
import structlog

from dataenginex.domains.ml.features import feature_store_registry
from dataenginex.foundation.plugin_contracts import BaseFeatureStore

logger = structlog.get_logger()


@feature_store_registry.decorator("builtin", is_default=True)
class BuiltinFeatureStore(BaseFeatureStore):
    """DuckDB-backed feature store.

    Args:
        database: Path to DuckDB file. Defaults to ``.dex/features.duckdb``.
    """

    def __init__(self, database: str = ".dex/features.duckdb", **kwargs: Any) -> None:
        db_path = Path(database)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(db_path))
        self._lock = threading.Lock()
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS _feature_groups (
                name VARCHAR PRIMARY KEY,
                entity_key VARCHAR,
                created_at TIMESTAMP DEFAULT current_timestamp
            )
        """)
        logger.debug("feature store ready", database=str(db_path))

    def save_features(
        self,
        feature_group: str,
        data: Any,
        entity_key: str,
    ) -> None:
        """Persist features for a feature group."""
        import pyarrow as pa

        if isinstance(data, list):
            if len(data) == 0:
                return
            tbl = pa.Table.from_pylist(data)
        elif isinstance(data, pa.Table):
            tbl = data
        else:
            msg = f"Unsupported data type: {type(data)}"
            raise TypeError(msg)

        with self._lock:
            # Register or update feature group metadata
            self._conn.execute(
                "INSERT OR REPLACE INTO _feature_groups (name, entity_key) VALUES (?, ?)",
                [feature_group, entity_key],
            )
            # Store features as a table (overwrite)
            safe_name = feature_group.replace('"', '""')
            self._conn.execute(
                f'CREATE OR REPLACE TABLE "{safe_name}" AS SELECT * FROM tbl'  # noqa: S608
            )
            logger.info(
                "features saved",
                feature_group=feature_group,
                entity_key=entity_key,
                rows=len(tbl),
            )

    def get_features(
        self,
        feature_group: str,
        entity_ids: list[str],
    ) -> Any:
        """Retrieve features by entity IDs."""
        # Get entity key column name
        row = self._conn.execute(
            "SELECT entity_key FROM _feature_groups WHERE name = ?",
            [feature_group],
        ).fetchone()
        if row is None:
            msg = f"Feature group '{feature_group}' not found"
            raise KeyError(msg)
        entity_key = row[0]

        # Build IN clause
        placeholders = ", ".join(["?"] * len(entity_ids))
        safe_name = feature_group.replace('"', '""')
        safe_key = entity_key.replace('"', '""')
        result = self._conn.execute(
            f'SELECT * FROM "{safe_name}" '  # noqa: S608
            f'WHERE CAST("{safe_key}" AS VARCHAR) IN ({placeholders})',
            entity_ids,
        )
        columns = [desc[0] for desc in result.description]
        return [dict(zip(columns, r, strict=True)) for r in result.fetchall()]

    def list_feature_groups(self) -> list[str]:
        """List all registered feature groups."""
        rows = self._conn.execute("SELECT name FROM _feature_groups ORDER BY name").fetchall()
        return [r[0] for r in rows]

    def feature_group_info(self) -> list[dict[str, str]]:
        """Return name, entity key, and row count for every feature group."""
        with self._lock:
            meta_rows = self._conn.execute(
                "SELECT name, entity_key FROM _feature_groups ORDER BY name"
            ).fetchall()
            result: list[dict[str, str]] = []
            for name, entity_key in meta_rows:
                safe = name.replace('"', '""')
                row = self._conn.execute(
                    f'SELECT COUNT(*) FROM "{safe}"'  # noqa: S608
                ).fetchone()
                count = f"{row[0]:,}" if row else ""
                result.append({"name": name, "entity": str(entity_key), "row_count": count})
        return result

    def close(self) -> None:
        """Close the DuckDB connection."""
        self._conn.close()
