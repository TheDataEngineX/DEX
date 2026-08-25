"""DuckDB engine — fast, local, single-node execution.

Phase 2: Full implementation of BaseEngine for DuckDB.
"""

from __future__ import annotations

import contextlib
import datetime
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.dataset as ds
import structlog

from dataenginex.engines.base import (
    BaseEngine,
    EngineCapabilities,
    EngineConfig,
    LoadResult,
    MergeResult,
    QualityResult,
    SCD2Result,
)
from dataenginex.engines.registry import engine_registry

logger = structlog.get_logger()

__all__ = ["DuckDBEngine"]

_CAPABILITIES = EngineCapabilities(
    name="duckdb",
    streaming=False,
    distributed=False,
    auto_cdc=False,
    iceberg_read=True,
    iceberg_write=False,
    delta_read=True,
    delta_write=True,
    mllib=False,
    catalyst=False,
    spark_connect=False,
)


def _infer_layer(pipeline_name: str) -> str:
    """Return the lakehouse layer implied by a pipeline name prefix."""
    if pipeline_name.startswith("bronze_"):
        return "bronze"
    if pipeline_name.startswith("gold_"):
        return "gold"
    return "silver"


def _is_delta_table(path: Path) -> bool:
    """Return True if path is a Delta table directory."""
    return (path / "_delta_log").exists()


class DuckDBEngine(BaseEngine):
    """DuckDB execution engine.

    Fast, local, single-node. Best for:
    - Local development
    - CI/CD testing
    - Small-to-medium datasets (<10GB)
    - Dashboards and interactive queries
    """

    def __init__(self) -> None:
        self._conn: Any = None
        self._config: EngineConfig | None = None

    def connect(self, config: EngineConfig) -> None:
        """Initialize DuckDB connection."""
        self._config = config
        path = config.path or ":memory:"
        self._conn = duckdb.connect(path)
        if config.threads:
            self._conn.execute(f"SET threads = {config.threads}")
        if config.memory_limit:
            self._conn.execute(f"SET memory_limit = '{config.memory_limit}'")
        logger.info("duckdb connected", path=path, threads=config.threads)

    def disconnect(self) -> None:
        """Close DuckDB connection."""
        if self._conn:
            with contextlib.suppress(Exception):
                self._conn.close()
            self._conn = None
            logger.info("duckdb disconnected")

    @property
    def conn(self) -> Any:
        """Return the DuckDB connection."""
        if self._conn is None:
            msg = "Engine not connected — call connect() first"
            raise RuntimeError(msg)
        return self._conn

    def extract(self, source_config: Any) -> Any:
        """Extract data from a source into a DuckDB table named 'bronze'.

        Args:
            source_config: dict with keys: connector_cls, connector_kwargs, name

        Returns:
            Row count extracted.
        """
        connector_cls = source_config["connector_cls"]
        connector_kwargs = source_config["connector_kwargs"]
        name = source_config.get("name", "unknown")

        connector = connector_cls(**connector_kwargs)
        try:
            connector.connect()
            read_table = str(connector_kwargs.get("default_file", ""))
            raw_data = connector.read(table=read_table)
        finally:
            connector.disconnect()

        if isinstance(raw_data, ds.Dataset):
            files = getattr(raw_data, "files", None)
            if files:
                quoted = ", ".join("'" + f.replace("'", "''") + "'" for f in files)
                self.conn.execute(
                    f"CREATE OR REPLACE TABLE bronze AS SELECT * FROM read_parquet([{quoted}])"
                )
            else:
                self.conn.register("_raw_src", raw_data)
                self.conn.execute("CREATE OR REPLACE TABLE bronze AS SELECT * FROM _raw_src")
            row_count = raw_data.count_rows()
        elif isinstance(raw_data, pa.Table):
            self.conn.register("_raw_src", raw_data)
            self.conn.execute("CREATE OR REPLACE TABLE bronze AS SELECT * FROM _raw_src")
            row_count = len(raw_data)
        else:
            arrow_table = pa.Table.from_pylist(raw_data)
            self.conn.register("_raw_src", arrow_table)
            self.conn.execute("CREATE OR REPLACE TABLE bronze AS SELECT * FROM _raw_src")
            row_count = len(arrow_table)

        logger.info("extract complete", source=name, rows=row_count)
        return row_count

    def transform(self, df: Any, steps: list[Any]) -> Any:
        """Apply transform steps to a DuckDB table.

        Args:
            df: Starting table name (e.g. "bronze").
            steps: List of (transform_cls, kwargs) tuples.

        Returns:
            Final table name after all transforms.
        """
        current_table = df
        for i, (transform_cls, kwargs) in enumerate(steps):
            transform = transform_cls(**kwargs)
            errors = transform.validate()
            if errors:
                msg = f"Transform validation failed at step {i}: {errors}"
                raise ValueError(msg)

            prev_table = current_table
            current_table = transform.apply(self.conn, current_table)
            # Drop previous table to save memory
            self.conn.execute(f"DROP TABLE IF EXISTS {prev_table}")
            logger.info("transform complete", step=i, type=transform_cls.__name__)

        return current_table

    def quality_check(self, df: Any, checks: Any) -> QualityResult:
        """Run quality checks on a DuckDB table.

        Args:
            df: Table name.
            checks: dict with completeness, uniqueness, row_count_min, custom_sql.

        Returns:
            QualityResult with pass/fail and scores.
        """
        from dataenginex.domains.analytics.quality.gates import check_quality

        if checks is None:
            return QualityResult(passed=True)

        table = df
        completeness = checks.get("completeness") if isinstance(checks, dict) else None
        uniqueness = checks.get("uniqueness") if isinstance(checks, dict) else None
        row_count_min = checks.get("row_count_min") if isinstance(checks, dict) else None
        custom_sql = checks.get("custom_sql") if isinstance(checks, dict) else None

        if isinstance(checks, dict):
            resolved_sql = custom_sql.replace("_data", table) if custom_sql else None
        else:
            resolved_sql = None
            completeness = getattr(checks, "completeness", None)
            uniqueness = getattr(checks, "uniqueness", None)
            row_count_min = getattr(checks, "row_count_min", None)
            custom_sql_raw = getattr(checks, "custom_sql", None)
            if custom_sql_raw:
                resolved_sql = custom_sql_raw.replace("_data", table)

        result = check_quality(
            self.conn,
            table,
            completeness=completeness,
            uniqueness=uniqueness,
            row_count_min=row_count_min,
            custom_sql=resolved_sql,
        )

        return QualityResult(
            passed=result.passed,
            completeness_score=result.completeness_score,
            uniqueness_score=result.uniqueness_score,
            custom_passed=result.custom_passed,
            details={
                "schema_violations": result.schema_violations,
                **result.details,
            },
        )

    def load(self, df: Any, target_config: Any) -> LoadResult:
        """Load a DuckDB table to the target location.

        Args:
            df: Table name.
            target_config: dict with layer, format, path, name, source, pipeline_name.

        Returns:
            LoadResult with row count and status.
        """
        table = df
        target_layer = target_config.get("layer", "silver")
        target_format = target_config.get("format", "parquet")
        output_name = target_config.get("name", "output")
        data_dir = Path(target_config.get("data_dir", ".dex/lakehouse"))
        source_name = target_config.get("source", "unknown")
        pipeline_name = target_config.get("pipeline_name", "unknown")

        count_row = self.conn.execute(f"SELECT count(*) FROM {table}").fetchone()
        rows = int(count_row[0]) if count_row else 0

        layer_dir = data_dir / target_layer
        layer_dir.mkdir(parents=True, exist_ok=True)
        output_path = layer_dir / (
            f"{output_name}.parquet" if target_format == "parquet" else output_name
        )

        ingested_at = datetime.datetime.now(datetime.UTC).isoformat()
        safe_source = source_name.replace("'", "''")
        safe_name = pipeline_name.replace("'", "''")
        select_with_meta = f"""
            SELECT
                *,
                '{ingested_at}'::TIMESTAMPTZ AS _dex_ingested_at,
                '{safe_name}'               AS _dex_pipeline,
                '{target_layer}'            AS _dex_layer,
                '{safe_source}'             AS _dex_source
            FROM {table}
        """

        if target_format == "delta":
            from dataenginex.providers.object_store.storage import DeltaStorage

            arrow_reader = self.conn.execute(select_with_meta).to_arrow_reader(100_000)
            storage = DeltaStorage(base_path=str(layer_dir), mode="overwrite")
            if not storage.write(arrow_reader, output_name):
                msg = f"pipeline '{pipeline_name}': delta write failed"
                raise RuntimeError(msg)
        elif target_format == "iceberg":
            # Iceberg write via Spark — not supported in pure DuckDB
            msg = "Iceberg write requires Spark engine"
            raise NotImplementedError(msg)
        else:
            self.conn.execute(f"COPY ({select_with_meta}) TO '{output_path}' (FORMAT PARQUET)")

        logger.info(
            "load complete",
            layer=target_layer,
            format=target_format,
            path=str(output_path),
            rows=rows,
        )
        return LoadResult(
            success=True,
            rows_output=rows,
            location=str(output_path),
            format=target_format,
        )

    def merge(
        self, target: str, source: str, keys: list[str], strategy: str = "upsert"
    ) -> MergeResult:
        """Merge source into target table using DuckDB SQL."""
        safe_target = target.replace('"', '""')
        safe_source = source.replace('"', '""')
        join_condition = " AND ".join(
            [f't."{k}" = s."{k}"' for k in keys]
        )

        # Count before merge
        before_count = self.conn.execute(
            f'SELECT COUNT(*) FROM "{safe_target}"'
        ).fetchone()
        target_rows_before = int(before_count[0]) if before_count else 0

        if strategy == "upsert":
            merge_sql = f"""
                MERGE INTO "{safe_target}" AS t
                USING "{safe_source}" AS s
                ON {join_condition}
                WHEN MATCHED THEN UPDATE SET *
                WHEN NOT MATCHED THEN INSERT *
            """
        elif strategy == "insert_only":
            merge_sql = f"""
                INSERT INTO "{safe_target}"
                SELECT * FROM "{safe_source}" s
                WHERE NOT EXISTS (
                    SELECT 1 FROM "{safe_target}" t WHERE {join_condition}
                )
            """
        else:
            msg = f"Unsupported merge strategy: {strategy}"
            raise ValueError(msg)

        self.conn.execute(merge_sql)

        # Count after merge
        after_count = self.conn.execute(
            f'SELECT COUNT(*) FROM "{safe_target}"'
        ).fetchone()
        target_rows_after = int(after_count[0]) if after_count else 0

        # Estimate inserted/updated from row count change
        rows_inserted = max(0, target_rows_after - target_rows_before)
        rows_updated = target_rows_before - rows_inserted if target_rows_before > 0 else 0

        return MergeResult(
            success=True,
            rows_inserted=rows_inserted,
            rows_updated=rows_updated,
        )

    def scd_type2(
        self, target: str, source: str, keys: list[str], valid_from: str = "valid_from"
    ) -> SCD2Result:
        """SCD Type 2 merge using DuckDB hash-based technique."""
        if not keys:
            msg = "SCD2 requires at least one key"
            raise ValueError(msg)

        key = keys[0]
        safe_key = key.replace('"', '""')
        ingested_at = datetime.datetime.now(datetime.UTC).isoformat()

        new_hashed = f"""
            SELECT *, hash(t)::VARCHAR AS _dex_row_hash FROM "{source}" t
        """

        target_exists = True
        try:
            self.conn.execute(f'SELECT _dex_is_current FROM "{target}" LIMIT 0')
        except Exception:
            target_exists = False

        if not target_exists:
            merged_sql = f"""
                SELECT
                    * EXCLUDE (_dex_row_hash),
                    '{ingested_at}'::TIMESTAMPTZ AS _dex_valid_from,
                    NULL::TIMESTAMPTZ            AS _dex_valid_to,
                    true                          AS _dex_is_current,
                    _dex_row_hash
                FROM ({new_hashed})
            """
        else:
            merged_sql = f"""
                WITH new_hashed AS ({new_hashed}),
                existing_current AS (
                    SELECT * FROM "{target}" WHERE _dex_is_current
                ),
                existing_historical AS (
                    SELECT * FROM "{target}" WHERE NOT _dex_is_current
                ),
                to_close AS (
                    SELECT e.* EXCLUDE (_dex_valid_to, _dex_is_current),
                           '{ingested_at}'::TIMESTAMPTZ AS _dex_valid_to,
                           false AS _dex_is_current
                    FROM existing_current e
                    LEFT JOIN new_hashed n ON e."{safe_key}" = n."{safe_key}"
                    WHERE n."{safe_key}" IS NULL OR e._dex_row_hash != n._dex_row_hash
                ),
                unchanged AS (
                    SELECT e.* FROM existing_current e
                    JOIN new_hashed n
                      ON e."{safe_key}" = n."{safe_key}" AND e._dex_row_hash = n._dex_row_hash
                ),
                new_or_changed AS (
                    SELECT
                        n.* EXCLUDE (_dex_row_hash),
                        '{ingested_at}'::TIMESTAMPTZ AS _dex_valid_from,
                        NULL::TIMESTAMPTZ            AS _dex_valid_to,
                        true                          AS _dex_is_current,
                        n._dex_row_hash
                    FROM new_hashed n
                    LEFT JOIN existing_current e ON n."{safe_key}" = e."{safe_key}"
                    WHERE e."{safe_key}" IS NULL OR n._dex_row_hash != e._dex_row_hash
                )
                SELECT * FROM to_close
                UNION ALL BY NAME SELECT * FROM unchanged
                UNION ALL BY NAME SELECT * FROM new_or_changed
                UNION ALL BY NAME SELECT * FROM existing_historical
            """

        self.conn.execute(f'CREATE OR REPLACE TABLE "{target}" AS {merged_sql}')
        total = self.conn.execute(f'SELECT COUNT(*) FROM "{target}"').fetchone()
        current = self.conn.execute(
            f'SELECT COUNT(*) FROM "{target}" WHERE _dex_is_current'
        ).fetchone()

        return SCD2Result(
            success=True,
            rows_inserted=int(current[0]) if current else 0,
            rows_updated=0,  # computed from diff
            rows_expired=(int(total[0]) if total else 0) - (int(current[0]) if current else 0),
        )

    def content_hash(self, df: Any) -> str:
        """Compute a content hash of the table using DuckDB's hash() + bit_xor()."""
        table = df
        try:
            row = self.conn.execute(
                f'SELECT count(*), bit_xor(hash(t)) FROM "{table}" t'
            ).fetchone()
        except Exception:
            return ""
        if row is None:
            return ""
        count, xor_hash = row
        return f"{table}:{count}:{xor_hash}"

    def read_table(self, table: str) -> Any:
        """Read a table as a list of dicts."""
        cursor = self.conn.execute(f'SELECT * FROM "{table}"')
        columns = [desc[0] for desc in (cursor.description or [])]
        rows = cursor.fetchall()
        return [dict(zip(columns, row, strict=True)) for row in rows]

    def write_table(self, df: Any, table: str, format: str = "parquet") -> None:
        """Write a DataFrame to a table."""
        if isinstance(df, pa.Table):
            self.conn.register("_write_src", df)
            self.conn.execute(f'CREATE OR REPLACE TABLE "{table}" AS SELECT * FROM _write_src')
        elif isinstance(df, ds.Dataset):
            files = getattr(df, "files", None)
            if files:
                quoted = ", ".join("'" + f.replace("'", "''") + "'" for f in files)
                self.conn.execute(
                    f'CREATE OR REPLACE TABLE "{table}" AS SELECT * FROM read_parquet([{quoted}])'
                )
            else:
                self.conn.register("_write_src", df)
                self.conn.execute(f'CREATE OR REPLACE TABLE "{table}" AS SELECT * FROM _write_src')
        else:
            self.conn.register("_write_src", df)
            self.conn.execute(f'CREATE OR REPLACE TABLE "{table}" AS SELECT * FROM _write_src')

    def execute_sql(self, sql: str) -> Any:
        """Execute raw SQL and return results as a list of dicts."""
        cursor = self.conn.execute(sql)
        columns = [desc[0] for desc in (cursor.description or [])]
        rows = cursor.fetchall()
        return [dict(zip(columns, row, strict=True)) for row in rows]

    def capabilities(self) -> EngineCapabilities:
        """Return DuckDB capabilities."""
        return _CAPABILITIES


# Auto-register on import
engine_registry.register("duckdb", DuckDBEngine())
