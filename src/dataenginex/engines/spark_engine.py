"""Spark engine — distributed, streaming, ML, Iceberg native.

Phase 3: Full implementation of BaseEngine for Spark.
"""

from __future__ import annotations

import datetime
from typing import Any

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

__all__ = ["SparkEngine"]

_CAPABILITIES = EngineCapabilities(
    name="spark",
    streaming=True,
    distributed=True,
    auto_cdc=True,
    iceberg_read=True,
    iceberg_write=True,
    delta_read=True,
    delta_write=True,
    mllib=True,
    catalyst=True,
    spark_connect=True,
)


class SparkEngine(BaseEngine):
    """Spark execution engine.

    Distributed, full-featured. Best for:
    - Large datasets (>10GB)
    - Streaming workloads
    - CDC/SCD Type 1/2
    - MLlib distributed training
    - Iceberg native writes
    - Multi-device LAN clusters
    """

    def __init__(self) -> None:
        self._spark: Any = None
        self._config: EngineConfig | None = None

    def connect(self, config: EngineConfig) -> None:
        """Initialize SparkSession."""
        try:
            from pyspark.sql import SparkSession
        except ImportError as err:
            msg = "pyspark not installed — run: pip install 'dataenginex[spark]'"
            raise ImportError(msg) from err

        self._config = config
        master = config.master or "local[*]"

        builder = SparkSession.builder.master(master).appName("dataenginex")

        # Iceberg catalog
        if config.file_format == "iceberg":
            builder = (
                builder.config(
                    "spark.sql.catalog.iceberg", "org.apache.iceberg.spark.SparkCatalog"
                )
                .config("spark.sql.catalog.iceberg.type", "hadoop")
                .config("spark.sql.catalog.iceberg.warehouse", config.warehouse or ".dex/lakehouse")
            )

        # Delta Lake — only when file_format is delta
        if config.file_format == "delta":
            builder = (
                builder.config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
                .config(
                    "spark.sql.catalog.spark_catalog",
                    "org.apache.spark.sql.delta.catalog.DeltaCatalog",
                )
            )

        # Resource config
        if config.executor_memory:
            builder = builder.config("spark.executor.memory", config.executor_memory)
        if config.executor_cores:
            builder = builder.config("spark.executor.cores", str(config.executor_cores))

        # Performance
        builder = (
            builder.config("spark.sql.shuffle.partitions", "200")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.cbo.enabled", "true")
        )

        self._spark = builder.getOrCreate()
        logger.info(
            "spark connected",
            master=master,
            version=self._spark.version,
            file_format=config.file_format,
        )

    def disconnect(self) -> None:
        """Stop SparkSession."""
        if self._spark:
            self._spark.stop()
            self._spark = None
            logger.info("spark disconnected")

    @property
    def spark(self) -> Any:
        """Return the SparkSession."""
        if self._spark is None:
            msg = "Engine not connected — call connect() first"
            raise RuntimeError(msg)
        return self._spark

    def extract(self, source_config: Any) -> Any:
        """Extract data from a source into a Spark DataFrame.

        Args:
            source_config: dict with keys: connector_cls, connector_kwargs, name

        Returns:
            Spark DataFrame.
        """
        import pyarrow as pa

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

        if isinstance(raw_data, pa.Table):
            arrow_table = raw_data
        elif hasattr(raw_data, "to_table"):
            arrow_table = raw_data.to_table()
        else:
            arrow_table = pa.Table.from_pylist(raw_data)

        # Convert Arrow to Spark DataFrame — prefer pandas path when available
        try:
            pdf = arrow_table.to_pandas()
            df = self.spark.createDataFrame(pdf)
        except Exception:
            # Fallback: convert to list of dicts for environments without pandas
            columns = arrow_table.column_names
            col_lists = [arrow_table[c].to_pylist() for c in columns]
            rows = [dict(zip(columns, row, strict=True)) for row in zip(*col_lists, strict=True)]
            df = self.spark.createDataFrame(rows, schema=columns)
        logger.info("extract complete", source=name, rows=df.count())
        return df

    def transform(self, df: Any, steps: list[Any]) -> Any:
        """Apply transform steps to a Spark DataFrame.

        Args:
            df: Input Spark DataFrame.
            steps: List of (transform_cls, kwargs) tuples.

        Returns:
            Transformed Spark DataFrame.
        """
        from dataenginex.spark.transforms.applier import SparkTransformApplier

        applier = SparkTransformApplier()
        current_table = "bronze"

        for i, (transform_cls, kwargs) in enumerate(steps):
            # Create a step config-like object for SparkTransformApplier
            step_config = type("StepConfig", (), {
                "type": kwargs.get("type", transform_cls.__name__),
                "condition": kwargs.get("condition"),
                "expression": kwargs.get("expression"),
                "name": kwargs.get("name"),
                "columns": kwargs.get("columns"),
                "key": kwargs.get("key"),
                "sql": kwargs.get("sql"),
                "mapping": kwargs.get("mapping"),
                "defaults": kwargs.get("defaults"),
                "group_by": kwargs.get("group_by"),
                "agg_exprs": kwargs.get("agg_exprs"),
                "partition_by": kwargs.get("partition_by"),
                "order_by": kwargs.get("order_by"),
                "options": kwargs.get("options", {}),
            })()

            df = applier.apply(df, step_config, current_table)
            logger.info("transform complete", step=i, type=transform_cls.__name__)

        return df

    def quality_check(self, df: Any, checks: Any) -> QualityResult:
        """Run quality checks on a Spark DataFrame.

        Args:
            df: Spark DataFrame.
            checks: dict with completeness, uniqueness, row_count_min, custom_sql.

        Returns:
            QualityResult with pass/fail and scores.
        """
        if checks is None:
            return QualityResult(passed=True)

        total_rows = df.count()
        if total_rows == 0:
            return QualityResult(passed=True)

        def _get(key: str) -> Any:
            if isinstance(checks, dict):
                return checks.get(key)
            return getattr(checks, key, None)

        completeness = _get("completeness")
        uniqueness = _get("uniqueness")
        row_count_min = _get("row_count_min")
        custom_sql = _get("custom_sql")

        reasons: list[str] = []
        self._check_row_count(total_rows, row_count_min, reasons)
        self._check_completeness(df, total_rows, completeness, reasons)
        self._check_uniqueness(df, total_rows, uniqueness, reasons)
        self._check_custom_sql(df, custom_sql, reasons)

        passed = len(reasons) == 0
        return QualityResult(
            passed=passed,
            completeness_score=1.0 if passed else 0.0,
            uniqueness_score=1.0 if passed else 0.0,
            custom_passed="custom_sql" not in str(reasons),
            details={"reasons": reasons},
        )

    def _check_row_count(self, total: int, minimum: Any, reasons: list[str]) -> None:
        if minimum is not None and total < minimum:
            reasons.append(f"row_count={total} < min {minimum}")

    def _check_completeness(
        self, df: Any, total_rows: int, threshold: Any, reasons: list[str]
    ) -> None:
        if threshold is None:
            return
        from pyspark.sql import functions as F

        total_cells = total_rows * len(df.columns)
        if total_cells == 0:
            return
        null_row = df.select(
            [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns]
        ).collect()[0]
        total_nulls = sum(v or 0 for v in null_row)
        score = (total_cells - total_nulls) / total_cells
        if score < threshold:
            reasons.append(f"completeness={score:.4f} < {threshold}")

    def _check_uniqueness(
        self, df: Any, total_rows: int, columns: Any, reasons: list[str]
    ) -> None:
        if columns is None:
            return
        distinct_count = df.dropDuplicates(columns).count()
        if distinct_count < total_rows:
            reasons.append(
                f"uniqueness={distinct_count / total_rows:.4f} (duplicates on {columns})"
            )

    def _check_custom_sql(self, df: Any, sql: Any, reasons: list[str]) -> None:
        if sql is None:
            return
        df.createOrReplaceTempView("_data")
        result_row = self.spark.sql(sql).collect()
        if not result_row or not result_row[0][0]:
            reasons.append("custom_sql check returned false/zero rows")

    def load(self, df: Any, target_config: Any) -> LoadResult:
        """Load a Spark DataFrame to the target location.

        Args:
            df: Spark DataFrame.
            target_config: dict with layer, format, path, name, source, pipeline_name.

        Returns:
            LoadResult with row count and status.
        """
        target_layer = target_config.get("layer", "silver")
        target_format = target_config.get("format", "parquet")
        output_name = target_config.get("name", "output")
        data_dir = target_config.get("data_dir", ".dex/lakehouse")

        rows_output = df.count()
        target_path = f"{data_dir}/{target_layer}/{output_name}"

        if target_format == "iceberg":
            df.writeTo(f"iceberg.{output_name}").createOrReplace()
        elif target_format == "delta":
            df.write.format("delta").mode("overwrite").save(target_path)
        else:
            df.write.parquet(target_path, mode="overwrite")

        logger.info(
            "load complete",
            layer=target_layer,
            format=target_format,
            path=target_path,
            rows=rows_output,
        )
        return LoadResult(
            success=True,
            rows_output=rows_output,
            location=target_path,
            format=target_format,
        )

    def merge(
        self, target: str, source: str, keys: list[str], strategy: str = "upsert"
    ) -> MergeResult:
        """Merge source into target using Delta Lake MERGE."""
        from delta.tables import DeltaTable

        target_table = DeltaTable.forPath(self.spark, target)
        source_df = self.spark.read.format("delta").load(source)

        join_condition = " AND ".join([f"target.{k} = source.{k}" for k in keys])

        if strategy == "upsert":
            (
                target_table.alias("target")
                .merge(source_df.alias("source"), join_condition)
                .whenMatchedUpdateAll()
                .whenNotMatchedInsertAll()
                .execute()
            )
        elif strategy == "insert_only":
            (
                target_table.alias("target")
                .merge(source_df.alias("source"), join_condition)
                .whenNotMatchedInsertAll()
                .execute()
            )
        else:
            msg = f"Unsupported merge strategy: {strategy}"
            raise ValueError(msg)

        return MergeResult(success=True)

    def scd_type2(
        self, target: str, source: str, keys: list[str], valid_from: str = "valid_from"
    ) -> SCD2Result:
        """SCD Type 2 using Spark SQL."""
        key = keys[0]
        ingested_at = datetime.datetime.now(datetime.UTC).isoformat()

        # Register source as temp view
        source_df = self.spark.read.format("delta").load(source)
        source_df.createOrReplaceTempView("scd2_source")

        # Check if target exists
        try:
            target_df = self.spark.read.format("delta").load(target)
            target_df.createOrReplaceTempView("scd2_target")

            merge_sql = f"""
                MERGE INTO scd2_target AS t
                USING scd2_source AS s
                ON t.{key} = s.{key} AND t._dex_is_current = true
                WHEN MATCHED AND t._dex_row_hash != hash(s.*) THEN
                    UPDATE SET _dex_valid_to = '{ingested_at}', _dex_is_current = false
                WHEN NOT MATCHED THEN
                    INSERT SET *, _dex_valid_from = '{ingested_at}', _dex_is_current = true,
                        _dex_row_hash = hash(s.*)
            """
            self.spark.sql(merge_sql)
        except Exception:
            # First run — create target from source
            self.spark.sql(f"""
                CREATE TABLE delta.`{target}` AS
                SELECT *, '{ingested_at}' AS _dex_valid_from,
                    NULL AS _dex_valid_to,
                    true AS _dex_is_current,
                    hash(*) AS _dex_row_hash
                FROM scd2_source
            """)

        return SCD2Result(success=True)

    def content_hash(self, df: Any) -> str:
        """Compute a content hash of the Spark DataFrame."""
        from pyspark.sql import functions as F

        row_count = df.count()
        if row_count == 0:
            return ""

        # Use md5 of all columns concatenated
        hash_col = F.md5(F.concat_ws("|", *[F.col(c).cast("string") for c in df.columns]))
        result = df.select(F.md5(F.concat_ws("", F.collect_list(hash_col)))).collect()
        return result[0][0] if result else ""

    def read_table(self, table: str) -> Any:
        """Read a table into a Spark DataFrame."""
        return self.spark.read.format("delta").load(table)

    def write_table(self, df: Any, table: str, format: str = "parquet") -> None:
        """Write a Spark DataFrame to a table."""
        if format == "iceberg":
            df.writeTo(f"iceberg.{table}").createOrReplace()
        elif format == "delta":
            df.write.format("delta").mode("overwrite").save(table)
        else:
            df.write.parquet(table, mode="overwrite")

    def execute_sql(self, sql: str) -> Any:
        """Execute raw SQL and return results as a list of dicts."""
        result = self.spark.sql(sql)
        return [row.asDict() for row in result.collect()]

    def capabilities(self) -> EngineCapabilities:
        """Return Spark capabilities."""
        return _CAPABILITIES


# Auto-register on import
engine_registry.register("spark", SparkEngine())
