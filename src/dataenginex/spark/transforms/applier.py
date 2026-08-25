"""Spark DataFrame transform applier — mirrors the DuckDB transform semantics
in src/dataenginex/domains/analytics/transforms/sql.py, one method per
TransformStepConfig.type."""

from __future__ import annotations

from typing import Any, cast

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

__all__ = ["SparkTransformApplier"]


class SparkTransformApplier:
    """Applies TransformStepConfig steps to a Spark DataFrame."""

    def apply(self, df: DataFrame, step: Any, current_table_name: str) -> DataFrame:
        method = getattr(self, f"_apply_{step.type}", None)
        if method is None:
            msg = f"Unsupported transform type for Spark engine: {step.type}"
            raise ValueError(msg)
        return cast(DataFrame, method(df, step, current_table_name))

    def _apply_filter(self, df: DataFrame, step: Any, _table: str) -> DataFrame:
        return df.filter(F.expr(step.condition))

    def _apply_derive(self, df: DataFrame, step: Any, _table: str) -> DataFrame:
        return df.withColumn(step.name, F.expr(step.expression))

    def _apply_cast(self, df: DataFrame, step: Any, _table: str) -> DataFrame:
        for col_name, target_type in step.columns.items():
            df = df.withColumn(col_name, F.col(col_name).cast(target_type))
        return df

    def _apply_deduplicate(self, df: DataFrame, step: Any, _table: str) -> DataFrame:
        key = [step.key] if isinstance(step.key, str) else step.key
        return df.dropDuplicates(key)

    def _apply_sql(self, df: DataFrame, step: Any, current_table_name: str) -> DataFrame:
        df.createOrReplaceTempView(current_table_name)
        resolved_sql = step.sql.replace("_data", current_table_name)
        return df.sparkSession.sql(resolved_sql)

    def _apply_rename(self, df: DataFrame, step: Any, _table: str) -> DataFrame:
        for old, new in step.mapping.items():
            df = df.withColumnRenamed(old, new)
        return df

    def _apply_drop_columns(self, df: DataFrame, step: Any, _table: str) -> DataFrame:
        return df.drop(*step.columns)

    def _apply_fill_null(self, df: DataFrame, step: Any, _table: str) -> DataFrame:
        return df.fillna(step.defaults)

    def _apply_aggregate(self, df: DataFrame, step: Any, _table: str) -> DataFrame:
        agg_cols = [F.expr(expr).alias(name) for name, expr in step.agg_exprs.items()]
        return df.groupBy(*step.group_by).agg(*agg_cols)

    def _apply_window(self, df: DataFrame, step: Any, _table: str) -> DataFrame:
        window = Window.partitionBy(*(step.partition_by or []))
        if step.order_by:
            window = window.orderBy(F.expr(step.order_by))
        return df.withColumn(step.name, F.expr(step.expression).over(window))

    def _apply_explode(self, df: DataFrame, step: Any, _table: str) -> DataFrame:
        # column/alias are not top-level TransformStepConfig fields — same
        # convention as the DuckDB ExplodeTransform, which takes them via
        # `options` (see runner._build_transform_kwargs).
        column = step.options["column"]
        path = column.split(".")
        top_level = path[0]
        alias = step.options.get("alias") or path[-1]
        keep = [c for c in df.columns if c != top_level]
        col_expr = F.col(column) if len(path) > 1 else F.col(top_level)
        return df.select(*keep, F.explode(col_expr).alias(alias))

    def _apply_json_normalize(self, df: DataFrame, step: Any, _table: str) -> DataFrame:
        # column/prefix are passed via `options`, same convention as explode above.
        column = step.options["column"]
        prefix = step.options.get("prefix", "")
        from pyspark.sql.types import StructType

        field_names = [f.name for f in cast(StructType, df.schema[column].dataType).fields]
        keep = [c for c in df.columns if c != column]
        derived = [F.col(f"{column}.{field}").alias(f"{prefix}{field}") for field in field_names]
        return df.select(*keep, *derived)
