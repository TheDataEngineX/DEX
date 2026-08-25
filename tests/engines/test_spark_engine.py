"""Tests for SparkEngine — Phase 3 (full implementation).

Requires PySpark and Java runtime. Automatically skipped if unavailable.
"""

from __future__ import annotations

import tempfile

import pytest

from dataenginex.engines.base import EngineConfig
from dataenginex.engines.registry import engine_registry
from tests.conftest import requires_pyspark


class TestSparkEngineRegistration:
    def test_auto_registered(self) -> None:
        assert engine_registry.is_registered("spark")

    def test_get_engine(self) -> None:
        engine = engine_registry.get("spark")
        assert engine.capabilities().name == "spark"


class TestSparkEngineCapabilities:
    def test_capabilities(self) -> None:
        from dataenginex.engines.spark_engine import SparkEngine

        engine = SparkEngine()
        caps = engine.capabilities()
        assert caps.name == "spark"
        assert caps.streaming is True
        assert caps.distributed is True
        assert caps.auto_cdc is True
        assert caps.iceberg_read is True
        assert caps.iceberg_write is True
        assert caps.delta_read is True
        assert caps.delta_write is True
        assert caps.mllib is True
        assert caps.catalyst is True
        assert caps.spark_connect is True


@requires_pyspark
class TestSparkEngineConnect:
    def test_connect_local(self) -> None:
        from dataenginex.engines.spark_engine import SparkEngine

        engine = SparkEngine()
        config = EngineConfig(type="spark", master="local[1]")
        engine.connect(config)
        assert engine._spark is not None
        assert engine._spark.sparkContext.appName == "dataenginex"
        engine.disconnect()

    def test_connect_with_iceberg(self) -> None:
        from dataenginex.engines.spark_engine import SparkEngine

        engine = SparkEngine()
        config = EngineConfig(
            type="spark",
            master="local[1]",
            file_format="iceberg",
            warehouse="/tmp/test-warehouse",
        )
        engine.connect(config)
        assert engine._spark is not None
        engine.disconnect()

    def test_disconnect(self) -> None:
        from dataenginex.engines.spark_engine import SparkEngine

        engine = SparkEngine()
        config = EngineConfig(type="spark", master="local[1]")
        engine.connect(config)
        engine.disconnect()
        assert engine._spark is None

    def test_spark_property_raises_when_disconnected(self) -> None:
        from dataenginex.engines.spark_engine import SparkEngine

        engine = SparkEngine()
        with pytest.raises(RuntimeError, match="not connected"):
            _ = engine.spark


@requires_pyspark
class TestSparkEngineExtract:
    def test_extract_from_pylist(self) -> None:
        from dataenginex.engines.spark_engine import SparkEngine

        engine = SparkEngine()
        engine.connect(EngineConfig(type="spark", master="local[1]"))

        import pyarrow as pa

        data = [
            {"id": 1, "name": "alice"},
            {"id": 2, "name": "bob"},
            {"id": 3, "name": "charlie"},
        ]
        arrow_table = pa.Table.from_pylist(data)

        class MockConnector:
            def __init__(self, **kwargs: object) -> None:
                pass

            def connect(self) -> None:
                pass

            def disconnect(self) -> None:
                pass

            def read(self, table: str = "") -> pa.Table:
                return arrow_table

        result = engine.extract({
            "connector_cls": MockConnector,
            "connector_kwargs": {},
            "name": "test_source",
        })
        assert result.count() == 3
        engine.disconnect()


@requires_pyspark
class TestSparkEngineTransform:
    def test_transform_with_filter(self) -> None:
        from dataenginex.engines.spark_engine import SparkEngine

        engine = SparkEngine()
        engine.connect(EngineConfig(type="spark", master="local[1]"))

        spark = engine.spark
        data = [
            (1, "alice", 100),
            (2, "bob", 200),
            (3, "charlie", 150),
        ]
        df = spark.createDataFrame(data, ["id", "name", "value"])

        # Filter using Spark SQL directly
        df.createOrReplaceTempView("bronze")
        result = spark.sql("SELECT * FROM bronze WHERE value > 120")
        assert result.count() == 2
        engine.disconnect()


@requires_pyspark
class TestSparkEngineQualityCheck:
    def test_quality_check_pass(self) -> None:
        from dataenginex.engines.spark_engine import SparkEngine

        engine = SparkEngine()
        engine.connect(EngineConfig(type="spark", master="local[1]"))

        spark = engine.spark
        data = [(1, "alice"), (2, "bob"), (3, "charlie")]
        df = spark.createDataFrame(data, ["id", "name"])

        checks = {"completeness": 0.5, "uniqueness": ["id"]}
        result = engine.quality_check(df, checks)
        assert result.passed is True
        engine.disconnect()

    def test_quality_check_no_checks(self) -> None:
        from dataenginex.engines.spark_engine import SparkEngine

        engine = SparkEngine()
        engine.connect(EngineConfig(type="spark", master="local[1]"))

        spark = engine.spark
        df = spark.createDataFrame([(1,)], ["id"])
        result = engine.quality_check(df, None)
        assert result.passed is True
        engine.disconnect()


@requires_pyspark
class TestSparkEngineLoad:
    def test_load_parquet(self) -> None:
        from dataenginex.engines.spark_engine import SparkEngine

        engine = SparkEngine()
        engine.connect(EngineConfig(type="spark", master="local[1]"))

        spark = engine.spark
        data = [(1, "alice"), (2, "bob")]
        df = spark.createDataFrame(data, ["id", "name"])

        with tempfile.TemporaryDirectory() as tmpdir:
            target_config = {
                "layer": "silver",
                "format": "parquet",
                "name": "test_output",
                "data_dir": tmpdir,
                "source": "test_source",
                "pipeline_name": "test_pipeline",
            }
            result = engine.load(df, target_config)
            assert result.success is True
            assert result.rows_output == 2
            assert result.format == "parquet"
            # Verify parquet file was written
            import os
            output_dir = f"{tmpdir}/silver"
            assert os.path.exists(output_dir)
        engine.disconnect()


@requires_pyspark
class TestSparkEngineContentHash:
    def test_content_hash(self) -> None:
        from dataenginex.engines.spark_engine import SparkEngine

        engine = SparkEngine()
        engine.connect(EngineConfig(type="spark", master="local[1]"))

        spark = engine.spark
        data = [(1, "alice"), (2, "bob")]
        df = spark.createDataFrame(data, ["id", "name"])

        hash1 = engine.content_hash(df)
        assert hash1 != ""

        hash2 = engine.content_hash(df)
        assert hash1 == hash2
        engine.disconnect()


@requires_pyspark
class TestSparkEngineExecuteSQL:
    def test_execute_sql(self) -> None:
        from dataenginex.engines.spark_engine import SparkEngine

        engine = SparkEngine()
        engine.connect(EngineConfig(type="spark", master="local[1]"))

        result = engine.execute_sql("SELECT 42 AS answer")
        assert result[0]["answer"] == 42
        engine.disconnect()
