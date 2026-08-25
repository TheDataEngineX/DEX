"""Tests for DuckDBEngine — Phase 2 (full implementation)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pyarrow as pa
import pytest

from dataenginex.engines.base import EngineConfig
from dataenginex.engines.duckdb_engine import DuckDBEngine
from dataenginex.engines.registry import engine_registry


class TestDuckDBEngineRegistration:
    def test_auto_registered(self) -> None:
        assert engine_registry.is_registered("duckdb")

    def test_get_engine(self) -> None:
        engine = engine_registry.get("duckdb")
        assert isinstance(engine, DuckDBEngine)


class TestDuckDBEngineConnect:
    def test_connect_memory(self) -> None:
        engine = DuckDBEngine()
        config = EngineConfig(type="duckdb", path=":memory:")
        engine.connect(config)
        assert engine._conn is not None
        engine.disconnect()

    def test_connect_with_threads(self) -> None:
        engine = DuckDBEngine()
        config = EngineConfig(type="duckdb", path=":memory:", threads=2)
        engine.connect(config)
        result = engine._conn.execute("SELECT current_setting('threads')").fetchone()
        assert result is not None
        engine.disconnect()

    def test_connect_with_memory_limit(self) -> None:
        engine = DuckDBEngine()
        config = EngineConfig(type="duckdb", path=":memory:", memory_limit="256MB")
        engine.connect(config)
        result = engine._conn.execute("SELECT current_setting('memory_limit')").fetchone()
        assert result is not None
        engine.disconnect()

    def test_disconnect(self) -> None:
        engine = DuckDBEngine()
        config = EngineConfig(type="duckdb", path=":memory:")
        engine.connect(config)
        engine.disconnect()
        assert engine._conn is None

    def test_conn_property_raises_when_disconnected(self) -> None:
        engine = DuckDBEngine()
        with pytest.raises(RuntimeError, match="not connected"):
            _ = engine.conn


class TestDuckDBEngineCapabilities:
    def test_capabilities(self) -> None:
        engine = DuckDBEngine()
        caps = engine.capabilities()
        assert caps.name == "duckdb"
        assert caps.streaming is False
        assert caps.distributed is False
        assert caps.auto_cdc is False
        assert caps.iceberg_read is True
        assert caps.iceberg_write is False
        assert caps.delta_read is True
        assert caps.delta_write is True
        assert caps.mllib is False
        assert caps.catalyst is False
        assert caps.spark_connect is False


class TestDuckDBEngineExtract:
    def test_extract_from_pylist(self) -> None:
        engine = DuckDBEngine()
        engine.connect(EngineConfig(type="duckdb", path=":memory:"))

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
        assert result == 3
        df = engine.execute_sql("SELECT * FROM bronze")
        assert len(df) == 3
        engine.disconnect()


class TestDuckDBEngineTransform:
    def test_transform_with_filter(self) -> None:
        engine = DuckDBEngine()
        engine.connect(EngineConfig(type="duckdb", path=":memory:"))

        engine.conn.execute("""
            CREATE OR REPLACE TABLE bronze AS
            SELECT * FROM (VALUES
                (1, 'alice', 100),
                (2, 'bob', 200),
                (3, 'charlie', 150)
            ) AS t(id, name, value)
        """)

        from dataenginex.domains.analytics.transforms.sql import FilterTransform

        steps = [(FilterTransform, {"condition": "value > 120"})]
        result = engine.transform("bronze", steps)
        assert result is not None
        df = engine.execute_sql(f"SELECT * FROM {result}")
        assert len(df) == 2
        engine.disconnect()


class TestDuckDBEngineQualityCheck:
    def test_quality_check_pass(self) -> None:
        engine = DuckDBEngine()
        engine.connect(EngineConfig(type="duckdb", path=":memory:"))

        engine.conn.execute("""
            CREATE OR REPLACE TABLE test_table AS
            SELECT * FROM (VALUES
                (1, 'alice'),
                (2, 'bob'),
                (3, 'charlie')
            ) AS t(id, name)
        """)

        checks = {"completeness": 0.5, "uniqueness": ["id"]}
        result = engine.quality_check("test_table", checks)
        assert result.passed is True
        engine.disconnect()

    def test_quality_check_no_checks(self) -> None:
        engine = DuckDBEngine()
        engine.connect(EngineConfig(type="duckdb", path=":memory:"))
        result = engine.quality_check("any_table", None)
        assert result.passed is True
        engine.disconnect()


class TestDuckDBEngineLoad:
    def test_load_parquet(self) -> None:
        engine = DuckDBEngine()
        engine.connect(EngineConfig(type="duckdb", path=":memory:"))

        engine.conn.execute("""
            CREATE OR REPLACE TABLE test_table AS
            SELECT * FROM (VALUES
                (1, 'alice'),
                (2, 'bob')
            ) AS t(id, name)
        """)

        with tempfile.TemporaryDirectory() as tmpdir:
            target_config = {
                "layer": "silver",
                "format": "parquet",
                "name": "test_output",
                "data_dir": tmpdir,
                "source": "test_source",
                "pipeline_name": "test_pipeline",
            }
            result = engine.load("test_table", target_config)
            assert result.success is True
            assert result.rows_output == 2
            assert result.format == "parquet"
            output_path = Path(tmpdir) / "silver" / "test_output.parquet"
            assert output_path.exists()
        engine.disconnect()


class TestDuckDBEngineMerge:
    def test_merge_upsert(self) -> None:
        engine = DuckDBEngine()
        engine.connect(EngineConfig(type="duckdb", path=":memory:"))

        engine.conn.execute("""
            CREATE OR REPLACE TABLE target_table AS
            SELECT * FROM (VALUES
                (1, 'alice_v1'),
                (2, 'bob_v1')
            ) AS t(id, name)
        """)
        engine.conn.execute("""
            CREATE OR REPLACE TABLE source_table AS
            SELECT * FROM (VALUES
                (2, 'bob_v2'),
                (3, 'charlie')
            ) AS t(id, name)
        """)

        result = engine.merge("target_table", "source_table", ["id"], "upsert")
        assert result.success is True
        # After merge: target has 3 rows (2 original + 1 new), 1 inserted
        target = engine.read_table("target_table")
        assert len(target) == 3
        engine.disconnect()


class TestDuckDBEngineSCD2:
    def test_scd2_first_run(self) -> None:
        engine = DuckDBEngine()
        engine.connect(EngineConfig(type="duckdb", path=":memory:"))

        engine.conn.execute("""
            CREATE OR REPLACE TABLE source_data AS
            SELECT * FROM (VALUES
                (1, 'alice'),
                (2, 'bob')
            ) AS t(id, name)
        """)

        result = engine.scd_type2("target_scd2", "source_data", ["id"])
        assert result.success is True
        assert result.rows_inserted == 2
        engine.disconnect()


class TestDuckDBEngineContentHash:
    def test_content_hash(self) -> None:
        engine = DuckDBEngine()
        engine.connect(EngineConfig(type="duckdb", path=":memory:"))

        engine.conn.execute("""
            CREATE OR REPLACE TABLE test_hash AS
            SELECT * FROM (VALUES
                (1, 'alice'),
                (2, 'bob')
            ) AS t(id, name)
        """)

        hash1 = engine.content_hash("test_hash")
        assert hash1 != ""
        assert "test_hash" in hash1

        hash2 = engine.content_hash("test_hash")
        assert hash1 == hash2
        engine.disconnect()


class TestDuckDBEngineReadWriteTable:
    def test_write_and_read_table(self) -> None:
        engine = DuckDBEngine()
        engine.connect(EngineConfig(type="duckdb", path=":memory:"))

        arrow_table = pa.Table.from_pylist([
            {"id": 1, "name": "alice"},
            {"id": 2, "name": "bob"},
        ])
        engine.write_table(arrow_table, "my_table")
        result = engine.read_table("my_table")
        assert len(result) == 2
        engine.disconnect()


class TestDuckDBEngineExecuteSQL:
    def test_execute_sql(self) -> None:
        engine = DuckDBEngine()
        engine.connect(EngineConfig(type="duckdb", path=":memory:"))

        result = engine.execute_sql("SELECT 42 AS answer")
        assert result[0]["answer"] == 42
        engine.disconnect()
