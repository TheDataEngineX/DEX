"""Tests for dex.yaml Pydantic schema models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dataenginex.config.schema import (
    AiConfig,
    DataConfig,
    DexConfig,
    MlConfig,
    PipelineConfig,
    ProjectConfig,
    SourceConfig,
    SparkConfig,
)


class TestProjectConfig:
    def test_minimal(self) -> None:
        cfg = ProjectConfig(name="test-project")
        assert cfg.name == "test-project"
        assert cfg.version == "0.1.0"

    def test_with_version(self) -> None:
        cfg = ProjectConfig(name="demo", version="1.0.0")
        assert cfg.version == "1.0.0"


class TestSourceConfig:
    def test_csv_source(self) -> None:
        cfg = SourceConfig(type="csv", path="data/input.csv")
        assert cfg.type == "csv"

    def test_duckdb_source(self) -> None:
        cfg = SourceConfig(type="duckdb", query="SELECT * FROM users")
        assert cfg.type == "duckdb"


class TestSparkConfig:
    def test_defaults(self) -> None:
        cfg = SparkConfig()
        assert cfg.master == "local[*]"
        assert cfg.warehouse == ".dex/lakehouse"
        assert cfg.file_format == "parquet"

    def test_custom(self) -> None:
        cfg = SparkConfig(
            master="spark://host:7077",
            warehouse="/mnt/iceberg",
            file_format="iceberg",
            executor_memory="4g",
            executor_cores=2,
        )
        assert cfg.master == "spark://host:7077"
        assert cfg.executor_memory == "4g"

    def test_is_frozen(self) -> None:
        cfg = SparkConfig()
        with pytest.raises(ValidationError):
            cfg.master = "changed"  # type: ignore[misc]


class TestPipelineConfig:
    def test_minimal_pipeline(self) -> None:
        cfg = PipelineConfig(
            source="raw_data",
            transforms=[],
            destination="silver_users",
        )
        assert cfg.source == "raw_data"
        assert len(cfg.transforms) == 0

    def test_default_engine_is_duckdb(self) -> None:
        cfg = PipelineConfig(source="raw")
        assert cfg.engine == "duckdb"

    def test_spark_engine(self) -> None:
        cfg = PipelineConfig(source="raw", engine="spark")
        assert cfg.engine == "spark"

    def test_auto_engine(self) -> None:
        cfg = PipelineConfig(source="raw", engine="auto")
        assert cfg.engine == "auto"

    def test_invalid_engine_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PipelineConfig(source="raw", engine="invalid")  # type: ignore[arg-type]

    def test_with_spark_config(self) -> None:
        cfg = PipelineConfig(
            source="raw",
            engine="spark",
            spark=SparkConfig(master="spark://cluster:7077"),
        )
        assert cfg.spark.master == "spark://cluster:7077"


class TestDataConfig:
    def test_with_sources_and_pipelines(self) -> None:
        cfg = DataConfig(
            sources={
                "users": SourceConfig(type="csv", path="data/users.csv"),
            },
            pipelines={
                "clean_users": PipelineConfig(
                    source="users",
                    transforms=[],
                    destination="silver_users",
                ),
            },
        )
        assert "users" in cfg.sources
        assert "clean_users" in cfg.pipelines

    def test_default_engine_is_duckdb(self) -> None:
        cfg = DataConfig()
        assert cfg.engine == "duckdb"

    def test_spark_engine(self) -> None:
        cfg = DataConfig(engine="spark")
        assert cfg.engine == "spark"

    def test_with_spark_config(self) -> None:
        cfg = DataConfig(
            engine="spark",
            spark=SparkConfig(master="spark://cluster:7077", file_format="iceberg"),
        )
        assert cfg.spark.master == "spark://cluster:7077"
        assert cfg.spark.file_format == "iceberg"


class TestMlConfig:
    def test_defaults(self) -> None:
        cfg = MlConfig()
        assert cfg.tracker == "builtin"


class TestAiConfig:
    def test_defaults(self) -> None:
        cfg = AiConfig()
        assert cfg.llm.provider == "ollama"


class TestDexConfig:
    def test_minimal_valid_config(self) -> None:
        cfg = DexConfig(project=ProjectConfig(name="minimal"))
        assert cfg.project.name == "minimal"
        assert cfg.data.engine == "duckdb"
        assert cfg.ml.tracker == "builtin"

    def test_all_sections_optional_except_project(self) -> None:
        with pytest.raises(ValidationError):
            DexConfig()  # type: ignore[call-arg]

    def test_no_server_field(self) -> None:
        cfg = DexConfig(project=ProjectConfig(name="srv"))
        assert not hasattr(cfg, "server")
