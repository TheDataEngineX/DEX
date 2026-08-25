"""Tests for dataenginex.providers.connectors.dbt — DbtConnector (DuckDB + Spark)."""

from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import MagicMock, patch

import duckdb
import pytest

from dataenginex.providers.connectors.dbt import (
    DbtConnector,
    _generate_duckdb_profile,
    _generate_spark_profile,
)


@pytest.fixture()
def dbt_project(tmp_path: Path) -> Path:
    (tmp_path / "dbt_project.yml").write_text("name: test_project\n")
    return tmp_path


@pytest.fixture()
def dbt_database(dbt_project: Path) -> Path:
    db_path = dbt_project / "dev.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute("CREATE TABLE users_cleaned AS SELECT 1 AS id, 'alice' AS name")
    conn.close()
    return db_path


def _ok_run(*args: object, **kwargs: object) -> CompletedProcess[str]:
    return CompletedProcess(args=[], returncode=0, stdout="", stderr="")


def _fail_run(*args: object, **kwargs: object) -> CompletedProcess[str]:
    return CompletedProcess(args=[], returncode=1, stdout="Error output", stderr="stderr msg")


# ---------------------------------------------------------------------------
# Profile generation
# ---------------------------------------------------------------------------

class TestGenerateDuckDBProfile:
    def test_basic(self) -> None:
        p = _generate_duckdb_profile("/tmp/test.duckdb")
        assert p["profiles"]["dex_project"]["outputs"]["dev"]["type"] == "duckdb"
        assert p["profiles"]["dex_project"]["outputs"]["dev"]["path"] == "/tmp/test.duckdb"

    def test_custom_target(self) -> None:
        p = _generate_duckdb_profile("/tmp/test.duckdb", target="prod")
        assert "prod" in p["profiles"]["dex_project"]["outputs"]


class TestGenerateSparkProfile:
    def test_basic(self) -> None:
        p = _generate_spark_profile("/tmp/warehouse")
        out = p["profiles"]["dex_project"]["outputs"]["dev"]
        assert out["type"] == "spark"
        assert out["method"] == "shell"
        assert out["schema"] == "default"

    def test_custom_schema(self) -> None:
        p = _generate_spark_profile("/tmp/warehouse", schema="analytics")
        assert p["profiles"]["dex_project"]["outputs"]["dev"]["schema"] == "analytics"


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------

class TestDbtConnectorImportGuard:
    def test_raises_import_error_when_dbt_missing(self, tmp_path: Path) -> None:
        with (
            patch("dataenginex.providers.connectors.dbt._DBT_AVAILABLE", False),
            pytest.raises(ImportError, match="dbt CLI not found"),
        ):
            DbtConnector(project_dir=str(tmp_path), model="my_model")

    def test_registered_as_dbt(self) -> None:
        from dataenginex.providers.connectors import connector_registry

        cls = connector_registry.get("dbt")
        assert cls is DbtConnector


# ---------------------------------------------------------------------------
# DuckDB engine (existing tests)
# ---------------------------------------------------------------------------

class TestDbtConnectorDuckDBConnect:
    def test_connect_succeeds(self, dbt_project: Path) -> None:
        with patch("dataenginex.providers.connectors.dbt._DBT_AVAILABLE", True):
            c = DbtConnector(project_dir=str(dbt_project), model="m", engine="duckdb")
            c.connect()

    def test_connect_raises_when_project_missing(self, tmp_path: Path) -> None:
        with patch("dataenginex.providers.connectors.dbt._DBT_AVAILABLE", True):
            c = DbtConnector(project_dir=str(tmp_path), model="m", engine="duckdb")
            with pytest.raises(FileNotFoundError, match="dbt project not found"):
                c.connect()

    def test_disconnect_is_idempotent(self, dbt_project: Path) -> None:
        with patch("dataenginex.providers.connectors.dbt._DBT_AVAILABLE", True):
            c = DbtConnector(project_dir=str(dbt_project), model="m")
            c.disconnect()
            c.disconnect()


class TestDbtConnectorDuckDBRead:
    def test_read_returns_rows(self, dbt_project: Path, dbt_database: Path) -> None:
        with (
            patch("dataenginex.providers.connectors.dbt._DBT_AVAILABLE", True),
            patch("subprocess.run", side_effect=_ok_run),
        ):
            c = DbtConnector(project_dir=str(dbt_project), model="users_cleaned")
            rows = c.read()
        assert len(rows) == 1
        assert rows[0]["name"] == "alice"

    def test_read_table_arg_overrides(self, dbt_project: Path, dbt_database: Path) -> None:
        with (
            patch("dataenginex.providers.connectors.dbt._DBT_AVAILABLE", True),
            patch("subprocess.run", side_effect=_ok_run) as mock_sub,
        ):
            c = DbtConnector(project_dir=str(dbt_project), model="default_model")
            c.read(table="users_cleaned")
        cmd = mock_sub.call_args[0][0]
        assert "users_cleaned" in cmd

    def test_read_dbt_failure_raises(self, dbt_project: Path) -> None:
        with (
            patch("dataenginex.providers.connectors.dbt._DBT_AVAILABLE", True),
            patch("subprocess.run", side_effect=_fail_run),
        ):
            c = DbtConnector(project_dir=str(dbt_project), model="m")
            with pytest.raises(RuntimeError, match="dbt run failed"):
                c.read()

    def test_read_missing_database_returns_default(self, dbt_project: Path) -> None:
        with (
            patch("dataenginex.providers.connectors.dbt._DBT_AVAILABLE", True),
            patch("subprocess.run", side_effect=_ok_run),
        ):
            c = DbtConnector(project_dir=str(dbt_project), model="m")
            rows = c.read(default=[])
        assert rows == []

    def test_read_missing_model_returns_default(
        self, dbt_project: Path, dbt_database: Path
    ) -> None:
        with (
            patch("dataenginex.providers.connectors.dbt._DBT_AVAILABLE", True),
            patch("subprocess.run", side_effect=_ok_run),
        ):
            c = DbtConnector(project_dir=str(dbt_project), model="nonexistent")
            rows = c.read(default=[])
        assert rows == []


class TestDbtConnectorDuckDBWrite:
    def test_write_raises(self, dbt_project: Path) -> None:
        with patch("dataenginex.providers.connectors.dbt._DBT_AVAILABLE", True):
            c = DbtConnector(project_dir=str(dbt_project), model="m")
            with pytest.raises(NotImplementedError, match="read-only"):
                c.write([{"x": 1}])


# ---------------------------------------------------------------------------
# Spark engine (new)
# ---------------------------------------------------------------------------

class TestDbtConnectorSparkConnect:
    def test_connect_succeeds(self, dbt_project: Path) -> None:
        with patch("dataenginex.providers.connectors.dbt._DBT_AVAILABLE", True):
            c = DbtConnector(
                project_dir=str(dbt_project),
                model="m",
                engine="spark",
                warehouse="/tmp/wh",
            )
            c.connect()
            assert c._profile_dir is not None
            assert (c._profile_dir / "profiles.yml").exists()

    def test_connect_writes_spark_profile(self, dbt_project: Path) -> None:
        with patch("dataenginex.providers.connectors.dbt._DBT_AVAILABLE", True):
            c = DbtConnector(
                project_dir=str(dbt_project),
                model="m",
                engine="spark",
                warehouse="/tmp/wh",
                schema="analytics",
            )
            c.connect()
            import yaml

            profile = yaml.safe_load((c._profile_dir / "profiles.yml").read_text())
            out = profile["profiles"]["dex_project"]["outputs"]["dev"]
            assert out["type"] == "spark"
            assert out["schema"] == "analytics"

    def test_connect_raises_when_project_missing(self, tmp_path: Path) -> None:
        with patch("dataenginex.providers.connectors.dbt._DBT_AVAILABLE", True):
            c = DbtConnector(
                project_dir=str(tmp_path),
                model="m",
                engine="spark",
            )
            with pytest.raises(FileNotFoundError, match="dbt project not found"):
                c.connect()


class TestDbtConnectorSparkRead:
    def test_read_uses_spark_session(self, dbt_project: Path) -> None:
        mock_spark = MagicMock()
        mock_df = MagicMock()
        mock_row = MagicMock()
        mock_row.asDict.return_value = {"id": 1, "name": "alice"}
        mock_df.collect.return_value = [mock_row]
        mock_spark.table.return_value = mock_df

        mock_pyspark = MagicMock()
        mock_pyspark.sql.SparkSession = MagicMock()
        mock_builder = mock_pyspark.sql.SparkSession.builder
        mock_chain = mock_builder.master.return_value
        mock_chain = mock_chain.appName.return_value
        mock_chain = mock_chain.config.return_value
        mock_chain.getOrCreate.return_value = mock_spark

        with (
            patch("dataenginex.providers.connectors.dbt._DBT_AVAILABLE", True),
            patch("subprocess.run", side_effect=_ok_run),
            patch.dict("sys.modules", {"pyspark": mock_pyspark, "pyspark.sql": mock_pyspark.sql}),
        ):
            c = DbtConnector(
                project_dir=str(dbt_project),
                model="users_cleaned",
                engine="spark",
            )
            c.connect()
            rows = c._read_model_spark("users_cleaned", default=None)

        assert len(rows) == 1
        assert rows[0]["name"] == "alice"


# ---------------------------------------------------------------------------
# Health check (both engines)
# ---------------------------------------------------------------------------

class TestDbtConnectorHealthCheck:
    def test_true_when_project_exists(self, dbt_project: Path) -> None:
        with patch("dataenginex.providers.connectors.dbt._DBT_AVAILABLE", True):
            c = DbtConnector(project_dir=str(dbt_project), model="m")
            assert c.health_check() is True

    def test_false_when_project_missing(self, tmp_path: Path) -> None:
        with patch("dataenginex.providers.connectors.dbt._DBT_AVAILABLE", True):
            c = DbtConnector(project_dir=str(tmp_path), model="m")
            assert c.health_check() is False
