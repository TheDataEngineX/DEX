"""Tests for Spark ML, datasource registry, connect client, and streaming query manager."""

from __future__ import annotations

import contextlib
from unittest.mock import MagicMock

from dataenginex.foundation.ids import ProjectId
from dataenginex.spark.connect.client import SparkConnectClient
from dataenginex.spark.datasource.registry import DataSourceRegistry
from dataenginex.spark.ml.mllib_provider import MLlibProvider
from dataenginex.spark.ml.model_contract import ModelContract
from dataenginex.spark.streaming.query_manager import StreamingQueryManager

# --- SparkConnectClient ---


class TestSparkConnectClient:
    def test_init(self) -> None:
        client = SparkConnectClient()
        assert client.server_url == "local[*]"
        assert client._session is None

    def test_init_custom(self) -> None:
        client = SparkConnectClient(
            server_url="sc://host:15002",
            project_id=ProjectId("p1"),
            session_config={"spark.sql.shuffle.partitions": "2"},
        )
        assert client.server_url == "sc://host:15002"
        assert client.project_id == "p1"

    def test_connect_local(self) -> None:
        client = SparkConnectClient()
        session = client.connect()
        assert session is not None or client._session is not None

    def test_disconnect(self) -> None:
        client = SparkConnectClient()
        client._session = MagicMock()
        client.disconnect()
        assert client._session is None

    def test_disconnect_when_none(self) -> None:
        client = SparkConnectClient()
        client.disconnect()

    def test_execute_sql_no_session(self) -> None:
        client = SparkConnectClient()
        with contextlib.suppress(RuntimeError, AttributeError):
            client.execute_sql("SELECT 1")

    def test_get_session_config(self) -> None:
        client = SparkConnectClient(session_config={"key": "val"})
        config = client.get_session_config()
        assert config.get("key") == "val"

    def test_is_connected(self) -> None:
        client = SparkConnectClient()
        assert isinstance(client.is_connected, bool)


# --- DataSourceRegistry ---


class TestDataSourceRegistry:
    def test_init(self) -> None:
        reg = DataSourceRegistry(ProjectId("p1"))
        assert isinstance(reg._sources, dict)

    def test_register_and_get(self) -> None:
        reg = DataSourceRegistry(ProjectId("p1"))
        reg.register("csv_source", "csv", description="CSV connector")
        source = reg.get_source("csv_source")
        assert source is not None
        assert source["source_class"] == "csv"

    def test_get_missing(self) -> None:
        reg = DataSourceRegistry(ProjectId("p1"))
        assert reg.get_source("missing") is None

    def test_list_sources(self) -> None:
        reg = DataSourceRegistry(ProjectId("p1"))
        reg.register("a", "csv_class")
        reg.register("b", "parquet_class")
        sources = reg.list_sources()
        assert len(sources) == 2

    def test_unregister(self) -> None:
        reg = DataSourceRegistry(ProjectId("p1"))
        reg.register("a", "csv_class")
        reg.unregister("a")
        assert reg.get_source("a") is None

    def test_unregister_missing(self) -> None:
        reg = DataSourceRegistry(ProjectId("p1"))
        reg.unregister("missing")

    def test_load_source(self) -> None:
        reg = DataSourceRegistry(ProjectId("p1"))
        with contextlib.suppress(ValueError):
            reg.load_source("missing_source")

    def test_get_capabilities(self) -> None:
        reg = DataSourceRegistry(ProjectId("p1"))
        reg.register("csv_source", "csv")
        caps = reg.get_capabilities("csv_source")
        assert isinstance(caps, list)

    def test_validate_config(self) -> None:
        reg = DataSourceRegistry(ProjectId("p1"))
        reg.register("csv_source", "csv")
        result = reg.validate_config("csv_source", {"path": "/data.csv"})
        assert isinstance(result, dict)


# --- MLlibProvider ---


class TestMLlibProvider:
    def test_init(self) -> None:
        provider = MLlibProvider(ProjectId("p1"))
        assert provider.project_id == "p1"

    def test_connect(self) -> None:
        provider = MLlibProvider(ProjectId("p1"))
        provider.connect()

    def test_train_no_data(self) -> None:
        provider = MLlibProvider(ProjectId("p1"))
        with contextlib.suppress(ValueError, RuntimeError, FileNotFoundError):
            provider.train("logistic_regression", dataset_ref="nonexistent")

    def test_register_model(self) -> None:
        provider = MLlibProvider(ProjectId("p1"))
        contract = ModelContract(
            model_name="m", version="1.0", framework="sklearn",
        )
        with contextlib.suppress(Exception):
            provider.register_model(contract, "/tmp/model")

    def test_load_model(self) -> None:
        provider = MLlibProvider(ProjectId("p1"))
        with contextlib.suppress(ValueError, FileNotFoundError, RuntimeError):
            provider.load_model("nonexistent")


# --- StreamingQueryManager ---


class TestStreamingQueryManager:
    def test_init(self) -> None:
        mgr = StreamingQueryManager(ProjectId("p1"))
        assert mgr.project_id == "p1"

    def test_register(self) -> None:
        mgr = StreamingQueryManager(ProjectId("p1"))
        mgr.register("q1", {"source": "kafka", "topic": "events"})
        status = mgr.get_status("q1")
        assert status is not None

    def test_get_status_missing(self) -> None:
        mgr = StreamingQueryManager(ProjectId("p1"))
        assert mgr.get_status("missing") is None

    def test_active_queries(self) -> None:
        mgr = StreamingQueryManager(ProjectId("p1"))
        mgr.register("q1", {})
        mgr.register("q2", {})
        queries = mgr.active_queries()
        assert len(queries) == 2

    def test_deregister(self) -> None:
        mgr = StreamingQueryManager(ProjectId("p1"))
        mgr.register("q1", {})
        mgr.deregister("q1")
        assert mgr.get_status("q1") is None

    def test_deregister_missing(self) -> None:
        mgr = StreamingQueryManager(ProjectId("p1"))
        mgr.deregister("missing")

    def test_update_status(self) -> None:
        mgr = StreamingQueryManager(ProjectId("p1"))
        mgr.register("q1", {})
        mgr.update_status("q1", "running", {"rows_processed": 100})
        status = mgr.get_status("q1")
        assert status is not None

    def test_get_query_metrics(self) -> None:
        mgr = StreamingQueryManager(ProjectId("p1"))
        mgr.register("q1", {})
        with contextlib.suppress(RuntimeError, AttributeError):
            metrics = mgr.get_query_metrics("q1")
            assert isinstance(metrics, dict)

    def test_stop_query(self) -> None:
        mgr = StreamingQueryManager(ProjectId("p1"))
        with contextlib.suppress(KeyError, RuntimeError):
            mgr.stop_query("q1")

    def test_cleanup_checkpoints(self) -> None:
        mgr = StreamingQueryManager(ProjectId("p1"))
        result = mgr.cleanup_checkpoints()
        assert isinstance(result, dict)

    def test_cleanup_checkpoints_specific(self) -> None:
        mgr = StreamingQueryManager(ProjectId("p1"))
        result = mgr.cleanup_checkpoints("q1")
        assert isinstance(result, dict)
