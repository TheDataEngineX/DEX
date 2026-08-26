"""Tests for Spark infrastructure modules (no PySpark required)."""

from __future__ import annotations

from dataenginex.foundation.ids import ProjectId, RunId
from dataenginex.spark.catalog.identifiers import SparkCatalogIdentifier
from dataenginex.spark.connect.server_manager import SparkServerManager, SparkServerState
from dataenginex.spark.connect.session_registry import SparkSessionRegistry
from dataenginex.spark.datasource.capabilities import DataSourceCapabilities
from dataenginex.spark.lineage.openlineage_projection import OpenLineageProjection
from dataenginex.spark.ml.model_contract import ModelContract
from dataenginex.spark.sql.portable_sql import Dialect, PortableSQLAdapter
from dataenginex.spark.streaming.checkpoint_registry import CheckpointRegistry
from dataenginex.spark.streaming.health import StreamingHealthMonitor

# --- SparkCatalogIdentifier ---


class TestSparkCatalogIdentifier:
    def test_defaults(self) -> None:
        ident = SparkCatalogIdentifier(table="t")
        assert ident.catalog == "spark_catalog"
        assert ident.namespace == "default"
        assert ident.table == "t"

    def test_to_spark_sql(self) -> None:
        ident = SparkCatalogIdentifier(catalog="cat", namespace="ns", table="t")
        assert ident.to_spark_sql() == "cat.ns.t"

    def test_to_hive_sql(self) -> None:
        ident = SparkCatalogIdentifier(catalog="cat", namespace="ns", table="t")
        assert ident.to_hive_sql() == "ns.t"

    def test_from_spark_sql_three_parts(self) -> None:
        ident = SparkCatalogIdentifier.from_spark_sql("cat.ns.t")
        assert ident.catalog == "cat"
        assert ident.namespace == "ns"
        assert ident.table == "t"

    def test_from_spark_sql_two_parts(self) -> None:
        ident = SparkCatalogIdentifier.from_spark_sql("ns.t")
        assert ident.catalog == "spark_catalog"
        assert ident.namespace == "ns"
        assert ident.table == "t"

    def test_from_spark_sql_one_part(self) -> None:
        ident = SparkCatalogIdentifier.from_spark_sql("t")
        assert ident.table == "t"


# --- SparkSessionRegistry ---


class TestSparkSessionRegistry:
    def test_register_and_get(self) -> None:
        reg = SparkSessionRegistry()
        key = reg.register(ProjectId("p1"), RunId("r1"), {"mode": "local"})
        assert key == "p1:r1"
        session = reg.get(ProjectId("p1"), RunId("r1"))
        assert session is not None
        assert session["config"] == {"mode": "local"}

    def test_get_missing(self) -> None:
        reg = SparkSessionRegistry()
        assert reg.get(ProjectId("p1"), RunId("r1")) is None

    def test_unregister(self) -> None:
        reg = SparkSessionRegistry()
        reg.register(ProjectId("p1"), RunId("r1"), {})
        reg.unregister(ProjectId("p1"), RunId("r1"))
        assert reg.get(ProjectId("p1"), RunId("r1")) is None

    def test_unregister_missing(self) -> None:
        reg = SparkSessionRegistry()
        reg.unregister(ProjectId("p1"), RunId("r1"))  # no-op

    def test_list_sessions(self) -> None:
        reg = SparkSessionRegistry()
        reg.register(ProjectId("p1"), RunId("r1"), {"a": 1})
        reg.register(ProjectId("p2"), RunId("r2"), {"b": 2})
        sessions = reg.list_sessions()
        assert len(sessions) == 2


# --- SparkServerManager ---


class TestSparkServerManager:
    def test_default_state(self) -> None:
        mgr = SparkServerManager()
        assert mgr.state == SparkServerState.STOPPED
        assert mgr.server_url == "local"

    def test_custom_url(self) -> None:
        mgr = SparkServerManager(server_url="http://localhost:15002")
        assert mgr.server_url == "http://localhost:15002"

    def test_start_stop(self) -> None:
        mgr = SparkServerManager()
        mgr.start()
        assert mgr.state == SparkServerState.RUNNING
        mgr.stop()
        assert mgr.state == SparkServerState.STOPPED

    def test_status(self) -> None:
        mgr = SparkServerManager()
        assert mgr.status() == SparkServerState.STOPPED


# --- StreamingHealthMonitor ---


class TestStreamingHealthMonitor:
    def test_check_health_empty(self) -> None:
        mon = StreamingHealthMonitor()
        result = mon.check_health("q1")
        assert result["status"] == "healthy"
        assert result["metrics"] == {}

    def test_record_and_check(self) -> None:
        mon = StreamingHealthMonitor()
        mon.record_metrics("q1", {"rows": 100})
        result = mon.check_health("q1")
        assert result["metrics"] == {"rows": 100}


# --- CheckpointRegistry ---


class TestCheckpointRegistry:
    def test_default_base_path(self) -> None:
        reg = CheckpointRegistry()
        assert reg.base_path == "/tmp/checkpoints"

    def test_custom_base_path(self) -> None:
        reg = CheckpointRegistry(base_path="/data/checkpoints")
        assert reg.base_path == "/data/checkpoints"

    def test_get_checkpoint_path(self) -> None:
        reg = CheckpointRegistry()
        assert reg.get_checkpoint_path("q1") == "/tmp/checkpoints/q1"

    def test_register_and_exists(self) -> None:
        reg = CheckpointRegistry()
        assert reg.exists("q1") is False
        reg.register("q1", "/check/q1")
        assert reg.exists("q1") is True


# --- PortableSQLAdapter ---


class TestPortableSQLAdapter:
    def test_portable_passthrough(self) -> None:
        adapter = PortableSQLAdapter(Dialect.PORTABLE)
        assert adapter.translate("SELECT 1") == "SELECT 1"

    def test_duckdb_to_spark_integer(self) -> None:
        adapter = PortableSQLAdapter(Dialect.SPARK)
        result = adapter.translate("SELECT CAST(x AS INTEGER) FROM t")
        assert "INT" in result

    def test_duckdb_to_spark_varchar(self) -> None:
        adapter = PortableSQLAdapter(Dialect.SPARK)
        result = adapter.translate("SELECT CAST(x AS VARCHAR) FROM t")
        assert "STRING" in result

    def test_duckdb_to_spark_current_date(self) -> None:
        adapter = PortableSQLAdapter(Dialect.SPARK)
        result = adapter.translate("SELECT CURRENT_DATE")
        assert "current_date()" in result

    def test_duckdb_to_spark_date_diff(self) -> None:
        adapter = PortableSQLAdapter(Dialect.SPARK)
        result = adapter.translate("SELECT DATE_DIFF(day, a, b)")
        assert "datediff" in result.lower()

    def test_duckdb_to_spark_string_split(self) -> None:
        adapter = PortableSQLAdapter(Dialect.SPARK)
        result = adapter.translate("SELECT STRING_SPLIT(x, ',')")
        assert "split(" in result.lower()

    def test_duckdb_to_spark_string_agg(self) -> None:
        adapter = PortableSQLAdapter(Dialect.SPARK)
        result = adapter.translate("SELECT STRING_AGG(x, ',')")
        assert "collect_list" in result.lower()

    def test_duckdb_to_spark_list_contains(self) -> None:
        adapter = PortableSQLAdapter(Dialect.SPARK)
        result = adapter.translate("SELECT LIST_CONTAINS(arr, 'x')")
        assert "array_contains" in result.lower()

    def test_duckdb_to_spark_list_len(self) -> None:
        adapter = PortableSQLAdapter(Dialect.SPARK)
        result = adapter.translate("SELECT LIST_LEN(arr)")
        assert "size(" in result.lower()

    def test_duckdb_to_spark_generate_series(self) -> None:
        adapter = PortableSQLAdapter(Dialect.SPARK)
        result = adapter.translate("SELECT GENERATE_SERIES(1, 10)")
        assert "sequence(" in result.lower()

    def test_duckdb_to_spark_json_extract(self) -> None:
        adapter = PortableSQLAdapter(Dialect.SPARK)
        result = adapter.translate("SELECT JSON_EXTRACT(j, '$.x')")
        assert "get_json_object" in result.lower()

    def test_spark_to_duckdb_int(self) -> None:
        adapter = PortableSQLAdapter(Dialect.DUCKDB)
        result = adapter.translate("SELECT CAST(x AS INT)")
        assert "INTEGER" in result

    def test_spark_to_duckdb_string(self) -> None:
        adapter = PortableSQLAdapter(Dialect.DUCKDB)
        result = adapter.translate("SELECT CAST(x AS STRING)")
        assert "VARCHAR" in result

    def test_spark_to_duckdb_current_date(self) -> None:
        adapter = PortableSQLAdapter(Dialect.DUCKDB)
        result = adapter.translate("SELECT current_date()")
        assert "CURRENT_DATE" in result

    def test_spark_to_duckdb_split(self) -> None:
        adapter = PortableSQLAdapter(Dialect.DUCKDB)
        result = adapter.translate("SELECT split(x, ',')")
        assert "STRING_SPLIT" in result

    def test_spark_to_duckdb_array_contains(self) -> None:
        adapter = PortableSQLAdapter(Dialect.DUCKDB)
        result = adapter.translate("SELECT array_contains(arr, 'x')")
        assert "LIST_CONTAINS" in result

    def test_spark_to_duckdb_size(self) -> None:
        adapter = PortableSQLAdapter(Dialect.DUCKDB)
        result = adapter.translate("SELECT size(arr)")
        assert "LIST_LEN" in result

    def test_spark_to_duckdb_get_json_object(self) -> None:
        adapter = PortableSQLAdapter(Dialect.DUCKDB)
        result = adapter.translate("SELECT get_json_object(j, '$.x')")
        assert "JSON_EXTRACT" in result


# --- DataSourceCapabilities ---


class TestDataSourceCapabilities:
    def test_defaults(self) -> None:
        caps = DataSourceCapabilities()
        assert caps.supports_batch_read is True
        assert caps.supports_batch_write is False
        assert caps.supports_streaming_read is False
        assert caps.max_parallelism == 1

    def test_custom(self) -> None:
        caps = DataSourceCapabilities(
            supports_batch_write=True, max_parallelism=8,
        )
        assert caps.supports_batch_write is True
        assert caps.max_parallelism == 8


# --- ModelContract ---


class TestModelContract:
    def test_required_fields(self) -> None:
        contract = ModelContract(
            model_name="m", version="1.0", framework="spark_mllib",
        )
        assert contract.model_name == "m"
        assert contract.metrics == {}

    def test_with_metrics(self) -> None:
        contract = ModelContract(
            model_name="m", version="1.0", framework="mlflow",
            metrics={"accuracy": 0.95},
        )
        assert contract.metrics["accuracy"] == 0.95


# --- OpenLineageProjection ---


class TestOpenLineageProjection:
    def test_init(self) -> None:
        proj = OpenLineageProjection(ProjectId("p1"))
        assert proj.project_id == "p1"
        assert proj._client is None

    def test_create_dataset(self) -> None:
        proj = OpenLineageProjection(ProjectId("p1"))
        ds = proj.create_dataset("my_table")
        assert ds["name"] == "my_table"
        assert ds["namespace"] == "dex-p1"

    def test_create_dataset_with_facets(self) -> None:
        proj = OpenLineageProjection(ProjectId("p1"))
        ds = proj.create_dataset("t", facets={"custom": True})
        assert ds["facets"]["custom"] is True

    def test_create_input_dataset(self) -> None:
        proj = OpenLineageProjection(ProjectId("p1"))
        ds = proj.create_input_dataset("input_table", source_type="parquet", path="/data/in")
        assert ds["facets"]["sourceType"] == "parquet"
        assert ds["facets"]["path"] == "/data/in"

    def test_create_input_dataset_no_path(self) -> None:
        proj = OpenLineageProjection(ProjectId("p1"))
        ds = proj.create_input_dataset("input_table")
        assert "path" not in ds["facets"]

    def test_create_output_dataset(self) -> None:
        proj = OpenLineageProjection(ProjectId("p1"))
        ds = proj.create_output_dataset("output_table", sink_type="delta", path="/data/out")
        assert ds["facets"]["sinkType"] == "delta"
        assert ds["facets"]["path"] == "/data/out"

    def test_create_output_dataset_no_path(self) -> None:
        proj = OpenLineageProjection(ProjectId("p1"))
        ds = proj.create_output_dataset("output_table")
        assert "path" not in ds["facets"]

    def test_create_schema_facet(self) -> None:
        proj = OpenLineageProjection(ProjectId("p1"))
        facet = proj.create_schema_facet([
            {"name": "id", "type": "int"},
            {"name": "name", "description": "user name"},
        ])
        assert len(facet["fields"]) == 2
        assert facet["fields"][0]["type"] == "int"
        assert facet["fields"][1]["description"] == "user name"

    def test_create_lineage_facet(self) -> None:
        proj = OpenLineageProjection(ProjectId("p1"))
        facet = proj.create_lineage_facet(["input1"], ["output1"])
        assert facet["inputs"] == ["input1"]
        assert facet["outputs"] == ["output1"]

    def test_create_run_event(self) -> None:
        proj = OpenLineageProjection(ProjectId("p1"))
        event = proj.create_run_event("run_1", "my_job", "START")
        assert event["eventType"] == "START"
        assert event["run"]["runId"] == "run_1"
        assert event["job"]["name"] == "my_job"
        assert event["job"]["namespace"] == "dex-p1"
        assert event["producer"] == "dex-dataenginex"

    def test_create_run_event_with_io(self) -> None:
        proj = OpenLineageProjection(ProjectId("p1"))
        event = proj.create_run_event(
            "r1", "job1", "COMPLETE",
            inputs=[{"name": "in1"}],
            outputs=[{"name": "out1"}],
        )
        assert event["inputs"] == [{"name": "in1"}]
        assert event["outputs"] == [{"name": "out1"}]

    def test_get_lineage_not_connected(self) -> None:
        proj = OpenLineageProjection(ProjectId("p1"))
        result = proj.get_lineage()
        assert "error" in result

    def test_disconnect_when_not_connected(self) -> None:
        proj = OpenLineageProjection(ProjectId("p1"))
        proj.disconnect()  # no-op, no error

    def test_connect_and_disconnect(self) -> None:
        proj = OpenLineageProjection(ProjectId("p1"))
        proj.connect()
        assert proj._client is not None
        proj.disconnect()
        assert proj._client is None
