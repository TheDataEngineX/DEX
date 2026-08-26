"""Tests for Spark catalog adapter and resource accounting (no PySpark)."""

from __future__ import annotations

from pathlib import Path

from dataenginex.foundation.ids import ProjectId
from dataenginex.spark.catalog.adapter import SparkCatalogAdapter
from dataenginex.spark.catalog.identifiers import SparkCatalogIdentifier
from dataenginex.spark.datasource.python_datasource import PythonDataSource
from dataenginex.spark.metrics.resource_accounting import ResourceAccounting

# --- SparkCatalogAdapter ---


class TestSparkCatalogAdapterInit:
    def test_default(self) -> None:
        adapter = SparkCatalogAdapter()
        assert adapter.catalog_type == "hive"
        assert adapter._client is None

    def test_file_type(self) -> None:
        adapter = SparkCatalogAdapter("file", {"lakehouse_path": "/tmp/lh"})
        assert adapter.catalog_type == "file"
        assert str(adapter._lakehouse_root) == "/tmp/lh"


class TestSparkCatalogAdapterConnect:
    def test_unsupported_type(self) -> None:
        adapter = SparkCatalogAdapter("unsupported")
        try:
            adapter.connect()
            raise AssertionError("should have raised")
        except ValueError as e:
            assert "Unsupported" in str(e)

    def test_hive_no_pyhive(self) -> None:
        adapter = SparkCatalogAdapter("hive")
        adapter.connect()
        assert adapter._client is None  # pyhive not installed

    def test_unity(self) -> None:
        adapter = SparkCatalogAdapter("unity")
        adapter.connect()
        assert adapter._client is None

    def test_file(self, tmp_path: Path) -> None:
        adapter = SparkCatalogAdapter("file", {"lakehouse_path": str(tmp_path)})
        adapter.connect()
        assert adapter._client is None

    def test_disconnect(self) -> None:
        adapter = SparkCatalogAdapter("hive")
        adapter.connect()
        adapter.disconnect()
        assert adapter._client is None


class TestSparkCatalogAdapterFileCatalog:
    def test_list_databases_empty(self, tmp_path: Path) -> None:
        adapter = SparkCatalogAdapter("file", {"lakehouse_path": str(tmp_path)})
        assert adapter.list_databases() == []

    def test_list_databases(self, tmp_path: Path) -> None:
        (tmp_path / "bronze").mkdir()
        (tmp_path / "silver").mkdir()
        adapter = SparkCatalogAdapter("file", {"lakehouse_path": str(tmp_path)})
        dbs = adapter.list_databases()
        assert len(dbs) == 2
        assert {d["name"] for d in dbs} == {"bronze", "silver"}

    def test_list_tables_empty(self, tmp_path: Path) -> None:
        adapter = SparkCatalogAdapter("file", {"lakehouse_path": str(tmp_path)})
        assert adapter.list_tables("bronze") == []

    def test_list_tables(self, tmp_path: Path) -> None:
        layer = tmp_path / "bronze"
        (layer / "users").mkdir(parents=True)
        (layer / "users" / "_delta_log").mkdir()
        adapter = SparkCatalogAdapter("file", {"lakehouse_path": str(tmp_path)})
        tables = adapter.list_tables("bronze")
        assert len(tables) == 1
        assert tables[0]["name"] == "users"

    def test_list_tables_non_file(self) -> None:
        adapter = SparkCatalogAdapter("hive")
        assert adapter.list_tables("default") == []

    def test_list_databases_non_file(self) -> None:
        adapter = SparkCatalogAdapter("hive")
        assert adapter.list_databases() == []

    def test_list_databases_hive_no_client(self) -> None:
        adapter = SparkCatalogAdapter("hive")
        assert adapter._list_databases_from_hive(None) == []

    def test_list_tables_hive_no_client(self) -> None:
        adapter = SparkCatalogAdapter("hive")
        assert adapter._list_tables_from_hive("default") == []

    def test_get_table_schema_hive_no_client(self) -> None:
        adapter = SparkCatalogAdapter("hive")
        ident = SparkCatalogIdentifier(table="t")
        result = adapter._get_table_schema_from_hive(ident)
        assert result == {"columns": [], "partition_columns": []}

    def test_get_table_metadata_hive_no_client(self) -> None:
        adapter = SparkCatalogAdapter("hive")
        ident = SparkCatalogIdentifier(table="t")
        assert adapter._get_table_metadata_from_hive(ident) == {}

    def test_get_table_schema_non_file(self) -> None:
        adapter = SparkCatalogAdapter("hive")
        ident = SparkCatalogIdentifier(table="t")
        assert adapter.get_table_schema(ident) == {"columns": [], "partition_columns": []}

    def test_get_table_metadata_non_file(self) -> None:
        adapter = SparkCatalogAdapter("hive")
        ident = SparkCatalogIdentifier(table="t")
        assert adapter._get_table_metadata(ident) == {}

    def test_get_table_schema_no_delta_log(self, tmp_path: Path) -> None:
        table_dir = tmp_path / "default" / "users"
        table_dir.mkdir(parents=True)
        adapter = SparkCatalogAdapter("file", {"lakehouse_path": str(tmp_path)})
        ident = SparkCatalogIdentifier(namespace="default", table="users")
        assert adapter.get_table_schema(ident) == {"columns": [], "partition_columns": []}

    def test_get_table_metadata_no_delta_log(self, tmp_path: Path) -> None:
        table_dir = tmp_path / "default" / "users"
        table_dir.mkdir(parents=True)
        adapter = SparkCatalogAdapter("file", {"lakehouse_path": str(tmp_path)})
        ident = SparkCatalogIdentifier(namespace="default", table="users")
        assert adapter._get_table_metadata(ident) == {}

    def test_spark_table_to_resource(self, tmp_path: Path) -> None:
        adapter = SparkCatalogAdapter("file", {"lakehouse_path": str(tmp_path)})
        ident = SparkCatalogIdentifier(namespace="default", table="users")
        resource = adapter.spark_table_to_resource(ident)
        assert resource["name"] == "users"
        assert resource["resource_type"] == "table"
        assert resource["catalog"] == "spark_catalog"


# --- PythonDataSource ---


class TestPythonDataSource:
    def test_init(self) -> None:
        ds = PythonDataSource("my_source", "my.module")
        assert ds.name == "my_source"
        assert ds.module_path == "my.module"

    def test_read_empty(self) -> None:
        ds = PythonDataSource("s", "m")
        assert ds.read({}) == []

    def test_load(self) -> None:
        ds = PythonDataSource("s", "m")
        ds.load({})  # no-op

    def test_write(self) -> None:
        ds = PythonDataSource("s", "m")
        ds.write([], {})  # no-op


# --- ResourceAccounting ---


class TestResourceAccounting:
    def test_record_and_get(self) -> None:
        ra = ResourceAccounting(ProjectId("p1"))
        ra.record_job_metrics("r1", {
            "job_id": 1, "stage_id": 0, "taskCount": 4,
            "inputBytes": 1024, "outputBytes": 512,
            "executorRunTime": 1000, "peakExecutionMemory": 1024**3,
        })
        metrics = ra.get_job_metrics("r1")
        assert metrics is not None
        assert metrics["job_id"] == 1
        assert metrics["input_bytes"] == 1024

    def test_get_missing(self) -> None:
        ra = ResourceAccounting(ProjectId("p1"))
        assert ra.get_job_metrics("missing") is None

    def test_resource_usage(self) -> None:
        ra = ResourceAccounting(ProjectId("p1"))
        ra.record_job_metrics("r1", {"executorRunTime": 2000, "peakExecutionMemory": 2 * 1024**3})
        usage = ra.get_resource_usage("r1")
        assert usage is not None
        assert usage["cpu_time_ms"] == 2000
        assert usage["memory_bytes"] == 2 * 1024**3
        assert usage["cost_estimate"] > 0

    def test_resource_usage_missing(self) -> None:
        ra = ResourceAccounting(ProjectId("p1"))
        assert ra.get_resource_usage("missing") is None

    def test_total_cost(self) -> None:
        ra = ResourceAccounting(ProjectId("p1"))
        assert ra.get_total_cost() == 0.0
        ra.record_job_metrics("r1", {"executorRunTime": 1000})
        assert ra.get_total_cost() > 0

    def test_usage_summary(self) -> None:
        ra = ResourceAccounting(ProjectId("p1"))
        ra.record_job_metrics("r1", {"executorRunTime": 1000})
        ra.record_job_metrics("r2", {"executorRunTime": 2000})
        summary = ra.get_usage_summary()
        assert summary["total_jobs"] == 2
        assert summary["total_cpu_time_ms"] == 3000

    def test_usage_summary_empty(self) -> None:
        ra = ResourceAccounting(ProjectId("p1"))
        summary = ra.get_usage_summary()
        assert summary["total_jobs"] == 0
        assert summary["avg_cost_per_job"] == 0

    def test_performance_metrics(self) -> None:
        ra = ResourceAccounting(ProjectId("p1"))
        ra.record_job_metrics("r1", {
            "inputBytes": 1024, "outputBytes": 512,
            "executorRunTime": 1000, "taskCount": 4,
            "shuffleRead": 256, "shuffleWrite": 128, "gcTime": 100,
        })
        perf = ra.get_performance_metrics("r1")
        assert "throughput_mbps" in perf
        assert "gc_ratio" in perf
        assert "shuffle_ratio" in perf
        assert "task_efficiency" in perf

    def test_performance_metrics_missing(self) -> None:
        ra = ResourceAccounting(ProjectId("p1"))
        assert ra.get_performance_metrics("missing") == {}

    def test_performance_metrics_zero_runtime(self) -> None:
        ra = ResourceAccounting(ProjectId("p1"))
        ra.record_job_metrics("r1", {"executorRunTime": 0})
        perf = ra.get_performance_metrics("r1")
        assert perf["throughput_mbps"] == 0
        assert perf["gc_ratio"] == 0

    def test_collect_spark_metrics(self) -> None:
        ra = ResourceAccounting(ProjectId("p1"))
        result = ra.collect_spark_metrics()
        # Returns either error dict (no pyspark) or success dict (pyspark available)
        assert "error" in result or "status" in result

    def test_reset(self) -> None:
        ra = ResourceAccounting(ProjectId("p1"))
        ra.record_job_metrics("r1", {"executorRunTime": 1000})
        assert ra.get_total_cost() > 0
        ra.reset()
        assert ra.get_total_cost() == 0.0
        assert ra.get_job_metrics("r1") is None
