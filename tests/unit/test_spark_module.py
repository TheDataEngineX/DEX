"""Tests for v0.7 Spark integration module."""

import pytest

from dataenginex.spark.catalog import SparkCatalogIdentifier
from dataenginex.spark.connect import SparkConnectClient, SparkServerManager, SparkSessionRegistry
from dataenginex.spark.datasource import DataSourceRegistry
from dataenginex.spark.lineage import OpenLineageProjection
from dataenginex.spark.metrics import ResourceAccounting
from dataenginex.spark.ml import MLlibProvider, ModelContract
from dataenginex.spark.sql import PortableSQLAdapter, SparkSQLExecutor
from dataenginex.spark.streaming import (
    CheckpointRegistry,
)


class TestSparkConnectClient:
    def test_create_client(self) -> None:
        client = SparkConnectClient(
            server_url="local[1]",
            project_id="proj:123",
        )
        assert client.server_url == "local[1]"

    def test_execute_sql_requires_connect(self) -> None:
        client = SparkConnectClient(server_url="local[1]", project_id="proj:123")
        with pytest.raises(RuntimeError, match="call connect"):
            client.execute_sql("SELECT 1")


class TestSparkServerManager:
    def test_server_lifecycle(self) -> None:
        mgr = SparkServerManager()
        assert mgr.status() == "stopped"
        mgr.start()
        assert mgr.status() == "running"
        mgr.stop()
        assert mgr.status() == "stopped"


class TestSparkSessionRegistry:
    def test_register_session(self) -> None:
        reg = SparkSessionRegistry()
        key = reg.register("proj:123", "run:456", {"mode": "local"})
        assert "proj:123:run:456" in key

    def test_get_session(self) -> None:
        reg = SparkSessionRegistry()
        reg.register("proj:123", "run:456", {"mode": "local"})
        session = reg.get("proj:123", "run:456")
        assert session is not None


class TestSparkCatalogIdentifier:
    def test_from_spark_sql(self) -> None:
        id = SparkCatalogIdentifier.from_spark_sql("catalog.namespace.table")
        assert id.catalog == "catalog"
        assert id.namespace == "namespace"
        assert id.table == "table"

    def test_to_spark_sql(self) -> None:
        id = SparkCatalogIdentifier(catalog="c", namespace="n", table="t")
        assert id.to_spark_sql() == "c.n.t"


class TestDataSourceRegistry:
    def test_register_connector(self) -> None:
        reg = DataSourceRegistry(project_id="proj:123")
        reg.register(
            name="s3",
            source_class="dataenginex.spark.datasource.python_datasource.PythonDataSource",
            capabilities=["read"],
        )
        connector = reg.get_source("s3")
        assert connector is not None
        assert connector["capabilities"] == ["read"]


class TestCheckpointRegistry:
    def test_checkpoint_path(self) -> None:
        reg = CheckpointRegistry()
        path = reg.get_checkpoint_path("query:123")
        assert "query:123" in path


class TestPortableSQLAdapter:
    def test_translate_portable(self) -> None:
        adapter = PortableSQLAdapter()
        result = adapter.translate("SELECT 1")
        assert result == "SELECT 1"


class TestSparkSQLExecutor:
    def test_execute(self) -> None:
        executor = SparkSQLExecutor(project_id="proj:123")
        result = executor.execute("SELECT 1")
        assert result["status"] == "executed"


class TestMLlibProvider:
    @pytest.mark.skip(reason="Requires mlflow which is not in base deps")
    def test_train_reports_error_for_unloadable_dataset(self) -> None:
        # Real behavior: with a live Spark session and no such dataset/table,
        # train() fails in _load_dataset() before algorithm selection is ever
        # reached, surfacing a Spark/py4j error referencing the dataset_ref.
        provider = MLlibProvider(project_id="proj:123")
        provider.connect()
        result = provider.train(
            algorithm="not_a_real_algorithm",
            dataset_ref="table:123",
        )
        assert result["status"] == "error"
        assert "table:123" in result["error"]


class TestModelContract:
    def test_contract_creation(self) -> None:
        contract = ModelContract(
            model_name="churn-model",
            version="1.0",
            framework="spark_mllib",
        )
        assert contract.model_name == "churn-model"


class TestOpenLineageProjection:
    def test_create_run_event(self) -> None:
        proj = OpenLineageProjection(project_id="proj:123")
        event = proj.create_run_event(run_id="run:123", job_name="job1")
        assert event["run"]["runId"] == "run:123"
        assert event["job"]["name"] == "job1"


class TestResourceAccounting:
    def test_record_job_metrics(self) -> None:
        acct = ResourceAccounting(project_id="proj:123")
        acct.record_job_metrics("run:123", {"executorRunTime": 2500, "peakExecutionMemory": 1024})
        metrics = acct.get_job_metrics("run:123")
        assert metrics is not None
        assert metrics["executor_run_time_ms"] == 2500


