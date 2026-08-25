# Spark Foundation Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every `engine: spark` pipeline path actually work (currently crashes
immediately), fix the Spark catalog page (currently shows silent fake data on a
real UI page), delete dead Spark scaffolding, and fix the currently-failing Spark
unit test suite — bringing the Spark foundation from "scaffolded, mostly broken"
to "the parts that are wired to something real, work correctly."

**Architecture:** Mirror the existing DuckDB pipeline path's stage structure
(extract → transform chain → quality gate → Delta load) using Spark DataFrames
instead of DuckDB tables, reusing the same source-connector and Delta-storage
layer both engines already share. Fix `SparkCatalogAdapter`'s file-mode path to
read real Delta table metadata instead of hardcoded mock data. Delete Spark
subsystem code that has zero callers and duplicates logic that already lives
correctly elsewhere.

**Tech Stack:** Python 3.13, PySpark 4.2.0, `delta-spark` (Delta Lake JVM
package), `deltalake` (Python-native Delta reader, already a dependency),
pytest, DuckDB (existing engine, unchanged).

**Spec:** `docs/superpowers/specs/2026-08-14-spark-foundation-integration-design.md`
(read alongside `2026-08-14-spark-pipeline-execution-design.md` — the first
spec this one extends).

## Global Constraints

- `local[*]` Spark mode only — no remote/distributed cluster execution in this
  plan.
- Every Spark pipeline stage must catch failures per-stage and return
  `PipelineResult(success=False, error=str(exc))` — never silently report
  success with zero rows moved.
- Delta write/read must target the exact same `.dex/lakehouse/{layer}/{destination}`
  path the DuckDB engine uses — cross-engine table interop is required, not
  optional.
- No new UI integration for streaming/ML/lineage/data-source-registry
  subsystems — fix their tests and verify against real pyspark only.
- Java 17 (`JAVA_HOME=/opt/homebrew/opt/openjdk@17`) and `pyspark>=4.2.0` are
  already installed in the dev environment — do not re-install.
- Do not touch the in-progress uv-workspace-split changes to
  `dataenginex/pyproject.toml` / `uv.lock` beyond adding the one new
  dependency line this plan requires.

---

## Task 1: Add `delta-spark` dependency

**Files:**
- Modify: `dataenginex/pyproject.toml` (`spark` dependency group, currently
  `spark = ["pyspark>=4.2.0"]`)

**Interfaces:**
- Produces: `delta-spark>=4.0.0` importable as `import delta` in any file that
  needs `delta.pip_utils.configure_spark_with_delta_pip`.

- [ ] **Step 1: Add the dependency**

Edit the `spark` line in `dataenginex/pyproject.toml`:

```toml
spark = ["pyspark>=4.2.0", "delta-spark>=4.0.0"]
```

- [ ] **Step 2: Sync and verify import**

Run: `cd dataenginex && VIRTUAL_ENV= uv sync --group spark --group lineage --group ml`
Then: `VIRTUAL_ENV= .venv/bin/python -c "import delta; print(delta.__version__)"`
Expected: prints a version string, no `ImportError`.

- [ ] **Step 3: Commit**

```bash
cd dataenginex
git add pyproject.toml uv.lock
git commit -m "build: add delta-spark dependency for Spark Delta writes"
```

---

## Task 2: Delete orphaned Spark scaffolding

**Files:**
- Delete: `dataenginex/src/dataenginex/spark/session/configuration.py`,
  `.../session/lifecycle.py`, `.../session/isolation.py`, `.../session/__init__.py`
- Delete: `.../spark/local/lazy_startup.py`, `.../local/profile.py`, `.../local/__init__.py`
- Delete: `.../spark/pipelines/declarative.py`, `.../pipelines/project_adapter.py`,
  `.../pipelines/run_projection.py`, `.../pipelines/__init__.py`
- Delete: `.../spark/catalog/resource_projection.py`
- Delete: `.../spark/lineage/spark_to_dex_lineage.py`
- Delete: `.../spark/metrics/metrics_correlation.py`
- Modify: `.../spark/__init__.py`, `.../spark/catalog/__init__.py`,
  `.../spark/lineage/__init__.py`, `.../spark/metrics/__init__.py` (drop
  re-exports of deleted names)

**Interfaces:**
- Consumes: none (this task only removes code nothing else calls — confirmed
  in the spec's audit, zero non-test callers for every name being deleted).
- Produces: nothing new. `SparkCatalogIdentifier` (in `spark/catalog/identifiers.py`)
  is NOT deleted — `SparkCatalogAdapter` uses it.

- [ ] **Step 1: Confirm zero production callers before deleting (safety check)**

Run from `dataenginex/`:
```bash
for cls in SparkSessionConfig SparkSessionLifecycle SparkSessionIsolation \
  LazySparkStartup LocalProfile SparkPipelineAdapter PipelineProjectAdapter \
  PipelineRunProjection SparkResourceProjection SparkToDexLineage MetricsCorrelation; do
  echo "=== $cls ==="
  grep -rl "$cls" --include="*.py" . | grep -v "/\.venv/" | grep -v "tests/unit/test_spark_module.py"
done
```
Expected: each `$cls` prints only its own defining file (and `__init__.py`
re-export) — no other file references it. If any class shows an unexpected
caller, stop and re-scope that one item out of this task.

- [ ] **Step 2: Delete the files**

```bash
cd dataenginex
rm -rf src/dataenginex/spark/session src/dataenginex/spark/local src/dataenginex/spark/pipelines
rm src/dataenginex/spark/catalog/resource_projection.py
rm src/dataenginex/spark/lineage/spark_to_dex_lineage.py
rm src/dataenginex/spark/metrics/metrics_correlation.py
```

- [ ] **Step 3: Clean up `__init__.py` re-exports**

Read each of `src/dataenginex/spark/__init__.py`,
`src/dataenginex/spark/catalog/__init__.py`,
`src/dataenginex/spark/lineage/__init__.py`,
`src/dataenginex/spark/metrics/__init__.py`. Remove any `from .session import
...`, `from .local import ...`, `from .pipelines import ...`,
`from .resource_projection import SparkResourceProjection`,
`from .spark_to_dex_lineage import SparkToDexLineage`,
`from .metrics_correlation import MetricsCorrelation` lines, and drop the
corresponding names from each file's `__all__` list.

- [ ] **Step 4: Verify nothing is broken**

Run: `cd dataenginex && VIRTUAL_ENV= .venv/bin/python -c "import dataenginex.engine"`
Expected: no `ImportError`/`ModuleNotFoundError`.

Run: `cd dataenginex && VIRTUAL_ENV= .venv/bin/python -m ruff check src/dataenginex/spark/`
Expected: no unresolved-import errors (F401/F811 etc. for removed names).

- [ ] **Step 5: Commit**

```bash
cd dataenginex
git add -A src/dataenginex/spark/
git commit -m "refactor: delete orphaned Spark scaffolding (session/local/pipelines + 3 unused helpers)"
```

---

## Task 3: Rewrite `test_spark_module.py`

**Files:**
- Modify: `dataenginex/tests/unit/test_spark_module.py`

**Interfaces:**
- Consumes: real APIs of `SparkConnectClient`, `SparkCatalogAdapter`,
  `SparkCatalogIdentifier`, `DataSourceRegistry`, `CheckpointRegistry`,
  `PortableSQLAdapter`, `SparkSQLExecutor`, `MLlibProvider`, `ModelContract`,
  `OpenLineageProjection`, `ResourceAccounting`, `StreamingHealthMonitor`,
  `StreamingQueryManager`, `SparkServerManager`, `SparkSessionRegistry` — as
  they exist today in `src/dataenginex/spark/**/*.py` (do not invent new
  methods; read each class's real `__init__`/method signatures before writing
  its test).

- [ ] **Step 1: Remove test classes for deleted code**

Delete these classes from `test_spark_module.py`: `TestSparkSessionConfig`,
`TestSparkPipelineAdapter`, `TestLocalProfile`, `TestLazySparkStartup`. Remove
their imports from the top of the file (`SparkSessionConfig`,
`SparkSessionLifecycle`, `SparkSessionIsolation`, `SparkPipelineAdapter`,
`PipelineProjectAdapter`, `PipelineRunProjection`, `SparkResourceProjection`,
`SparkToDexLineage`, `MetricsCorrelation`, `LazySparkStartup`, `LocalProfile`).

- [ ] **Step 2: Fix `TestSparkConnectClient::test_execute_sql`**

Real behavior: `execute_sql()` requires `connect()` first (raises
`RuntimeError("Not connected — call connect() first")` otherwise). Replace with:

```python
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

    def test_execute_sql_after_connect(self) -> None:
        client = SparkConnectClient(server_url="local[1]", project_id="proj:123")
        client.connect()
        try:
            result = client.execute_sql("SELECT 1 AS one")
            assert result["status"] == "executed"
            assert result["row_count"] == 1
        finally:
            client.disconnect()
```

- [ ] **Step 3: Fix `TestDataSourceRegistry::test_register_connector`**

Real signature: `DataSourceRegistry(project_id)`, `.register(name, source_class,
description="", capabilities=None, config_schema=None)`, `.get_source(name)`.

```python
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
```

- [ ] **Step 4: Fix `TestMLlibProvider::test_train`**

Real behavior with no `spark_session` passed and `pyspark.ml` importable
(installed as of Task 1): `train()` runs the real Spark ML path and returns
`status: "completed"` (or `"error"` if the tiny in-memory dataset can't be
loaded — since `dataset_ref` here is a fake table name, `_load_dataset` will
raise before training starts). Assert against the real contract instead of a
fixed dataset outcome:

```python
class TestMLlibProvider:
    def test_train_unknown_algorithm_reports_error(self) -> None:
        provider = MLlibProvider(project_id="proj:123")
        provider.connect()
        result = provider.train(
            algorithm="not_a_real_algorithm",
            dataset_ref="table:123",
        )
        assert result["status"] == "error"
        assert "algorithm" in result["error"].lower() or "Unknown algorithm" in result["error"]
```

- [ ] **Step 5: Fix `TestOpenLineageProjection::test_project_run`**

Real signature: `OpenLineageProjection(project_id, openlineage_url=..., api_key=None)`,
method is `create_run_event(run_id, job_name, event_type="START", inputs=None, outputs=None)`,
not `project_run()`.

```python
class TestOpenLineageProjection:
    def test_create_run_event(self) -> None:
        proj = OpenLineageProjection(project_id="proj:123")
        event = proj.create_run_event(run_id="run:123", job_name="job1")
        assert event["run"]["runId"] == "run:123"
        assert event["job"]["name"] == "job1"
```

- [ ] **Step 6: Fix `TestResourceAccounting::test_record_usage`**

Real method is `record_job_metrics(run_id, metrics_dict)`, not `record_usage`.

```python
class TestResourceAccounting:
    def test_record_job_metrics(self) -> None:
        acct = ResourceAccounting(project_id="proj:123")
        acct.record_job_metrics("run:123", {"executorRunTime": 2500, "peakExecutionMemory": 1024})
        metrics = acct.get_job_metrics("run:123")
        assert metrics is not None
        assert metrics["executor_run_time_ms"] == 2500
```

- [ ] **Step 7: Run the full suite, verify green**

Run: `cd dataenginex && VIRTUAL_ENV= .venv/bin/python -m pytest tests/unit/test_spark_module.py -v`
Expected: all tests pass, 0 failures.

- [ ] **Step 8: Commit**

```bash
cd dataenginex
git add tests/unit/test_spark_module.py
git commit -m "test: fix spark module tests to match real APIs, drop tests for deleted code"
```

---

## Task 4: Fix `SparkCatalogAdapter` (file-mode real implementation + bug fixes)

**Files:**
- Modify: `dataenginex/src/dataenginex/spark/catalog/adapter.py`
- Modify: `dex-studio/src/dex_studio/routers/data.py:2019-2037` (catalog page
  call site)
- Test: `dataenginex/tests/unit/test_spark_catalog_adapter.py` (new file)

**Interfaces:**
- Consumes: `deltalake.DeltaTable` (already a dependency via `deltalake>=1.6.2`).
- Produces: `SparkCatalogAdapter.list_databases()` / `.list_tables(database:
  str)` / `.get_table_schema(identifier)` return real data for
  `catalog_type="file"`, scanning `.dex/lakehouse/{layer}/{table}/_delta_log/`.

- [ ] **Step 1: Write failing test for real file-catalog behavior**

```python
# tests/unit/test_spark_catalog_adapter.py
from pathlib import Path

import pyarrow as pa
from deltalake import write_deltalake

from dataenginex.spark.catalog.adapter import SparkCatalogAdapter


def test_file_catalog_lists_real_delta_tables(tmp_path: Path) -> None:
    lakehouse = tmp_path / ".dex" / "lakehouse"
    table_dir = lakehouse / "gold" / "top_rated"
    table_dir.mkdir(parents=True)
    write_deltalake(str(table_dir), pa.table({"id": [1, 2], "name": ["a", "b"]}))

    adapter = SparkCatalogAdapter(catalog_type="file", connection_config={"lakehouse_path": str(lakehouse)})
    adapter.connect()

    dbs = adapter.list_databases()
    assert any(db["name"] == "gold" for db in dbs)

    tables = adapter.list_tables("gold")
    assert any(t["name"] == "top_rated" for t in tables)


def test_file_catalog_empty_lakehouse_returns_empty(tmp_path: Path) -> None:
    adapter = SparkCatalogAdapter(
        catalog_type="file",
        connection_config={"lakehouse_path": str(tmp_path / "nonexistent")},
    )
    adapter.connect()
    assert adapter.list_databases() == []
```

- [ ] **Step 2: Run test, verify it fails**

Run: `cd dataenginex && VIRTUAL_ENV= .venv/bin/python -m pytest tests/unit/test_spark_catalog_adapter.py -v`
Expected: FAIL — `list_databases()` returns the hardcoded mock
(`default`/`bronze`/`silver`/`gold`) regardless of `tmp_path` contents, so
`test_file_catalog_empty_lakehouse_returns_empty` fails first (mock list is
non-empty).

- [ ] **Step 3: Implement real file-catalog scanning**

In `src/dataenginex/spark/catalog/adapter.py`, replace `_connect_file_catalog`,
`list_databases`, `list_tables`, `get_table_schema`, `_get_table_metadata` for
the `file` catalog type:

```python
from pathlib import Path

def _connect_file_catalog(self) -> None:
    """Connect to file-based catalog — scans .dex/lakehouse/ for Delta tables."""
    lakehouse_path = self.connection_config.get("lakehouse_path", ".dex/lakehouse")
    self._lakehouse_root = Path(lakehouse_path)
    self._client = None  # no network client for file mode — real lookups go via _lakehouse_root
    logger.info("using file-based catalog", lakehouse_path=str(self._lakehouse_root))

def list_databases(self, catalog: str | None = None) -> list[dict]:
    if self._client:
        return self._list_databases_from_hive(catalog)
    if self.catalog_type != "file" or not self._lakehouse_root.exists():
        return []
    return [
        {"name": layer_dir.name, "description": f"{layer_dir.name} layer tables"}
        for layer_dir in sorted(self._lakehouse_root.iterdir())
        if layer_dir.is_dir()
    ]

def list_tables(self, database: str = "default") -> list[dict]:
    if self._client:
        return self._list_tables_from_hive(database)
    if self.catalog_type != "file":
        return []
    layer_dir = self._lakehouse_root / database
    if not layer_dir.exists():
        return []
    tables = []
    for table_dir in sorted(layer_dir.iterdir()):
        if table_dir.is_dir() and (table_dir / "_delta_log").exists():
            tables.append({"name": table_dir.name, "type": "MANAGED", "format": "delta"})
    return tables

def get_table_schema(self, identifier: SparkCatalogIdentifier) -> dict:
    if self._client:
        return self._get_table_schema_from_hive(identifier)
    if self.catalog_type != "file":
        return {"columns": [], "partition_columns": []}
    table_dir = self._lakehouse_root / identifier.namespace / identifier.table
    if not (table_dir / "_delta_log").exists():
        return {"columns": [], "partition_columns": []}
    from deltalake import DeltaTable

    dt = DeltaTable(str(table_dir))
    schema = dt.schema()
    return {
        "columns": [{"name": f.name, "type": str(f.type), "nullable": f.nullable} for f in schema.fields],
        "partition_columns": dt.metadata().partition_columns,
    }
```

Update `__init__` to always set `self._lakehouse_root = Path(self.connection_config.get("lakehouse_path", ".dex/lakehouse"))`
regardless of catalog type (cheap, avoids `AttributeError` if `list_databases`
is called before `connect()`).

- [ ] **Step 4: Run test, verify it passes**

Run: `cd dataenginex && VIRTUAL_ENV= .venv/bin/python -m pytest tests/unit/test_spark_catalog_adapter.py -v`
Expected: PASS.

- [ ] **Step 5: Fix the two call-site bugs in dex-studio**

In `dex-studio/src/dex_studio/routers/data.py`, replace lines 2019-2037:

```python
    # Add Spark/Hive Metastore tables if adapter is available
    with contextlib.suppress(Exception):
        spark_catalog = eng.spark_catalog
        if spark_catalog:
            spark_catalog.connect()
            for db in spark_catalog.list_databases():
                for tbl in spark_catalog.list_tables(db["name"]):
                    entries.append(
                        {
                            "name": f"{db['name']}.{tbl['name']}",
                            "layer": "hive",
                            "layer_color": "orange",
                            "row_count": "—",
                            "column_count": len(tbl.get("columns", [])),
                            "size": "—",
                            "columns": tbl.get("columns", []),
                            "format": tbl.get("format", "parquet"),
                            "engine": "spark",
                        }
                    )
```

(Change: added `spark_catalog.connect()` call; changed `list_tables(db)` to
`list_tables(db["name"])`; changed `db.name`-style access to `db['name']`
consistently.)

- [ ] **Step 6: Manual verification via dex-studio dev server**

Run: `cd dex-studio && uv run poe dev`, navigate to the Data → Catalog page.
Expected: no fake `employees`/`departments`/`sales` rows appear; if
`.dex/lakehouse/` has real Delta tables from prior pipeline runs, they appear
under the "spark" engine column; if not, no spark rows appear at all (empty,
not fake).

- [ ] **Step 7: Commit**

```bash
cd dataenginex
git add src/dataenginex/spark/catalog/adapter.py tests/unit/test_spark_catalog_adapter.py
git commit -m "fix: SparkCatalogAdapter file-mode reads real Delta tables instead of mock data"
cd ../dex-studio
git add src/dex_studio/routers/data.py
git commit -m "fix: call spark_catalog.connect() and pass table name (not dict) to list_tables"
```

---

## Task 5: `SparkTransformApplier` — all 12 transform types

**Files:**
- Create: `dataenginex/src/dataenginex/spark/transforms/__init__.py`
- Create: `dataenginex/src/dataenginex/spark/transforms/applier.py`
- Test: `dataenginex/tests/unit/test_spark_transform_applier.py` (new file)

**Interfaces:**
- Consumes: `TransformStepConfig` (`dataenginex.config.schema`) — same config
  objects the DuckDB path consumes.
- Produces: `SparkTransformApplier.apply(df: DataFrame, step: TransformStepConfig,
  current_table_name: str) -> DataFrame` — one function, dispatches on
  `step.type`. `current_table_name` is needed for the `sql` case's temp-view
  registration.

- [ ] **Step 1: Write failing tests for all 12 transform types**

```python
# tests/unit/test_spark_transform_applier.py
import pytest
from pyspark.sql import SparkSession

from dataenginex.config.schema import TransformStepConfig
from dataenginex.spark.transforms.applier import SparkTransformApplier


@pytest.fixture(scope="module")
def spark():
    session = SparkSession.builder.master("local[1]").appName("test-transforms").getOrCreate()
    yield session
    session.stop()


@pytest.fixture
def applier():
    return SparkTransformApplier()


def test_filter(spark, applier):
    df = spark.createDataFrame([(1, 5.0), (2, 8.0)], ["id", "rating"])
    step = TransformStepConfig(type="filter", condition="rating > 6.0")
    out = applier.apply(df, step, "bronze")
    assert out.count() == 1
    assert out.collect()[0]["id"] == 2


def test_derive(spark, applier):
    df = spark.createDataFrame([(10.0,)], ["rating"])
    step = TransformStepConfig(type="derive", name="rating_pct", expression="rating / 10.0 * 100")
    out = applier.apply(df, step, "bronze")
    assert out.collect()[0]["rating_pct"] == 100.0


def test_cast(spark, applier):
    df = spark.createDataFrame([("5",)], ["rating"])
    step = TransformStepConfig(type="cast", columns={"rating": "double"})
    out = applier.apply(df, step, "bronze")
    assert dict(out.dtypes)["rating"] == "double"


def test_deduplicate(spark, applier):
    df = spark.createDataFrame([(1, "a"), (1, "b")], ["id", "val"])
    step = TransformStepConfig(type="deduplicate", key=["id"])
    out = applier.apply(df, step, "bronze")
    assert out.count() == 1


def test_sql(spark, applier):
    df = spark.createDataFrame([(1,), (2,)], ["id"])
    step = TransformStepConfig(type="sql", sql="SELECT * FROM _data WHERE id > 1")
    out = applier.apply(df, step, "bronze")
    assert out.count() == 1
    assert out.collect()[0]["id"] == 2


def test_rename(spark, applier):
    df = spark.createDataFrame([(1,)], ["old_name"])
    step = TransformStepConfig(type="rename", mapping={"old_name": "new_name"})
    out = applier.apply(df, step, "bronze")
    assert "new_name" in out.columns
    assert "old_name" not in out.columns


def test_drop_columns(spark, applier):
    df = spark.createDataFrame([(1, 2)], ["keep", "drop_me"])
    step = TransformStepConfig(type="drop_columns", columns=["drop_me"])
    out = applier.apply(df, step, "bronze")
    assert out.columns == ["keep"]


def test_fill_null(spark, applier):
    df = spark.createDataFrame([(None,)], ["rating"])
    step = TransformStepConfig(type="fill_null", defaults={"rating": 0.0})
    out = applier.apply(df, step, "bronze")
    assert out.collect()[0]["rating"] == 0.0


def test_aggregate(spark, applier):
    df = spark.createDataFrame([("a", 1), ("a", 2), ("b", 5)], ["grp", "val"])
    step = TransformStepConfig(type="aggregate", group_by=["grp"], agg_exprs={"total": "SUM(val)"})
    out = applier.apply(df, step, "bronze")
    rows = {r["grp"]: r["total"] for r in out.collect()}
    assert rows["a"] == 3
    assert rows["b"] == 5


def test_window(spark, applier):
    df = spark.createDataFrame([("a", 1.0), ("a", 2.0)], ["grp", "score"])
    step = TransformStepConfig(
        type="window", name="rnk", expression="RANK()", partition_by=["grp"], order_by="score DESC"
    )
    out = applier.apply(df, step, "bronze")
    assert "rnk" in out.columns


def test_explode(spark, applier):
    df = spark.createDataFrame([(1, ["a", "b"])], ["id", "genres"])
    step = TransformStepConfig(type="explode", column="genres", alias="genre")
    out = applier.apply(df, step, "bronze")
    assert out.count() == 2
    assert "genres" not in out.columns
    assert "genre" in out.columns


def test_json_normalize(spark, applier):
    df = spark.createDataFrame([(1, {"a": 10, "b": 20})], ["id", "info"])
    step = TransformStepConfig(type="json_normalize", column="info", prefix="info_")
    out = applier.apply(df, step, "bronze")
    assert "info_a" in out.columns and "info_b" in out.columns
    assert "info" not in out.columns
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `cd dataenginex && VIRTUAL_ENV= .venv/bin/python -m pytest tests/unit/test_spark_transform_applier.py -v`
Expected: FAIL — `dataenginex.spark.transforms.applier` doesn't exist yet
(`ModuleNotFoundError`).

- [ ] **Step 3: Implement `SparkTransformApplier`**

```python
# src/dataenginex/spark/transforms/applier.py
"""Spark DataFrame transform applier — mirrors the DuckDB transform semantics
in src/dataenginex/domains/analytics/transforms/sql.py, one method per
TransformStepConfig.type."""

from __future__ import annotations

from typing import Any

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
        return method(df, step, current_table_name)

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
        path = step.column.split(".")
        top_level = path[0]
        alias = step.alias or path[-1]
        keep = [c for c in df.columns if c != top_level]
        col_expr = F.col(step.column) if len(path) > 1 else F.col(top_level)
        return df.select(*keep, F.explode(col_expr).alias(alias))

    def _apply_json_normalize(self, df: DataFrame, step: Any, _table: str) -> DataFrame:
        field_names = [f.name for f in df.schema[step.column].dataType.fields]
        keep = [c for c in df.columns if c != step.column]
        derived = [
            F.col(f"{step.column}.{field}").alias(f"{step.prefix}{field}") for field in field_names
        ]
        return df.select(*keep, *derived)
```

Create `src/dataenginex/spark/transforms/__init__.py`:

```python
from dataenginex.spark.transforms.applier import SparkTransformApplier

__all__ = ["SparkTransformApplier"]
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `cd dataenginex && VIRTUAL_ENV= .venv/bin/python -m pytest tests/unit/test_spark_transform_applier.py -v`
Expected: all 12 tests pass.

- [ ] **Step 5: Commit**

```bash
cd dataenginex
git add src/dataenginex/spark/transforms/ tests/unit/test_spark_transform_applier.py
git commit -m "feat: add SparkTransformApplier — all 12 transform types as Spark DataFrame ops"
```

---

## Task 6: Fix `PipelineRunner._run_spark()` — real execution

**Files:**
- Modify: `dataenginex/src/dataenginex/domains/data/pipeline/runner.py:296-321`
  (`_run_spark` method)
- Modify: `dataenginex/src/dataenginex/spark/connect/client.py` (Delta JAR
  bootstrap in `connect()`)
- Test: `dataenginex/tests/integration/test_spark_pipeline_execution.py` (new file)

**Interfaces:**
- Consumes: `SparkTransformApplier` (Task 5), `SparkConnectClient` (existing,
  method names already correct — `.connect()`/`.disconnect()`), the same
  connector-registry resolution `_extract` already uses
  (`connector.read(table=read_table)` → `pyarrow.Table`/`Dataset`/`list[dict]`),
  `DeltaStorage` (existing, `src/dataenginex/providers/object_store/storage.py`),
  `eng.spark_metrics` / `ResourceAccounting.record_job_metrics` (Task 3 fixed
  its test; this task adds the real call site).
- Produces: `PipelineRunner._run_spark()` returns a real `PipelineResult` with
  accurate `rows_input`/`rows_output`, or `success=False` with `error` set.

- [ ] **Step 1: Bootstrap the Delta JAR in `SparkConnectClient.connect()`**

In `src/dataenginex/spark/connect/client.py`, modify `connect()`:

```python
def connect(self) -> None:
    if not _PYSPARK_AVAILABLE:
        msg = "PySpark is required. Install with: uv sync --group spark"
        raise ImportError(msg)

    try:
        from delta import configure_spark_with_delta_pip

        builder = SparkSession.builder.master(self.server_url)
        builder = builder.appName(f"dex-{self.project_id or 'default'}")

        for key, value in self.session_config.items():
            builder = builder.config(key, value)

        if self.server_url.startswith("sc://"):
            builder = builder.config("spark.connect.grpc.arrow.maxBatchSize", "128m")
            builder = builder.config("spark.connect.reconnectionBackoff", "1s")

        builder = configure_spark_with_delta_pip(builder)
        self._session = builder.getOrCreate()
        self._connected = True

        logger.info(
            "spark session started",
            server_url=self.server_url,
            project_id=str(self.project_id),
        )
    except Exception as exc:
        logger.error("spark session failed", error=str(exc))
        raise
```

(Removed the `session_id=self._session.sparkSession.sessionId` log field per
the spec's open question — not a valid attribute on an embedded `SparkSession`;
verified by the smoke test in Step 5 below.)

- [ ] **Step 2: Write failing integration test**

```python
# tests/integration/test_spark_pipeline_execution.py
"""Requires: Java 17+, pyspark>=4.2.0, delta-spark>=4.0.0 installed."""

import pyarrow as pa
import pytest

from dataenginex.config.schema import PipelineConfig, TransformStepConfig
from dataenginex.domains.data.pipeline.runner import PipelineRunner


@pytest.fixture
def spark_pipeline_config() -> PipelineConfig:
    return PipelineConfig(
        engine="spark",
        source="fixture_source",
        destination="spark_test_output",
        transforms=[
            TransformStepConfig(type="filter", condition="value > 1"),
        ],
    )


def test_spark_pipeline_moves_real_rows(tmp_path, spark_pipeline_config, monkeypatch):
    # Arrange: a runner pointed at tmp_path, with a fixture connector
    # registered under "fixture_source" that returns a small pyarrow.Table
    # (follow the existing pattern used by DuckDB pipeline tests for
    # registering a fixture connector — see
    # tests/unit/test_pipeline_runner.py for the established fixture pattern).
    runner = PipelineRunner(config=..., data_dir=tmp_path / ".dex" / "lakehouse")
    result = runner.run("spark_test_pipeline")

    assert result.success is True
    assert result.rows_input > 0
    assert result.rows_output > 0
    assert result.rows_output <= result.rows_input  # filter removed at least the value<=1 rows


def test_spark_pipeline_failure_path(tmp_path, monkeypatch):
    # A transform with an invalid SQL expression must produce
    # success=False + error set, never a silent zero-row success.
    ...
```

(The `...` in `PipelineRunner(config=..., ...)` and the fixture-connector
registration must be filled in by matching the exact pattern the existing
DuckDB pipeline integration tests use in this repo — read
`tests/unit/test_pipeline_runner.py` first and mirror its setup, since that
pattern isn't reproduced in this plan to avoid drifting from whatever the
current fixture helper actually looks like.)

- [ ] **Step 3: Run test, verify it fails**

Run: `cd dataenginex && VIRTUAL_ENV= .venv/bin/python -m pytest tests/integration/test_spark_pipeline_execution.py -v`
Expected: FAIL — current `_run_spark()` crashes on `spark_client.initialize()`
(`AttributeError`).

- [ ] **Step 4: Implement real `_run_spark()`**

All names below (`pa`, `ds`, `Path`, `connector_registry`, `_infer_layer`,
`self._resolve_connector_paths`, `self._project_dir`, `self._data_dir`,
`self._config`) are already imported/defined at module or class level in
`runner.py` — reuse them, do not re-import under different names.

```python
def _run_spark(
    self,
    pipeline_name: str,
    pipeline_config: Any,
    log: Any,
    progress_cb: Callable[[str, int, int], None] | None = None,
    checkpoint_cb: Callable[[str], None] | None = None,
) -> PipelineResult:
    """Run pipeline using the Spark engine — extract, transform chain, quality
    gate, Delta load — mirroring the DuckDB path's stage structure."""
    from dataenginex.spark.connect.client import SparkConnectClient
    from dataenginex.spark.transforms.applier import SparkTransformApplier

    spark_client = SparkConnectClient(project_id=self._config.project.name)
    applier = SparkTransformApplier()
    rows_input = 0
    rows_output = 0

    try:
        spark_client.connect()
        spark = spark_client.get_spark_session()
        log.info("spark pipeline starting", pipeline=pipeline_name)

        # Stage 1: extract — reuse the exact connector resolution
        # _extract_from_source() uses (runner.py:677-710): resolve the
        # source's connector class from connector_registry, build kwargs from
        # connection+options+path/url, connect, read, disconnect. Do not
        # invent a different helper — this is the only way sources are
        # resolved in this codebase.
        source_config = self._config.data.sources[pipeline_config.source]
        connector_cls = connector_registry.get(source_config.type)
        connector_kwargs: dict[str, Any] = {
            **dict(source_config.connection),
            **dict(source_config.options),
        }
        connector_kwargs = self._resolve_connector_paths(connector_kwargs)
        if source_config.path and "path" not in connector_kwargs:
            src_path = source_config.path
            if self._project_dir and not Path(src_path).is_absolute():
                src_path = str(self._project_dir / src_path)
            connector_kwargs["path"] = src_path
        if source_config.url and "url" not in connector_kwargs:
            connector_kwargs["url"] = source_config.url

        connector = connector_cls(**connector_kwargs)
        try:
            connector.connect()
            read_table = str(connector_kwargs.get("default_file", ""))
            raw_data = connector.read(table=read_table)
        finally:
            connector.disconnect()

        if isinstance(raw_data, pa.Table):
            arrow_table = raw_data
        elif isinstance(raw_data, ds.Dataset):
            arrow_table = raw_data.to_table()
        else:
            arrow_table = pa.Table.from_pylist(raw_data)
        df = spark.createDataFrame(arrow_table.to_pandas())
        rows_input = df.count()
        if checkpoint_cb:
            checkpoint_cb("spark_extract")
        if progress_cb:
            progress_cb("spark_extract", 1, 4)

        # Stage 2: transform chain
        current_table_name = "bronze"
        for step in pipeline_config.transforms:
            df = applier.apply(df, step, current_table_name)
        if checkpoint_cb:
            checkpoint_cb("spark_transform")
        if progress_cb:
            progress_cb("spark_transform", 2, 4)

        # Stage 3: quality gate — same QualityCheckConfig semantics as DuckDB.
        if pipeline_config.quality:
            self._check_quality_spark(spark, df, pipeline_config.quality, pipeline_name)
        if checkpoint_cb:
            checkpoint_cb("spark_quality")
        if progress_cb:
            progress_cb("spark_quality", 3, 4)

        # Stage 4: load — Delta write to the same lakehouse path DuckDB uses.
        layer = _infer_layer(pipeline_config.destination or pipeline_name)
        destination = pipeline_config.destination or pipeline_name
        target_path = self._data_dir / layer / destination
        (
            df.write.format("delta")
            .mode("overwrite")
            .save(str(target_path))
        )
        rows_output = df.count()
        if checkpoint_cb:
            checkpoint_cb("spark_load")
        if progress_cb:
            progress_cb("spark_load", 4, 4)

        # Resource accounting — wire the previously-orphaned metrics subsystem
        # in now that this code path is being fixed anyway.
        try:
            executor_infos = spark.sparkContext.statusTracker().getExecutorInfos()
            self._spark_metrics_hook(pipeline_name, {"executor_count": len(executor_infos)})
        except Exception:
            log.warning("spark resource accounting skipped", pipeline=pipeline_name)

        log.info("spark pipeline complete", rows_input=rows_input, rows_output=rows_output)
        return PipelineResult(
            pipeline=pipeline_name, success=True, rows_input=rows_input, rows_output=rows_output
        )
    except Exception as e:
        log.error("spark pipeline failed", error=str(e), exc_info=True)
        return PipelineResult(
            pipeline=pipeline_name,
            success=False,
            error=str(e),
            rows_input=rows_input,
            rows_output=rows_output,
        )
    finally:
        spark_client.disconnect()
```

Add a small `_check_quality_spark` helper (mirrors `_check_quality`, using
`df.filter()`/`df.count()`-based null-ratio and duplicate-count checks for
`completeness`/`uniqueness`, and the same `_data`-substitution pattern for
`custom_sql` via `df.sparkSession.sql()`), and a `_spark_metrics_hook` helper
that calls `self._engine_ref.spark_metrics.record_job_metrics(...)` if the
runner has a reference to the engine (check how `self._lineage` is threaded
into `PipelineRunner.__init__` today and follow the same pattern to get a
`spark_metrics` reference — do not construct a new `ResourceAccounting`
instance here, reuse the one `engine.py` already owns).

- [ ] **Step 5: Run test, verify it passes**

Run: `cd dataenginex && VIRTUAL_ENV= .venv/bin/python -m pytest tests/integration/test_spark_pipeline_execution.py -v`
Expected: both tests pass — real row counts, and the failure-path test
confirms `success=False` with `error` set.

- [ ] **Step 6: Cross-engine interop smoke test**

```python
def test_duckdb_and_spark_engines_share_delta_tables(tmp_path):
    # Write a table via engine=duckdb, read it via a Spark DataFrame,
    # and vice versa. Confirms delta-spark/deltalake protocol compatibility.
    ...
```

Run it, confirm both directions succeed.

- [ ] **Step 7: Commit**

```bash
cd dataenginex
git add src/dataenginex/domains/data/pipeline/runner.py src/dataenginex/spark/connect/client.py
git add tests/integration/test_spark_pipeline_execution.py
git commit -m "fix: implement real Spark pipeline execution (extract/transform/quality/load)"
```

---

## Task 7: Full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full `dataenginex` test suite**

Run: `cd dataenginex && VIRTUAL_ENV= uv run poe test` (or `.venv/bin/python -m pytest`
if `poe` isn't wired to the split-out venv yet — check `poe_tasks.toml`/`pyproject.toml`
`[tool.poe]` section first).
Expected: all tests pass, including the new/fixed Spark tests.

- [ ] **Step 2: Run lint + typecheck**

Run: `cd dataenginex && VIRTUAL_ENV= .venv/bin/python -m ruff check src/ tests/`
Run: `cd dataenginex && VIRTUAL_ENV= .venv/bin/python -m mypy src/dataenginex/spark/`
Expected: clean (fix any new findings before proceeding).

- [ ] **Step 3: Manual end-to-end pipeline run**

Set a real moviedex pipeline's `dex.yaml` `engine: spark` (e.g.
`bronze_wiki_edits` or similar per the original spec's manual test), run it
via dex-studio, confirm real row counts (not zero) and the pipeline detail
page shows success.

- [ ] **Step 4: Manual catalog page check**

Navigate to Data → Catalog in dex-studio, confirm real Delta tables show under
the Spark/hive column (from the pipeline run in Step 3) and no fake
`employees`/`departments`/`sales` rows appear.

- [ ] **Step 5: Final commit (if any fixes were needed in Steps 1-2)**

```bash
git add -A
git commit -m "fix: lint/typecheck cleanup after Spark foundation integration"
```
