"""
Test configuration and fixtures for DataEngineX test suite.

Shared fixtures live here; test files create their own local fixtures
and ``TestClient`` instances as needed.

PySpark fixtures require the ``data`` dependency group
(``uv sync --group data``).  Tests that depend on Spark are
automatically skipped when PySpark is not installed.
"""

from __future__ import annotations

import os
import shutil
from typing import Any

import pytest

# Skip test files that import modules removed in v0.7.0 restructure.
# These tests need rewriting against the new domain/store APIs.
collect_ignore: list[str] = [
    "unit/test_data.py",
    "unit/test_scheduler.py",
    "unit/test_run_history.py",
    "unit/test_drift_scheduler.py",
    "unit/test_lineage_advanced.py",
    "unit/test_lineage_persistence.py",
    "unit/test_spark_catalog_adapter.py",
    "unit/test_spark_transform_applier.py",
    "integration/test_lineage_integration.py",
    "integration/test_cli_run.py",
    "integration/test_config_cli.py",
    "integration/test_spark_pipeline_execution.py",
    "unit/test_secops_engine_and_cli.py",
    "engines/test_iceberg.py",
]

# ---------------------------------------------------------------------------
# PySpark availability guard
# ---------------------------------------------------------------------------

try:
    from pyspark.sql import SparkSession  # type: ignore[import-untyped]

    _HAS_PYSPARK = True
except ImportError:
    _HAS_PYSPARK = False


def _has_java_runtime() -> bool:
    """Return True when a functional Java runtime exists for PySpark.

    macOS ships a /usr/bin/java stub that just prints an install prompt and
    exits non-zero, so we probe with -version rather than just checking PATH.
    """
    import subprocess  # noqa: PLC0415

    java_home = os.environ.get("JAVA_HOME")
    candidates: list[str] = []
    if java_home:
        candidates.append(os.path.join(java_home, "bin", "java"))
    java_on_path = shutil.which("java")
    if java_on_path:
        candidates.append(java_on_path)
    for java_bin in candidates:
        if not (os.path.isfile(java_bin) and os.access(java_bin, os.X_OK)):
            continue
        try:
            result = subprocess.run(
                [java_bin, "-version"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                return True
        except (OSError, subprocess.TimeoutExpired):
            pass
    return False


_HAS_JAVA_RUNTIME = _has_java_runtime()
_HAS_SPARK_TEST_RUNTIME = _HAS_PYSPARK and _HAS_JAVA_RUNTIME

requires_pyspark = pytest.mark.skipif(
    not _HAS_SPARK_TEST_RUNTIME,
    reason=(
        "Spark test runtime unavailable: requires PySpark and Java "
        "(install via `uv sync --group data` and set JAVA_HOME or add java to PATH)"
    ),
)


# ---------------------------------------------------------------------------
# PySpark session fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def spark() -> Any:
    """Session-scoped local-mode SparkSession.

    Reused across all tests in a single pytest run to avoid the overhead
    of starting / stopping the JVM repeatedly.

    Automatically skipped when PySpark is not available.
    """
    if not _HAS_PYSPARK:
        pytest.skip("PySpark not installed")
    if not _HAS_JAVA_RUNTIME:
        pytest.skip("Java runtime unavailable for PySpark (set JAVA_HOME or add java to PATH)")

    session: Any = (
        SparkSession.builder.master("local[1]")
        .appName("DataEngineX-test")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.default.parallelism", "1")
        .config("spark.sql.warehouse.dir", "/tmp/dex-test-warehouse")
        .config("spark.driver.extraJavaOptions", "-Dderby.system.home=/tmp/dex-derby")
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture()
def spark_df_jobs(spark: Any) -> Any:
    """Small Spark DataFrame with sample job-posting rows.

    Columns: job_id, source, company_name, job_title, salary_min, salary_max
    """
    if not _HAS_PYSPARK:
        pytest.skip("PySpark not installed")

    data = [
        ("j1", "linkedin", "Acme Corp", "Software Engineer", 80000, 140000),
        ("j2", "indeed", "Beta Inc", "Data Scientist", 90000, 160000),
        ("j3", "linkedin", "Gamma LLC", "ML Engineer", 100000, 180000),
        ("j4", "glassdoor", "Delta Co", "DevOps Engineer", 85000, 150000),
        ("j5", "indeed", "Acme Corp", "Backend Developer", 75000, 130000),
    ]
    columns = ["job_id", "source", "company_name", "job_title", "salary_min", "salary_max"]
    return spark.createDataFrame(data, schema=columns)


@pytest.fixture()
def spark_df_weather(spark: Any) -> Any:
    """Small Spark DataFrame with sample weather rows.

    Columns: city, date, temp_c, humidity, wind_speed_kmh
    """
    if not _HAS_PYSPARK:
        pytest.skip("PySpark not installed")

    data = [
        ("Seattle", "2025-01-01", 5.0, 82.0, 15.5),
        ("Seattle", "2025-01-02", 4.5, 80.0, 12.0),
        ("New York", "2025-01-01", -2.0, 60.0, 25.0),
        ("New York", "2025-01-02", -1.5, 65.0, 20.0),
        ("Austin", "2025-01-01", 12.0, 45.0, 10.0),
    ]
    columns = ["city", "date", "temp_c", "humidity", "wind_speed_kmh"]
    return spark.createDataFrame(data, schema=columns)


@pytest.fixture()
def spark_df_empty(spark: Any) -> Any:
    """Empty Spark DataFrame with a string-typed ``id`` column."""
    if not _HAS_PYSPARK:
        pytest.skip("PySpark not installed")

    from pyspark.sql.types import (  # type: ignore[import-untyped]
        StringType,
        StructField,
        StructType,
    )

    schema = StructType([StructField("id", StringType(), nullable=True)])
    return spark.createDataFrame([], schema=schema)
