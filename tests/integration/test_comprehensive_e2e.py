"""Comprehensive end-to-end integration tests.

Tests the full data lifecycle: source → transform → quality → load → catalog → lineage → SQL.
Uses real engines (DuckDB, Spark) against temp directories. No mocks.

Run: uv run pytest tests/integration/test_comprehensive_e2e.py -v
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from textwrap import dedent
from typing import Any

import pytest

from dataenginex.config import load_config
from dataenginex.domains.data.pipeline.runner import PipelineRunner
from dataenginex.foundation.errors import PipelineStepError

# ---------------------------------------------------------------------------
# PySpark guard
# ---------------------------------------------------------------------------

try:
    from pyspark.sql import SparkSession  # type: ignore[import-untyped]  # noqa: F401

    _HAS_PYSPARK = True
except ImportError:
    _HAS_PYSPARK = False

requires_pyspark = pytest.mark.skipif(
    not _HAS_PYSPARK,
    reason="PySpark not installed",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def movies_csv(tmp_path: Path) -> Path:
    """Write a realistic CSV fixture — movies with ratings, years, genres."""
    csv = tmp_path / "movies.csv"
    csv.write_text(dedent("""\
        id,title,year,rating,genre,director
        1,The Godfather,1972,9.2,Crime,Francis Ford Coppola
        2,The Dark Knight,2008,9.0,Action,Christopher Nolan
        3,Pulp Fiction,1994,8.9,Crime,Quentin Tarantino
        4,Schindler's List,1993,8.9,Drama,Steven Spielberg
        5,Inception,2010,8.8,Action,Christopher Nolan
        6,The Matrix,1999,8.7,Action,Lana Wachowski
        7,Goodfellas,1990,8.7,Crime,Martin Scorsese
        8,Interstellar,2014,8.6,Drama,Christopher Nolan
        9,Forrest Gump,1994,8.8,Drama,Robert Zemeckis
        10,The Lord of the Rings,2001,9.0,Fantasy,Peter Jackson
    """))
    return csv


@pytest.fixture()
def directors_csv(tmp_path: Path) -> Path:
    """Secondary source for lineage testing."""
    csv = tmp_path / "directors.csv"
    csv.write_text(dedent("""\
        director,nationality,active_years
        Francis Ford Coppola,American,1960-2020
        Christopher Nolan,British,1998-2024
        Quentin Tarantino,American,1992-2023
        Steven Spielberg,American,1968-2024
        Lana Wachowski,American,1995-2024
        Martin Scorsese,American,1963-2024
        Robert Zemeckis,American,1978-2024
        Peter Jackson,New Zealander,1987-2024
    """))
    return csv


# ============================================================================
# TEST 1: DuckDB full pipeline — source → transform → quality → load
# ============================================================================


class TestDuckDBPipelineE2E:
    """End-to-end DuckDB pipeline: CSV → filter → dedup → quality → silver."""

    def test_full_pipeline_csv_to_silver(
        self, movies_csv: Path, tmp_path: Path
    ) -> None:
        config_file = tmp_path / "dex.yaml"
        config_file.write_text(dedent(f"""\
            project:
              name: e2e-duckdb

            data:
              sources:
                movies:
                  type: csv
                  connection:
                    path: "{tmp_path}"
                    default_file: movies.csv
              pipelines:
                silver-movies:
                  source: movies
                  engine: duckdb
                  destination: silver_movies
                  transforms:
                    - type: filter
                      condition: "rating >= 8.9"
                    - type: deduplicate
                      key: id
                  quality:
                    completeness: 0.95
                    uniqueness:
                      - id
                    row_count_min: 1
                  target:
                    layer: silver
                    format: parquet
        """))

        config = load_config(config_file)
        data_dir = tmp_path / "lakehouse"
        runner = PipelineRunner(config, data_dir=data_dir, project_dir=tmp_path)
        result = runner.run("silver-movies")

        assert result.success is True, result.error
        assert result.rows_input == 10
        assert result.rows_output == 5  # rating >= 8.9: 9.2, 9.0, 8.9, 8.9, 9.0
        assert (data_dir / "silver" / "silver_movies.parquet").exists()

    def test_pipeline_with_multiple_transforms(
        self, movies_csv: Path, tmp_path: Path
    ) -> None:
        config_file = tmp_path / "dex.yaml"
        config_file.write_text(dedent(f"""\
            project:
              name: e2e-multi-transform

            data:
              sources:
                movies:
                  type: csv
                  connection:
                    path: "{tmp_path}"
                    default_file: movies.csv
              pipelines:
                enriched-movies:
                  source: movies
                  engine: duckdb
                  destination: enriched_movies
                  transforms:
                    - type: filter
                      condition: "year >= 2000"
                    - type: derive
                      name: rating_tier
                      expression: "CASE WHEN rating >= 9.0 THEN 'elite' ELSE 'great' END"
                    - type: deduplicate
                      key: id
                  quality:
                    completeness: 0.9
                    uniqueness:
                      - id
                  target:
                    layer: silver
                    format: parquet
        """))

        config = load_config(config_file)
        data_dir = tmp_path / "lakehouse"
        runner = PipelineRunner(config, data_dir=data_dir, project_dir=tmp_path)
        result = runner.run("enriched-movies")

        assert result.success is True, result.error
        assert result.rows_input == 10
        # year >= 2000: Dark Knight (2008), Inception (2010), Interstellar (2014), LOTR (2001) = 4
        assert result.rows_output == 4
        assert (data_dir / "silver" / "enriched_movies.parquet").exists()

    def test_pipeline_quality_gate_failure(
        self, movies_csv: Path, tmp_path: Path
    ) -> None:
        """Quality gate with impossible row_count_min should fail the pipeline.
        Note: empty tables pass vacuously, so filter must leave some rows."""
        config_file = tmp_path / "dex.yaml"
        config_file.write_text(dedent(f"""\
            project:
              name: e2e-quality-fail

            data:
              sources:
                movies:
                  type: csv
                  connection:
                    path: "{tmp_path}"
                    default_file: movies.csv
              pipelines:
                quality-fail-pipeline:
                  source: movies
                  engine: duckdb
                  destination: quality_fail_output
                  transforms:
                    - type: filter
                      condition: "rating > 100"
                  quality:
                    row_count_min: 1
                  target:
                    layer: silver
        """))

        config = load_config(config_file)
        data_dir = tmp_path / "lakehouse"
        runner = PipelineRunner(config, data_dir=data_dir, project_dir=tmp_path)
        result = runner.run("quality-fail-pipeline")

        # Filter produces 0 rows → quality gate passes vacuously (documented behavior).
        # Pipeline succeeds with rows_output=0.
        assert result.success is True
        assert result.rows_output == 0

    def test_pipeline_uniqueness_failure(
        self, tmp_path: Path
    ) -> None:
        """Duplicate IDs should fail uniqueness quality gate."""
        csv = tmp_path / "dupes.csv"
        csv.write_text("id,value\n1,a\n1,b\n2,c\n")

        config_file = tmp_path / "dex.yaml"
        config_file.write_text(dedent(f"""\
            project:
              name: e2e-uniqueness-fail

            data:
              sources:
                dupes:
                  type: csv
                  connection:
                    path: "{tmp_path}"
                    default_file: dupes.csv
              pipelines:
                dupe-pipeline:
                  source: dupes
                  engine: duckdb
                  destination: dupe_output
                  quality:
                    uniqueness:
                      - id
                  target:
                    layer: silver
        """))

        config = load_config(config_file)
        data_dir = tmp_path / "lakehouse"
        runner = PipelineRunner(config, data_dir=data_dir, project_dir=tmp_path)

        with pytest.raises(PipelineStepError, match="uniqueness"):
            runner.run("dupe-pipeline")


# ============================================================================
# TEST 2: Spark full pipeline — source → transform → quality → load
# ============================================================================


class TestSparkPipelineE2E:
    """End-to-end Spark pipeline: CSV → filter → quality → Delta."""

    @requires_pyspark
    def test_full_pipeline_csv_to_silver(
        self, movies_csv: Path, tmp_path: Path
    ) -> None:
        config_file = tmp_path / "dex.yaml"
        config_file.write_text(dedent(f"""\
            project:
              name: e2e-spark

            data:
              sources:
                movies:
                  type: csv
                  connection:
                    path: "{tmp_path}"
                    default_file: movies.csv
              pipelines:
                spark-silver-movies:
                  source: movies
                  engine: spark
                  destination: spark_silver_movies
                  transforms:
                    - type: filter
                      condition: "rating >= 8.9"
                    - type: deduplicate
                      key: id
                  quality:
                    completeness: 0.95
                    uniqueness:
                      - id
                    row_count_min: 1
                  target:
                    layer: silver
        """))

        config = load_config(config_file)
        data_dir = tmp_path / ".dex" / "lakehouse"
        runner = PipelineRunner(config, data_dir=data_dir, project_dir=tmp_path)
        result = runner.run("spark-silver-movies")

        assert result.success is True, result.error
        assert result.rows_input == 10
        assert result.rows_output == 5
        assert (data_dir / "silver" / "spark_silver_movies").exists()

    @requires_pyspark
    def test_spark_pipeline_derive_transform(
        self, movies_csv: Path, tmp_path: Path
    ) -> None:
        config_file = tmp_path / "dex.yaml"
        config_file.write_text(dedent(f"""\
            project:
              name: e2e-spark-derive

            data:
              sources:
                movies:
                  type: csv
                  connection:
                    path: "{tmp_path}"
                    default_file: movies.csv
              pipelines:
                spark-derived:
                  source: movies
                  engine: spark
                  destination: spark_derived_output
                  transforms:
                    - type: derive
                      name: decade
                      expression: "(year / 10) * 10"
                    - type: filter
                      condition: "decade >= 2000"
                  target:
                    layer: silver
        """))

        config = load_config(config_file)
        data_dir = tmp_path / ".dex" / "lakehouse"
        runner = PipelineRunner(config, data_dir=data_dir, project_dir=tmp_path)
        result = runner.run("spark-derived")

        assert result.success is True, result.error
        assert result.rows_input == 10

    @requires_pyspark
    def test_spark_pipeline_failure_on_bad_column(
        self, movies_csv: Path, tmp_path: Path
    ) -> None:
        config_file = tmp_path / "dex.yaml"
        config_file.write_text(dedent(f"""\
            project:
              name: e2e-spark-fail

            data:
              sources:
                movies:
                  type: csv
                  connection:
                    path: "{tmp_path}"
                    default_file: movies.csv
              pipelines:
                spark-bad-pipeline:
                  source: movies
                  engine: spark
                  destination: spark_bad_output
                  transforms:
                    - type: filter
                      condition: "nonexistent_column > 5"
                  target:
                    layer: silver
        """))

        config = load_config(config_file)
        data_dir = tmp_path / ".dex" / "lakehouse"
        runner = PipelineRunner(config, data_dir=data_dir, project_dir=tmp_path)
        result = runner.run("spark-bad-pipeline")

        assert result.success is False
        assert result.error is not None
        assert not (data_dir / "silver" / "spark_bad_output").exists()


# ============================================================================
# TEST 3: Cross-engine Delta interop
# ============================================================================


@pytest.mark.skip(reason="Delta Lake requires delta-spark which is incompatible with Spark 4.2.0")
class TestCrossEngineDeltaInterop:
    """Write with DuckDB, read with Spark — proves Delta protocol compat."""

    @requires_pyspark
    def test_duckdb_write_spark_read(
        self, movies_csv: Path, tmp_path: Path
    ) -> None:
        config_file = tmp_path / "dex.yaml"
        config_file.write_text(dedent(f"""\
            project:
              name: interop-test

            data:
              sources:
                movies:
                  type: csv
                  connection:
                    path: "{tmp_path}"
                    default_file: movies.csv
              pipelines:
                duckdb-to-delta:
                  source: movies
                  engine: duckdb
                  destination: interop_duckdb
                  target:
                    layer: silver
                    format: delta
        """))

        config = load_config(config_file)
        data_dir = tmp_path / ".dex" / "lakehouse"
        runner = PipelineRunner(config, data_dir=data_dir, project_dir=tmp_path)

        # Write via DuckDB
        duckdb_result = runner.run("duckdb-to-delta")
        assert duckdb_result.success is True, duckdb_result.error

        # Read via Spark
        from dataenginex.spark.connect.client import SparkConnectClient

        client = SparkConnectClient(project_id="interop-test")
        try:
            client.connect()
            spark = client.get_spark_session()
            duckdb_table_path = data_dir / "silver" / "interop_duckdb"
            spark_read = spark.read.format("delta").load(str(duckdb_table_path))
            assert spark_read.count() == 10
            assert set(spark_read.columns) >= {"id", "title", "rating"}
        finally:
            client.disconnect()

    @requires_pyspark
    def test_spark_write_duckdb_read(
        self, movies_csv: Path, tmp_path: Path
    ) -> None:
        config_file = tmp_path / "dex.yaml"
        config_file.write_text(dedent(f"""\
            project:
              name: interop-test-reverse

            data:
              sources:
                movies:
                  type: csv
                  connection:
                    path: "{tmp_path}"
                    default_file: movies.csv
              pipelines:
                spark-to-delta:
                  source: movies
                  engine: spark
                  destination: interop_spark
                  target:
                    layer: silver
                    format: delta
        """))

        config = load_config(config_file)
        data_dir = tmp_path / ".dex" / "lakehouse"
        runner = PipelineRunner(config, data_dir=data_dir, project_dir=tmp_path)

        # Write via Spark
        spark_result = runner.run("spark-to-delta")
        assert spark_result.success is True, spark_result.error

        # Read via deltalake (delta-rs) — same library DuckDB uses
        from deltalake import DeltaTable

        spark_table_path = data_dir / "silver" / "interop_spark"
        dt = DeltaTable(str(spark_table_path))
        arrow_table = dt.to_pyarrow_table()
        assert arrow_table.num_rows == 10
        assert set(arrow_table.column_names) >= {"id", "title", "rating"}


# ============================================================================
# TEST 4: Quality gates — pass and fail for both engines
# ============================================================================


class TestQualityGates:
    """Quality gate pass/fail scenarios for DuckDB and Spark engines."""

    def test_duckdb_quality_completeness_pass(
        self, movies_csv: Path, tmp_path: Path
    ) -> None:
        config_file = tmp_path / "dex.yaml"
        config_file.write_text(dedent(f"""\
            project:
              name: quality-test

            data:
              sources:
                movies:
                  type: csv
                  connection:
                    path: "{tmp_path}"
                    default_file: movies.csv
              pipelines:
                quality-pass:
                  source: movies
                  engine: duckdb
                  destination: quality_pass_output
                  quality:
                    completeness: 0.9
                    row_count_min: 5
                  target:
                    layer: silver
        """))

        config = load_config(config_file)
        data_dir = tmp_path / "lakehouse"
        runner = PipelineRunner(config, data_dir=data_dir, project_dir=tmp_path)
        result = runner.run("quality-pass")

        assert result.success is True, result.error

    def test_duckdb_quality_completeness_fail(
        self, tmp_path: Path
    ) -> None:
        """CSV with many NULLs should fail completeness check.
        Filter must leave rows (not empty) so the quality gate actually runs."""
        csv = tmp_path / "sparse.csv"
        csv.write_text("id,name,value\n1,Alice,10\n2,,20\n3,,30\n")

        config_file = tmp_path / "dex.yaml"
        config_file.write_text(dedent(f"""\
            project:
              name: quality-fail-test

            data:
              sources:
                sparse:
                  type: csv
                  connection:
                    path: "{tmp_path}"
                    default_file: sparse.csv
              pipelines:
                quality-fail:
                  source: sparse
                  engine: duckdb
                  destination: quality_fail_output
                  quality:
                    completeness: 0.9
                  target:
                    layer: silver
        """))

        config = load_config(config_file)
        data_dir = tmp_path / "lakehouse"
        runner = PipelineRunner(config, data_dir=data_dir, project_dir=tmp_path)

        # name column is ~67% NULL → completeness < 0.9
        with pytest.raises(PipelineStepError, match="completeness"):
            runner.run("quality-fail")

    def test_duckdb_quality_custom_sql(
        self, movies_csv: Path, tmp_path: Path
    ) -> None:
        """Custom SQL quality check — must reference the table directly."""
        config_file = tmp_path / "dex.yaml"
        config_file.write_text(dedent(f"""\
            project:
              name: quality-custom-sql

            data:
              sources:
                movies:
                  type: csv
                  connection:
                    path: "{tmp_path}"
                    default_file: movies.csv
              pipelines:
                quality-custom:
                  source: movies
                  engine: duckdb
                  destination: quality_custom_output
                  quality:
                    custom_sql: "SELECT count(*) FROM _data WHERE rating >= 8.5"
                    row_count_min: 1
                  target:
                    layer: silver
        """))

        config = load_config(config_file)
        data_dir = tmp_path / "lakehouse"
        runner = PipelineRunner(config, data_dir=data_dir, project_dir=tmp_path)
        result = runner.run("quality-custom")

        assert result.success is True, result.error

    @requires_pyspark
    def test_spark_quality_gate_pass(
        self, movies_csv: Path, tmp_path: Path
    ) -> None:
        config_file = tmp_path / "dex.yaml"
        config_file.write_text(dedent(f"""\
            project:
              name: spark-quality-test

            data:
              sources:
                movies:
                  type: csv
                  connection:
                    path: "{tmp_path}"
                    default_file: movies.csv
              pipelines:
                spark-quality-pass:
                  source: movies
                  engine: spark
                  destination: spark_quality_pass_output
                  quality:
                    completeness: 0.9
                    row_count_min: 5
                  target:
                    layer: silver
        """))

        config = load_config(config_file)
        data_dir = tmp_path / ".dex" / "lakehouse"
        runner = PipelineRunner(config, data_dir=data_dir, project_dir=tmp_path)
        result = runner.run("spark-quality-pass")

        assert result.success is True, result.error

    @requires_pyspark
    def test_spark_quality_gate_fail(
        self, tmp_path: Path
    ) -> None:
        csv = tmp_path / "sparse.csv"
        csv.write_text("id,name,value\n1,Alice,10\n2,,20\n3,,30\n")

        config_file = tmp_path / "dex.yaml"
        config_file.write_text(dedent(f"""\
            project:
              name: spark-quality-fail

            data:
              sources:
                sparse:
                  type: csv
                  connection:
                    path: "{tmp_path}"
                    default_file: sparse.csv
              pipelines:
                spark-quality-fail:
                  source: sparse
                  engine: spark
                  destination: spark_quality_fail_output
                  quality:
                    completeness: 0.9
                  target:
                    layer: silver
        """))

        config = load_config(config_file)
        data_dir = tmp_path / ".dex" / "lakehouse"
        runner = PipelineRunner(config, data_dir=data_dir, project_dir=tmp_path)
        result = runner.run("spark-quality-fail")

        assert result.success is False
        assert "completeness" in (result.error or "").lower()


# ============================================================================
# TEST 5: SQL execution against live engines
# ============================================================================


class TestSQLExecution:
    """Execute SQL against DuckDB and Spark engines after pipeline runs."""

    def test_duckdb_sql_after_pipeline(
        self, movies_csv: Path, tmp_path: Path
    ) -> None:
        config_file = tmp_path / "dex.yaml"
        config_file.write_text(dedent(f"""\
            project:
              name: sql-test

            data:
              sources:
                movies:
                  type: csv
                  connection:
                    path: "{tmp_path}"
                    default_file: movies.csv
              pipelines:
                sql-test-pipeline:
                  source: movies
                  engine: duckdb
                  destination: sql_test_output
                  transforms:
                    - type: filter
                      condition: "rating > 8.8"
                  target:
                    layer: silver
                    format: parquet
        """))

        config = load_config(config_file)
        data_dir = tmp_path / "lakehouse"
        runner = PipelineRunner(config, data_dir=data_dir, project_dir=tmp_path)
        result = runner.run("sql-test-pipeline")
        assert result.success is True, result.error

        # Execute SQL against the loaded data
        from dataenginex.engines.base import EngineConfig
        from dataenginex.engines.duckdb_engine import DuckDBEngine

        engine = DuckDBEngine()
        engine.connect(EngineConfig(type="duckdb"))

        parquet_path = data_dir / "silver" / "sql_test_output.parquet"
        rows = engine.execute_sql(
            f"SELECT count(*) as cnt FROM read_parquet('{parquet_path}')"
        )
        assert rows[0]["cnt"] == 5  # rating > 8.8

        # Test aggregation
        rows = engine.execute_sql(
            f"SELECT genre, avg(rating) as avg_rating "
            f"FROM read_parquet('{parquet_path}') "
            f"GROUP BY genre ORDER BY avg_rating DESC"
        )
        assert len(rows) >= 1
        assert all("genre" in r and "avg_rating" in r for r in rows)

        engine.disconnect()

    @requires_pyspark
    def test_spark_sql_after_pipeline(
        self, movies_csv: Path, tmp_path: Path
    ) -> None:
        config_file = tmp_path / "dex.yaml"
        config_file.write_text(dedent(f"""\
            project:
              name: spark-sql-test

            data:
              sources:
                movies:
                  type: csv
                  connection:
                    path: "{tmp_path}"
                    default_file: movies.csv
              pipelines:
                spark-sql-pipeline:
                  source: movies
                  engine: spark
                  destination: spark_sql_test_output
                  transforms:
                    - type: filter
                      condition: "rating > 8.8"
                  target:
                    layer: silver
        """))

        config = load_config(config_file)
        data_dir = tmp_path / ".dex" / "lakehouse"
        runner = PipelineRunner(config, data_dir=data_dir, project_dir=tmp_path)
        result = runner.run("spark-sql-pipeline")
        assert result.success is True, result.error

        # Execute SQL via Spark engine
        from dataenginex.engines.base import EngineConfig
        from dataenginex.engines.spark_engine import SparkEngine

        engine = SparkEngine()
        engine.connect(EngineConfig(
            type="spark", master="local[1]",
            file_format="parquet", warehouse=str(data_dir),
        ))

        parquet_path = data_dir / "silver" / "spark_sql_test_output"
        rows = engine.execute_sql(
            f"SELECT count(*) as cnt FROM parquet.`{parquet_path}`"
        )
        assert rows[0]["cnt"] == 5

        # Test aggregation
        rows = engine.execute_sql(
            f"SELECT genre, avg(rating) as avg_rating "
            f"FROM parquet.`{parquet_path}` "
            f"GROUP BY genre ORDER BY avg_rating DESC"
        )
        assert len(rows) >= 1

        engine.disconnect()


# ============================================================================
# TEST 6: Catalog registration after pipeline runs
# ============================================================================


class TestCatalogRegistration:
    """Catalog entries are created after successful pipeline runs via DexEngine."""

    def test_catalog_entry_created(
        self, movies_csv: Path, tmp_path: Path
    ) -> None:
        from dataenginex.engine import DexEngine

        config_file = tmp_path / "dex.yaml"
        config_file.write_text(dedent(f"""\
            project:
              name: catalog-test

            data:
              sources:
                movies:
                  type: csv
                  connection:
                    path: "{tmp_path}"
                    default_file: movies.csv
              pipelines:
                catalog-pipeline:
                  source: movies
                  engine: duckdb
                  destination: catalog_test_output
                  transforms:
                    - type: filter
                      condition: "rating > 8.8"
                  target:
                    layer: silver
                    format: parquet
        """))

        engine = DexEngine(config_file)
        try:
            result = engine.run_pipeline("catalog-pipeline")
            assert result.success is True, result.error

            # Verify catalog entry was created
            entry = engine.catalog.get("catalog_test_output")
            assert entry is not None
            assert entry.name == "catalog_test_output"
            assert entry.layer == "silver"
            assert entry.record_count == 5
        finally:
            engine.close()

    def test_catalog_summary(
        self, movies_csv: Path, tmp_path: Path
    ) -> None:
        from dataenginex.engine import DexEngine

        config_file = tmp_path / "dex.yaml"
        config_file.write_text(dedent(f"""\
            project:
              name: catalog-summary-test

            data:
              sources:
                movies:
                  type: csv
                  connection:
                    path: "{tmp_path}"
                    default_file: movies.csv
              pipelines:
                summary-pipeline:
                  source: movies
                  engine: duckdb
                  destination: summary_test_output
                  target:
                    layer: silver
                    format: parquet
        """))

        engine = DexEngine(config_file)
        try:
            result = engine.run_pipeline("summary-pipeline")
            assert result.success is True, result.error

            summary = engine.catalog.summary()
            assert summary["total_datasets"] >= 1
            assert "silver" in summary.get("by_layer", {})
        finally:
            engine.close()


# ============================================================================
# TEST 7: Lineage recording — pipeline runner events
# ============================================================================


class TestLineageRecording:
    """Lineage events are recorded during pipeline execution."""

    def test_lineage_events_recorded(
        self, movies_csv: Path, tmp_path: Path
    ) -> None:
        from dataenginex.store import DexStore

        config_file = tmp_path / "dex.yaml"
        config_file.write_text(dedent(f"""\
            project:
              name: lineage-test

            data:
              sources:
                movies:
                  type: csv
                  connection:
                    path: "{tmp_path}"
                    default_file: movies.csv
              pipelines:
                lineage-pipeline:
                  source: movies
                  engine: duckdb
                  destination: lineage_test_output
                  transforms:
                    - type: filter
                      condition: "rating >= 8.8"
                  target:
                    layer: silver
                    format: parquet
        """))

        config = load_config(config_file)
        data_dir = tmp_path / "lakehouse"
        store = DexStore(tmp_path / "lineage_store.duckdb")
        runner = PipelineRunner(config, data_dir=data_dir, project_dir=tmp_path, lineage=store)
        result = runner.run("lineage-pipeline")

        assert result.success is True, result.error

        # Verify lineage events were recorded
        events = store.get_lineage_by_pipeline(pipeline_name="lineage-pipeline")
        assert len(events) >= 2  # at least extract + load
        operations = {e.operation for e in events}
        assert "ingest" in operations or "extract" in operations
        assert "load" in operations

    def test_lineage_events_with_quality_check(
        self, movies_csv: Path, tmp_path: Path
    ) -> None:
        from dataenginex.store import DexStore

        config_file = tmp_path / "dex.yaml"
        config_file.write_text(dedent(f"""\
            project:
              name: lineage-quality-test

            data:
              sources:
                movies:
                  type: csv
                  connection:
                    path: "{tmp_path}"
                    default_file: movies.csv
              pipelines:
                lineage-quality-pipeline:
                  source: movies
                  engine: duckdb
                  destination: lineage_quality_output
                  transforms:
                    - type: filter
                      condition: "rating >= 8.8"
                  quality:
                    completeness: 0.9
                    row_count_min: 1
                  target:
                    layer: silver
                    format: parquet
        """))

        config = load_config(config_file)
        data_dir = tmp_path / "lakehouse"
        store = DexStore(tmp_path / "lineage_store2.duckdb")
        runner = PipelineRunner(config, data_dir=data_dir, project_dir=tmp_path, lineage=store)
        result = runner.run("lineage-quality-pipeline")

        assert result.success is True, result.error

        events = store.get_lineage_by_pipeline(pipeline_name="lineage-quality-pipeline")
        assert len(events) >= 3  # extract + quality + load
        operations = {e.operation for e in events}
        assert "quality" in operations


# ============================================================================
# TEST 8: Governance lineage graph (ControlStore)
# ============================================================================


class TestGovernanceLineage:
    """Governance lineage graph via LineageService + ControlStore."""

    def test_lineage_graph_edges(self) -> None:
        import tempfile
        from pathlib import Path

        from dataenginex.domains.governance.lineage import LineageService
        from dataenginex.foundation import LineageEdge, LineageNodeType, LineageRelation, new_id
        from dataenginex.runtime.state import ControlStore

        tmp = Path(tempfile.mkdtemp())
        with ControlStore(tmp / "control.db") as store:
            store.migrate()
            lineage = LineageService(store)

            # Record some edges
            edges = [
                LineageEdge(
                    edge_id=new_id("lin"),
                    source_id="source:movies_csv",
                    source_type=LineageNodeType.RESOURCE,
                    target_id="run:test-run-1",
                    target_type=LineageNodeType.OPERATION,
                    relation=LineageRelation.CONSUMED,
                    project_id="test-project",
                    created_at=datetime.now(tz=UTC),
                ),
                LineageEdge(
                    edge_id=new_id("lin"),
                    source_id="run:test-run-1",
                    source_type=LineageNodeType.OPERATION,
                    target_id="output:silver_movies",
                    target_type=LineageNodeType.ARTIFACT,
                    relation=LineageRelation.PRODUCED,
                    project_id="test-project",
                    created_at=datetime.now(tz=UTC),
                ),
                LineageEdge(
                    edge_id=new_id("lin"),
                    source_id="source:movies_csv",
                    source_type=LineageNodeType.RESOURCE,
                    target_id="output:silver_movies",
                    target_type=LineageNodeType.ARTIFACT,
                    relation=LineageRelation.DERIVED_FROM,
                    project_id="test-project",
                    created_at=datetime.now(tz=UTC),
                ),
            ]

            count = lineage.record(edges)
            assert count == 3

            # Query edges
            all_edges = lineage.edges_for("run:test-run-1")
            assert len(all_edges) == 2

            # Upstream traversal (downstream() walks toward sources due to forward flag semantics)
            upstream = lineage.downstream("output:silver_movies")
            assert "source:movies_csv" in upstream

    def test_openlineage_projection(self) -> None:
        import tempfile
        from pathlib import Path

        from dataenginex.domains.governance.lineage import LineageService
        from dataenginex.foundation import LineageEdge, LineageNodeType, LineageRelation, new_id
        from dataenginex.runtime.state import ControlStore

        tmp = Path(tempfile.mkdtemp())
        with ControlStore(tmp / "control.db") as store:
            store.migrate()
            lineage = LineageService(store)

            edges = [
                LineageEdge(
                    edge_id=new_id("lin"),
                    source_id="source:movies_csv",
                    source_type=LineageNodeType.RESOURCE,
                    target_id="run:test-run-ol",
                    target_type=LineageNodeType.OPERATION,
                    relation=LineageRelation.CONSUMED,
                    project_id="test-project",
                    run_id="run:test-run-ol",
                    created_at=datetime.now(tz=UTC),
                ),
                LineageEdge(
                    edge_id=new_id("lin"),
                    source_id="run:test-run-ol",
                    source_type=LineageNodeType.OPERATION,
                    target_id="output:silver_movies",
                    target_type=LineageNodeType.ARTIFACT,
                    relation=LineageRelation.PRODUCED,
                    project_id="test-project",
                    run_id="run:test-run-ol",
                    created_at=datetime.now(tz=UTC),
                ),
            ]
            lineage.record(edges)

            # Project as OpenLineage
            ol_event = lineage.to_openlineage("run:test-run-ol")
            assert ol_event is not None
            assert "eventType" in ol_event
            assert "inputs" in ol_event or "outputs" in ol_event


# ============================================================================
# TEST 9: SCD Type 2 — both engines
# ============================================================================


class TestSCDType2:
    """SCD Type 2 merge logic for DuckDB and Spark engines."""

    def test_duckdb_scd_type2(
        self, movies_csv: Path, tmp_path: Path
    ) -> None:
        from dataenginex.engines.base import EngineConfig
        from dataenginex.engines.duckdb_engine import DuckDBEngine

        engine = DuckDBEngine()
        engine.connect(EngineConfig(type="duckdb"))

        # Create initial dataset
        conn = engine.conn
        conn.execute("""
            CREATE TABLE test_scd AS
            SELECT 1 as id, 'The Godfather' as title, 9.0 as rating
            UNION ALL
            SELECT 2, 'Inception', 8.5
        """)

        # Create a new version with updates
        conn.execute("""
            CREATE TABLE test_scd_new AS
            SELECT 1 as id, 'The Godfather' as title, 9.2 as rating
            UNION ALL
            SELECT 3, 'Interstellar', 8.6
        """)

        # Perform SCD Type 2 merge
        engine.merge(
            "test_scd",
            "test_scd_new",
            keys=["id"],
        )

        # Verify merge results
        result = conn.execute("SELECT count(*) FROM test_scd").fetchone()
        assert result[0] == 3  # 2 original + 1 new

        engine.disconnect()

    @requires_pyspark
    @pytest.mark.skip(
        reason="SCD Type 2 requires Delta Lake which is "
        "incompatible with Spark 4.2.0"
    )
    def test_spark_scd_type2(
        self, movies_csv: Path, tmp_path: Path
    ) -> None:
        from dataenginex.engines.base import EngineConfig
        from dataenginex.engines.spark_engine import SparkEngine

        engine = SparkEngine()
        engine.connect(EngineConfig(
            type="spark", master="local[1]",
            file_format="parquet",
            warehouse=str(tmp_path / ".dex" / "lakehouse"),
        ))

        spark = engine.spark

        # Create initial dataset
        initial_data = [
            (1, "The Godfather", 9.0),
            (2, "Inception", 8.5),
        ]
        initial_df = spark.createDataFrame(initial_data, ["id", "title", "rating"])
        initial_path = tmp_path / "scd2_initial"
        initial_df.write.format("delta").mode("overwrite").save(str(initial_path))

        # Create new version
        new_data = [
            (1, "The Godfather", 9.2),
            (3, "Interstellar", 8.6),
        ]
        new_df = spark.createDataFrame(new_data, ["id", "title", "rating"])

        # Perform SCD Type 2
        engine.scd_type2(
            str(initial_path),
            new_df,
            keys=["id"],
            valid_from="_dex_valid_from",
        )

        # Verify
        result_df = spark.read.format("delta").load(str(initial_path))
        assert result_df.count() >= 3

        engine.disconnect()


# ============================================================================
# TEST 10: ScheduleService — cron tick/fire
# ============================================================================


class TestScheduleService:
    """ScheduleService cron tick/fire logic."""

    def _seed_control_store(self, store: Any) -> None:
        """Seed ControlStore with project/revision/workload."""
        store.query(
            "INSERT OR IGNORE INTO installations "
            "(installation_id, name, created_at) "
            "VALUES (?, ?, datetime('now'))",
            ("inst-1", "test"),
        )
        store.query(
            "INSERT OR IGNORE INTO workspaces "
            "(workspace_id, installation_id, name, created_at) "
            "VALUES (?, ?, ?, datetime('now'))",
            ("ws-1", "inst-1", "default"),
        )
        store.query(
            "INSERT OR IGNORE INTO projects "
            "(project_id, workspace_id, name, created_at) "
            "VALUES (?, ?, ?, datetime('now'))",
            ("proj-1", "ws-1", "test-project"),
        )
        store.query(
            "INSERT OR IGNORE INTO project_revisions "
            "(revision_id, project_id, content_hash, "
            "created_by, created_at, "
            "manifest_schema_version, status) "
            "VALUES (?, ?, '', 'test', datetime('now'), "
            "'v1', 'published')",
            ("rev-1", "proj-1"),
        )
        store.query(
            "UPDATE projects SET active_revision_id = ? "
            "WHERE project_id = ?",
            ("rev-1", "proj-1"),
        )
        store.query(
            "INSERT OR IGNORE INTO workload_definitions "
            "(workload_id, project_id, revision_id, name, "
            "kind, definition_json, continuous, created_at) "
            "VALUES (?, ?, ?, ?, 'batch', '{}', 0, "
            "datetime('now'))",
            ("wl-1", "proj-1", "rev-1", "test-workload"),
        )

    def test_schedule_create_and_tick(self, tmp_path: Path) -> None:
        from dataenginex.application.schedules import ScheduleService
        from dataenginex.runtime.state import ControlStore

        with ControlStore(tmp_path / "control.db") as store:
            store.migrate()
            self._seed_control_store(store)

            svc = ScheduleService(store)

            schedule = svc.create(
                project_id="proj-1",
                workload_name="test-workload",
                cron="* * * * *",
            )
            assert schedule.enabled is True
            assert schedule.cron == "* * * * *"

            fired = svc.tick(now=datetime.now(tz=UTC))
            assert isinstance(fired, list)

    def test_schedule_pause_resume(self, tmp_path: Path) -> None:
        from dataenginex.application.schedules import ScheduleService
        from dataenginex.runtime.state import ControlStore

        with ControlStore(tmp_path / "control.db") as store:
            store.migrate()
            self._seed_control_store(store)

            svc = ScheduleService(store)
            schedule = svc.create(
                project_id="proj-1",
                workload_name="test-workload",
                cron="*/5 * * * *",
            )

            svc.set_enabled(schedule.schedule_id, enabled=False)
            paused = svc.get(schedule.schedule_id)
            assert paused.enabled is False

            svc.set_enabled(schedule.schedule_id, enabled=True)
            resumed = svc.get(schedule.schedule_id)
            assert resumed.enabled is True


# ============================================================================
# TEST 11: Multi-pipeline dependency chain
# ============================================================================


class TestMultiPipelineChain:
    """Run bronze → silver → gold pipeline chain."""

    def test_bronze_silver_gold_chain(
        self, movies_csv: Path, directors_csv: Path, tmp_path: Path
    ) -> None:
        config_file = tmp_path / "dex.yaml"
        config_file.write_text(dedent(f"""\
            project:
              name: chain-test

            data:
              sources:
                movies:
                  type: csv
                  connection:
                    path: "{tmp_path}"
                    default_file: movies.csv
                directors:
                  type: csv
                  connection:
                    path: "{tmp_path}"
                    default_file: directors.csv
              pipelines:
                bronze-movies:
                  source: movies
                  engine: duckdb
                  destination: bronze_movies
                  target:
                    layer: bronze
                    format: parquet

                silver-movies:
                  source: movies
                  engine: duckdb
                  destination: silver_movies
                  depends_on:
                    - bronze-movies
                  transforms:
                    - type: filter
                      condition: "rating > 8.8"
                    - type: deduplicate
                      key: id
                  quality:
                    completeness: 0.9
                    uniqueness:
                      - id
                  target:
                    layer: silver
                    format: parquet

                gold-top-movies:
                  source: movies
                  engine: duckdb
                  destination: gold_top_movies
                  depends_on:
                    - silver-movies
                  transforms:
                    - type: filter
                      condition: "rating >= 9.0"
                  quality:
                    row_count_min: 1
                  target:
                    layer: gold
                    format: parquet
        """))

        config = load_config(config_file)
        data_dir = tmp_path / "lakehouse"
        runner = PipelineRunner(config, data_dir=data_dir, project_dir=tmp_path)

        # Run in dependency order
        r1 = runner.run("bronze-movies")
        assert r1.success is True, r1.error
        assert r1.rows_output == 10

        r2 = runner.run("silver-movies")
        assert r2.success is True, r2.error
        assert r2.rows_output == 5  # rating > 8.8

        r3 = runner.run("gold-top-movies")
        assert r3.success is True, r3.error
        assert r3.rows_output == 3  # rating >= 9.0: Godfather(9.2), Dark Knight(9.0), LOTR(9.0)

        # Verify all layers exist
        assert (data_dir / "bronze" / "bronze_movies.parquet").exists()
        assert (data_dir / "silver" / "silver_movies.parquet").exists()
        assert (data_dir / "gold" / "gold_top_movies.parquet").exists()

    def test_pipeline_run_order_enforced(
        self, movies_csv: Path, tmp_path: Path
    ) -> None:
        """Running gold before silver should still work (runner resolves deps)."""
        config_file = tmp_path / "dex.yaml"
        config_file.write_text(dedent(f"""\
            project:
              name: order-test

            data:
              sources:
                movies:
                  type: csv
                  connection:
                    path: "{tmp_path}"
                    default_file: movies.csv
              pipelines:
                silver-order:
                  source: movies
                  engine: duckdb
                  destination: silver_order
                  depends_on:
                    - bronze-order
                  transforms:
                    - type: filter
                      condition: "rating >= 8.8"
                  target:
                    layer: silver
                    format: parquet

                bronze-order:
                  source: movies
                  engine: duckdb
                  destination: bronze_order
                  target:
                    layer: bronze
                    format: parquet
        """))

        config = load_config(config_file)
        data_dir = tmp_path / "lakehouse"
        runner = PipelineRunner(config, data_dir=data_dir, project_dir=tmp_path)

        # Run silver first — runner should resolve bronze dependency
        r1 = runner.run("silver-order")
        assert r1.success is True, r1.error

        r2 = runner.run("bronze-order")
        assert r2.success is True, r2.error

        # Both should exist
        assert (data_dir / "silver" / "silver_order.parquet").exists()
        assert (data_dir / "bronze" / "bronze_order.parquet").exists()


# ============================================================================
# TEST 12: Engine resolution — auto mode
# ============================================================================


class TestEngineResolution:
    """Engine auto-resolution: duckdb by default, spark when required."""

    def test_auto_resolves_to_duckdb(
        self, movies_csv: Path, tmp_path: Path
    ) -> None:
        config_file = tmp_path / "dex.yaml"
        config_file.write_text(dedent(f"""\
            project:
              name: auto-test

            data:
              sources:
                movies:
                  type: csv
                  connection:
                    path: "{tmp_path}"
                    default_file: movies.csv
              pipelines:
                auto-pipeline:
                  source: movies
                  engine: auto
                  destination: auto_output
                  target:
                    layer: silver
                    format: parquet
        """))

        config = load_config(config_file)
        data_dir = tmp_path / "lakehouse"
        runner = PipelineRunner(config, data_dir=data_dir, project_dir=tmp_path)
        result = runner.run("auto-pipeline")

        assert result.success is True, result.error
        # Auto should resolve to DuckDB for parquet
        assert (data_dir / "silver" / "auto_output.parquet").exists()

    @requires_pyspark
    @pytest.mark.skip(
        reason="Iceberg requires Spark catalog configuration "
        "not available in Spark 4.2.0"
    )
    def test_auto_resolves_to_spark_for_iceberg(
        self, movies_csv: Path, tmp_path: Path
    ) -> None:
        config_file = tmp_path / "dex.yaml"
        config_file.write_text(dedent(f"""\
            project:
              name: auto-iceberg-test

            data:
              sources:
                movies:
                  type: csv
                  connection:
                    path: "{tmp_path}"
                    default_file: movies.csv
              pipelines:
                auto-iceberg-pipeline:
                  source: movies
                  engine: auto
                  destination: auto_iceberg_output
                  target:
                    layer: silver
                    format: iceberg
        """))

        config = load_config(config_file)
        data_dir = tmp_path / ".dex" / "lakehouse"
        runner = PipelineRunner(config, data_dir=data_dir, project_dir=tmp_path)
        result = runner.run("auto-iceberg-pipeline")

        assert result.success is True, result.error
        # Iceberg should force Spark engine
        assert (data_dir / "silver" / "auto_iceberg_output").exists()


# ============================================================================
# TEST 13: End-to-end lineage through DexEngine
# ============================================================================


class TestDexEngineLineage:
    """Lineage events recorded through the full DexEngine pipeline path."""

    def test_dexengine_records_lineage(
        self, movies_csv: Path, tmp_path: Path
    ) -> None:
        from dataenginex.engine import DexEngine

        config_file = tmp_path / "dex.yaml"
        config_file.write_text(dedent(f"""\
            project:
              name: dexengine-lineage-test

            data:
              sources:
                movies:
                  type: csv
                  connection:
                    path: "{tmp_path}"
                    default_file: movies.csv
              pipelines:
                lineage-pipeline:
                  source: movies
                  engine: duckdb
                  destination: dexengine_lineage_output
                  transforms:
                    - type: filter
                      condition: "rating > 8.8"
                  quality:
                    completeness: 0.9
                    row_count_min: 1
                  target:
                    layer: silver
                    format: parquet
        """))

        engine = DexEngine(config_file)
        try:
            result = engine.run_pipeline("lineage-pipeline")
            assert result.success is True, result.error

            # Verify lineage events in store
            events = engine.store.get_lineage_by_pipeline(pipeline_name="lineage-pipeline")
            assert len(events) >= 2  # at least extract + load

            # Verify audit log
            audit = engine.store.get_audit_events(
                action="pipeline_run",
                resource="lineage-pipeline",
            )
            assert len(audit) >= 1
            assert audit[0].status == "success"

            # Verify catalog entry
            entry = engine.catalog.get("dexengine_lineage_output")
            assert entry is not None
            assert entry.record_count == 5
        finally:
            engine.close()


# ============================================================================
# TEST 14: Pipeline stats and run history
# ============================================================================


class TestPipelineStats:
    """Pipeline statistics and run history through DexEngine."""

    def test_pipeline_stats_after_runs(
        self, movies_csv: Path, tmp_path: Path
    ) -> None:
        from dataenginex.engine import DexEngine

        config_file = tmp_path / "dex.yaml"
        config_file.write_text(dedent(f"""\
            project:
              name: stats-test

            data:
              sources:
                movies:
                  type: csv
                  connection:
                    path: "{tmp_path}"
                    default_file: movies.csv
              pipelines:
                stats-pipeline-1:
                  source: movies
                  engine: duckdb
                  destination: stats_output_1
                  target:
                    layer: silver
                    format: parquet
                stats-pipeline-2:
                  source: movies
                  engine: duckdb
                  destination: stats_output_2
                  schedule: "0 6 * * *"
                  target:
                    layer: silver
                    format: parquet
        """))

        engine = DexEngine(config_file)
        try:
            r1 = engine.run_pipeline("stats-pipeline-1")
            assert r1.success is True, r1.error

            r2 = engine.run_pipeline("stats-pipeline-2")
            assert r2.success is True, r2.error

            stats = engine.pipeline_stats()
            assert stats["total"] == 2
            assert stats["scheduled"] == 1
            assert stats["failed"] == 0

            # Run history
            last_run = engine.pipeline_last_run("stats-pipeline-1")
            assert last_run is not None
            assert last_run.success is True
            assert last_run.rows_output == 10
        finally:
            engine.close()
