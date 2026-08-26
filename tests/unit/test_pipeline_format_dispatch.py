"""Tests for target.format dispatch (Parquet vs Delta) in PipelineRunner."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("deltalake")

from dataenginex.config import load_config  # noqa: E402
from dataenginex.domains.data.pipeline.runner import PipelineRunner


def _write_csv(tmp_path: Path) -> None:
    (tmp_path / "movies.csv").write_text(
        "id,title,rating\n1,Matrix,8.7\n2,Jaws,7.0\n3,Inception,8.8\n4,Bad Movie,2.0\n"
    )


def _write_config(tmp_path: Path, pipelines_yaml: str) -> Path:
    config_file = tmp_path / "dex.yaml"
    config_file.write_text(f"""
project:
  name: test-project

data:
  sources:
    movies:
      type: csv
      connection:
        path: "{tmp_path}"
        default_file: "movies.csv"
  pipelines:
{pipelines_yaml}
""")
    return config_file


def test_load_writes_audit_columns_without_intermediate_table(tmp_path: Path) -> None:
    """Regression test for the sink-write memory fix: audit columns
    (_dex_ingested_at, _dex_pipeline, _dex_layer, _dex_source) are appended
    directly in the COPY's SELECT rather than via an intermediate
    "..._with_meta" DuckDB table — verify the output still carries the
    correct values with the new single-pass query."""
    import duckdb

    _write_csv(tmp_path)
    config_file = _write_config(
        tmp_path,
        """    bronze_movies:
      source: movies
      target:
        layer: bronze
""",
    )
    data_dir = tmp_path / "data"
    runner = PipelineRunner(load_config(config_file), data_dir=data_dir)
    result = runner.run("bronze_movies")

    assert result.success is True
    out_path = data_dir / "bronze" / "bronze_movies.parquet"
    con = duckdb.connect(":memory:")
    row = con.execute(
        f"SELECT DISTINCT _dex_pipeline, _dex_layer, _dex_source FROM '{out_path}'"
    ).fetchone()
    assert row == ("bronze_movies", "bronze", "movies")
    ingested_count = con.execute(
        f"SELECT count(*) FROM '{out_path}' WHERE _dex_ingested_at IS NOT NULL"
    ).fetchone()
    assert ingested_count == (4,)


def test_no_format_key_still_writes_parquet(tmp_path: Path) -> None:
    """A pipeline with no target.format keeps writing Parquet — zero behavior change."""
    _write_csv(tmp_path)
    config_file = _write_config(
        tmp_path,
        """    bronze_movies:
      source: movies
      target:
        layer: bronze
""",
    )
    data_dir = tmp_path / "data"
    runner = PipelineRunner(load_config(config_file), data_dir=data_dir)
    result = runner.run("bronze_movies")

    assert result.success is True
    assert (data_dir / "bronze" / "bronze_movies.parquet").exists()
    assert not (data_dir / "bronze" / "bronze_movies").exists()


def test_format_delta_writes_delta_table(tmp_path: Path) -> None:
    """target.format: delta writes a Delta table directory, not a .parquet file."""
    _write_csv(tmp_path)
    config_file = _write_config(
        tmp_path,
        """    bronze_movies:
      source: movies
      target:
        layer: bronze
        format: delta
""",
    )
    data_dir = tmp_path / "data"
    runner = PipelineRunner(load_config(config_file), data_dir=data_dir)
    result = runner.run("bronze_movies")

    assert result.success is True
    assert result.rows_output == 4
    delta_dir = data_dir / "bronze" / "bronze_movies"
    assert (delta_dir / "_delta_log").exists()
    assert not (data_dir / "bronze" / "bronze_movies.parquet").exists()


def test_downstream_pipeline_reads_delta_written_upstream(tmp_path: Path) -> None:
    """A downstream pipeline can extract a Delta-written upstream pipeline's output."""
    _write_csv(tmp_path)
    config_file = _write_config(
        tmp_path,
        """    bronze_movies:
      source: movies
      target:
        layer: bronze
        format: delta
    silver_movies:
      source: bronze_movies
      depends_on: [bronze_movies]
      target:
        layer: silver
""",
    )
    data_dir = tmp_path / "data"
    runner = PipelineRunner(load_config(config_file), data_dir=data_dir)

    bronze_result = runner.run("bronze_movies")
    assert bronze_result.success is True

    silver_result = runner.run("silver_movies")
    assert silver_result.success is True, silver_result.error
    assert silver_result.rows_input == 4
    assert silver_result.rows_output == 4
    # Downstream pipeline itself has no format override — stays Parquet.
    assert (data_dir / "silver" / "silver_movies.parquet").exists()
