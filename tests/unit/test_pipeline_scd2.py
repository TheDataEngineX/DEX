"""Tests for SCD Type 2 loading: history preserved via _dex_valid_from/
_dex_valid_to/_dex_is_current, keyed by target.scd_key."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

pytest.importorskip("deltalake")

from dataenginex.config import load_config  # noqa: E402
from dataenginex.domains.data.pipeline.runner import PipelineRunner


def _write_csv(tmp_path: Path, rows: str) -> None:
    (tmp_path / "movies.csv").write_text(f"id,title,rating\n{rows}")


def _write_config(tmp_path: Path) -> Path:
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
    bronze_movies:
      source: movies
      target:
        layer: bronze
        format: delta
        scd_type: "2"
        scd_key: id
""")
    return config_file


def test_scd2_first_run_marks_all_rows_current(tmp_path: Path) -> None:
    _write_csv(tmp_path, "1,Matrix,8.7\n2,Jaws,7.0\n")
    config_file = _write_config(tmp_path)
    data_dir = tmp_path / "data"
    runner = PipelineRunner(load_config(config_file), data_dir=data_dir)

    result = runner.run("bronze_movies")
    assert result.success is True

    out_path = data_dir / "bronze" / "bronze_movies"
    con = duckdb.connect(":memory:")
    rows = con.execute(
        f"SELECT id, rating, _dex_is_current FROM delta_scan('{out_path}') ORDER BY id"
    ).fetchall()
    assert rows == [(1, 8.7, True), (2, 7.0, True)]


def _write_config_no_scd(tmp_path: Path) -> Path:
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
    bronze_movies:
      source: movies
      target:
        layer: bronze
        format: delta
""")
    return config_file


def test_scd2_enabled_on_table_with_no_prior_history_starts_fresh(tmp_path: Path) -> None:
    """A table written before scd_type was turned on has no _dex_is_current
    column yet — enabling SCD2 must not crash trying to read it as history."""
    _write_csv(tmp_path, "1,Matrix,8.7\n2,Jaws,7.0\n")
    data_dir = tmp_path / "data"
    runner = PipelineRunner(load_config(_write_config_no_scd(tmp_path)), data_dir=data_dir)
    first = runner.run("bronze_movies")
    assert first.success is True

    _write_csv(tmp_path, "1,Matrix,9.5\n2,Jaws,7.0\n")
    runner = PipelineRunner(load_config(_write_config(tmp_path)), data_dir=data_dir)
    second = runner.run("bronze_movies")
    assert second.success is True

    out_path = data_dir / "bronze" / "bronze_movies"
    con = duckdb.connect(":memory:")
    rows = con.execute(
        f"SELECT id, rating, _dex_is_current FROM delta_scan('{out_path}') ORDER BY id"
    ).fetchall()
    assert rows == [(1, 9.5, True), (2, 7.0, True)]


def test_scd2_changed_row_gets_new_version_old_row_closed(tmp_path: Path) -> None:
    _write_csv(tmp_path, "1,Matrix,8.7\n2,Jaws,7.0\n")
    config_file = _write_config(tmp_path)
    data_dir = tmp_path / "data"
    runner = PipelineRunner(load_config(config_file), data_dir=data_dir)
    runner.run("bronze_movies")

    # Rating for id=1 changes; id=2 unchanged; id=3 is new.
    _write_csv(tmp_path, "1,Matrix,9.5\n2,Jaws,7.0\n3,Inception,8.8\n")
    result = runner.run("bronze_movies")
    assert result.success is True

    out_path = data_dir / "bronze" / "bronze_movies"
    con = duckdb.connect(":memory:")
    rows = con.execute(
        f"SELECT id, rating, _dex_is_current FROM delta_scan('{out_path}') "
        "ORDER BY id, _dex_is_current"
    ).fetchall()

    # id=1: one closed historical row (old rating) + one current row (new rating)
    id1_rows = [r for r in rows if r[0] == 1]
    assert len(id1_rows) == 2
    assert (1, 8.7, False) in id1_rows
    assert (1, 9.5, True) in id1_rows

    # id=2: unchanged, still a single current row
    id2_rows = [r for r in rows if r[0] == 2]
    assert id2_rows == [(2, 7.0, True)]

    # id=3: new, single current row
    id3_rows = [r for r in rows if r[0] == 3]
    assert id3_rows == [(3, 8.8, True)]
