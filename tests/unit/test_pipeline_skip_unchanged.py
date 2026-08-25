"""Tests for skip-if-unchanged: a pipeline whose source content hasn't changed
since its last run should skip transform/quality/load rather than redo
identical work."""

from __future__ import annotations

from pathlib import Path

from dataenginex.config import load_config
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
""")
    return config_file


def test_second_run_with_unchanged_source_is_skipped(tmp_path: Path) -> None:
    _write_csv(tmp_path, "1,Matrix,8.7\n2,Jaws,7.0\n")
    config_file = _write_config(tmp_path)
    data_dir = tmp_path / "data"
    runner = PipelineRunner(load_config(config_file), data_dir=data_dir)

    first = runner.run("bronze_movies")
    assert first.success is True
    assert first.skipped is False
    assert first.steps_completed >= 0
    assert first.rows_output == 2

    second = runner.run("bronze_movies")
    assert second.success is True
    assert second.skipped is True
    assert second.steps_completed == 0
    assert second.rows_output == 2


def test_run_with_changed_source_is_not_skipped(tmp_path: Path) -> None:
    _write_csv(tmp_path, "1,Matrix,8.7\n2,Jaws,7.0\n")
    config_file = _write_config(tmp_path)
    data_dir = tmp_path / "data"
    runner = PipelineRunner(load_config(config_file), data_dir=data_dir)

    first = runner.run("bronze_movies")
    assert first.skipped is False

    _write_csv(tmp_path, "1,Matrix,8.7\n2,Jaws,7.0\n3,Inception,8.8\n")
    second = runner.run("bronze_movies")
    assert second.success is True
    assert second.skipped is False
    assert second.rows_output == 3
