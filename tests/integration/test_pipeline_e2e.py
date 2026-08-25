"""End-to-end pipeline integration test."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from dataenginex.config import load_config
from dataenginex.domains.data.pipeline.runner import PipelineRunner


@pytest.fixture()
def e2e_config(tmp_path: Path) -> Path:
    """Full pipeline config with real CSV data."""
    movies_csv = Path("examples/movies.csv")
    if not movies_csv.exists():
        pytest.skip("examples/movies.csv not found")
    shutil.copy(movies_csv, tmp_path / "movies.csv")

    config = tmp_path / "dex.yaml"
    config.write_text(f"""
project:
  name: e2e-test
  version: "0.1.0"

data:
  sources:
    movies:
      type: csv
      connection:
        path: "{tmp_path}"
        default_file: movies.csv
  pipelines:
    ingest-movies:
      source: movies
      transforms:
        - type: filter
          condition: "rating > 5.0"
        - type: deduplicate
          key: id
      quality:
        completeness: 0.9
        uniqueness:
          - id
      target:
        layer: silver
""")
    return config


class TestPipelineE2E:
    def test_full_pipeline_csv_to_silver(self, e2e_config: Path, tmp_path: Path) -> None:
        config = load_config(e2e_config)
        data_dir = tmp_path / "lakehouse"
        runner = PipelineRunner(config, data_dir=data_dir)
        result = runner.run("ingest-movies")

        assert result.success is True
        assert result.rows_input == 8  # all rows from CSV
        assert result.rows_output == 6  # filtered: rating > 5.0 removes 2
        assert (data_dir / "silver" / "ingest-movies.parquet").exists()
