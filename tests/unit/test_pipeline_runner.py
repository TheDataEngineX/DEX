"""Tests for PipelineRunner."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from dataenginex.config import load_config
from dataenginex.config.schema import DexConfig
from dataenginex.domains.data.pipeline.runner import PipelineRunner


class RecordingLineage:
    """Minimal ``LineageBackend`` that keeps events in memory.

    The runner is what is under test here, not a storage backend, so this
    asserts against the protocol the runner actually calls rather than against
    whichever concrete store happens to be wired in.
    """

    def __init__(self) -> None:
        self.events: list[SimpleNamespace] = []

    def record(self, **kwargs: object) -> SimpleNamespace:
        event = SimpleNamespace(**kwargs)
        self.events.append(event)
        return event

    @property
    def all_events(self) -> list[SimpleNamespace]:
        return self.events


@pytest.fixture()
def sample_config(tmp_path: Path) -> Path:
    """Create a minimal dex.yaml with a CSV pipeline."""
    csv_file = tmp_path / "movies.csv"
    csv_file.write_text(
        "id,title,rating\n1,Matrix,8.7\n2,Jaws,7.0\n3,Inception,8.8\n4,Bad Movie,2.0\n"
    )

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
    return config_file


@pytest.fixture()
def simple_config(sample_config: Path) -> DexConfig:
    """Return a loaded DexConfig from the sample config path."""
    return load_config(sample_config)


class TestPipelineRunner:
    def test_run_single_pipeline(self, sample_config: Path, tmp_path: Path) -> None:
        config = load_config(sample_config)
        runner = PipelineRunner(config, data_dir=tmp_path / "data")
        result = runner.run("ingest-movies")
        assert result.success is True
        assert result.rows_output > 0

    def test_run_pipeline_not_found(self, sample_config: Path, tmp_path: Path) -> None:
        config = load_config(sample_config)
        runner = PipelineRunner(config, data_dir=tmp_path / "data")
        with pytest.raises(KeyError, match="nonexistent"):
            runner.run("nonexistent")

    def test_run_all_pipelines(self, sample_config: Path, tmp_path: Path) -> None:
        config = load_config(sample_config)
        runner = PipelineRunner(config, data_dir=tmp_path / "data")
        results = runner.run_all()
        assert len(results) == 1
        assert all(r.success for r in results.values())

    def test_dry_run(self, sample_config: Path, tmp_path: Path) -> None:
        config = load_config(sample_config)
        runner = PipelineRunner(config, data_dir=tmp_path / "data")
        result = runner.run("ingest-movies", dry_run=True)
        assert result.success is True
        assert result.dry_run is True

    def test_output_parquet_written(self, sample_config: Path, tmp_path: Path) -> None:
        config = load_config(sample_config)
        data_dir = tmp_path / "data"
        runner = PipelineRunner(config, data_dir=data_dir)
        runner.run("ingest-movies")
        assert (data_dir / "silver" / "ingest-movies.parquet").exists()

    def test_lineage_recorded_on_run(self, tmp_path: Path, simple_config: DexConfig) -> None:
        """Pipeline run records lineage events for extract and load."""
        lineage = RecordingLineage()
        runner = PipelineRunner(simple_config, data_dir=tmp_path / "lakehouse", lineage=lineage)
        result = runner.run("ingest-movies")
        assert result.success
        events = lineage.all_events
        assert len(events) >= 2
        ops = [e.operation for e in events]
        assert "ingest" in ops
        assert "load" in ops
