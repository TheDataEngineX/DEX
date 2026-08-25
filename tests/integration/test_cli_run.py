"""Integration tests for `dex run` command."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from dataenginex.cli.main import dex


class TestDexRun:
    def test_run_single_pipeline(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "movies.csv"
        csv_file.write_text("id,title,rating\n1,Matrix,8.7\n2,Jaws,7.0\n")

        config = tmp_path / "dex.yaml"
        config.write_text(f"""
project:
  name: test
data:
  sources:
    movies:
      type: csv
      connection:
        path: "{tmp_path}"
        default_file: movies.csv
  pipelines:
    ingest:
      source: movies
      transforms:
        - type: filter
          condition: "rating > 5.0"
      target:
        layer: silver
""")
        runner = CliRunner()
        args = ["run", "ingest", "--config", str(config), "--state-dir", str(tmp_path / "state")]
        result = runner.invoke(dex, args)
        assert result.exit_code == 0

    def test_run_all(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,value\n1,10\n2,20\n")

        config = tmp_path / "dex.yaml"
        config.write_text(f"""
project:
  name: test
data:
  sources:
    src:
      type: csv
      connection:
        path: "{tmp_path}"
        default_file: data.csv
  pipelines:
    p1:
      source: src
      target:
        layer: silver
""")
        runner = CliRunner()
        args = ["run", "--all", "--config", str(config), "--state-dir", str(tmp_path / "state")]
        result = runner.invoke(dex, args)
        assert result.exit_code == 0

    def test_run_dry_run(self, tmp_path: Path) -> None:
        config = tmp_path / "dex.yaml"
        config.write_text("""
project:
  name: test
data:
  sources:
    src:
      type: csv
      connection:
        path: "."
        default_file: data.csv
  pipelines:
    p1:
      source: src
      target:
        layer: silver
""")
        runner = CliRunner()
        result = runner.invoke(dex, ["run", "p1", "--dry-run", "--config", str(config)])
        assert result.exit_code == 0
        assert "dry" in result.output.lower()

    def test_run_missing_pipeline(self, tmp_path: Path) -> None:
        config = tmp_path / "dex.yaml"
        config.write_text("""
project:
  name: test
data:
  sources: {}
  pipelines: {}
""")
        runner = CliRunner()
        result = runner.invoke(dex, ["run", "nonexistent", "--config", str(config)])
        assert result.exit_code != 0
