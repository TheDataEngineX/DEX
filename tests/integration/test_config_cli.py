"""Integration tests: config loading + CLI validate end-to-end."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from click.testing import CliRunner

from dataenginex.cli.main import dex
from dataenginex.config.loader import load_config, validate_config

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"


class TestExampleDexYaml:
    """The shipped examples/dex.yaml must load and validate cleanly."""

    def test_load_example_config(self) -> None:
        cfg = load_config(EXAMPLES_DIR / "dex.yaml")
        assert cfg.project.name == "demo-project"

    def test_cross_reference_validation_passes(self) -> None:
        cfg = load_config(EXAMPLES_DIR / "dex.yaml")
        issues = validate_config(cfg)
        # Registry warnings are acceptable; only hard errors are failures
        hard_errors = [e for e in issues if not e.startswith("Warning:")]
        assert hard_errors == []

    def test_sources_populated(self) -> None:
        cfg = load_config(EXAMPLES_DIR / "dex.yaml")
        assert "raw_users" in cfg.data.sources
        assert "raw_events" in cfg.data.sources

    def test_pipelines_populated(self) -> None:
        cfg = load_config(EXAMPLES_DIR / "dex.yaml")
        assert "clean_users" in cfg.data.pipelines
        assert "user_events" in cfg.data.pipelines

    def test_ml_experiment_populated(self) -> None:
        cfg = load_config(EXAMPLES_DIR / "dex.yaml")
        assert "churn_model" in cfg.ml.experiments

    def test_ai_agent_populated(self) -> None:
        cfg = load_config(EXAMPLES_DIR / "dex.yaml")
        assert "assistant" in cfg.ai.agents


MANIFEST = dedent("""\
    apiVersion: dex/v1alpha1
    kind: Project
    metadata:
      name: cli-fixture
    spec:
      profile: lite
      capabilities:
        required: [data.batch]
      limits:
        cpu: 1
        memory: 1GiB
        working_storage: 1GiB
      resources:
        - name: users_csv
          type: dataset
          classification: internal
          config:
            path: data/users.csv
            format: csv
      workloads:
        - name: load_users
          kind: batch
          operations:
            - type: ingest
              name: read_users
              outputs: [users_csv]
""")

REFERENCE_PROJECTS = Path(__file__).resolve().parents[4] / "reference-projects"


class TestCliValidate:
    """``dex validate`` runs the §6.8 compiler (§12.5).

    The command exists so a user can find out whether a project will publish
    *before* trying to publish it. That is only true if it runs the same
    pipeline the publisher runs — a lighter check would certify projects the
    publisher then rejects, which is worse than no check at all.
    """

    def test_validate_a_reference_project(self) -> None:
        runner = CliRunner()
        result = runner.invoke(dex, ["validate", str(REFERENCE_PROJECTS / "store-analytics")])
        assert result.exit_code == 0
        assert "Project is valid" in result.output

    def test_validate_reports_the_content_hash(self) -> None:
        """§6.8 stage 11 output. Without it the user cannot tell two
        successful validations of different content apart."""
        runner = CliRunner()
        result = runner.invoke(dex, ["validate", str(REFERENCE_PROJECTS / "store-analytics")])
        assert "sha256:" in result.output

    def test_a_manifest_path_works_too(self, tmp_path: Path) -> None:
        """The project is the directory, but `dex validate dex.yaml` is what
        fingers type. Accepting it costs one line."""
        (tmp_path / "dex.yaml").write_text(MANIFEST)
        runner = CliRunner()
        result = runner.invoke(dex, ["validate", str(tmp_path / "dex.yaml")])
        assert result.exit_code == 0

    def test_validate_missing_path(self) -> None:
        runner = CliRunner()
        result = runner.invoke(dex, ["validate", "/nonexistent/project"])
        assert result.exit_code != 0

    def test_an_invalid_project_fails_closed(self, tmp_path: Path) -> None:
        """Exit code, not just output. A script that gates a deploy on
        ``dex validate`` reads the status, and a validator that prints errors
        while exiting zero is a validator that certifies broken projects."""
        (tmp_path / "dex.yaml").write_text(MANIFEST.replace("profile: lite", "profile: nonsense"))
        runner = CliRunner()
        result = runner.invoke(dex, ["validate", str(tmp_path)])
        assert result.exit_code == 1
        assert "not publishable" in result.output

    def test_a_broken_yaml_file_is_an_error_not_a_traceback(self, tmp_path: Path) -> None:
        (tmp_path / "dex.yaml").write_text(": : : invalid {{{")
        runner = CliRunner()
        result = runner.invoke(dex, ["validate", str(tmp_path)])
        assert result.exit_code == 1


class TestCliVersion:
    def test_version_output(self) -> None:
        runner = CliRunner()
        result = runner.invoke(dex, ["version"])
        assert result.exit_code == 0
        assert "DataEngineX" in result.output

    def test_version_flag(self) -> None:
        runner = CliRunner()
        result = runner.invoke(dex, ["--version"])
        assert result.exit_code == 0
