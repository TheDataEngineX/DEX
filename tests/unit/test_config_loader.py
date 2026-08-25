"""Tests for YAML config loading, env var resolution, and layering."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from dataenginex.config.loader import load_config, resolve_env_vars, validate_config
from dataenginex.config.schema import (
    AgentConfig,
    AiConfig,
    DexConfig,
    MlConfig,
    ProjectConfig,
    TrackerConfig,
)
from dataenginex.foundation.errors import ConfigError


class TestResolveEnvVars:
    def test_simple_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DB_HOST", "localhost")
        result = resolve_env_vars("host: ${DB_HOST}")
        assert result == "host: localhost"

    def test_var_with_default(self) -> None:
        result = resolve_env_vars("port: ${UNSET_PORT_XYZ:-5432}")
        assert result == "port: 5432"

    def test_var_with_default_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_PORT", "9999")
        result = resolve_env_vars("port: ${MY_PORT:-5432}")
        assert result == "port: 9999"

    def test_multiple_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOST", "db.local")
        monkeypatch.setenv("PORT", "3306")
        result = resolve_env_vars("url: ${HOST}:${PORT}")
        assert result == "url: db.local:3306"

    def test_unset_var_no_default_raises(self) -> None:
        with pytest.raises(ConfigError, match="NONEXISTENT_VAR_ABC"):
            resolve_env_vars("val: ${NONEXISTENT_VAR_ABC}")


class TestLoadConfig:
    def test_load_minimal_yaml(self, tmp_path: Path) -> None:
        yaml_content = dedent("""\
            project:
              name: test-project
        """)
        config_file = tmp_path / "dex.yaml"
        config_file.write_text(yaml_content)

        cfg = load_config(config_file)
        assert isinstance(cfg, DexConfig)
        assert cfg.project.name == "test-project"

    def test_load_with_env_vars(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PROJECT_NAME", "env-project")
        yaml_content = dedent("""\
            project:
              name: ${PROJECT_NAME}
        """)
        config_file = tmp_path / "dex.yaml"
        config_file.write_text(yaml_content)

        cfg = load_config(config_file)
        assert cfg.project.name == "env-project"

    def test_load_nonexistent_file_raises(self) -> None:
        with pytest.raises(ConfigError, match="not found"):
            load_config(Path("/nonexistent/dex.yaml"))

    def test_load_invalid_yaml_raises(self, tmp_path: Path) -> None:
        config_file = tmp_path / "dex.yaml"
        config_file.write_text(": : : invalid yaml {{{")

        with pytest.raises(ConfigError, match="parse"):
            load_config(config_file)

    def test_load_with_overlay(self, tmp_path: Path) -> None:
        base = dedent("""\
            project:
              name: my-app
              version: "1.0"
        """)
        overlay = dedent("""\
            project:
              version: "2.0"
        """)
        (tmp_path / "dex.yaml").write_text(base)
        (tmp_path / "dex.prod.yaml").write_text(overlay)

        cfg = load_config(
            tmp_path / "dex.yaml",
            overlay=tmp_path / "dex.prod.yaml",
        )
        assert cfg.project.name == "my-app"
        assert cfg.project.version == "2.0"


class TestValidateConfig:
    def test_valid_config_returns_empty(self) -> None:
        cfg = DexConfig(
            project={"name": "valid"},  # type: ignore[arg-type]
        )
        errors = validate_config(cfg)
        assert errors == []

    def test_pipeline_references_missing_source(self) -> None:
        cfg = DexConfig(
            project={"name": "bad-ref"},  # type: ignore[arg-type]
            data={  # type: ignore[arg-type]
                "pipelines": {
                    "clean": {
                        "source": "nonexistent_source",
                        "transforms": [],
                    }
                }
            },
        )
        errors = validate_config(cfg)
        assert any("nonexistent_source" in e for e in errors)


class TestRegistryValidation:
    def test_warns_on_unknown_tracker_backend(self) -> None:
        config = DexConfig(
            project=ProjectConfig(name="test"),
            ml=MlConfig(tracking=TrackerConfig(backend="nonexistent")),
        )
        warnings = validate_config(config)
        assert any("tracker" in w.lower() and "nonexistent" in w for w in warnings)

    def test_warns_on_unknown_agent_runtime(self) -> None:
        config = DexConfig(
            project=ProjectConfig(name="test"),
            ai=AiConfig(
                agents={"bot": AgentConfig(runtime="nonexistent", system_prompt="test")},
            ),
        )
        warnings = validate_config(config)
        assert any("agent" in w.lower() and "nonexistent" in w for w in warnings)

    def test_valid_config_no_warnings(self) -> None:
        config = DexConfig(project=ProjectConfig(name="test"))
        warnings = validate_config(config)
        assert len(warnings) == 0
