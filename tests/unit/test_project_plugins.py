"""Tests for dataenginex.domains.plugins.loader.load_project_plugins."""

from __future__ import annotations

from pathlib import Path

from dataenginex.domains.plugins.loader import load_project_plugins


def test_no_plugins_dir_returns_empty_list(tmp_path: Path) -> None:
    result = load_project_plugins(tmp_path)
    assert result == []


def test_loads_python_files_from_plugins_dir(tmp_path: Path) -> None:
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    (plugins_dir / "my_plugin.py").write_text("LOADED = True\n")

    result = load_project_plugins(tmp_path)

    assert result == ["_dex_project_plugin_my_plugin"]


def test_skips_files_starting_with_underscore(tmp_path: Path) -> None:
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    (plugins_dir / "_helper.py").write_text("x = 1\n")

    result = load_project_plugins(tmp_path)

    assert result == []


def test_broken_plugin_is_skipped_not_raised(tmp_path: Path) -> None:
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    (plugins_dir / "broken.py").write_text("raise RuntimeError('boom')\n")
    (plugins_dir / "good.py").write_text("x = 1\n")

    result = load_project_plugins(tmp_path)

    assert result == ["_dex_project_plugin_good"]
