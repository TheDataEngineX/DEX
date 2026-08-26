"""Tests for CLI commands: main, run, train, studio, secops, runtime."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from dataenginex.cli.main import dex, version
from dataenginex.cli.runtime import runtime
from dataenginex.cli.secops import secops


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


class TestMainDex:
    def test_dex_help(self, runner: CliRunner) -> None:
        result = runner.invoke(dex, ["--help"])
        assert result.exit_code == 0
        assert "DataEngineX" in result.output

    def test_dex_version(self, runner: CliRunner) -> None:
        result = runner.invoke(dex, ["version"])
        assert result.exit_code == 0
        assert "DataEngineX" in result.output

    def test_version_command(self, runner: CliRunner) -> None:
        result = runner.invoke(version)
        assert result.exit_code == 0
        assert "DataEngineX" in result.output
        assert "Python" in result.output


class TestSecops:
    def test_secops_help(self, runner: CliRunner) -> None:
        result = runner.invoke(secops, ["--help"])
        assert result.exit_code == 0
        assert "PrivacyGuard" in result.output

    def test_secops_scan_no_pii(self, runner: CliRunner) -> None:
        result = runner.invoke(secops, ["scan", "hello world"])
        assert result.exit_code == 0
        assert "No PII detected" in result.output

    def test_secops_scan_with_pii(self, runner: CliRunner) -> None:
        result = runner.invoke(secops, ["scan", "my email is test@example.com"])
        assert result.exit_code == 0
        assert "Detections" in result.output

    @patch("dataenginex.bootstrap.build_lite_gateway")
    @patch("dataenginex.bootstrap.open_control_store")
    def test_secops_status_empty(
        self, mock_store_fn: MagicMock, mock_gateway_fn: MagicMock, runner: CliRunner
    ) -> None:
        mock_store_inst = MagicMock()
        mock_store_fn.return_value.__enter__ = MagicMock(return_value=mock_store_inst)
        mock_store_fn.return_value.__exit__ = MagicMock(return_value=False)
        mock_gw = MagicMock()
        mock_gw.list_policies.return_value = MagicMock(items=[])
        mock_gateway_fn.return_value = mock_gw

        result = runner.invoke(secops, ["status"])
        assert result.exit_code == 0


class TestRuntime:
    def test_runtime_help(self, runner: CliRunner) -> None:
        result = runner.invoke(runtime, ["--help"])
        assert result.exit_code == 0
        assert "control plane" in result.output.lower()

    @patch("dataenginex.runtime.queue.Scheduler")
    @patch("dataenginex.bootstrap.build_lite_gateway")
    @patch("dataenginex.bootstrap.open_control_store")
    def test_runtime_serve_once(
        self,
        mock_store_fn: MagicMock,
        mock_gateway_fn: MagicMock,
        mock_scheduler_cls: MagicMock,
        runner: CliRunner,
    ) -> None:
        mock_store_inst = MagicMock()
        mock_store_fn.return_value.__enter__ = MagicMock(return_value=mock_store_inst)
        mock_store_fn.return_value.__exit__ = MagicMock(return_value=False)
        mock_gw = MagicMock()
        mock_gateway_fn.return_value = mock_gw
        mock_sched = MagicMock()
        mock_scheduler_cls.return_value = mock_sched

        result = runner.invoke(runtime, ["serve", "--once"])
        assert result.exit_code == 0

    @patch("dataenginex.bootstrap.build_lite_gateway")
    @patch("dataenginex.bootstrap.open_control_store")
    def test_runtime_tick(
        self, mock_store_fn: MagicMock, mock_gateway_fn: MagicMock, runner: CliRunner
    ) -> None:
        mock_store_inst = MagicMock()
        mock_store_fn.return_value.__enter__ = MagicMock(return_value=mock_store_inst)
        mock_store_fn.return_value.__exit__ = MagicMock(return_value=False)
        mock_gw = MagicMock()
        mock_gw.schedules.tick.return_value = []
        mock_gateway_fn.return_value = mock_gw

        result = runner.invoke(runtime, ["tick"])
        assert result.exit_code == 0
        assert "Nothing due" in result.output

    @patch("dataenginex.bootstrap.build_lite_gateway")
    @patch("dataenginex.bootstrap.open_control_store")
    def test_runtime_schedules_empty(
        self, mock_store_fn: MagicMock, mock_gateway_fn: MagicMock, runner: CliRunner
    ) -> None:
        mock_store_inst = MagicMock()
        mock_store_fn.return_value.__enter__ = MagicMock(return_value=mock_store_inst)
        mock_store_fn.return_value.__exit__ = MagicMock(return_value=False)
        mock_gw = MagicMock()
        mock_gw.schedules.list_for_project.return_value = []
        mock_gateway_fn.return_value = mock_gw

        result = runner.invoke(runtime, ["schedules", "--project", "proj_test"])
        assert result.exit_code == 0
        assert "No schedules" in result.output
