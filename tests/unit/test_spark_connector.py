"""Tests for dataenginex.providers.connectors.spark — SparkConnector."""

from __future__ import annotations

import sys
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


def _make_arrow_table(data: list[dict[str, Any]]) -> MagicMock:
    """Build a mock Arrow table from a list of dicts."""
    cols: list[str] = list(data[0].keys()) if data else []
    mock = MagicMock()
    mock.schema.names = cols
    mock.num_rows = len(data)
    mock.to_pydict.return_value = {col: [row[col] for row in data] for col in cols}
    return mock


def _make_spark_session(arrow_table: MagicMock | None = None) -> MagicMock:
    """Return a mock SparkSession whose read chain returns a mock DataFrame."""
    session = MagicMock()
    df = MagicMock()
    df.toArrow.return_value = arrow_table or _make_arrow_table([])

    reader = MagicMock()
    reader.format.return_value = reader
    reader.options.return_value = reader
    reader.load.return_value = df
    session.read = reader

    builder = MagicMock()
    builder.master.return_value = builder
    builder.appName.return_value = builder
    builder.getOrCreate.return_value = session
    session.builder = builder

    return session


@pytest.fixture()
def mock_pyspark(monkeypatch: pytest.MonkeyPatch) -> Generator[MagicMock]:
    """Inject a mock pyspark into sys.modules and patch _PYSPARK_AVAILABLE."""
    mock_session = _make_spark_session()

    mock_pyspark_mod = MagicMock()
    mock_sql_mod = MagicMock()
    mock_sql_mod.SparkSession = MagicMock()
    mock_sql_mod.SparkSession.builder = mock_session.builder

    with (
        patch.dict(sys.modules, {"pyspark": mock_pyspark_mod, "pyspark.sql": mock_sql_mod}),
        patch("dataenginex.providers.connectors.spark._PYSPARK_AVAILABLE", True),
    ):
        yield mock_session


class TestSparkConnectorImportGuard:
    def test_raises_import_error_when_pyspark_missing(self) -> None:
        with patch("dataenginex.providers.connectors.spark._PYSPARK_AVAILABLE", False):
            from dataenginex.providers.connectors.spark import SparkConnector

            with pytest.raises(ImportError, match="PySpark is required"):
                SparkConnector()

    def test_registered_as_spark(self) -> None:
        from dataenginex.providers.connectors import connector_registry
        from dataenginex.providers.connectors.spark import SparkConnector

        assert connector_registry.get("spark") is SparkConnector


class TestSparkConnectorConnect:
    def test_connect_creates_session(self, mock_pyspark: MagicMock) -> None:
        with patch("dataenginex.providers.connectors.spark._PYSPARK_AVAILABLE", True):
            from dataenginex.providers.connectors.spark import SparkConnector

            c = SparkConnector()
            c.connect()
            assert c._spark is not None
            c.disconnect()

    def test_disconnect_clears_session(self, mock_pyspark: MagicMock) -> None:
        with patch("dataenginex.providers.connectors.spark._PYSPARK_AVAILABLE", True):
            from dataenginex.providers.connectors.spark import SparkConnector

            c = SparkConnector()
            c.connect()
            c.disconnect()
            assert c._spark is None

    def test_disconnect_before_connect_is_safe(self, mock_pyspark: MagicMock) -> None:
        with patch("dataenginex.providers.connectors.spark._PYSPARK_AVAILABLE", True):
            from dataenginex.providers.connectors.spark import SparkConnector

            SparkConnector().disconnect()  # should not raise

    def test_health_check_false_when_disconnected(self, mock_pyspark: MagicMock) -> None:
        with patch("dataenginex.providers.connectors.spark._PYSPARK_AVAILABLE", True):
            from dataenginex.providers.connectors.spark import SparkConnector

            assert SparkConnector().health_check() is False


class TestSparkConnectorRead:
    def _connector(self, path: str | None = "/data/test") -> Any:
        from dataenginex.providers.connectors.spark import SparkConnector

        return SparkConnector(path=path)

    def test_read_not_connected_raises(self, mock_pyspark: MagicMock) -> None:
        with patch("dataenginex.providers.connectors.spark._PYSPARK_AVAILABLE", True):
            c = self._connector()
            with pytest.raises(RuntimeError, match="Not connected"):
                c.read()

    def test_read_no_path_raises(self, mock_pyspark: MagicMock) -> None:
        with patch("dataenginex.providers.connectors.spark._PYSPARK_AVAILABLE", True):
            c = self._connector(path=None)
            c._spark = mock_pyspark
            with pytest.raises(ValueError, match="No path specified"):
                c.read()

    def test_read_returns_rows(self, mock_pyspark: MagicMock) -> None:
        data = [{"id": 1, "name": "alice"}, {"id": 2, "name": "bob"}]
        arrow = _make_arrow_table(data)

        df = MagicMock()
        df.toArrow.return_value = arrow
        mock_pyspark.read.format.return_value.options.return_value.load.return_value = df
        mock_pyspark.read.format.return_value.load.return_value = df

        with patch("dataenginex.providers.connectors.spark._PYSPARK_AVAILABLE", True):
            c = self._connector()
            c._spark = mock_pyspark
            rows = c.read()

        assert len(rows) == 2
        assert rows[0]["name"] == "alice"

    def test_read_uses_table_arg_over_path(self, mock_pyspark: MagicMock) -> None:
        data = [{"x": 42}]
        arrow = _make_arrow_table(data)

        df = MagicMock()
        df.toArrow.return_value = arrow
        mock_pyspark.read.format.return_value.options.return_value.load.return_value = df
        mock_pyspark.read.format.return_value.load.return_value = df

        with patch("dataenginex.providers.connectors.spark._PYSPARK_AVAILABLE", True):
            c = self._connector(path="/default/path")
            c._spark = mock_pyspark
            c.read(table="/override/path")

        mock_pyspark.read.format.return_value.load.assert_called_with("/override/path")

    def test_read_exception_with_default_returns_default(self, mock_pyspark: MagicMock) -> None:
        mock_pyspark.read.format.return_value.load.side_effect = RuntimeError("oops")
        mock_pyspark.read.format.return_value.options.return_value.load.side_effect = RuntimeError(
            "oops"
        )

        with patch("dataenginex.providers.connectors.spark._PYSPARK_AVAILABLE", True):
            c = self._connector()
            c._spark = mock_pyspark
            assert c.read(default=[]) == []

    def test_read_exception_without_default_raises(self, mock_pyspark: MagicMock) -> None:
        mock_pyspark.read.format.return_value.load.side_effect = RuntimeError("boom")
        mock_pyspark.read.format.return_value.options.return_value.load.side_effect = RuntimeError(
            "boom"
        )

        with patch("dataenginex.providers.connectors.spark._PYSPARK_AVAILABLE", True):
            c = self._connector()
            c._spark = mock_pyspark
            with pytest.raises(RuntimeError, match="boom"):
                c.read()


class TestSparkConnectorWrite:
    def test_write_not_connected_raises(self, mock_pyspark: MagicMock) -> None:
        with patch("dataenginex.providers.connectors.spark._PYSPARK_AVAILABLE", True):
            from dataenginex.providers.connectors.spark import SparkConnector

            c = SparkConnector()
            with pytest.raises(RuntimeError, match="Not connected"):
                c.write([{"x": 1}])

    def test_write_unsupported_type_raises(self, mock_pyspark: MagicMock) -> None:
        with patch("dataenginex.providers.connectors.spark._PYSPARK_AVAILABLE", True):
            from dataenginex.providers.connectors.spark import SparkConnector

            c = SparkConnector()
            c._spark = mock_pyspark
            with pytest.raises(TypeError):
                c.write("not a list")

    def test_write_empty_list_is_noop(self, mock_pyspark: MagicMock) -> None:
        with patch("dataenginex.providers.connectors.spark._PYSPARK_AVAILABLE", True):
            from dataenginex.providers.connectors.spark import SparkConnector

            c = SparkConnector()
            c._spark = mock_pyspark
            c.write([])  # should not raise or call createDataFrame
            mock_pyspark.createDataFrame.assert_not_called()
