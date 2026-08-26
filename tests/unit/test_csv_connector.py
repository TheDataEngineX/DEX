"""CSV connector tests — conformance + CSV-specific tests."""

from __future__ import annotations

import pytest

from dataenginex.providers.connectors.csv import CsvConnector
from tests.conformance.test_connector import ConnectorConformanceTests


class TestCsvConnector(ConnectorConformanceTests):
    @pytest.fixture()
    def connector(self, tmp_path):  # type: ignore[override]
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("id,name\n1,alice\n2,bob\n")
        return CsvConnector(path=str(tmp_path), default_file="test.csv")

    def test_read_specific_file(self, tmp_path) -> None:
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("x,y\n1,2\n3,4\n")
        conn = CsvConnector(path=str(tmp_path))
        conn.connect()
        result = conn.read(table="data.csv")
        assert len(result) == 2
        conn.disconnect()

    def test_read_missing_file_with_default(self, tmp_path) -> None:
        conn = CsvConnector(path=str(tmp_path))
        conn.connect()
        result = conn.read(table="missing.csv", default=[])
        assert result == []
        conn.disconnect()
