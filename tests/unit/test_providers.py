"""Tests for providers: REST, SSE, Delta, MLflow registry."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from dataenginex.providers.connectors.rest import RestApiConnector
from dataenginex.providers.connectors.sse import SseConnector


class TestRestApiConnector:
    def test_init(self) -> None:
        conn = RestApiConnector(url="https://example.com/api")
        assert conn._url == "https://example.com/api"
        assert conn._max_pages == 1

    def test_write_raises(self) -> None:
        conn = RestApiConnector(url="https://example.com/api")
        with pytest.raises(NotImplementedError, match="read-only"):
            conn.write([{"a": 1}])

    def test_read_not_connected(self) -> None:
        conn = RestApiConnector(url="https://example.com/api")
        with pytest.raises(RuntimeError, match="not connected"):
            conn.read()

    def test_extract_records_with_key(self) -> None:
        conn = RestApiConnector(url="https://example.com/api", records_key="results")
        body = {"results": [{"id": 1}, {"id": 2}], "total": 2}
        records = conn._extract_records(body)
        assert len(records) == 2

    def test_extract_records_list(self) -> None:
        conn = RestApiConnector(url="https://example.com/api")
        records = conn._extract_records([{"id": 1}, {"id": 2}])
        assert len(records) == 2

    def test_extract_records_dict_no_key(self) -> None:
        conn = RestApiConnector(url="https://example.com/api")
        records = conn._extract_records({"id": 1})
        assert records == [{"id": 1}]

    def test_extract_records_empty_key(self) -> None:
        conn = RestApiConnector(url="https://example.com/api", records_key="items")
        records = conn._extract_records({"other": []})
        assert records == []

    def test_health_check_success(self) -> None:
        conn = RestApiConnector(url="https://example.com/api")
        with patch("dataenginex.providers.connectors.rest.httpx") as mock_httpx:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_httpx.head.return_value = mock_resp
            assert conn.health_check() is True

    def test_health_check_failure(self) -> None:
        conn = RestApiConnector(url="https://example.com/api")
        with patch("dataenginex.providers.connectors.rest.httpx") as mock_httpx:
            mock_httpx.head.side_effect = Exception("connection refused")
            assert conn.health_check() is False

    def test_connect_disconnect(self) -> None:
        conn = RestApiConnector(url="https://example.com/api")
        conn.connect()
        assert conn._client is not None
        conn.disconnect()
        assert conn._client is None


class TestSseConnector:
    def test_init(self) -> None:
        conn = SseConnector(url="https://example.com/stream")
        assert conn._url == "https://example.com/stream"
        assert conn._window == 30

    def test_write_raises(self) -> None:
        conn = SseConnector(url="https://example.com/stream")
        with pytest.raises(NotImplementedError, match="read-only"):
            conn.write([{"a": 1}])

    def test_read_not_connected(self) -> None:
        conn = SseConnector(url="https://example.com/stream")
        with pytest.raises(RuntimeError, match="not connected"):
            conn.read()

    def test_matches(self) -> None:
        conn = SseConnector(url="https://example.com/stream", filter={"wiki": "enwiki"})
        assert conn._matches({"wiki": "enwiki", "type": "edit"}) is True
        assert conn._matches({"wiki": "frwiki"}) is False

    def test_matches_empty_filter(self) -> None:
        conn = SseConnector(url="https://example.com/stream")
        assert conn._matches({"anything": True}) is True

    def test_process_event_block_valid(self) -> None:
        conn = SseConnector(url="https://example.com/stream")
        collected: list[dict[str, Any]] = []
        conn._process_event_block(['{"type": "edit"}'], collected)
        assert len(collected) == 1

    def test_process_event_block_invalid_json(self) -> None:
        conn = SseConnector(url="https://example.com/stream")
        collected: list[dict[str, Any]] = []
        conn._process_event_block(["not json"], collected)
        assert len(collected) == 0

    def test_process_event_block_filtered_out(self) -> None:
        conn = SseConnector(url="https://example.com/stream", filter={"wiki": "enwiki"})
        collected: list[dict[str, Any]] = []
        conn._process_event_block(['{"wiki": "frwiki"}'], collected)
        assert len(collected) == 0

    def test_disconnect(self) -> None:
        conn = SseConnector(url="https://example.com/stream")
        conn._events = [{"data": 1}]
        conn._ready = True
        conn.disconnect()
        assert conn._events == []
        assert conn._ready is False

    def test_health_check_success(self) -> None:
        conn = SseConnector(url="https://example.com/stream")
        with patch("dataenginex.providers.connectors.sse.httpx") as mock_httpx:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_httpx.get.return_value = mock_resp
            assert conn.health_check() is True

    def test_health_check_failure(self) -> None:
        conn = SseConnector(url="https://example.com/stream")
        with patch("dataenginex.providers.connectors.sse.httpx") as mock_httpx:
            mock_httpx.get.side_effect = Exception("timeout")
            assert conn.health_check() is False


class TestDeltaConnector:
    def test_import_without_deltalake(self) -> None:
        from dataenginex.providers.connectors.delta import DeltaConnector
        conn = DeltaConnector(path="/tmp/test_delta")
        assert conn._connected is False

    def test_connect_requires_deltalake(self) -> None:
        from dataenginex.providers.connectors.delta import _HAS_DELTALAKE, DeltaConnector
        conn = DeltaConnector(path="/tmp/test_delta")
        if not _HAS_DELTALAKE:
            with pytest.raises(RuntimeError, match="deltalake"):
                conn.connect()

    def test_read_not_connected(self) -> None:
        from dataenginex.providers.connectors.delta import DeltaConnector
        conn = DeltaConnector(path="/tmp/test_delta")
        with pytest.raises(RuntimeError, match="Not connected"):
            conn.read()

    def test_write_not_connected(self) -> None:
        from dataenginex.providers.connectors.delta import DeltaConnector
        conn = DeltaConnector(path="/tmp/test_delta")
        with pytest.raises(RuntimeError, match="Not connected"):
            conn.write([{"a": 1}])

    def test_vacuum_not_connected(self) -> None:
        from dataenginex.providers.connectors.delta import DeltaConnector
        conn = DeltaConnector(path="/tmp/test_delta")
        with pytest.raises(RuntimeError, match="Not connected"):
            conn.vacuum()

    def test_health_check_not_connected(self) -> None:
        from dataenginex.providers.connectors.delta import DeltaConnector
        conn = DeltaConnector(path="/tmp/test_delta")
        assert conn.health_check() is False


class TestMLflowRegistry:
    def test_import_without_mlflow(self) -> None:
        from dataenginex.providers.model.mlflow_registry import MLflowRegistryError
        assert issubclass(MLflowRegistryError, RuntimeError)
