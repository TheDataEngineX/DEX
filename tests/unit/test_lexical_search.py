"""Tests for dataenginex.domains.ai.lexical_search — ElasticsearchBackend."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from dataenginex.domains.ai.lexical_search import ElasticsearchBackend
from dataenginex.providers.vector.vectorstore import Document


def _backend() -> ElasticsearchBackend:
    return ElasticsearchBackend(
        hosts=["http://localhost:9200"], index_name="test_idx", timeout_seconds=1.0
    )


class TestElasticsearchBackendSearch:
    def test_search_returns_results_on_success(self) -> None:
        backend = _backend()
        backend._client = MagicMock()
        backend._client.search.return_value = {
            "hits": {
                "hits": [
                    {"_id": "1", "_score": 2.5, "_source": {"text": "python", "genre": "code"}},
                ]
            }
        }
        results = backend.search("python", top_k=5)
        assert len(results) == 1
        assert results[0].document.id == "1"
        assert results[0].document.text == "python"
        assert results[0].document.metadata == {"genre": "code"}
        assert results[0].score == 2.5

    def test_search_returns_empty_on_connection_error(self) -> None:
        backend = _backend()
        backend._client = MagicMock()
        backend._client.search.side_effect = ConnectionError("es unreachable")
        # Must never raise — callers rely on graceful degradation.
        assert backend.search("python", top_k=5) == []


class TestElasticsearchBackendIndex:
    def test_index_skips_empty(self) -> None:
        backend = _backend()
        assert backend.index([]) == 0

    def test_index_success(self) -> None:
        backend = _backend()
        backend._client = MagicMock()
        with patch("elasticsearch.helpers.bulk", return_value=(2, [])) as bulk_mock:
            count = backend.index([Document(id="1", text="a"), Document(id="2", text="b")])
        assert count == 2
        assert bulk_mock.called

    def test_index_returns_zero_on_error(self) -> None:
        backend = _backend()
        backend._client = MagicMock()
        with patch("elasticsearch.helpers.bulk", side_effect=ConnectionError("down")):
            count = backend.index([Document(id="1", text="a")])
        assert count == 0


class TestElasticsearchBackendHealthCheck:
    def test_health_check_true(self) -> None:
        backend = _backend()
        backend._client = MagicMock()
        backend._client.ping.return_value = True
        assert backend.health_check() is True

    def test_health_check_false_on_exception(self) -> None:
        backend = _backend()
        backend._client = MagicMock()
        backend._client.ping.side_effect = ConnectionError("down")
        assert backend.health_check() is False
