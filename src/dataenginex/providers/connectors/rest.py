"""REST API connector — synchronous HTTP polling for JSON endpoints.

Registered as ``type: rest`` in dex.yaml sources.  Designed for paginated
or single-page JSON APIs where the response is a list (or a dict with a
list under a known key).

Example dex.yaml::

    sources:
      tmdb_trending:
        type: rest
        url: https://api.themoviedb.org/3/trending/movie/day
        connection:
          params:
            api_key: "YOUR_TMDB_API_KEY"
            language: en-US
        options:
          records_key: results
          timeout: 30
"""

from __future__ import annotations

from typing import Any

import httpx
import structlog

from dataenginex.foundation.plugin_contracts import BaseConnector
from dataenginex.providers.connectors import connector_registry

logger = structlog.get_logger()


@connector_registry.decorator("rest")
class RestApiConnector(BaseConnector):
    """Synchronous REST connector backed by httpx.

    Args:
        url: Full API endpoint URL.
        params: Query-string parameters (e.g. ``{"api_key": "xxx"}``).
        headers: HTTP headers (e.g. ``{"Authorization": "Bearer xxx"}``).
        records_key: Key in the JSON response that holds the list of records.
            When ``None`` the root response is expected to be a list.
        timeout: HTTP timeout in seconds (default 30).
        page_param: Query parameter name used for pagination (default ``"page"``).
        max_pages: Maximum pages to fetch (default 1 — single request).
    """

    def __init__(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        records_key: str | None = None,
        timeout: float = 30.0,
        page_param: str = "page",
        max_pages: int = 1,
        **kwargs: Any,
    ) -> None:
        self._url = url
        self._params = params or {}
        self._headers = headers or {}
        self._records_key = records_key
        self._timeout = timeout
        self._page_param = page_param
        self._max_pages = max_pages
        self._client: httpx.Client | None = None

    def connect(self) -> None:
        # ponytail: httpx native retry transport — 3 retries on connection errors
        transport = httpx.HTTPTransport(retries=3)
        self._client = httpx.Client(
            headers=self._headers,
            timeout=self._timeout,
            transport=transport,
        )
        logger.debug("rest connector ready", url=self._url)

    def disconnect(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def _extract_records(self, body: Any) -> list[dict[str, Any]]:
        if self._records_key:
            return body.get(self._records_key, []) if isinstance(body, dict) else []
        if isinstance(body, list):
            return body
        return [body]

    def read(self, *, table: str | None = None, **kwargs: Any) -> list[dict[str, Any]]:
        if self._client is None:
            msg = "RestApiConnector not connected — call connect() first"
            raise RuntimeError(msg)

        all_records: list[dict[str, Any]] = []

        for page in range(1, self._max_pages + 1):
            params = {**self._params}
            if self._max_pages > 1:
                params[self._page_param] = page

            try:
                resp = self._client.get(self._url, params=params)
                resp.raise_for_status()
                body = resp.json()
            except httpx.HTTPStatusError as exc:
                logger.error(
                    "rest connector http error", status=exc.response.status_code, url=self._url
                )
                raise
            except Exception as exc:
                logger.error("rest connector request failed", url=self._url, error=str(exc))
                raise

            records = self._extract_records(body)
            all_records.extend(records)
            logger.debug("rest connector page fetched", page=page, records=len(records))

            if not records:
                break

        logger.info("rest connector read complete", url=self._url, total=len(all_records))
        return all_records

    def write(self, data: Any, *, table: str = "", **kwargs: Any) -> None:
        raise NotImplementedError("RestApiConnector is read-only")

    def health_check(self) -> bool:
        try:
            resp = httpx.head(self._url, params=self._params, timeout=5)
            return resp.status_code < 500
        except Exception:
            return False
