"""Server-Sent Events (SSE) connector — windowed micro-batch ingestion.

Registered as ``type: sse`` in dex.yaml sources.  Opens a persistent SSE
connection, collects events for a fixed time window, then disconnects and
returns the batch.  Running on a short cron (e.g. ``*/15 * * * *``) gives
near-real-time ingestion without the complexity of a persistent stream process.

The Wikimedia EventStreams endpoint is the primary target::

    https://stream.wikimedia.org/v2/stream/recentchange

Example dex.yaml::

    sources:
      wiki_movie_edits:
        type: sse
        url: https://stream.wikimedia.org/v2/stream/recentchange
        options:
          window_seconds: 60
          filter:
            wiki: enwiki
            namespace: 0
            type: edit
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

import httpx
import structlog

from dataenginex.foundation.plugin_contracts import BaseConnector
from dataenginex.providers.connectors import connector_registry

logger = structlog.get_logger()

_DEFAULT_WINDOW = 30  # seconds per pipeline run


@connector_registry.decorator("sse")
class SseConnector(BaseConnector):
    """Windowed SSE connector — collects a time-bounded batch from an SSE stream.

    Args:
        url: SSE stream endpoint URL.
        window_seconds: How long to listen per pipeline run (default 30s).
        filter: Dict of ``field: value`` pairs that ALL must match on the event
            JSON payload to keep the event.  Missing fields are treated as
            non-matching (event dropped).
        headers: Optional HTTP headers (auth, accept, etc.).
        max_events: Hard cap on events per window regardless of time (default 10 000).
        timeout: HTTP connection timeout in seconds (default 10).
    """

    def __init__(
        self,
        url: str,
        window_seconds: int = _DEFAULT_WINDOW,
        filter: dict[str, Any] | None = None,  # noqa: A002
        headers: dict[str, str] | None = None,
        max_events: int = 10_000,
        timeout: float = 10.0,
        **kwargs: Any,
    ) -> None:
        self._url = url
        self._window = window_seconds
        self._filter = filter or {}
        self._headers = {"Accept": "text/event-stream", **(headers or {})}
        self._max_events = max_events
        self._timeout = timeout
        self._events: list[dict[str, Any]] = []
        self._ready = False

    def _matches(self, payload: dict[str, Any]) -> bool:
        """Return True if the event payload passes all filter criteria."""
        return all(payload.get(key) == expected for key, expected in self._filter.items())

    def _process_event_block(self, data_buf: list[str], collected: list[dict[str, Any]]) -> None:
        """Parse a completed SSE event block and append to collected if it matches."""
        raw_json = "\n".join(data_buf)
        try:
            payload = json.loads(raw_json)
            if isinstance(payload, dict) and self._matches(payload):
                collected.append(payload)
        except json.JSONDecodeError:
            pass

    def _collect(self) -> None:
        """Background thread: open SSE connection, collect events for the window."""
        deadline = time.monotonic() + self._window
        collected: list[dict[str, Any]] = []
        data_buf: list[str] = []

        try:
            # connect timeout is short; read timeout must be None so sparse SSE
            # streams don't abort between events during the collection window.
            sse_timeout = httpx.Timeout(connect=self._timeout, read=None, write=None, pool=None)
            transport = httpx.HTTPTransport(retries=3)
            with (
                httpx.Client(
                    headers=self._headers, timeout=sse_timeout, transport=transport
                ) as client,
                client.stream("GET", self._url) as response,
            ):
                response.raise_for_status()
                for raw_line in response.iter_lines():
                    if time.monotonic() >= deadline:
                        break
                    if len(collected) >= self._max_events:
                        break

                    line = raw_line.strip()

                    if line.startswith("data:"):
                        data_buf.append(line[5:].strip())
                    elif line == "" and data_buf:
                        self._process_event_block(data_buf, collected)
                        data_buf = []
                    # Lines starting with "event:", "id:", ":" (comment) are ignored
        except Exception as exc:
            logger.warning("sse connector stream error", url=self._url, error=str(exc))

        self._events = collected
        self._ready = True
        logger.info(
            "sse window complete",
            url=self._url,
            events=len(collected),
            window_s=self._window,
        )

    # ── BaseConnector ─────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Open the SSE stream and collect one window of events (blocking)."""
        self._events = []
        self._ready = False

        # Run collection in a thread so we can enforce the deadline cleanly
        t = threading.Thread(target=self._collect, daemon=True)
        t.start()
        # Wait up to window + connection_timeout before giving up
        t.join(timeout=self._window + self._timeout + 5)

        if not self._ready:
            logger.warning("sse connector timed out waiting for collection thread")

    def disconnect(self) -> None:
        self._events = []
        self._ready = False

    def read(self, *, table: str | None = None, **kwargs: Any) -> list[dict[str, Any]]:
        if not self._ready:
            msg = "SseConnector not connected — call connect() first"
            raise RuntimeError(msg)
        return list(self._events)

    def write(self, data: Any, *, table: str = "", **kwargs: Any) -> None:
        raise NotImplementedError("SseConnector is read-only")

    def health_check(self) -> bool:
        try:
            resp = httpx.get(self._url, headers=self._headers, timeout=5)
            # SSE endpoints return 200 with text/event-stream content-type
            return resp.status_code == 200
        except Exception:
            return False
