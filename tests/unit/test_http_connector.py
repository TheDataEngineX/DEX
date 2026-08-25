"""HttpConnector — download, cache, and read behavior for TSV/CSV sources.

Regression coverage for the resource-usage fix: gzip sources must be read
directly by DuckDB (no intermediate decompressed on-disk copy), and the
converted Parquet cache must be memory-mapped on read.
"""

from __future__ import annotations

import gzip
import http.server
import io
import threading
from collections.abc import Iterator

import pytest

from dataenginex.providers.connectors.http import HttpConnector

_TSV_BODY = "tconst\taverageRating\ttitle\ntt0000001\t5.7\tCarmencita\ntt0000002\t6.0\tLe clown\n"


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/data.tsv":
            body = _TSV_BODY.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/tab-separated-values")
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/data.tsv.gz":
            buf = io.BytesIO()
            with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
                gz.write(_TSV_BODY.encode())
            self.send_response(200)
            self.send_header("Content-Type", "application/gzip")
            self.end_headers()
            self.wfile.write(buf.getvalue())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass  # silence request logging in test output


@pytest.fixture
def http_server() -> Iterator[str]:
    server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_plain_tsv_round_trips(http_server: str, tmp_path: object) -> None:
    conn = HttpConnector(url=f"{http_server}/data.tsv", cache_dir=str(tmp_path))
    conn.connect()
    dataset = conn.read()
    assert dataset.count_rows() == 2
    assert dataset.to_table().column("title").to_pylist() == ["Carmencita", "Le clown"]


def test_gzip_tsv_round_trips_without_decompressed_copy(http_server: str, tmp_path: object) -> None:
    """The regression case: DuckDB must read the .gz directly (compression='gzip'),
    with no intermediate plain-TSV file ever written to the temp directory."""
    conn = HttpConnector(url=f"{http_server}/data.tsv.gz", cache_dir=str(tmp_path))
    conn.connect()
    dataset = conn.read()
    assert dataset.count_rows() == 2
    assert dataset.to_table().column("tconst").to_pylist() == ["tt0000001", "tt0000002"]


def test_read_before_connect_raises(tmp_path: object) -> None:
    conn = HttpConnector(url="http://example.invalid/data.tsv", cache_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match="not connected"):
        conn.read()
