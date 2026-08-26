"""DeltaStorage.write() must tolerate additive schema drift by default.

Bronze pipelines capture external API data as-is, and real API responses
have optional/variable-shape nested fields — a later batch can legitimately
have a field the first batch never saw. Previously write_deltalake() was
called with no schema_mode, so the first batch whose inferred schema didn't
exactly match the existing table's schema hard-failed the whole write
(observed live: bronze_tmdb_movie_details failing every run past the first
137-row batch once TMDB returned a movie with an extra nested field).
"""

from __future__ import annotations

from pathlib import Path

import pytest

deltalake = pytest.importorskip("deltalake")
from dataenginex.providers.object_store.storage import DeltaStorage  # noqa: E402


def test_write_tolerates_new_field_in_later_batch(tmp_path: Path) -> None:
    storage = DeltaStorage(base_path=str(tmp_path))

    storage.write([{"id": 1, "title": "A"}], path="movies")
    # Second batch has an extra field the first batch's schema never had.
    storage.write([{"id": 2, "title": "B", "tagline": "new field"}], path="movies")

    import duckdb

    conn = duckdb.connect(":memory:")
    rows = conn.execute(
        f"SELECT id, title, tagline FROM delta_scan('{tmp_path}/movies') ORDER BY id"
    ).fetchall()
    assert rows == [(1, "A", None), (2, "B", "new field")]


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        test_write_tolerates_new_field_in_later_batch(Path(tmp))
    print("ok")
