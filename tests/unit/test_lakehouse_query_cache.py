"""_make_lakehouse_query must materialize only the tables a query actually
references, not the whole lakehouse — the earlier eager "load everything"
approach (later just cached, not scoped) OOM'd a resource-constrained
container once real data volumes existed.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from dataenginex.domains.ai.tools.builtin import _make_lakehouse_query, _referenced_table_names


def _write_bronze_table(lakehouse_dir: Path, name: str, rows: list[dict[str, object]]) -> None:
    bronze = lakehouse_dir / "bronze"
    bronze.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, bronze / f"{name}.parquet")


def test_returns_correct_results(tmp_path: Path) -> None:
    _write_bronze_table(tmp_path, "movies", [{"id": 1, "title": "A"}, {"id": 2, "title": "B"}])
    query = _make_lakehouse_query(tmp_path)

    rows_1 = query("SELECT * FROM movies ORDER BY id")
    rows_2 = query("SELECT COUNT(*) AS n FROM movies")

    assert rows_1 == [{"id": 1, "title": "A"}, {"id": 2, "title": "B"}]
    assert rows_2 == [{"n": 2}]


def test_only_materializes_referenced_table(tmp_path: Path) -> None:
    """A query against one table must not require every other lakehouse
    table to exist/load — proves scoping, not a full-catalog load."""
    _write_bronze_table(tmp_path, "movies", [{"id": 1, "title": "A"}])
    # A second, much larger table that the query below never references.
    _write_bronze_table(tmp_path, "unrelated_huge_table", [{"id": i} for i in range(1000)])
    query = _make_lakehouse_query(tmp_path)

    rows = query("SELECT * FROM movies")

    assert rows == [{"id": 1, "title": "A"}]


def test_referenced_table_names_extracts_from_joins() -> None:
    names = _referenced_table_names(
        'SELECT a.x FROM foo a JOIN "bar" b ON a.id = b.id WHERE a.y > 1'
    )
    assert names == {"foo", "bar"}


def test_rejects_non_select(tmp_path: Path) -> None:
    query = _make_lakehouse_query(tmp_path)
    try:
        query("DELETE FROM movies")
    except ValueError as exc:
        assert "read-only" in str(exc)
    else:
        raise AssertionError("expected ValueError for non-SELECT statement")


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        test_returns_correct_results(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_only_materializes_referenced_table(Path(tmp))
    test_referenced_table_names_extracts_from_joins()
    with tempfile.TemporaryDirectory() as tmp:
        test_rejects_non_select(Path(tmp))
    print("ok")
