"""Tests for the explode transform (JSON array un-nesting)."""

from __future__ import annotations

import duckdb
import pytest

from dataenginex.domains.analytics.transforms.sql import ExplodeTransform, JsonNormalizeTransform


@pytest.fixture
def conn() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(":memory:")


def test_explode_splits_list_into_rows(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        "CREATE TABLE movies AS SELECT * FROM (VALUES "
        "(1, ['alice', 'bob']), (2, ['carol'])"
        ") AS t(movie_id, cast_names)"
    )
    transform = ExplodeTransform(column="cast_names", alias="cast_name")

    output = transform.apply(conn, "movies")

    rows = conn.execute(
        f"SELECT movie_id, cast_name FROM {output} ORDER BY movie_id, cast_name"
    ).fetchall()
    assert rows == [(1, "alice"), (1, "bob"), (2, "carol")]


def test_explode_drops_rows_with_empty_array(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        "CREATE TABLE movies AS SELECT * FROM (VALUES "
        "(1, ['alice']), (2, CAST([] AS VARCHAR[]))"
        ") AS t(movie_id, cast_names)"
    )
    transform = ExplodeTransform(column="cast_names", alias="cast_name")

    output = transform.apply(conn, "movies")

    rows = conn.execute(f"SELECT movie_id FROM {output}").fetchall()
    assert rows == [(1,)]


def test_validate_rejects_empty_column() -> None:
    transform = ExplodeTransform(column="")
    assert transform.validate() == ["explode requires a column name"]


def test_validate_accepts_valid_column() -> None:
    transform = ExplodeTransform(column="cast_names")
    assert transform.validate() == []


def test_explode_handles_nested_struct_field_path(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        "CREATE TABLE movies AS SELECT * FROM (VALUES "
        "(1, {'cast': ['alice', 'bob'], 'crew': ['dir1']}), "
        "(2, {'cast': ['carol'], 'crew': ['dir2']})"
        ") AS t(movie_id, credits)"
    )
    transform = ExplodeTransform(column="credits.cast", alias="cast_member")

    output = transform.apply(conn, "movies")

    rows = conn.execute(
        f"SELECT movie_id, cast_member FROM {output} ORDER BY movie_id, cast_member"
    ).fetchall()
    assert rows == [(1, "alice"), (1, "bob"), (2, "carol")]

    cols = [r[0] for r in conn.execute(f"DESCRIBE {output}").fetchall()]
    assert "credits" not in cols
    assert "cast_member" in cols


def test_explode_rejects_non_list_input(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("CREATE TABLE malformed(movie_id INTEGER, credits VARCHAR)")
    transform = ExplodeTransform(column="credits")

    with pytest.raises(ValueError, match="must be a LIST"):
        transform.apply(conn, "malformed")


def test_json_normalize_flattens_struct(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        "CREATE TABLE movies AS SELECT "
        "1 AS id, {'imdb_id': 'tt1', 'wikidata_id': 'Q1'} AS external_ids"
    )
    transform = JsonNormalizeTransform(column="external_ids", prefix="external_")

    output = transform.apply(conn, "movies")

    assert conn.execute(
        f"SELECT id, external_imdb_id, external_wikidata_id FROM {output}"
    ).fetchall() == [(1, "tt1", "Q1")]


def test_json_normalize_validate_rejects_empty_column() -> None:
    assert JsonNormalizeTransform(column="").validate() == ["json_normalize requires a column name"]
