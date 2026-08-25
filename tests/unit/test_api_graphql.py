"""Tests for dataenginex.api.graphql — the generic gold-table GraphQL builder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dataenginex.api.graphql import GoldTable, build_schema

deltalake = pytest.importorskip("deltalake")
from dataenginex.providers.object_store.storage import DeltaStorage  # noqa: E402


class _FakeEngine:
    """Minimal stand-in for DexEngine — only what build_schema() needs."""

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = project_dir

    def warehouse_table_schema(self, table_name: str, layer: str) -> list[dict[str, Any]]:
        path = Path(self.project_dir) / ".dex" / "lakehouse" / layer / f"{table_name}.parquet"
        if not path.exists():
            return []
        with duckdb.connect(":memory:") as conn:
            conn.execute(f"CREATE VIEW v AS SELECT * FROM read_parquet('{path}')")
            rows = conn.execute("DESCRIBE v").fetchall()
        return [{"column_name": r[0], "column_type": r[1], "nullable": r[3] != "NO"} for r in rows]


def _write_table(project_dir: Path, layer: str, name: str, table: pa.Table) -> None:
    d = project_dir / ".dex" / "lakehouse" / layer
    d.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, d / f"{name}.parquet")


@pytest.fixture
def engine(tmp_path: Path) -> _FakeEngine:
    _write_table(
        tmp_path,
        "gold",
        "widgets",
        pa.table(
            {
                "widget_id": [1, 2, 3],
                "name": ["alpha", "beta", "gamma"],
                "score": [1.5, 2.5, 3.5],
            }
        ),
    )
    return _FakeEngine(tmp_path)


def test_lists_all_rows(engine: _FakeEngine) -> None:
    schema = build_schema(engine, {"Widget": GoldTable(table="widgets", id_column="widget_id")})
    result = schema.execute_sync("{ widget { widgetId name score } }")

    assert result.errors is None
    assert result.data is not None
    rows = result.data["widget"]
    assert len(rows) == 3
    assert {"widgetId": 1, "name": "alpha", "score": 1.5} in rows


def test_filters_by_id_column(engine: _FakeEngine) -> None:
    schema = build_schema(engine, {"Widget": GoldTable(table="widgets", id_column="widget_id")})
    result = schema.execute_sync('{ widget(id: "2") { widgetId name } }')

    assert result.errors is None
    assert result.data is not None
    rows = result.data["widget"]
    assert rows == [{"widgetId": 2, "name": "beta"}]


def test_plain_string_config_uses_first_column_as_id(engine: _FakeEngine) -> None:
    """Passing a bare table name (no GoldTable) infers id_column from the first column."""
    schema = build_schema(engine, {"Widget": "widgets"})
    result = schema.execute_sync('{ widget(id: "1") { name } }')

    assert result.errors is None
    assert result.data is not None
    assert result.data["widget"] == [{"name": "alpha"}]


def test_unknown_table_is_skipped_not_fatal(engine: _FakeEngine) -> None:
    """A table with no materialized parquet (pipeline hasn't run) is skipped, not an error."""
    schema = build_schema(engine, {"Widget": "widgets", "Ghost": "does_not_exist"})
    result = schema.execute_sync("{ widget { name } __typename }")
    assert result.errors is None
    assert len(result.data["widget"]) == 3


def test_generic_helper_knows_nothing_about_table_semantics(engine: _FakeEngine) -> None:
    """Sanity check: build_schema is driven entirely by the caller-supplied mapping."""
    schema = build_schema(engine, {"Anything": GoldTable(table="widgets", id_column="name")})
    result = schema.execute_sync('{ anything(id: "gamma") { widgetId score } }')
    assert result.errors is None
    assert result.data["anything"] == [{"widgetId": 3, "score": 3.5}]


def test_reads_delta_table(tmp_path: Path) -> None:
    gold = tmp_path / ".dex" / "lakehouse" / "gold"
    gold.mkdir(parents=True)
    assert DeltaStorage(base_path=str(gold)).write(
        [{"movie_id": 1, "title": "Delta Movie"}],
        "movies",
    )

    class _DeltaEngine:
        project_dir = tmp_path

        def warehouse_table_schema(self, table_name: str, layer: str) -> list[dict[str, Any]]:
            return [
                {"name": "movie_id", "dtype": "BIGINT", "nullable": False},
                {"name": "title", "dtype": "VARCHAR", "nullable": True},
            ]

    schema = build_schema(
        _DeltaEngine(),
        {"Movie": GoldTable(table="movies", id_column="movie_id")},
    )
    result = schema.execute_sync('{ movie(id: "1") { movieId title } }')
    assert result.errors is None
    assert result.data == {"movie": [{"movieId": 1, "title": "Delta Movie"}]}
