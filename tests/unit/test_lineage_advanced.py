"""Tests for warehouse lineage: PostgresLineage, JSON persistence, and more."""

from __future__ import annotations

from typing import Any

from dataenginex.warehouse.lineage import (
    LineageEvent,
    PersistentLineage,
    PostgresLineage,
)


class TestPostgresLineage:
    def test_fallback_on_no_asyncpg(self, tmp_path: Any) -> None:
        fallback = tmp_path / "fallback.json"
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pl = PostgresLineage("postgresql://invalid:5432/test", fallback_path=str(fallback))
        assert pl._pg_ok is False
        assert pl._fallback is not None

    def test_run_method(self, tmp_path: Any) -> None:
        import warnings

        fallback = tmp_path / "fb.json"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pl = PostgresLineage("postgresql://invalid:5432/test", fallback_path=str(fallback))

        async def _coro() -> None:
            pass

        coro = _coro()
        try:
            pl._run(coro)
        except Exception:
            pass
        finally:
            coro.close()


class TestPersistentLineageAdvanced:
    def test_thread_safety(self) -> None:
        pl = PersistentLineage()
        import concurrent.futures

        def _record(i: int) -> LineageEvent:
            return pl.record(
                operation="op",
                source="s",
                destination="d",
                input_count=i,
                output_count=i,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            futures = [ex.submit(_record, i) for i in range(20)]
            results = [f.result() for f in futures]
        assert len(results) == 20
        assert len(pl.all_events) == 20

    def test_get_children_empty(self) -> None:
        pl = PersistentLineage()
        assert pl.get_children("nonexistent") == []

    def test_get_by_layer_empty(self) -> None:
        pl = PersistentLineage()
        assert pl.get_by_layer("bronze") == []

    def test_get_by_pipeline_empty(self) -> None:
        pl = PersistentLineage()
        assert pl.get_by_pipeline("p1") == []
