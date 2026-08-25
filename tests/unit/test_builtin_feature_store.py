"""Tests for the built-in DuckDB-backed feature store."""

from __future__ import annotations

import pytest

from dataenginex.domains.ml.features.builtin import BuiltinFeatureStore
from tests.conformance.test_feature_store import FeatureStoreConformanceTests


class TestBuiltinFeatureStore(FeatureStoreConformanceTests):
    @pytest.fixture()
    def feature_store(self, tmp_path):
        store = BuiltinFeatureStore(database=str(tmp_path / "features.duckdb"))
        yield store
        store.close()

    def test_multiple_feature_groups(self, tmp_path) -> None:
        store = BuiltinFeatureStore(database=str(tmp_path / "multi.duckdb"))
        store.save_features(
            "users",
            [{"uid": "1", "age": 25}],
            entity_key="uid",
        )
        store.save_features(
            "items",
            [{"iid": "a", "price": 9.99}],
            entity_key="iid",
        )
        groups = store.list_feature_groups()
        assert "users" in groups
        assert "items" in groups
        store.close()

    def test_overwrite_feature_group(self, tmp_path) -> None:
        store = BuiltinFeatureStore(database=str(tmp_path / "overwrite.duckdb"))
        store.save_features(
            "data",
            [{"id": "1", "val": 10}],
            entity_key="id",
        )
        store.save_features(
            "data",
            [{"id": "1", "val": 20}, {"id": "2", "val": 30}],
            entity_key="id",
        )
        result = store.get_features("data", ["1"])
        assert result[0]["val"] == 20
        store.close()

    def test_get_multiple_entities(self, tmp_path) -> None:
        store = BuiltinFeatureStore(database=str(tmp_path / "multi_get.duckdb"))
        store.save_features(
            "fg",
            [{"id": "a", "v": 1}, {"id": "b", "v": 2}, {"id": "c", "v": 3}],
            entity_key="id",
        )
        result = store.get_features("fg", ["a", "c"])
        assert len(result) == 2
        store.close()
