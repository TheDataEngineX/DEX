"""Tests for storage abstraction — list_objects, exists, get_storage factory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dataenginex.providers.object_store.storage import JsonStorage, get_storage

# ---------------------------------------------------------------------------
# JsonStorage.list_objects / .exists
# ---------------------------------------------------------------------------


class TestJsonStorageListObjects:
    """Verify JsonStorage.list_objects returns relative paths without extension."""

    def test_empty_directory(self, tmp_path: Path) -> None:
        store = JsonStorage(base_path=str(tmp_path))
        assert store.list_objects() == []

    def test_lists_json_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.json").write_text(json.dumps({"x": 1}))
        (tmp_path / "b.json").write_text(json.dumps({"x": 2}))
        store = JsonStorage(base_path=str(tmp_path))
        result = sorted(store.list_objects())
        assert result == ["a", "b"]

    def test_lists_nested_files(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.json").write_text(json.dumps({"x": 1}))
        store = JsonStorage(base_path=str(tmp_path))
        result = store.list_objects()
        assert "sub/nested" in result

    def test_with_prefix(self, tmp_path: Path) -> None:
        sub = tmp_path / "layer"
        sub.mkdir()
        (sub / "data.json").write_text(json.dumps({"x": 1}))
        (tmp_path / "other.json").write_text(json.dumps({"y": 2}))
        store = JsonStorage(base_path=str(tmp_path))
        result = store.list_objects(prefix="layer")
        assert result == ["layer/data"]

    def test_nonexistent_prefix(self, tmp_path: Path) -> None:
        store = JsonStorage(base_path=str(tmp_path))
        assert store.list_objects(prefix="does_not_exist") == []


class TestJsonStorageExists:
    """Verify JsonStorage.exists checks for .json file presence."""

    def test_exists_true(self, tmp_path: Path) -> None:
        (tmp_path / "record.json").write_text(json.dumps({"k": "v"}))
        store = JsonStorage(base_path=str(tmp_path))
        assert store.exists("record") is True

    def test_exists_false(self, tmp_path: Path) -> None:
        store = JsonStorage(base_path=str(tmp_path))
        assert store.exists("missing") is False


# ---------------------------------------------------------------------------
# get_storage factory
# ---------------------------------------------------------------------------


class TestGetStorage:
    """Verify URI-based storage backend creation."""

    def test_file_scheme(self, tmp_path: Path) -> None:
        store = get_storage(f"file://{tmp_path}")
        assert isinstance(store, JsonStorage)

    def test_no_scheme(self, tmp_path: Path) -> None:
        store = get_storage(str(tmp_path))
        assert isinstance(store, JsonStorage)

    def test_unsupported_scheme(self) -> None:
        with pytest.raises(ValueError, match="Unsupported storage URI scheme"):
            get_storage("ftp://example.com/data")

    def test_s3_scheme(self) -> None:
        """S3Storage is created (even without credentials)."""
        from dataenginex.providers.object_store.storage import S3Storage

        store = get_storage("s3://my-bucket/prefix", endpoint_url="http://localhost:1")
        assert isinstance(store, S3Storage)

    def test_gs_scheme(self) -> None:
        """GCSStorage is created (even without credentials)."""
        from dataenginex.providers.object_store.storage import GCSStorage

        store = get_storage("gs://my-bucket/prefix", api_endpoint="http://localhost:1")
        assert isinstance(store, GCSStorage)

    def test_bq_scheme(self) -> None:
        """BigQueryStorage is created via bq:// URI."""
        from unittest.mock import MagicMock

        from dataenginex.providers.object_store.storage import BigQueryStorage

        mock_client = MagicMock()
        store = get_storage("bq://my-project/my_dataset", client=mock_client)
        assert isinstance(store, BigQueryStorage)


# ---------------------------------------------------------------------------
# BigQueryStorage unit tests (mocked client)
# ---------------------------------------------------------------------------


class TestBigQueryStorage:
    """BigQueryStorage unit tests with a mocked google.cloud.bigquery.Client."""

    @pytest.fixture()
    def bq_store(self) -> Any:
        from unittest.mock import MagicMock, patch

        from dataenginex.providers.object_store.storage import BigQueryStorage

        mock_client = MagicMock()
        mock_bq_module = MagicMock()
        patcher_flag = patch("dataenginex.providers.object_store.storage._HAS_BIGQUERY", True)
        patcher_mod = patch("dataenginex.providers.object_store.storage.bq_client", mock_bq_module)
        patcher_flag.start()
        patcher_mod.start()

        store = BigQueryStorage(
            project_id="test-project",
            dataset="test_ds",
            location="US",
            client=mock_client,
        )

        yield store

        patcher_mod.stop()
        patcher_flag.stop()

    def test_table_ref(self, bq_store: Any) -> None:
        assert bq_store._table_ref("my_table") == "test-project.test_ds.my_table"
        assert bq_store._table_ref("other_ds.tbl") == "test-project.other_ds.tbl"

    def test_write_success(self, bq_store: Any) -> None:
        mock_job = bq_store._client.load_table_from_json.return_value
        mock_job.result.return_value = None

        result = bq_store.write([{"col": "val"}], "my_table")
        assert result is True
        bq_store._client.load_table_from_json.assert_called_once()

    def test_write_failure(self, bq_store: Any) -> None:
        bq_store._client.load_table_from_json.side_effect = RuntimeError("fail")
        assert bq_store.write([{"col": "val"}], "my_table") is False

    def test_read_success(self, bq_store: Any) -> None:
        from unittest.mock import MagicMock

        row = MagicMock()
        row.__iter__ = lambda self: iter([("col", "val")])
        row.keys.return_value = ["col"]
        row.__getitem__ = lambda self, k: "val"
        bq_store._client.list_rows.return_value = [row]

        data = bq_store.read("my_table")
        assert isinstance(data, list)

    def test_read_failure(self, bq_store: Any) -> None:
        bq_store._client.list_rows.side_effect = RuntimeError("fail")
        assert bq_store.read("my_table") is None

    def test_delete_success(self, bq_store: Any) -> None:
        assert bq_store.delete("my_table") is True
        bq_store._client.delete_table.assert_called_once()

    def test_delete_failure(self, bq_store: Any) -> None:
        bq_store._client.delete_table.side_effect = RuntimeError("fail")
        assert bq_store.delete("my_table") is False

    def test_list_objects(self, bq_store: Any) -> None:
        from unittest.mock import MagicMock

        t1 = MagicMock()
        t1.table_id = "alpha"
        t2 = MagicMock()
        t2.table_id = "beta"
        bq_store._client.list_tables.return_value = [t1, t2]

        assert bq_store.list_objects() == ["alpha", "beta"]

    def test_list_objects_with_prefix(self, bq_store: Any) -> None:
        from unittest.mock import MagicMock

        t1 = MagicMock()
        t1.table_id = "raw_users"
        t2 = MagicMock()
        t2.table_id = "gold_metrics"
        bq_store._client.list_tables.return_value = [t1, t2]

        assert bq_store.list_objects(prefix="raw") == ["raw_users"]

    def test_exists_true(self, bq_store: Any) -> None:
        assert bq_store.exists("my_table") is True
        bq_store._client.get_table.assert_called_once()

    def test_exists_false(self, bq_store: Any) -> None:
        class NotFound(Exception):
            pass

        bq_store._client.get_table.side_effect = NotFound("table not found")
        assert bq_store.exists("missing_table") is False

    def test_no_client_operations(self) -> None:
        """When _client is None, operations return safe defaults."""
        from dataenginex.providers.object_store.storage import BigQueryStorage

        store = BigQueryStorage.__new__(BigQueryStorage)
        store.project_id = "p"
        store.dataset = "d"
        store.location = "US"
        store._client = None

        assert store.write([{"x": 1}], "t") is False
        assert store.read("t") is None
        assert store.delete("t") is False
        assert store.list_objects() == []
        assert store.exists("t") is False
