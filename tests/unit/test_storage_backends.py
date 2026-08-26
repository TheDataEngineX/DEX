"""Tests for storage backends: JsonStorage, ParquetStorage, get_storage."""

from __future__ import annotations

from pathlib import Path

import pytest

from dataenginex.providers.object_store.storage import JsonStorage, get_storage


class TestJsonStorage:
    def test_write_and_read(self, tmp_path: Path) -> None:
        storage = JsonStorage(base_path=str(tmp_path / "data"))
        records = [{"id": 1, "name": "alice"}, {"id": 2, "name": "bob"}]
        assert storage.write(records, "users") is True
        result = storage.read("users")
        assert len(result) == 2
        assert result[0]["name"] == "alice"

    def test_read_nonexistent(self, tmp_path: Path) -> None:
        storage = JsonStorage(base_path=str(tmp_path / "data"))
        assert storage.read("nonexistent") is None

    def test_delete(self, tmp_path: Path) -> None:
        storage = JsonStorage(base_path=str(tmp_path / "data"))
        storage.write([{"id": 1}], "to_delete")
        assert storage.delete("to_delete") is True
        assert storage.read("to_delete") is None

    def test_delete_nonexistent(self, tmp_path: Path) -> None:
        storage = JsonStorage(base_path=str(tmp_path / "data"))
        assert storage.delete("nonexistent") is True

    def test_list_objects(self, tmp_path: Path) -> None:
        storage = JsonStorage(base_path=str(tmp_path / "data"))
        storage.write([{"id": 1}], "file1")
        storage.write([{"id": 2}], "file2")
        files = storage.list_objects()
        assert len(files) == 2

    def test_write_empty(self, tmp_path: Path) -> None:
        storage = JsonStorage(base_path=str(tmp_path / "data"))
        assert storage.write([], "empty") is True


class TestGetStorage:
    def test_file_uri(self, tmp_path: Path) -> None:
        backend = get_storage(f"file://{tmp_path / 'data'}")
        assert isinstance(backend, JsonStorage)

    def test_no_scheme(self, tmp_path: Path) -> None:
        backend = get_storage(str(tmp_path / "data"))
        assert isinstance(backend, JsonStorage)

    def test_unknown_uri(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            get_storage("unknown://bucket/path")
