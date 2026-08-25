"""Integration tests for storage backends — local filesystem, S3, and GCS.

Tests run against **local emulators** by default (LocalStack + fake-gcs-server)
or against real cloud services when ``DEX_TEST_S3_BUCKET`` / ``DEX_TEST_GCS_BUCKET``
environment variables are set.

Quick start (emulators)
-----------------------
.. code-block:: bash

    docker compose -f docker-compose.test.yml up -d
    uv run pytest tests/integration/test_storage_real.py -v
    docker compose -f docker-compose.test.yml down

Real cloud
----------
Set ``DEX_TEST_S3_BUCKET``, ``DEX_TEST_GCS_BUCKET`` (and credentials) to test
against live AWS / GCP services.
"""

from __future__ import annotations

import json
import os
import socket
import uuid
from pathlib import Path
from typing import Any

import pytest

from dataenginex.providers.object_store.storage import (
    GCSStorage,
    JsonStorage,
    S3Storage,
    get_storage,
)

# ---------------------------------------------------------------------------
# Configuration — emulators vs real cloud
# ---------------------------------------------------------------------------

# S3 config: env var overrides, or defaults to LocalStack on localhost:4566
_S3_BUCKET = os.environ.get("DEX_TEST_S3_BUCKET", "dex-test-bucket")
_S3_REGION = os.environ.get("DEX_TEST_S3_REGION", "us-east-1")
_S3_ENDPOINT = os.environ.get("DEX_TEST_S3_ENDPOINT", "http://localhost:4566")

# GCS config: env var overrides, or defaults to fake-gcs-server on localhost:4443
_GCS_BUCKET = os.environ.get("DEX_TEST_GCS_BUCKET", "dex-test-bucket")
_GCS_PROJECT = os.environ.get("DEX_TEST_GCS_PROJECT", "test-project")
_GCS_ENDPOINT = os.environ.get("DEX_TEST_GCS_ENDPOINT", "http://localhost:4443")

# Unique prefix per test run to avoid collisions
_RUN_ID = uuid.uuid4().hex[:8]


# ---------------------------------------------------------------------------
# Connectivity helpers
# ---------------------------------------------------------------------------


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Return True if a TCP connection to *host*:*port* succeeds."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _s3_emulator_available() -> bool:
    """Check if the S3 emulator (LocalStack) is reachable."""
    if os.environ.get("DEX_TEST_S3_BUCKET"):
        return True  # Real cloud — assume credentials are valid
    return _port_open("localhost", 4566)


def _gcs_emulator_available() -> bool:
    """Check if the GCS emulator (fake-gcs-server) is reachable."""
    if os.environ.get("DEX_TEST_GCS_BUCKET"):
        return True  # Real cloud
    return _port_open("localhost", 4443)


requires_s3 = pytest.mark.skipif(
    not _s3_emulator_available(),
    reason=(
        "S3 not available — start emulator with "
        "'docker compose -f docker-compose.test.yml up -d' "
        "or set DEX_TEST_S3_BUCKET for real AWS"
    ),
)

requires_gcs = pytest.mark.skipif(
    not _gcs_emulator_available(),
    reason=(
        "GCS not available — start emulator with "
        "'docker compose -f docker-compose.test.yml up -d' "
        "or set DEX_TEST_GCS_BUCKET for real GCP"
    ),
)


def _is_emulator(env_var: str) -> bool:
    """True when using local emulator (env var not explicitly set)."""
    return not os.environ.get(env_var)


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_RECORDS: list[dict[str, Any]] = [
    {"id": 1, "name": "alpha", "score": 0.91},
    {"id": 2, "name": "beta", "score": 0.85},
    {"id": 3, "name": "gamma", "score": 0.77},
]


# ===================================================================
# Local file storage (always runs)
# ===================================================================


class TestLocalFileStorageIntegration:
    """Full CRUD lifecycle against local filesystem using JsonStorage."""

    def test_write_read_cycle(self, tmp_path: Path) -> None:
        store = JsonStorage(base_path=str(tmp_path))
        path = f"test/{_RUN_ID}/records"

        assert store.write(SAMPLE_RECORDS, path) is True

        data = store.read(path)
        assert data is not None
        assert len(data) == 3
        assert data[0]["name"] == "alpha"

    def test_exists_after_write(self, tmp_path: Path) -> None:
        store = JsonStorage(base_path=str(tmp_path))
        path = f"test/{_RUN_ID}/exists_check"

        assert store.exists(path) is False
        store.write(SAMPLE_RECORDS, path)
        assert store.exists(path) is True

    def test_list_objects(self, tmp_path: Path) -> None:
        store = JsonStorage(base_path=str(tmp_path))
        prefix = f"test/{_RUN_ID}/listing"

        store.write([{"a": 1}], f"{prefix}/file_a")
        store.write([{"b": 2}], f"{prefix}/file_b")
        store.write([{"c": 3}], f"{prefix}/sub/file_c")

        objects = sorted(store.list_objects(prefix=prefix))
        assert len(objects) == 3
        assert f"{prefix}/file_a" in objects
        assert f"{prefix}/file_b" in objects
        assert f"{prefix}/sub/file_c" in objects

    def test_delete_and_verify(self, tmp_path: Path) -> None:
        store = JsonStorage(base_path=str(tmp_path))
        path = f"test/{_RUN_ID}/to_delete"

        store.write(SAMPLE_RECORDS, path)
        assert store.exists(path) is True

        assert store.delete(path) is True
        assert store.exists(path) is False
        assert store.read(path) is None

    def test_get_storage_factory_file(self, tmp_path: Path) -> None:
        store = get_storage(f"file://{tmp_path}")
        assert isinstance(store, JsonStorage)

        path = f"factory/{_RUN_ID}/data"
        assert store.write(SAMPLE_RECORDS, path) is True
        data = store.read(path)
        assert data is not None
        assert len(data) == 3

    def test_overwrite_existing(self, tmp_path: Path) -> None:
        store = JsonStorage(base_path=str(tmp_path))
        path = f"test/{_RUN_ID}/overwrite"

        store.write([{"v": 1}], path)
        store.write([{"v": 2}], path)

        data = store.read(path)
        assert data is not None
        assert data[0]["v"] == 2

    def test_single_dict_normalised(self, tmp_path: Path) -> None:
        store = JsonStorage(base_path=str(tmp_path))
        path = f"test/{_RUN_ID}/single_dict"

        store.write({"key": "value"}, path)
        data = store.read(path)
        assert data is not None
        assert isinstance(data, list)
        assert data[0]["key"] == "value"


# ===================================================================
# AWS S3 storage (emulator or real)
# ===================================================================


@requires_s3
class TestS3StorageIntegration:
    """Full CRUD lifecycle against S3 (LocalStack or real AWS)."""

    @pytest.fixture(autouse=True)
    def _setup_s3(self) -> Any:
        endpoint = _S3_ENDPOINT if _is_emulator("DEX_TEST_S3_BUCKET") else None
        self.prefix = f"dex-test/{_RUN_ID}"
        self.store = S3Storage(
            bucket=_S3_BUCKET,
            prefix=self.prefix,
            region=_S3_REGION,
            endpoint_url=endpoint,
        )
        yield
        self._cleanup()

    def _cleanup(self) -> None:
        """Best-effort removal of all objects created during the test."""
        try:
            objects = self.store.list_objects()
            for key in objects:
                if self.store._client is not None:
                    self.store._client.delete_object(Bucket=_S3_BUCKET, Key=key)
        except Exception:
            pass

    def test_write_read_cycle(self) -> None:
        path = "records"
        assert self.store.write(SAMPLE_RECORDS, path) is True

        data = self.store.read(path)
        assert data is not None
        assert len(data) == 3
        assert data[0]["name"] == "alpha"
        assert data[2]["score"] == 0.77

    def test_exists_after_write(self) -> None:
        path = "exists_check"
        assert self.store.exists(path) is False

        self.store.write(SAMPLE_RECORDS, path)
        assert self.store.exists(path) is True

    def test_list_objects(self) -> None:
        self.store.write([{"a": 1}], "list/file_a")
        self.store.write([{"b": 2}], "list/file_b")

        objects = self.store.list_objects()
        assert len(objects) >= 2
        keys_str = str(objects)
        assert "file_a" in keys_str
        assert "file_b" in keys_str

    def test_delete_and_verify(self) -> None:
        path = "to_delete"
        self.store.write(SAMPLE_RECORDS, path)
        assert self.store.exists(path) is True

        assert self.store.delete(path) is True
        assert self.store.exists(path) is False

    def test_read_nonexistent(self) -> None:
        data = self.store.read("does_not_exist_abc123")
        assert data is None

    def test_overwrite_existing(self) -> None:
        path = "overwrite"
        self.store.write([{"v": 1}], path)
        self.store.write([{"v": 2}], path)

        data = self.store.read(path)
        assert data is not None
        assert data[0]["v"] == 2

    def test_single_dict_write(self) -> None:
        path = "single_dict"
        self.store.write({"key": "value"}, path)

        data = self.store.read(path)
        assert data is not None
        assert data[0]["key"] == "value"


@requires_s3
class TestS3GetStorageFactory:
    """Test get_storage factory with S3 URIs."""

    @pytest.fixture(autouse=True)
    def _setup_s3(self) -> Any:
        self.prefix = f"dex-test/{_RUN_ID}/factory"
        endpoint = _S3_ENDPOINT if _is_emulator("DEX_TEST_S3_BUCKET") else None
        self.store = S3Storage(
            bucket=_S3_BUCKET,
            prefix=self.prefix,
            region=_S3_REGION,
            endpoint_url=endpoint,
        )
        yield
        try:
            if self.store._client is not None:
                objects = self.store.list_objects()
                for key in objects:
                    self.store._client.delete_object(Bucket=_S3_BUCKET, Key=key)
        except Exception:
            pass

    def test_factory_creates_s3storage(self) -> None:
        endpoint = _S3_ENDPOINT if _is_emulator("DEX_TEST_S3_BUCKET") else None
        kwargs: dict[str, Any] = {}
        if endpoint:
            kwargs["endpoint_url"] = endpoint
        store = get_storage(f"s3://{_S3_BUCKET}/{self.prefix}", **kwargs)
        assert isinstance(store, S3Storage)

    def test_factory_roundtrip(self) -> None:
        path = "roundtrip"
        assert self.store.write(SAMPLE_RECORDS, path) is True

        data = self.store.read(path)
        assert data is not None
        assert len(data) == 3


# ===================================================================
# GCP GCS storage (emulator or real)
# ===================================================================


@requires_gcs
class TestGCSStorageIntegration:
    """Full CRUD lifecycle against GCS (fake-gcs-server or real GCP)."""

    @pytest.fixture(autouse=True)
    def _setup_gcs(self) -> Any:
        endpoint = _GCS_ENDPOINT if _is_emulator("DEX_TEST_GCS_BUCKET") else None
        self.prefix = f"dex-test/{_RUN_ID}"

        # Create the bucket in the emulator if needed
        if endpoint:
            self._ensure_emulator_bucket(endpoint)

        self.store = GCSStorage(
            bucket=_GCS_BUCKET,
            prefix=self.prefix,
            project=_GCS_PROJECT,
            api_endpoint=endpoint,
        )
        yield
        self._cleanup()

    @staticmethod
    def _ensure_emulator_bucket(endpoint: str) -> None:
        """Create the test bucket in the fake-gcs-server if it doesn't exist."""
        import urllib.error
        import urllib.request

        url = f"{endpoint}/storage/v1/b"
        body = json.dumps({"name": _GCS_BUCKET}).encode()
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=5)  # noqa: S310
        except urllib.error.HTTPError as exc:
            if exc.code != 409:  # 409 = already exists — fine
                raise

    def _cleanup(self) -> None:
        """Best-effort removal of all objects created during the test."""
        try:
            objects = self.store.list_objects()
            for blob_name in objects:
                if self.store._bucket is not None:
                    self.store._bucket.blob(blob_name).delete()
        except Exception:
            pass

    def test_write_read_cycle(self) -> None:
        path = "records"
        assert self.store.write(SAMPLE_RECORDS, path) is True

        data = self.store.read(path)
        assert data is not None
        assert len(data) == 3
        assert data[0]["name"] == "alpha"
        assert data[2]["score"] == 0.77

    def test_exists_after_write(self) -> None:
        path = "exists_check"
        assert self.store.exists(path) is False

        self.store.write(SAMPLE_RECORDS, path)
        assert self.store.exists(path) is True

    def test_list_objects(self) -> None:
        self.store.write([{"a": 1}], "list/file_a")
        self.store.write([{"b": 2}], "list/file_b")

        objects = self.store.list_objects()
        assert len(objects) >= 2
        keys_str = str(objects)
        assert "file_a" in keys_str
        assert "file_b" in keys_str

    def test_delete_and_verify(self) -> None:
        path = "to_delete"
        self.store.write(SAMPLE_RECORDS, path)
        assert self.store.exists(path) is True

        assert self.store.delete(path) is True
        assert self.store.exists(path) is False

    def test_read_nonexistent(self) -> None:
        data = self.store.read("does_not_exist_abc123")
        assert data is None

    def test_overwrite_existing(self) -> None:
        path = "overwrite"
        self.store.write([{"v": 1}], path)
        self.store.write([{"v": 2}], path)

        data = self.store.read(path)
        assert data is not None
        assert data[0]["v"] == 2

    def test_single_dict_write(self) -> None:
        path = "single_dict"
        self.store.write({"key": "value"}, path)

        data = self.store.read(path)
        assert data is not None
        assert data[0]["key"] == "value"


@requires_gcs
class TestGCSGetStorageFactory:
    """Test get_storage factory with GCS URIs."""

    @pytest.fixture(autouse=True)
    def _setup_gcs(self) -> Any:
        self.prefix = f"dex-test/{_RUN_ID}/factory"
        endpoint = _GCS_ENDPOINT if _is_emulator("DEX_TEST_GCS_BUCKET") else None

        if endpoint:
            TestGCSStorageIntegration._ensure_emulator_bucket(endpoint)

        self.store = GCSStorage(
            bucket=_GCS_BUCKET,
            prefix=self.prefix,
            project=_GCS_PROJECT,
            api_endpoint=endpoint,
        )
        yield
        try:
            if self.store._bucket is not None:
                objects = self.store.list_objects()
                for blob_name in objects:
                    self.store._bucket.blob(blob_name).delete()
        except Exception:
            pass

    def test_factory_creates_gcsstorage(self) -> None:
        endpoint = _GCS_ENDPOINT if _is_emulator("DEX_TEST_GCS_BUCKET") else None
        kwargs: dict[str, Any] = {}
        if endpoint:
            kwargs["api_endpoint"] = endpoint
        store = get_storage(f"gs://{_GCS_BUCKET}/{self.prefix}", **kwargs)
        assert isinstance(store, GCSStorage)

    def test_factory_roundtrip(self) -> None:
        path = "roundtrip"
        assert self.store.write(SAMPLE_RECORDS, path) is True

        data = self.store.read(path)
        assert data is not None
        assert len(data) == 3
