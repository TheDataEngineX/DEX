"""
Concrete storage backends for the DataEngineX lakehouse.

``JsonStorage``, ``ParquetStorage``, ``S3Storage``, and ``GCSStorage``
implement the ``StorageBackend`` ABC from
``dataenginex.domains.data.medallion`` so they can be used
interchangeably by the ``DualStorage`` layer.

``ParquetStorage`` delegates to *pyarrow* when available; otherwise it
falls back to ``JsonStorage`` with a logged warning.

``S3Storage`` and ``GCSStorage`` are stub implementations that log
operations but require ``boto3`` / ``google-cloud-storage`` at runtime.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import structlog

from dataenginex import _json
from dataenginex.domains.data.medallion import StorageBackend, StorageFormat

logger = structlog.get_logger()

__all__ = [
    "BigQueryStorage",
    "DeltaStorage",
    "GCSStorage",
    "JsonStorage",
    "ParquetStorage",
    "S3Storage",
    "get_storage",
]

# Try importing pyarrow — optional heavyweight dependency
try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    _HAS_PYARROW = True
except ImportError:
    _HAS_PYARROW = False


# ---------------------------------------------------------------------------
# JSON storage (always available)
# ---------------------------------------------------------------------------


class JsonStorage(StorageBackend):
    """Simple JSON-file storage for development and testing.

    Each ``write`` call serialises *data* (list of dicts) as a JSON array.
    """

    def __init__(self, base_path: str = "data") -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info("json storage initialised", path=str(self.base_path))

    def write(
        self,
        data: Any,
        path: str,
        format: StorageFormat = StorageFormat.PARQUET,
    ) -> bool:
        """Serialize *data* as JSON and write to *path*."""
        try:
            full = self.base_path / f"{path}.json"
            full.parent.mkdir(parents=True, exist_ok=True)
            records = self._normalise(data)
            tmp = full.with_suffix(".tmp")
            tmp.write_text(_json.dumps(records, indent=2, default=str))
            os.replace(tmp, full)
            logger.info("wrote records", count=len(records), path=str(full))
            return True
        except Exception as exc:
            logger.error("json storage write failed", exc=str(exc))
            return False

    def read(self, path: str, format: StorageFormat = StorageFormat.PARQUET) -> Any:
        """Read and deserialize a JSON file at *path*."""
        try:
            full = self.base_path / f"{path}.json"
            if not full.exists():
                logger.warning("file not found", path=str(full))
                return None
            return _json.loads(full.read_text())
        except Exception as exc:
            logger.error("json storage read failed", exc=str(exc))
            return None

    def delete(self, path: str) -> bool:
        """Delete the JSON file at *path* if it exists."""
        try:
            full = self.base_path / f"{path}.json"
            if full.exists():
                full.unlink()
                logger.info("deleted file", path=str(full))
            return True
        except Exception as exc:
            logger.error("json storage delete failed", exc=str(exc))
            return False

    @staticmethod
    def _normalise(data: Any) -> list[dict[str, Any]]:
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        return [{"value": data}]

    def list_objects(self, prefix: str = "") -> list[str]:
        """List JSON entries under *prefix*."""
        target = self.base_path / prefix
        if not target.exists():
            return []
        return [
            str(p.relative_to(self.base_path).with_suffix(""))
            for p in target.rglob("*.json")
            if p.is_file()
        ]

    def exists(self, path: str) -> bool:
        """Return ``True`` if *path* has a corresponding JSON file."""
        return (self.base_path / f"{path}.json").exists()


# ---------------------------------------------------------------------------
# Parquet storage (requires pyarrow)
# ---------------------------------------------------------------------------


class ParquetStorage(StorageBackend):
    """Parquet file storage backed by *pyarrow*.

    Falls back to ``JsonStorage`` when *pyarrow* is not installed.
    """

    def __init__(self, base_path: str = "data", compression: str = "zstd") -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.compression = compression

        if _HAS_PYARROW:
            logger.info("parquet storage initialised", path=str(self.base_path))
        else:
            logger.warning("pyarrow not installed — parquet storage will use json fallback")
            self._fallback = JsonStorage(str(self.base_path))

    def write(
        self,
        data: Any,
        path: str,
        format: StorageFormat = StorageFormat.PARQUET,
    ) -> bool:
        """Write *data* as a Parquet file at *path*."""
        if not _HAS_PYARROW:
            return self._fallback.write(data, path, format)

        try:
            full = self.base_path / f"{path}.parquet"
            full.parent.mkdir(parents=True, exist_ok=True)
            records = self._to_records(data)
            if not records:
                logger.warning("no records to write", path=str(full))
                return False
            table = pa.Table.from_pylist(records)
            tmp = full.with_suffix(".tmp")
            pq.write_table(table, str(tmp), compression=self.compression)
            os.replace(tmp, full)
            logger.info("wrote records", count=len(records), path=str(full))
            return True
        except Exception as exc:
            logger.error("parquet storage write failed", exc=str(exc))
            return False

    def read(self, path: str, format: StorageFormat = StorageFormat.PARQUET) -> Any:
        """Read a Parquet file at *path* and return records."""
        if not _HAS_PYARROW:
            return self._fallback.read(path, format)

        try:
            full = self.base_path / f"{path}.parquet"
            if not full.exists():
                logger.warning("parquet file not found", path=str(full))
                return None
            table = pq.read_table(str(full))
            return table.to_pylist()
        except Exception as exc:
            logger.error("parquet storage read failed", exc=str(exc))
            return None

    def delete(self, path: str) -> bool:
        """Delete the Parquet file at *path* if it exists."""
        if not _HAS_PYARROW:
            return self._fallback.delete(path)

        try:
            full = self.base_path / f"{path}.parquet"
            if full.exists():
                full.unlink()
                logger.info("deleted file", path=str(full))
            return True
        except Exception as exc:
            logger.error("parquet storage delete failed", exc=str(exc))
            return False

    @staticmethod
    def _to_records(data: Any) -> list[dict[str, Any]]:
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        return []

    def list_objects(self, prefix: str = "") -> list[str]:
        """List Parquet entries under *prefix*."""
        if not _HAS_PYARROW:
            return self._fallback.list_objects(prefix)
        target = self.base_path / prefix
        if not target.exists():
            return []
        return [
            str(p.relative_to(self.base_path).with_suffix(""))
            for p in target.rglob("*.parquet")
            if p.is_file()
        ]

    def exists(self, path: str) -> bool:
        """Return ``True`` if *path* has a corresponding Parquet file."""
        if not _HAS_PYARROW:
            return self._fallback.exists(path)
        return (self.base_path / f"{path}.parquet").exists()


# ---------------------------------------------------------------------------
# Delta Lake storage (requires deltalake + pyarrow)
# ---------------------------------------------------------------------------

try:
    import deltalake as _deltalake_mod  # noqa: F401 — presence check only

    _HAS_DELTALAKE = True
except ImportError:
    _HAS_DELTALAKE = False


class DeltaStorage(StorageBackend):
    """Delta Lake storage backend — ACID transactions, time travel, schema evolution.

    Requires ``deltalake>=0.24.0`` (``pip install 'dataenginex[delta]'``).
    Logs a warning and returns ``False``/``None`` when absent — same
    graceful-degradation pattern as the other cloud backends.

    Each table lives at ``{base_path}/{path}/`` (a Delta table directory
    with ``_delta_log/`` transaction log).  Concurrent writers use
    optimistic concurrency control — conflicts retry, never corrupt.

    Parameters
    ----------
    base_path:
        Root directory under which all Delta tables are stored.
    mode:
        Default write mode passed to ``write_deltalake()``:
        ``"append"`` (default), ``"overwrite"``, ``"error"``, ``"ignore"``.
        Can be overridden per ``write()`` call via kwargs.
    """

    def __init__(self, base_path: str = "data", mode: str = "append") -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        if _HAS_DELTALAKE:
            logger.info("delta storage initialised", path=str(self.base_path))
        else:
            logger.warning(
                "deltalake not installed — delta storage unavailable",
                hint="pip install 'dataenginex[delta]'",
            )

    def _table_path(self, path: str) -> str:
        return str(self.base_path / path)

    def write(
        self,
        data: Any,
        path: str,
        format: StorageFormat = StorageFormat.PARQUET,
        **kwargs: Any,
    ) -> bool:
        """Write *data* as a Delta table at *path*.

        ``**kwargs`` are forwarded to ``write_deltalake()``
        (e.g. ``mode=``, ``schema_mode=``, ``partition_by=``).
        """
        if not _HAS_DELTALAKE:
            logger.error("delta storage: deltalake not installed")
            return False
        try:
            import pyarrow as _pa  # noqa: PLC0415
            from deltalake import write_deltalake  # noqa: PLC0415

            # write_deltalake() accepts a pyarrow.Table natively — skip the
            # to_records()/from_pylist() round-trip for callers that already
            # have Arrow data (e.g. an ingest operation), which for large tables
            # (millions of rows) would otherwise box every cell as a Python
            # object twice for no benefit.
            count: int | str
            if isinstance(data, _pa.RecordBatchReader):
                # Streamed straight through — never materialized as a single
                # Table on either side, so row count is unknown until the
                # writer has consumed it.
                arrow_table = data
                count = "streaming"
            elif isinstance(data, _pa.Table):
                if data.num_rows == 0:
                    logger.warning("no records to write", path=path)
                    return False
                arrow_table = data
                count = arrow_table.num_rows
            else:
                records = self._to_records(data)
                if not records:
                    logger.warning("no records to write", path=path)
                    return False
                arrow_table = _pa.Table.from_pylist(records)
                count = len(records)
            table_path = self._table_path(path)
            logger.info("write_deltalake starting", count=count, path=table_path)
            write_deltalake(
                table_path,
                arrow_table,
                mode=kwargs.pop("mode", self.mode),
                # Bronze captures external API/source data as-is, and those
                # shapes drift over time (e.g. an optional nested field only
                # some records have) — without schema evolution, the first
                # batch whose inferred schema doesn't exactly match the
                # existing table's fixed schema hard-fails the whole write,
                # rather than additively picking up the new field. Callers
                # writing to schema-stable tables can still override this.
                schema_mode=kwargs.pop("schema_mode", "merge"),
                **kwargs,
            )
            logger.info("write_deltalake finished", count=count, path=table_path)
            logger.info("wrote delta records", count=count, path=table_path)
            return True
        except Exception as exc:
            logger.error("delta storage write failed", exc=str(exc))
            return False

    def read(self, path: str, format: StorageFormat = StorageFormat.PARQUET) -> Any:
        """Read all rows from the Delta table at *path*.

        Returns a list of dicts — same shape as the other backends.
        """
        if not _HAS_DELTALAKE:
            logger.error("delta storage: deltalake not installed")
            return None
        try:
            from deltalake import DeltaTable  # noqa: PLC0415

            table_path = self._table_path(path)
            if not (Path(table_path) / "_delta_log").exists():
                logger.warning("delta table not found", path=table_path)
                return None
            dt = DeltaTable(table_path)
            return dt.to_pyarrow_table().to_pylist()
        except Exception as exc:
            logger.error("delta storage read failed", exc=str(exc))
            return None

    def file_uris(self, path: str) -> list[str]:
        """Return the active Parquet files in the current Delta snapshot."""
        if not _HAS_DELTALAKE:
            return []
        try:
            from deltalake import DeltaTable  # noqa: PLC0415

            table_path = self._table_path(path)
            if not (Path(table_path) / "_delta_log").exists():
                return []
            return [str(uri) for uri in DeltaTable(table_path).file_uris()]
        except Exception as exc:
            logger.error("delta storage file listing failed", exc=str(exc))
            return []

    def parquet_scan_sql(self, path: str) -> str:
        """Return a DuckDB relation expression for the current Delta snapshot."""
        files = self.file_uris(path)
        if not files:
            raise FileNotFoundError(f"Delta table has no active Parquet files: {path}")
        quoted = []
        for file_uri in files:
            safe_uri = file_uri.replace("'", "''")
            quoted.append(f"'{safe_uri}'")
        return f"read_parquet([{', '.join(quoted)}])"

    def delete(self, path: str) -> bool:
        """Remove the Delta table directory at *path* entirely."""
        try:
            import shutil  # noqa: PLC0415

            table_path = Path(self._table_path(path))
            if table_path.exists():
                shutil.rmtree(table_path)
                logger.info("deleted delta table", path=str(table_path))
            return True
        except Exception as exc:
            logger.error("delta storage delete failed", exc=str(exc))
            return False

    def list_objects(self, prefix: str = "") -> list[str]:
        """Return paths of all Delta tables under *prefix* (relative to base_path)."""
        try:
            target = self.base_path / prefix if prefix else self.base_path
            if not target.exists():
                return []
            return [
                str(p.parent.relative_to(self.base_path))
                for p in target.rglob("_delta_log")
                if p.is_dir()
            ]
        except Exception as exc:
            logger.error("delta storage list_objects failed", exc=str(exc))
            return []

    def exists(self, path: str) -> bool:
        """Return ``True`` if a Delta table exists at *path*."""
        return (Path(self._table_path(path)) / "_delta_log").exists()

    @staticmethod
    def _to_records(data: Any) -> list[dict[str, Any]]:
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        return []


# ---------------------------------------------------------------------------
# S3 storage (requires boto3)
# ---------------------------------------------------------------------------

try:
    import boto3

    _HAS_BOTO3 = True
except ImportError:
    _HAS_BOTO3 = False


class S3Storage(StorageBackend):
    """AWS S3 object storage backend.

    Reads/writes JSON-serialised records to an S3 bucket.  Requires
    ``boto3`` at runtime.

    Parameters
    ----------
    bucket:
        S3 bucket name.
    prefix:
        Key prefix for all objects (default ``""``).
    region:
        AWS region (default ``"us-east-1"``).
    endpoint_url:
        Custom endpoint for S3-compatible services (e.g. LocalStack).
        When ``None``, the default AWS endpoint is used.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        region: str = "us-east-1",
        endpoint_url: str | None = None,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self.region = region
        self.endpoint_url = endpoint_url

        if not _HAS_BOTO3:
            logger.warning("boto3 not installed — s3 storage operations will fail")
            self._client = None
        else:
            client_kwargs: dict[str, Any] = {"region_name": region}
            if endpoint_url:
                client_kwargs["endpoint_url"] = endpoint_url
            self._client = boto3.client("s3", **client_kwargs)
            logger.info("s3 storage initialised", uri=f"s3://{bucket}/{prefix}")

    def _key(self, path: str) -> str:
        return f"{self.prefix}/{path}" if self.prefix else path

    def write(
        self,
        data: Any,
        path: str,
        format: StorageFormat = StorageFormat.PARQUET,
    ) -> bool:
        """Write JSON-serialised data to S3."""
        if self._client is None:
            logger.error("s3 storage: boto3 not available")
            return False
        try:
            records = data if isinstance(data, list) else [data]
            body = _json.dumps(records, default=str).encode()
            key = self._key(f"{path}.json")
            self._client.put_object(Bucket=self.bucket, Key=key, Body=body)
            logger.info("wrote records to s3", count=len(records), uri=f"s3://{self.bucket}/{key}")
            return True
        except Exception as exc:
            logger.error("s3 storage write failed", exc=str(exc))
            return False

    def read(self, path: str, format: StorageFormat = StorageFormat.PARQUET) -> Any:
        """Read JSON data from S3."""
        if self._client is None:
            logger.error("s3 storage: boto3 not available")
            return None
        try:
            key = self._key(f"{path}.json")
            response = self._client.get_object(Bucket=self.bucket, Key=key)
            body = response["Body"].read().decode()
            return _json.loads(body)
        except Exception as exc:
            logger.error("s3 storage read failed", exc=str(exc))
            return None

    def delete(self, path: str) -> bool:
        """Delete object from S3."""
        if self._client is None:
            logger.error("s3 storage: boto3 not available")
            return False
        try:
            key = self._key(f"{path}.json")
            self._client.delete_object(Bucket=self.bucket, Key=key)
            logger.info("deleted s3 object", uri=f"s3://{self.bucket}/{key}")
            return True
        except Exception as exc:
            logger.error("s3 storage delete failed", exc=str(exc))
            return False

    def list_objects(self, prefix: str = "") -> list[str]:
        """List S3 objects under *prefix*."""
        if self._client is None:
            return []
        try:
            full_prefix = self._key(prefix)
            resp = self._client.list_objects_v2(Bucket=self.bucket, Prefix=full_prefix)
            return [obj["Key"] for obj in resp.get("Contents", [])]
        except Exception as exc:
            logger.error("s3 storage list_objects failed", exc=str(exc))
            return []

    def exists(self, path: str) -> bool:
        """Return ``True`` if *path* exists in S3."""
        if self._client is None:
            return False
        try:
            self._client.head_object(Bucket=self.bucket, Key=self._key(f"{path}.json"))
            return True
        except self._client.exceptions.NoSuchKey:
            return False
        except Exception as exc:
            # Surface auth/permission errors instead of silently returning False
            error_code = getattr(
                getattr(exc, "response", None),
                "Error",
                {},
            ).get("Code", "")
            if error_code in ("404", "NoSuchKey"):
                return False
            logger.error("s3 storage exists check failed", exc=str(exc))
            raise


# ---------------------------------------------------------------------------
# GCS storage (requires google-cloud-storage)
# ---------------------------------------------------------------------------

try:
    from google.cloud import storage as gcs_storage

    _HAS_GCS = True
except ImportError:
    _HAS_GCS = False


# ---------------------------------------------------------------------------
# BigQuery storage (requires google-cloud-bigquery)
# ---------------------------------------------------------------------------

try:
    from google.cloud import bigquery as bq_client

    _HAS_BIGQUERY = True
except ImportError:
    bq_client = None
    _HAS_BIGQUERY = False


class BigQueryStorage(StorageBackend):
    """Google BigQuery storage backend.

    Reads/writes JSON rows to BigQuery tables.  Requires
    ``google-cloud-bigquery`` at runtime.

    Path convention: ``dataset.table`` — the *path* argument is split on
    the first ``"."`` to derive the dataset and table identifiers.

    Parameters
    ----------
    project_id:
        GCP project ID.
    dataset:
        Default BigQuery dataset for operations when *path* does not
        contain a ``"."`` separator.  Defaults to ``"dex"``.
    location:
        BigQuery dataset location (default ``"US"``).
    client:
        Optional pre-configured ``bigquery.Client`` (useful for tests).
    """

    def __init__(
        self,
        project_id: str,
        dataset: str = "dex",
        location: str = "US",
        client: Any = None,
    ) -> None:
        self.project_id = project_id
        self.dataset = dataset
        self.location = location

        if client is not None:
            self._client = client
            logger.info("bigquery storage initialised with injected client", project=project_id)
        elif not _HAS_BIGQUERY:
            logger.warning("google-cloud-bigquery not installed — install dataenginex[cloud]")
            self._client = None
        else:
            self._client = bq_client.Client(project=project_id, location=location)
            logger.info("bigquery storage initialised", project=project_id)

    def _table_ref(self, path: str) -> str:
        """Resolve *path* to ``project.dataset.table``."""
        if "." in path:
            ds, table = path.split(".", 1)
        else:
            ds, table = self.dataset, path
        return f"{self.project_id}.{ds}.{table}"

    def write(
        self,
        data: Any,
        path: str,
        format: StorageFormat = StorageFormat.BIGQUERY,
    ) -> bool:
        """Load *data* (list of dicts) into a BigQuery table."""
        if self._client is None:
            logger.error("bigquery storage: google-cloud-bigquery not available")
            return False
        try:
            records = data if isinstance(data, list) else [data]
            table_ref = self._table_ref(path)
            job_config: Any = None
            try:
                from google.cloud import bigquery as bq  # noqa: PLC0415

                job_config = bq.LoadJobConfig(
                    source_format=bq.SourceFormat.NEWLINE_DELIMITED_JSON,
                    autodetect=True,
                    write_disposition=bq.WriteDisposition.WRITE_APPEND,
                )
            except ImportError:
                pass  # injected client (tests) — job_config left as None
            job = self._client.load_table_from_json(
                records,
                table_ref,
                job_config=job_config,
            )
            job.result()  # block until complete
            logger.info("wrote records to bigquery", count=len(records), table=table_ref)
            return True
        except Exception as exc:
            logger.error("bigquery storage write failed", exc=str(exc))
            return False

    def read(self, path: str, format: StorageFormat = StorageFormat.BIGQUERY) -> Any:
        """Query all rows from a BigQuery table."""
        if self._client is None:
            logger.error("bigquery storage: google-cloud-bigquery not available")
            return None
        try:
            table_ref = self._table_ref(path)
            rows = self._client.list_rows(table_ref)
            return [dict(row) for row in rows]
        except Exception as exc:
            logger.error("bigquery storage read failed", exc=str(exc))
            return None

    def delete(self, path: str) -> bool:
        """Delete a BigQuery table."""
        if self._client is None:
            logger.error("bigquery storage: google-cloud-bigquery not available")
            return False
        try:
            table_ref = self._table_ref(path)
            self._client.delete_table(table_ref, not_found_ok=True)
            logger.info("deleted bigquery table", table=table_ref)
            return True
        except Exception as exc:
            logger.error("bigquery storage delete failed", exc=str(exc))
            return False

    def list_objects(self, prefix: str = "") -> list[str]:
        """List tables in the dataset, optionally filtered by *prefix*."""
        if self._client is None:
            return []
        try:
            dataset_ref = f"{self.project_id}.{self.dataset}"
            tables = self._client.list_tables(dataset_ref)
            names = [t.table_id for t in tables]
            if prefix:
                names = [n for n in names if n.startswith(prefix)]
            return names
        except Exception as exc:
            logger.error("bigquery storage list_objects failed", exc=str(exc))
            return []

    def exists(self, path: str) -> bool:
        """Return ``True`` if the BigQuery table exists."""
        if self._client is None:
            return False
        try:
            table_ref = self._table_ref(path)
            self._client.get_table(table_ref)
            return True
        except Exception as exc:
            error_type = type(exc).__name__
            if error_type == "NotFound":
                return False
            logger.error("bigquery storage exists check failed", exc=str(exc))
            raise


class GCSStorage(StorageBackend):
    """Google Cloud Storage backend.

    Reads/writes JSON-serialised records to a GCS bucket.  Requires
    ``google-cloud-storage`` at runtime.

    Parameters
    ----------
    bucket:
        GCS bucket name.
    prefix:
        Key prefix for all objects (default ``""``).
    project:
        GCP project ID (optional, uses ADC default).
    api_endpoint:
        Custom API endpoint for GCS-compatible services (e.g.
        ``fake-gcs-server``).  When ``None``, the default Google
        endpoint is used.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        project: str | None = None,
        api_endpoint: str | None = None,
    ) -> None:
        self.bucket_name = bucket
        self.prefix = prefix.rstrip("/")

        if not _HAS_GCS:
            logger.warning("google-cloud-storage not installed — gcs storage operations will fail")
            self._bucket = None
        else:
            if api_endpoint:
                # Use anonymous credentials + ClientOptions for local emulators
                from google.api_core.client_options import ClientOptions  # noqa: PLC0415
                from google.auth.credentials import (
                    AnonymousCredentials,  # noqa: PLC0415, E501
                )

                client = gcs_storage.Client(
                    project=project or "test-project",
                    credentials=AnonymousCredentials(),
                    client_options=ClientOptions(api_endpoint=api_endpoint),
                )
            else:
                client = gcs_storage.Client(project=project)
            self._bucket = client.bucket(bucket)
            logger.info("gcs storage initialised", uri=f"gs://{bucket}/{prefix}")

    def _blob_name(self, path: str) -> str:
        return f"{self.prefix}/{path}" if self.prefix else path

    def write(
        self,
        data: Any,
        path: str,
        format: StorageFormat = StorageFormat.PARQUET,
    ) -> bool:
        """Write JSON-serialised data to GCS."""
        if self._bucket is None:
            logger.error("gcs storage: google-cloud-storage not available")
            return False
        try:
            records = data if isinstance(data, list) else [data]
            body = _json.dumps(records, default=str)
            blob = self._bucket.blob(self._blob_name(f"{path}.json"))
            blob.upload_from_string(body, content_type="application/json")
            logger.info(
                "wrote records to gcs",
                count=len(records),
                uri=f"gs://{self.bucket_name}/{blob.name}",
            )
            return True
        except Exception as exc:
            logger.error("gcs storage write failed", exc=str(exc))
            return False

    def read(self, path: str, format: StorageFormat = StorageFormat.PARQUET) -> Any:
        """Read JSON data from GCS."""
        if self._bucket is None:
            logger.error("gcs storage: google-cloud-storage not available")
            return None
        try:
            blob = self._bucket.blob(self._blob_name(f"{path}.json"))
            body = blob.download_as_text()
            return _json.loads(body)
        except Exception as exc:
            logger.error("gcs storage read failed", exc=str(exc))
            return None

    def delete(self, path: str) -> bool:
        """Delete object from GCS."""
        if self._bucket is None:
            logger.error("gcs storage: google-cloud-storage not available")
            return False
        try:
            blob = self._bucket.blob(self._blob_name(f"{path}.json"))
            blob.delete()
            logger.info("deleted gcs object", uri=f"gs://{self.bucket_name}/{blob.name}")
            return True
        except Exception as exc:
            logger.error("gcs storage delete failed", exc=str(exc))
            return False

    def list_objects(self, prefix: str = "") -> list[str]:
        """List GCS objects under *prefix*."""
        if self._bucket is None:
            return []
        try:
            full_prefix = self._blob_name(prefix)
            return [blob.name for blob in self._bucket.list_blobs(prefix=full_prefix)]
        except Exception as exc:
            logger.error("gcs storage list_objects failed", exc=str(exc))
            return []

    def exists(self, path: str) -> bool:
        """Return ``True`` if *path* exists in GCS."""
        if self._bucket is None:
            return False
        try:
            result: bool = self._bucket.blob(self._blob_name(f"{path}.json")).exists()
            return result
        except (AttributeError, TypeError):
            return False
        except Exception as exc:
            # Surface auth/permission errors instead of silently returning False
            logger.error("gcs storage exists check failed", exc=str(exc))
            raise


# ---------------------------------------------------------------------------
# Storage factory
# ---------------------------------------------------------------------------


def get_storage(uri: str, **kwargs: Any) -> StorageBackend:
    """Create a :class:`StorageBackend` from a URI scheme.

    Supported schemes:

    * ``file://`` (or no scheme) → :class:`JsonStorage`
    * ``s3://bucket/prefix``     → :class:`S3Storage`
    * ``gs://bucket/prefix``     → :class:`GCSStorage`

    Extra *kwargs* are forwarded to the backend constructor.

    Raises
    ------
    ValueError
        If the URI scheme is not supported.
    """
    from urllib.parse import urlparse

    parsed = urlparse(uri)
    if parsed.scheme in ("", "file"):
        path = parsed.path or "data"
        return JsonStorage(base_path=path, **kwargs)
    if parsed.scheme == "s3":
        return S3Storage(
            bucket=parsed.netloc,
            prefix=parsed.path.lstrip("/"),
            **kwargs,
        )
    if parsed.scheme == "gs":
        return GCSStorage(
            bucket=parsed.netloc,
            prefix=parsed.path.lstrip("/"),
            **kwargs,
        )
    if parsed.scheme == "bq":
        return BigQueryStorage(
            project_id=parsed.netloc,
            dataset=parsed.path.lstrip("/") or "dex",
            **kwargs,
        )
    if parsed.scheme == "delta":
        base = (parsed.netloc + parsed.path).lstrip("/") or "data"
        return DeltaStorage(base_path=base, **kwargs)
    msg = f"Unsupported storage URI scheme: {parsed.scheme!r}"
    raise ValueError(msg)
