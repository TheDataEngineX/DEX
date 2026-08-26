"""Delta Lake connector — ACID reads/writes via delta-rs.

Reads return a list of dicts (same contract as all other connectors).
Writes use ``write_deltalake()`` with optimistic concurrency — concurrent
writers retry on conflict instead of corrupting data.

Requires ``deltalake>=0.24.0``:
    pip install 'dataenginex[delta]'
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

from dataenginex.foundation.plugin_contracts import BaseConnector
from dataenginex.providers.connectors import connector_registry

logger = structlog.get_logger()

try:
    from deltalake import DeltaTable
    from deltalake import write_deltalake as _write_deltalake

    _HAS_DELTALAKE = True
except ImportError:
    _HAS_DELTALAKE = False


@connector_registry.decorator("delta")
class DeltaConnector(BaseConnector):
    """Delta Lake connector backed by delta-rs.

    Supports ACID writes, time travel reads, schema evolution, and
    VACUUM for GDPR-compliant row deletion.

    Args:
        path: Path to the Delta table directory.
        mode: Default write mode: ``"append"`` (default), ``"overwrite"``,
            ``"error"``, or ``"ignore"``.
        version: Optional table version for time-travel reads
            (``None`` = latest).
    """

    def __init__(
        self,
        path: str = ".",
        mode: str = "append",
        version: int | None = None,
        **kwargs: Any,
    ) -> None:
        self._path = Path(path)
        self._mode = mode
        self._version = version
        self._connected = False

    def connect(self) -> None:
        if not _HAS_DELTALAKE:
            msg = "deltalake package required — pip install 'dataenginex[delta]'"
            raise RuntimeError(msg)
        self._connected = True
        logger.debug("delta connector ready", path=str(self._path))

    def disconnect(self) -> None:
        self._connected = False

    def read(
        self,
        *,
        table: str | None = None,
        version: int | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Read rows from the Delta table.

        Args:
            table: Ignored — path is fixed at construction time.
            version: Override the version for this read only (time-travel);
                defaults to ``self._version``.
        """
        if not self._connected:
            msg = "Not connected — call connect() first"
            raise RuntimeError(msg)
        table_path = str(self._path)
        read_version = version if version is not None else self._version
        try:
            dt = DeltaTable(table_path, version=read_version)
            rows: list[dict[str, Any]] = dt.to_pyarrow_table().to_pylist()
            logger.info("delta read", path=table_path, rows=len(rows), version=read_version)
            return rows
        except Exception as exc:
            logger.error("delta read failed", path=table_path, exc=str(exc))
            raise

    def write(
        self,
        data: Any,
        *,
        table: str | None = None,
        mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Write *data* to the Delta table.

        Args:
            data: ``list[dict]`` or ``pyarrow.Table``.
            table: Ignored — path is fixed at construction time.
            mode: Override write mode for this call only.
            **kwargs: Forwarded to ``write_deltalake()``
                (e.g. ``partition_by=``, ``schema_mode=``).
        """
        if not self._connected:
            msg = "Not connected — call connect() first"
            raise RuntimeError(msg)

        import pyarrow as pa  # noqa: PLC0415

        if isinstance(data, list):
            if not data:
                logger.warning("no records to write", path=str(self._path))
                return
            arrow_table = pa.Table.from_pylist(data)
        elif isinstance(data, pa.Table):
            arrow_table = data
        else:
            msg = f"Unsupported data type for delta write: {type(data)}"
            raise TypeError(msg)

        table_path = str(self._path)
        write_mode = mode or self._mode
        _write_deltalake(table_path, arrow_table, mode=write_mode, **kwargs)
        logger.info(
            "delta written",
            path=table_path,
            rows=len(arrow_table),
            mode=write_mode,
        )

    def health_check(self) -> bool:
        if not self._connected or not _HAS_DELTALAKE:
            return False
        return (self._path / "_delta_log").exists()

    def vacuum(self, retention_hours: int = 168) -> None:
        """Remove old Parquet files no longer referenced by the transaction log.

        Default retention is 7 days (168 hours).  Required for GDPR
        compliance when rows are deleted via ``DELETE`` operations.
        """
        if not self._connected:
            msg = "Not connected — call connect() first"
            raise RuntimeError(msg)
        dt = DeltaTable(str(self._path))
        dt.vacuum(retention_hours=retention_hours, dry_run=False)
        logger.info(
            "delta vacuum complete",
            path=str(self._path),
            retention_hours=retention_hours,
        )
