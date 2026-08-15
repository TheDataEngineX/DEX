"""Data source capabilities (§20.6)."""

from __future__ import annotations

from dataenginex.foundation.projects import FrozenModel

__all__ = ["DataSourceCapabilities"]


class DataSourceCapabilities(FrozenModel):
    """Declared capabilities of a data source connector (§20.6)."""

    supports_batch_read: bool = True
    supports_batch_write: bool = False
    supports_streaming_read: bool = False
    supports_streaming_write: bool = False
    supports_pushdown: bool = False
    supports_row_level_ops: bool = False
    max_parallelism: int = 1
