"""Object storage providers — filesystem, S3, GCS, BigQuery (§5.3).

Imports boto3, google-cloud, and pyarrow, which is exactly why it is here and
not in a domain. Partitioning ships alongside because a partition layout is a
physical key convention, not a statement about the data.
"""

from __future__ import annotations

__all__: list[str] = []
