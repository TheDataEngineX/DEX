"""Data domain — what an ingestion or storage operation *means* (§5.3).

Medallion layering, catalog entries, schema registration, profiling, and the
pipeline graph. None of it names a storage SDK: the bytes are written by
``providers/``, and swapping Parquet for Delta must not reach in here.

Deliberately does not re-export its submodules. The pre-0.6 ``data/__init__``
pulled the whole layer into one namespace, which meant importing a profiler
also imported every connector. Callers name the module they need.
"""

from __future__ import annotations

__all__: list[str] = []
