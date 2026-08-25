"""Provider implementations — the mechanism layer (§5.3, §5.5).

Providers implement foundation contracts and may depend on external libraries.
That permission is what makes this layer different from every other one: boto3,
deltalake, pyarrow, duckdb, qdrant, and sentence-transformers live here or they
live nowhere.

The seam between this layer and ``domains/`` is mechanism versus semantics, and
it is observable rather than a matter of taste. A file that imports a storage
SDK is describing *how bytes are written*; a file that imports nothing but the
domain model is describing *what an operation means*. The first belongs here,
the second belongs in ``domains/``.

That distinction is what makes a backend swap real. DuckDB to PostgreSQL, or
filesystem to S3, has to be a wiring change in ``bootstrap/`` and nothing else.
If Parquet layout lived in a domain, swapping the store would mean editing that
domain — and the contract would have bought nothing.

Nothing here is imported by ``foundation``, ``application``, or ``domains``.
They depend on the Protocols in ``foundation/contracts.py``; ``bootstrap/`` is
the only place a concrete provider is named.
"""

__all__: list[str] = []
