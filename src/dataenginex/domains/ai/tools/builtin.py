"""Built-in tools for agent runtimes.

Provides tools that agents can invoke: SQL queries, ML inference,
semantic search, pipeline status, etc.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from dataenginex import _json
from dataenginex.domains.ai.tools import ToolSpec, tool_registry

if TYPE_CHECKING:
    pass

# Names available after register_builtin_tools() — used by the config validator.
BUILTIN_TOOL_NAMES: frozenset[str] = frozenset(
    {"query", "list_tools", "echo", "predict", "search_similar", "rag_search"}
)

logger = structlog.get_logger()


# ── SQL / lakehouse ────────────────────────────────────────────────────────────

# This tool is reachable from HTTP-exposed agent/native-call routes, so the SQL
# it runs must be treated as untrusted input. DuckDB's enable_external_access
# blocks filesystem/network access (read_csv/read_parquet/httpfs/ATTACH/etc.)
# for the connection it's set on; the statement-shape check below is a second,
# cheap layer against non-SELECT statements (COPY/INSTALL/PRAGMA/...).
_READ_ONLY_PREFIXES = ("SELECT", "WITH")


def _require_read_only(sql: str) -> None:
    stripped = sql.strip().lstrip("(").strip()
    if ";" in stripped[:-1]:  # allow a single optional trailing semicolon
        raise ValueError("query tool only allows a single statement")
    if not stripped.upper().startswith(_READ_ONLY_PREFIXES):
        raise ValueError("query tool only allows read-only SELECT/WITH statements")


def _query_sql(sql: str, database: str = ":memory:") -> list[dict[str, Any]]:
    """Execute a read-only SQL query via DuckDB and return results."""
    import duckdb

    _require_read_only(sql)
    conn = duckdb.connect(database)
    try:
        conn.execute("SET enable_external_access=false")
        result = conn.execute(sql)
        description = result.description or []
        if not description:
            return []
        columns = [desc[0] for desc in description]
        return [dict(zip(columns, row, strict=True)) for row in result.fetchall()]
    finally:
        conn.close()


_TABLE_REF_RE = re.compile(r"\b(?:FROM|JOIN)\s+\"?([A-Za-z_][A-Za-z0-9_]*)\"?", re.IGNORECASE)


def _referenced_table_names(sql: str) -> set[str]:
    """Best-effort extraction of table names a SELECT/WITH statement reads.

    Not a real SQL parser — a regex over FROM/JOIN clauses — so it can miss
    exotic syntax (e.g. a table name inside a subquery alias collision) or
    over-match a CTE name as if it were a lakehouse table. Both failure
    modes are safe here: a missed table just fails at materialization time
    with its usual "table not found" error (same as before this existed); an
    over-matched CTE name simply finds no matching file and is skipped.
    """
    return {m.group(1) for m in _TABLE_REF_RE.finditer(sql)}


def _materialize_tables(conn: Any, layer_path: Path, names: set[str]) -> set[str]:
    """Materialize tables from Parquet/Delta, return remaining names."""
    remaining = set(names)
    for name in list(remaining):
        pf = layer_path / f"{name}.parquet"
        if pf.exists():
            safe = str(pf).replace("'", "''")
            with contextlib.suppress(Exception):
                conn.execute(
                    f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM read_parquet('{safe}')"
                )
                remaining.discard(name)
            continue
        delta_dir = layer_path / name
        if delta_dir.is_dir() and (delta_dir / "_delta_log").exists():
            with contextlib.suppress(Exception):
                from dataenginex.providers.object_store.storage import DeltaStorage

                scan = DeltaStorage(base_path=str(layer_path)).parquet_scan_sql(name)
                conn.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM {scan}')
                remaining.discard(name)
    return remaining


def _make_lakehouse_query(lakehouse_dir: Path) -> Callable[[str], list[dict[str, Any]]]:
    """Return a query function that materializes only the Parquet/Delta
    tables each query actually references, not the whole lakehouse.

    Earlier versions materialized every bronze/silver/gold table on every
    call (a real OOM with tens-of-millions-of-row sources), then a 30s-TTL
    cache reduced that to once per TTL window — but the *size* of a single
    full materialization still grows with the lakehouse itself, so once
    real data volumes were reached (e.g. bronze_titles at 12.6M rows, a
    bronze_tmdb_movie_details batch with rich nested JSON per row), even one
    materialization was enough to OOM a resource-constrained container.
    Scoping to referenced tables only bounds cost to what a query actually
    needs, regardless of how large the rest of the lakehouse grows.
    """

    def _materialize_all(conn: Any, names: set[str]) -> None:
        remaining = set(names)
        for layer in ("bronze", "silver", "gold"):
            if not remaining:
                return
            layer_path = lakehouse_dir / layer
            if not layer_path.exists():
                continue
            remaining = _materialize_tables(conn, layer_path, remaining)

    def _query_with_lakehouse(sql: str) -> list[dict[str, Any]]:
        _require_read_only(sql)
        import duckdb

        # A fresh connection per call, not cached/reused: DuckDB refuses to
        # re-enable external_access once disabled on a connection ("Cannot
        # enable external access while database is running"), so a
        # long-lived connection can only ever materialize tables it already
        # knew about at creation time — defeating scoping to just what each
        # query references. Scoping the materialization (below) instead of
        # loading the whole lakehouse is what actually bounds the cost, so
        # paying connection-setup overhead per call is cheap by comparison.
        conn = duckdb.connect(":memory:")
        try:
            _materialize_all(conn, _referenced_table_names(sql))
            # Lock the connection down to the tables just materialized —
            # untrusted `sql` below can no longer touch the filesystem/network.
            conn.execute("SET enable_external_access=false")
            result = conn.execute(sql)
            description = result.description or []
            if not description:
                return []
            columns = [desc[0] for desc in description]
            return [dict(zip(columns, row, strict=True)) for row in result.fetchall()]
        finally:
            conn.close()

    return _query_with_lakehouse


# ── ML inference ───────────────────────────────────────────────────────────────


def _make_predict(
    models_dir: Path,
    registry_path: Path | None = None,
) -> Callable[..., Any]:
    """Return a predict function that loads and calls sklearn models by name.

    Looks up the model artifact path from registry.json (if present) or falls
    back to ``<models_dir>/<model_name>_v*.pkl`` glob.
    """

    def _predict(model_name: str, features: dict[str, Any]) -> Any:
        import pickle

        artifact: Path | None = None

        # Try registry first
        reg = registry_path or models_dir / "registry.json"
        if reg.exists():
            with contextlib.suppress(Exception):
                data = _json.loads(reg.read_text())
                versions = data.get(model_name, [])
                if versions:
                    # Prefer production stage, else latest
                    prod = [v for v in versions if v.get("stage") == "production"]
                    entry = (prod or versions)[-1]
                    artifact = Path(entry["artifact_path"])

        # Fallback: glob for <model_name>_v*.pkl
        if artifact is None or not artifact.exists():
            candidates = sorted(models_dir.glob(f"{model_name}_v*.pkl"))
            if candidates:
                artifact = candidates[-1]

        if artifact is None or not artifact.exists():
            available = [p.stem.rsplit("_v", 1)[0] for p in models_dir.glob("*_v*.pkl")]
            return {"error": f"Model '{model_name}' not found. Available: {list(set(available))}"}

        try:
            with artifact.open("rb") as f:
                model = pickle.load(f)  # noqa: S301

            import pandas as pd

            df = pd.DataFrame([features])
            prediction = model.predict(df)
            result: Any = prediction.tolist() if hasattr(prediction, "tolist") else list(prediction)
            return {"model": model_name, "prediction": result, "artifact": artifact.name}
        except Exception as exc:
            return {"error": f"Prediction failed: {exc}"}

    return _predict


# ── Semantic search ────────────────────────────────────────────────────────────


def _make_search_similar(
    vector_store: Any,
    embed_fn: Any | None = None,
    lexical_backend: Any = None,
) -> Callable[..., list[dict[str, Any]]]:
    """Return hybrid search backed by the vector store and optional lexical backend."""

    def _search_similar(query: str, top_k: int = 5) -> list[dict[str, Any]]:
        try:
            from dataenginex.providers.vector.vectorstore import RAGPipeline, SearchResult

            rag = RAGPipeline(store=vector_store, embed_fn=embed_fn, dimension=384)
            dense_results = rag.query(query, top_k=top_k * 2)
            lexical_results = (
                lexical_backend.search(query, top_k=top_k * 2)
                if lexical_backend is not None and lexical_backend.health_check()
                else []
            )
            if lexical_results:
                by_id: dict[str, Any] = {}
                scores: dict[str, float] = {}
                for ranking in (dense_results, lexical_results):
                    for rank, result in enumerate(ranking):
                        doc_id = result.document.id
                        by_id[doc_id] = result.document
                        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (60 + rank + 1)
                ranked = sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)[:top_k]
                results = [
                    SearchResult(document=by_id[doc_id], score=scores[doc_id]) for doc_id in ranked
                ]
                method = "hybrid"
            else:
                results = dense_results[:top_k]
                method = "dense"
            if not results:
                return [{"info": "Vector store is empty — run a gold pipeline to populate it."}]
            return [
                {
                    "id": r.document.id,
                    "text": r.document.text,
                    "score": round(r.score, 4),
                    "method": method,
                    **r.document.metadata,
                }
                for r in results
            ]
        except Exception as exc:
            return [{"error": str(exc)}]

    return _search_similar


# ── Builtins ───────────────────────────────────────────────────────────────────


def _list_tools() -> list[str]:
    return tool_registry.list()


def _echo(message: str) -> str:
    return message


# ── Registration ───────────────────────────────────────────────────────────────


def register_builtin_tools(
    lakehouse_dir: Path | None = None,
    models_dir: Path | None = None,
    vector_store: Any = None,
    embed_fn: Any | None = None,
    lexical_backend: Any = None,
) -> None:
    """Register all built-in tools.

    Args:
        lakehouse_dir: When provided, the ``query`` tool pre-registers every
            parquet file in bronze/silver/gold as a DuckDB view.
        models_dir: When provided, registers a ``predict`` tool that loads
            sklearn models from the model registry.
        vector_store: When provided, registers a ``search_similar`` tool for
            semantic search over embedded lakehouse documents.
        embed_fn: Embedding callable for ``search_similar``. Falls back to
            hash-based embedding when ``None``.
    """
    query_fn: Callable[..., Any]
    if lakehouse_dir:
        query_fn = _make_lakehouse_query(lakehouse_dir)
        query_desc = (
            "Execute a SQL query against the lakehouse. "
            "All bronze/silver/gold tables are pre-registered as views."
        )
        query_params: dict[str, str] = {"sql": "str"}
    else:
        query_fn = _query_sql
        query_desc = "Execute a SQL query via DuckDB"
        query_params = {"sql": "str", "database": "str (optional)"}

    builtins: list[ToolSpec] = [
        ToolSpec(name="query", description=query_desc, fn=query_fn, parameters=query_params),
        ToolSpec(
            name="list_tools",
            description="List all available tools",
            fn=_list_tools,
            parameters={},
        ),
        ToolSpec(
            name="echo",
            description="Echo a message back",
            fn=_echo,
            parameters={"message": "str"},
        ),
    ]

    if models_dir and models_dir.exists():
        registry_path = models_dir / "registry.json"
        predict_fn = _make_predict(models_dir, registry_path if registry_path.exists() else None)
        builtins.append(
            ToolSpec(
                name="predict",
                description=(
                    "Run ML model inference. "
                    "Args: model_name (str), features (dict of feature→value). "
                    "Available models: "
                    + ", ".join(
                        p.stem.rsplit("_v", 1)[0] for p in sorted(models_dir.glob("*_v*.pkl"))
                    )
                ),
                fn=predict_fn,
                parameters={"model_name": "str", "features": "dict"},
            )
        )

    if vector_store is not None:
        search_fn = _make_search_similar(vector_store, embed_fn, lexical_backend)
        builtins.extend(
            [
                ToolSpec(
                    name="search_similar",
                    description=(
                        "Semantic similarity search over the movie catalog. "
                        "Finds movies similar to a natural-language query. "
                        "Args: query (str), top_k (int, default 5)."
                    ),
                    fn=search_fn,
                    parameters={"query": "str", "top_k": "int (optional, default 5)"},
                ),
                ToolSpec(
                    name="rag_search",
                    description="RAG search over the indexed lakehouse documents.",
                    fn=search_fn,
                    parameters={"query": "str", "top_k": "int (optional, default 5)"},
                ),
            ]
        )

    for spec in builtins:
        tool_registry.register(spec)
