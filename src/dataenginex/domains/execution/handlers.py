"""Handlers that make the operation types actually do something (§4.7).

Until this module existed the execution path was complete but hollow: a run
could be queued, claimed, leased, planned, and completed — and execute nothing,
because ``InProcessBackend`` had no handler for any operation type and failed
every plan with "no handler registered".

Each handler bridges one operation type to the domain code that already
implements it. The bridge is thin on purpose: resolve the bound resources, call
into ``domains/`` or ``providers/``, return the logical names produced. The
domain code neither knows nor cares that a worker invoked it.

**Operations exchange data through a per-attempt DuckDB workspace.** ``ingest``
lands rows in a table named by its declared output; ``transform`` reads that
table and writes another; ``validate`` checks one. The workspace lives under the
attempt's artifact namespace, so two attempts of the same run cannot see each
other's intermediate tables — the isolation §14.3's commit protocol assumes but
cannot provide on its own.

Four types are deliberately absent: ``publish``, ``export``, ``notify``, and
``delete`` — the ones whose side-effect class is ``external_write`` or
``destructive``. A stub that returned success would be worse than the current
failure, because the run would be recorded as having sent something it never
sent.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from dataenginex.domains.analytics.quality import ColumnSpec, check_quality
from dataenginex.domains.analytics.transforms import transform_registry
from dataenginex.domains.execution.backends import InProcessBackend, OperationHandler
from dataenginex.foundation import ExecutionContext, ExecutionPlan, Operation
from dataenginex.foundation.plugin_contracts import BaseConnector

log = structlog.get_logger().bind(src="handlers")

__all__ = [
    "ConnectorFactory",
    "HandlerError",
    "ResultSink",
    "default_handlers",
    "handle_ingest",
    "handle_lakehouse_inventory",
    "handle_schema_inspect",
    "handle_sql_preview",
    "handle_table_stats",
    "handle_transform",
    "handle_validate",
    "register_default_handlers",
    "workspace_path",
]

# Given a resource's declared config, produce something that can read it. Passed
# in rather than imported: ``domains/`` may not name a concrete provider (§5.5),
# and a handler that reaches into ``providers/`` directly would make "swap the
# backend at the wiring" false — the swap would mean editing this file.
ConnectorFactory = Callable[[Mapping[str, str]], BaseConnector]

# Where an interactive handler puts what it produced. Injected for the same
# reason as the connector factory: writing to the control store from here would
# make ``domains/`` depend on ``runtime/``.
ResultSink = Callable[[ExecutionContext, dict[str, Any], int, bool], None]


class HandlerError(RuntimeError):
    """An operation could not run as declared.

    Raised for a *declaration* problem — a resource that was never declared, a
    transform type nobody registered. An underlying library failing propagates
    as itself, so the recorded error class stays diagnostic.
    """


# --- shared plumbing --------------------------------------------------------


def workspace_path(context: ExecutionContext) -> Path:
    """The DuckDB file a project's operations exchange tables through.

    Keyed on **project and revision**, so what one workload ingests is what the
    next one — and a SQL preview, and the catalog page — can read. An earlier
    version keyed this on the attempt, which isolated retries beautifully and
    made the workspace useless: every run started empty, so no workload could
    build on another's output and no preview could see anything at all.

    Isolation between attempts is the commit protocol's job (§14.3), not the
    file path's. Scoping to the revision as well as the project matters though:
    a published change to a table's shape must not silently reuse tables the
    previous revision's definition produced.
    """
    root = Path(os.environ.get("DEX_WORKSPACE_DIR") or tempfile.gettempdir()) / "dex-workspaces"
    root.mkdir(parents=True, exist_ok=True)
    token = context.capability
    return root / f"{token.project_id}__{token.revision_id}.duckdb"


def _open_workspace(context: ExecutionContext, *, read_only: bool = False) -> Any:
    """Open this attempt's workspace.

    *read_only* is the enforcement behind SQL preview, not a hint: DuckDB
    refuses writes on the connection, so a preview cannot modify anything
    whatever the user typed. An empty file has to exist first — DuckDB will not
    open a missing database read-only, and "no such file" is a confusing way to
    say "nothing has been ingested yet".
    """
    import duckdb

    path = workspace_path(context)
    if read_only and not path.exists():
        duckdb.connect(str(path)).close()
    return duckdb.connect(str(path), read_only=read_only)


def _resource(plan: ExecutionPlan, name: str) -> dict[str, str]:
    """The declared config for one bound resource name.

    Missing means the workload named a resource its revision does not declare.
    It fails with the name rather than falling back to a default, because a
    default here reads from somewhere nobody asked for.
    """
    raw = plan.inputs.get(name)
    if raw is None:
        raise HandlerError(
            f"operation is bound to resource {name!r}, which this revision does not declare"
        )
    config: dict[str, str] = json.loads(raw)
    return config


def _connect(connectors: ConnectorFactory, config: Mapping[str, str]) -> BaseConnector:
    """Build and open a connector for one resource.

    The factory is supplied by whoever wired the worker. This layer never names
    a provider, so it also cannot report which ones exist — a factory that
    cannot serve a type raises, and the message is the factory's to write.
    """
    connector = connectors(config)
    connector.connect()
    return connector


def _sole_output(operation: Operation, *, fallback: str) -> str:
    """The one table this operation writes.

    More than one declared output is refused rather than silently using the
    first: a workload that expects two tables and gets one is data loss nobody
    notices until something downstream reads a table that was never written.
    """
    if len(operation.bound_outputs) > 1:
        raise HandlerError(
            f"operation {operation.name!r} declares {len(operation.bound_outputs)} outputs; "
            "this handler writes exactly one"
        )
    return operation.bound_outputs[0] if operation.bound_outputs else fallback


def _sole_input(operation: Operation) -> str:
    """The one table this operation reads."""
    if not operation.bound_inputs:
        raise HandlerError(f"operation {operation.name!r} declares no input to read")
    return operation.bound_inputs[0]


def _each(plan: ExecutionPlan, operation_type: str) -> tuple[Operation, ...]:
    """The operations of one type in this plan, in declared order."""
    return tuple(op for op in plan.operations if op.operation_type == operation_type)


def _quote(name: str) -> str:
    """Quote an identifier for DuckDB.

    Table names come from the manifest, which constrains them to a safe
    character class — but they reach SQL by string interpolation, and a quoted
    identifier is the difference between that being merely true today and being
    structurally true.
    """
    return '"{}"'.format(name.replace('"', '""'))


# --- ingest -----------------------------------------------------------------


def handle_ingest(
    plan: ExecutionPlan, context: ExecutionContext, *, connectors: ConnectorFactory
) -> tuple[str, ...]:
    """Read each bound source through its connector into the workspace.

    Rows land in a DuckDB table named by the operation's declared output, so a
    downstream ``transform`` finds them under the name the manifest used.
    """
    produced: list[str] = []
    connection = _open_workspace(context)
    try:
        for operation in _each(plan, "ingest"):
            for source in operation.bound_inputs:
                config = _resource(plan, source)
                connector = _connect(connectors, config)
                try:
                    rows = connector.read(table=config.get("table") or source)
                finally:
                    connector.disconnect()

                target = _sole_output(operation, fallback=source)
                _write_rows(connection, target, rows)
                produced.append(target)
                log.info("ingested", operation=operation.name, source=source, target=target)
    finally:
        connection.close()
    return tuple(produced)


def _write_rows(connection: Any, table: str, rows: Any) -> None:
    """Land rows in the workspace as a table.

    An empty read still creates the table, empty. Skipping it would make a
    downstream transform fail with "table not found" — an error about the wrong
    thing, since the real answer is that the source had no rows.
    """
    import pyarrow as pa

    arrow = rows if isinstance(rows, pa.Table) else pa.Table.from_pylist(list(rows))
    connection.register("_dex_ingest", arrow)
    try:
        connection.execute(
            f"CREATE OR REPLACE TABLE {_quote(table)} AS SELECT * FROM _dex_ingest"  # noqa: S608
        )
    finally:
        connection.unregister("_dex_ingest")


# --- transform --------------------------------------------------------------


def handle_transform(plan: ExecutionPlan, context: ExecutionContext) -> tuple[str, ...]:
    """Apply each declared transform to its input table.

    The transform type comes from the operation's own parameters rather than
    the resource: two steps may apply different transforms to the same table.
    """
    produced: list[str] = []
    connection = _open_workspace(context)
    try:
        for operation in _each(plan, "transform"):
            kind = operation.parameters.get("transform") or "sql"
            settings = {k: v for k, v in operation.parameters.items() if k != "transform"}

            try:
                cls = transform_registry.get(kind)
            except KeyError as exc:
                raise HandlerError(f"no transform registered as {kind!r}") from exc

            try:
                transform = cls(**settings)
            except TypeError as exc:
                # A missing constructor argument is a manifest mistake, not a
                # bug in the transform. Saying so beats a bare TypeError about
                # a parameter name the author never typed.
                raise HandlerError(
                    f"transform {operation.name!r} of type {kind!r} is missing "
                    f"required parameters: {exc}"
                ) from exc

            problems = transform.validate()
            if problems:
                raise HandlerError(
                    f"transform {operation.name!r} is misconfigured: {'; '.join(problems)}"
                )

            source = _sole_input(operation)
            # Unquoted: a transform derives its output name by suffixing what it
            # is given (``orders_raw`` -> ``orders_raw_filtered``), so a quoted
            # identifier would produce ``"orders_raw"_filtered``. Manifest names
            # are constrained to a safe character class, which is what makes
            # that acceptable here.
            result = transform.apply(connection, source)
            target = _sole_output(operation, fallback=result)
            if target != result:
                connection.execute(
                    f"CREATE OR REPLACE TABLE {_quote(target)} AS "  # noqa: S608
                    f"SELECT * FROM {_quote(result)}"
                )
            produced.append(target)
            log.info("transformed", operation=operation.name, kind=kind, target=target)
    finally:
        connection.close()
    return tuple(produced)


# --- validate ---------------------------------------------------------------


def handle_validate(plan: ExecutionPlan, context: ExecutionContext) -> tuple[str, ...]:
    """Check each bound table against the operation's declared expectations.

    A failing check raises. A quality gate that logs and continues is not a
    gate: the run must not complete successfully when the data did not meet
    what the project declared.
    """
    produced: list[str] = []
    connection = _open_workspace(context)
    try:
        for operation in _each(plan, "validate"):
            for name in operation.bound_inputs:
                result = check_quality(
                    connection,
                    _quote(name),
                    completeness=_ratio(operation.parameters.get("completeness")),
                    uniqueness=_names(operation.parameters.get("unique")) or None,
                    row_count_min=_count(operation.parameters.get("row_count_min")),
                    schema=_schema(operation) or None,
                )
                if not result.passed:
                    raise HandlerError(
                        f"quality gate {operation.name!r} failed on {name!r}: "
                        f"completeness={result.completeness_score:.3f} "
                        f"uniqueness={result.uniqueness_score:.3f} "
                        f"violations={result.schema_violations}"
                    )
                produced.append(name)
                log.info("validated", operation=operation.name, table=name)
    finally:
        connection.close()
    return tuple(produced)


def _schema(operation: Operation) -> list[ColumnSpec]:
    """Column expectations declared as flat parameters.

    ``required`` names the columns that must exist and hold no nulls. A
    manifest is flat, so the list arrives comma-separated rather than nested.
    """
    required = _names(operation.parameters.get("required"))
    return [ColumnSpec(name=name, nullable=False) for name in required]


def _names(value: str | None) -> list[str]:
    return [part.strip() for part in (value or "").split(",") if part.strip()]


def _ratio(value: str | None) -> float | None:
    """A declared threshold, or None when the project did not set one.

    A malformed number is refused rather than defaulted: silently treating
    ``completeness: "ninety"`` as "no threshold" turns a typo into a gate that
    passes everything.
    """
    if value is None or not value.strip():
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise HandlerError(f"completeness must be a number, got {value!r}") from exc


def _count(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise HandlerError(f"row_count_min must be a whole number, got {value!r}") from exc


# --- interactive (§7.3) -----------------------------------------------------
#
# These three back the pages a person is sitting in front of: the SQL console,
# the schema panel, the catalog's table stats. They run on a worker like any
# other operation — that is the whole point, and the §17 Phase 1 criterion "no
# workload executes in Studio process".
#
# All three read through the same per-attempt DuckDB workspace, and all three
# cap what they return. An uncapped preview is a denial of service on the
# machine the user is already waiting on.


def _preview_rows(
    connection: Any, sql: str, limit: int, params: list[Any] | None = None
) -> tuple[list[str], list[list[Any]], bool]:
    """Run a query and return at most *limit* rows.

    Fetches one row beyond the cap so "there are more" can be reported as a
    fact rather than guessed from a row count that happens to equal the limit.
    """
    result = connection.execute(sql, params) if params else connection.execute(sql)
    columns = [str(d[0]) for d in result.description or []]
    rows = result.fetchmany(limit + 1)
    truncated = len(rows) > limit
    # JSON has no date type; a datetime would fail to serialise on the way to
    # the browser, and str() is what the UI displays anyway.
    body = [[_jsonable(value) for value in row] for row in rows[:limit]]
    return columns, body, truncated


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def handle_sql_preview(
    plan: ExecutionPlan,
    context: ExecutionContext,
    *,
    sink: ResultSink,
    max_rows: int,
) -> tuple[str, ...]:
    """Run a user's query against the project's data and store the rows.

    The query is read-only by construction: it runs on a connection opened
    read-only, so a preview cannot create, drop, or update anything. Relying on
    inspecting the SQL text instead would be a blocklist, and blocklists on SQL
    lose to the next syntax nobody thought of.
    """
    produced: list[str] = []
    for operation in _each(plan, "sql_preview"):
        sql = operation.parameters.get("sql", "").strip()
        if not sql:
            raise HandlerError("sql_preview requires a 'sql' parameter")

        # Bound values, never interpolated. A caller building a query around a
        # filesystem path needs these: a filename containing a quote would
        # otherwise close the literal and change the statement.
        raw_params = operation.parameters.get("params")
        params = json.loads(raw_params) if raw_params else None

        connection = _open_workspace(context, read_only=True)
        try:
            columns, rows, truncated = _preview_rows(connection, sql, max_rows, params)
        finally:
            connection.close()

        sink(context, {"columns": columns, "rows": rows}, len(rows), truncated)
        produced.append(operation.name or "sql_preview")
        log.info("sql preview", rows=len(rows), truncated=truncated)
    return tuple(produced)


def handle_schema_inspect(
    plan: ExecutionPlan,
    context: ExecutionContext,
    *,
    sink: ResultSink,
    connectors: ConnectorFactory,
    max_rows: int,
) -> tuple[str, ...]:
    """Describe a declared resource: its columns, and a few sample rows.

    Reads through the resource's own connector rather than the workspace,
    because the question is what the *source* looks like — a workspace table
    only exists after something ingested it.
    """
    produced: list[str] = []
    for operation in _each(plan, "schema_inspect"):
        name = _sole_input(operation)
        config = _resource(plan, name)
        connector = _connect(connectors, config)
        try:
            rows = list(connector.read(table=config.get("table") or name))
        finally:
            connector.disconnect()

        sample = rows[: min(max_rows, 20)]
        # Union of keys rather than the first row's: a connector reading JSON
        # can return records with different shapes, and describing only the
        # first one silently hides every other column.
        columns: list[str] = []
        for row in sample:
            for key in row:
                if key not in columns:
                    columns.append(key)

        sink(
            context,
            {
                "resource": name,
                "columns": [{"name": c, "type": _guess_type(sample, c)} for c in columns],
                "sample_rows": [
                    {k: _jsonable(v) for k, v in row.items()} for row in sample
                ],
            },
            len(rows),
            len(rows) > len(sample),
        )
        produced.append(name)
        log.info("schema inspected", resource=name, columns=len(columns))
    return tuple(produced)


def _guess_type(rows: list[dict[str, Any]], column: str) -> str:
    """Name the column's type from the first non-null value seen.

    A guess, and labelled as one. The connector contract does not carry a
    schema, so the alternative is showing nothing — which is less useful and no
    more honest, as long as nothing downstream treats this as authoritative.
    """
    for row in rows:
        value = row.get(column)
        if value is not None:
            return type(value).__name__
    return "unknown"


def handle_table_stats(
    plan: ExecutionPlan,
    context: ExecutionContext,
    *,
    sink: ResultSink,
    max_rows: int,
) -> tuple[str, ...]:
    """Row count and per-column null counts for a workspace table."""
    del max_rows
    produced: list[str] = []
    for operation in _each(plan, "table_stats"):
        table = _sole_input(operation)
        connection = _open_workspace(context, read_only=True)
        try:
            count_row = connection.execute(
                f"SELECT count(*) FROM {_quote(table)}"  # noqa: S608
            ).fetchone()
            total = int(count_row[0]) if count_row else 0
            described = connection.execute(f"DESCRIBE {_quote(table)}").fetchall()
            columns = [
                {
                    "name": str(row[0]),
                    "type": str(row[1]),
                    "nulls": _null_count(connection, table, str(row[0])),
                }
                for row in described
            ]
        finally:
            connection.close()

        sink(context, {"table": table, "row_count": total, "columns": columns}, total, False)
        produced.append(table)
        log.info("table stats", table=table, rows=total)
    return tuple(produced)


def _null_count(connection: Any, table: str, column: str) -> int:
    row = connection.execute(
        f"SELECT count(*) FROM {_quote(table)} WHERE {_quote(column)} IS NULL"  # noqa: S608
    ).fetchone()
    return int(row[0]) if row else 0


_LAYERS = ("bronze", "silver", "gold")


def handle_lakehouse_inventory(
    plan: ExecutionPlan,
    context: ExecutionContext,
    *,
    sink: ResultSink,
    max_rows: int,
) -> tuple[str, ...]:
    """Describe what is on disk in a project's lakehouse.

    Backs the warehouse, lakehouse, and catalog pages, which between them used
    to call four engine methods that globbed ``.dex/lakehouse`` and opened
    DuckDB *in the web process* — the same §17 Phase 1 violation the SQL console
    had, arrived at through the filesystem rather than through SQL.

    The root comes from the plan rather than from a project directory the worker
    happens to be able to see. A worker on another machine has no ``.dex`` to
    guess at, and a path it inferred would be a path nobody granted.
    """
    del max_rows
    produced: list[str] = []
    for operation in _each(plan, "lakehouse_inventory"):
        root = operation.parameters.get("lakehouse_root", "")
        if not root:
            raise HandlerError("lakehouse_inventory requires a 'lakehouse_root' parameter")
        lakehouse = Path(root)

        wanted = _names(operation.parameters.get("layers")) or list(_LAYERS)
        layers: list[dict[str, Any]] = []
        tables: list[dict[str, Any]] = []

        connection = _open_workspace(context, read_only=True)
        try:
            for layer in wanted:
                found = _layer_tables(connection, lakehouse / layer, layer)
                layers.append({"name": layer, "table_count": len(found)})
                tables.extend(found)
        finally:
            connection.close()

        sink(context, {"layers": layers, "tables": tables}, len(tables), False)
        produced.extend(t["name"] for t in tables)
        log.info("lakehouse inventoried", layers=len(layers), tables=len(tables))
    return tuple(produced)


def _layer_tables(connection: Any, layer_path: Path, layer: str) -> list[dict[str, Any]]:
    """Every parquet file and Delta directory in one layer, described."""
    if not layer_path.is_dir():
        return []

    entries = sorted(layer_path.glob("*.parquet"))
    entries += sorted(p for p in layer_path.iterdir() if (p / "_delta_log").is_dir())

    described: list[dict[str, Any]] = []
    for entry in entries:
        try:
            described.append(_describe_table(connection, entry, layer))
        except OSError:
            # A file that vanished mid-scan is not a reason to fail the whole
            # inventory — the other tables are still describable.
            log.warning("skipped unreadable lakehouse entry", path=str(entry))
    return described


def _describe_table(connection: Any, entry: Path, layer: str) -> dict[str, Any]:
    stat = entry.stat()
    size_bytes = (
        stat.st_size
        if entry.is_file()
        else sum(p.stat().st_size for p in entry.rglob("*") if p.is_file())
    )
    table_format = "parquet" if entry.is_file() else "delta"

    # Parameterised: the path comes from a directory listing, but a file named
    # with a quote would otherwise close the string literal.
    scan = "read_parquet(?)" if table_format == "parquet" else "delta_scan(?)"

    row_count: int | None = None
    columns: list[dict[str, str]] = []
    with contextlib.suppress(Exception):
        row = connection.execute(f"SELECT count(*) FROM {scan}", [str(entry)]).fetchone()  # noqa: S608
        row_count = int(row[0]) if row else None
        # Described in the same pass. The warehouse page wants a column count
        # per table, and fetching it separately meant one engine call per table
        # — the N+1 that made a twenty-table page slow.
        described = connection.execute(
            f"DESCRIBE SELECT * FROM {scan}",  # noqa: S608
            [str(entry)],
        ).fetchall()
        columns = [{"name": str(c[0]), "type": str(c[1])} for c in described]

    return {
        "name": entry.stem,
        "path": str(entry),
        "layer": layer,
        "size_bytes": size_bytes,
        "size": _human_size(size_bytes),
        "row_count": row_count,
        "columns": columns,
        "column_count": len(columns),
        "updated_at": datetime.fromtimestamp(stat.st_mtime, tz=UTC).strftime("%b %d %H:%M"),
        "format": table_format,
    }


def _human_size(size_bytes: int) -> str:
    if size_bytes < 1_048_576:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / 1_048_576:.1f} MB"


# --- registration -----------------------------------------------------------


def _max_rows(plan: ExecutionPlan) -> int:
    """The row cap this plan declared, or the conservative default.

    A malformed value falls back rather than raising: the cap exists to protect
    the machine, and refusing to run because it was mistyped protects nothing.
    """
    try:
        return max(1, int(plan.parameters.get("max_rows", "1000")))
    except ValueError:
        return 1000


def default_handlers(
    connectors: ConnectorFactory, sink: ResultSink | None = None
) -> dict[str, OperationHandler]:
    """Every operation type this installation can actually execute.

    Deliberately partial. The omitted types write outside the installation and
    have no implementation yet; leaving them unregistered means a project that
    declares one fails at its first run with "no handler registered" — a true
    statement — rather than succeeding without acting.

    Without a *sink* the interactive handlers are left out entirely rather than
    registered and silently discarding their output. A SQL preview that runs,
    reports success, and stores nothing is the worst of both: the user waits,
    and then sees nothing, with no error to explain it.
    """

    def ingest(plan: ExecutionPlan, context: ExecutionContext) -> tuple[str, ...]:
        return handle_ingest(plan, context, connectors=connectors)

    handlers: dict[str, OperationHandler] = {
        "ingest": ingest,
        "transform": handle_transform,
        "validate": handle_validate,
    }

    if sink is None:
        return handlers

    def sql_preview(plan: ExecutionPlan, context: ExecutionContext) -> tuple[str, ...]:
        return handle_sql_preview(plan, context, sink=sink, max_rows=_max_rows(plan))

    def schema_inspect(plan: ExecutionPlan, context: ExecutionContext) -> tuple[str, ...]:
        return handle_schema_inspect(
            plan, context, sink=sink, connectors=connectors, max_rows=_max_rows(plan)
        )

    def table_stats(plan: ExecutionPlan, context: ExecutionContext) -> tuple[str, ...]:
        return handle_table_stats(plan, context, sink=sink, max_rows=_max_rows(plan))

    def lakehouse_inventory(plan: ExecutionPlan, context: ExecutionContext) -> tuple[str, ...]:
        return handle_lakehouse_inventory(plan, context, sink=sink, max_rows=_max_rows(plan))

    handlers["sql_preview"] = sql_preview
    handlers["schema_inspect"] = schema_inspect
    handlers["table_stats"] = table_stats
    handlers["lakehouse_inventory"] = lakehouse_inventory
    return handlers


def register_default_handlers(
    backend: InProcessBackend,
    connectors: ConnectorFactory,
    sink: ResultSink | None = None,
) -> InProcessBackend:
    """Attach the default handlers to a backend and return it."""
    for operation_type, handler in default_handlers(connectors, sink).items():
        backend.register(operation_type, handler)
    return backend
