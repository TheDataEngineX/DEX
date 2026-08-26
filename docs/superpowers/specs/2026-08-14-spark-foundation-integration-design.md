# Spark Foundation Integration — Design

**Date:** 2026-08-14 (session 2)
**Status:** Draft — full-repo audit complete, not yet implemented
**Supersedes/extends:** `2026-08-14-spark-pipeline-execution-design.md` (that spec covers
the pipeline-runner fix only; this one covers all 8 Spark subsystems wired into
`DexEngine` plus the ones that turned out to be dead code). The runner spec's
architecture section is still correct except where corrected below (transform
count, `_data` mechanism, Delta write dependency) — read both.

## Context

Session 1 (this repo, earlier today) scoped the pipeline-runner crash and paused
for cost/scope reasons. This session: installed Java 17 + `pyspark` 4.2.0 in the
dev environment (confirmed working — local `SparkSession`, `count()`, `stop()` all
succeed), then read every file under `dataenginex/src/dataenginex/spark/` (40
files, 3168 lines) plus every call site that touches them, to answer "what's
actually real vs. scaffolding" with evidence instead of guesses.

**Environment note (pre-existing, not part of this work):** `dataenginex/pyproject.toml`
and `uv.lock` have a large uncommitted change in progress that splits the old
`dataenginex-workspace` uv-workspace root (which depended on both `dataenginex`
and `dex-studio` as members) into a standalone `dataenginex` package. That change
is left untouched by this spec. One consequence: `dataenginex`'s venv is now
separate from `dex-studio`'s, so `uv sync --group spark` only affects the
`dataenginex` venv, which is correct — dex-studio doesn't need pyspark, it talks
to `dataenginex` as a library dependency, not the reverse.

## Audit findings

`DexEngine._init_spark()` (`src/dataenginex/engine.py:1286`) eagerly constructs
8 Spark subsystem objects at engine startup, exposed as `eng.spark_*`
attributes. Every one of these `__init__` methods is cheap/lazy (no real Spark
connection happens until a method is explicitly called), so eager construction
itself isn't a problem. What each one actually *does* when called varies a lot:

| `eng.` attribute | Class | Real callers today | Verdict |
|---|---|---|---|
| `spark_client` | `SparkConnectClient` | `PipelineRunner._run_spark()` (broken — wrong method names) | **Real code, broken caller.** Fix per existing runner spec. |
| `spark_sql` | `SparkSQLExecutor` | dex-studio SQL console (`routers/data.py:_run_sql_spark`) | **Real code, reachable.** `_execute_local` fallback always returns 0 rows when `spark_client` isn't connected — same root cause as `spark_client` above. |
| `spark_catalog` | `SparkCatalogAdapter` | dex-studio catalog page (`routers/data.py:2021`) | **Real Hive code path, but never used.** See below — currently injects silent mock data into a real UI page. |
| `spark_streaming` | `StreamingQueryManager` | none | **Real, functional-looking code (Kafka/Delta/rate sources, Delta/Kafka/console sinks via Structured Streaming). Zero callers**, and out of scope per the approved runner spec ("streaming pipelines stay on their current Kafka-consumer path"). |
| `spark_ml` | `MLlibProvider` | none — dex-studio's actual "Train model" button (`routers/intelligence.py:run_finetune`) calls `dex_studio.tools.builtins._tool_finetune`, a separate sklearn-based path that doesn't touch this class at all | **Real, fairly complete MLlib+MLflow wrapper (with sklearn fallback). Genuinely orphaned** — the working training UI bypasses it entirely. |
| `spark_metrics` | `ResourceAccounting` | none | **Real code, zero callers.** Natural fit to wire into the runner fix (below) since that code is being touched anyway. |
| `spark_lineage` | `OpenLineageProjection` | none (only constructed if `openlineage_url` config is set, which it never is — no OpenLineage server exists anywhere in this stack) | **Real HTTP-based OpenLineage client, targets infrastructure that doesn't exist here.** Same shape of problem as Hive Metastore below. |
| `spark_datasource_registry` | `DataSourceRegistry` | none | **Internally consistent generic plugin loader** (not actually a Spark DataSource V2 registration, despite the docstring) — nothing ever calls `.register()`. |

### Orphaned scaffolding (not wired into `engine.py` at all)

These files are imported by exactly one place outside their own module:
`tests/unit/test_spark_module.py` — and that test file is **currently red**:
5 of 21 tests fail because the tests were written against a different, earlier
version of these classes' APIs (wrong constructor args, wrong method names,
wrong expected return values — e.g. `DataSourceRegistry()` called with no
`project_id`, `ResourceAccounting.record_usage()` which doesn't exist,
`MLlibProvider.train()` asserted to return `status: "submitted"` when real code
returns `"completed"`/`"error"`). Confirmed by running the suite.

- `spark/session/` (`configuration.py`, `lifecycle.py`, `isolation.py`) — session
  config/state-machine/scope-check objects. **Not used by `SparkConnectClient`**,
  which builds and configures its own `SparkSession` directly. Keeping both
  means two competing, inconsistent ways to configure a session.
- `spark/local/` (`lazy_startup.py`, `profile.py`) — `LazySparkStartup.ensure_started()`
  is a no-op (`# ponytail: actual lazy startup deferred to runtime`) that just
  flips a bool. Not called anywhere.
- `spark/pipelines/` (`declarative.py`, `project_adapter.py`, `run_projection.py`) —
  Spark Declarative Pipelines (SDP) adapter. The runner fix (existing spec)
  does **not** go through SDP — it builds its own extract/transform/quality/load
  stages directly, same shape as the DuckDB path. `discover_pipelines()` is a
  stub (`return []`), `validate()` is a stub (`return True`).
- `spark/catalog/resource_projection.py` (`SparkResourceProjection`) — not used
  by `spark/catalog/adapter.py` or anywhere else.
- `spark/lineage/spark_to_dex_lineage.py` (`SparkToDexLineage`) — not used by
  `OpenLineageProjection` or anywhere else.
- `spark/metrics/metrics_correlation.py` (`MetricsCorrelation`) — not used by
  `ResourceAccounting` or anywhere else.

**Decision: delete these files and their test classes**, rather than "fix them
in place." They don't do anything real today, nothing calls them, and the
functionality they claim (session config, pipeline discovery, resource/lineage
correlation) is either already handled correctly elsewhere (`SparkConnectClient`,
`PipelineRunner`) or has no current consumer. Fixing their internals to be
"correct" would just produce a second, unused, competing implementation —
that's a maintenance trap, not a fix. `spark/catalog/identifiers.py`
(`SparkCatalogIdentifier`) is the one exception in `catalog/` — it's actually
used by `adapter.py`, stays.

## Scope

### 1. Pipeline runner fix (`_run_spark`) — as previously specced, with 3 corrections

The existing runner spec undercounts the transform types and gets the `_data`
mechanism slightly wrong. Corrections, confirmed against
`src/dataenginex/domains/analytics/transforms/sql.py` (the real DuckDB
implementations):

- **12 transform types, not 10** — the existing spec missed `explode` (array
  un-nesting via UNNEST, drops the exploded struct/list column) and
  `json_normalize` (flattens a STRUCT column into top-level columns). Spark
  equivalents: `explode` → `df.select(*keep, F.explode(col).alias(alias))`
  (or `F.explode_outer` if empty/null arrays should produce a null row instead
  of being dropped — DuckDB's `UNNEST` **drops** them, so match with plain
  `explode`, not `explode_outer`); `json_normalize` → `df.select(*keep,
  *[col(f"{column}.{f}").alias(prefix+f) for f in field_names])` where
  `field_names` come from the DataFrame's schema for that struct column
  (`df.schema[column].dataType.names`).
- **`_data` is literal string substitution, not a real temp view.** The DuckDB
  `sql` transform does `self._sql.replace("_data", input_table)` then
  `CREATE TABLE ... AS (resolved_sql)` — there's no view actually named
  `_data`. Mirror exactly: `SparkTransformApplier`'s `sql` case should
  `df.createOrReplaceTempView(current_table_name)`, then
  `spark.sql(step.sql.replace("_data", current_table_name))` — same
  substitution semantics, not a hardcoded `_data` view.
- **Quality gate's `custom_sql`** uses the identical `.replace("_data", table)`
  pattern (`runner.py:962`) — same substitution approach applies there too.

**Extract**: `runner.py:_extract` calls `connector.read(table=read_table)`,
which returns a `pyarrow.Table`/`pyarrow.dataset.Dataset` or `list[dict]`
(runner.py:708-726). Reuse this — don't re-implement source resolution. For
Spark, after getting the same object back from the connector registry, convert
via `spark.createDataFrame(arrow_table.to_pandas())` (safest cross-version
path; avoid relying on `spark.createDataFrame(pa_table)` directly since Arrow-native
DataFrame construction support varies by PySpark point release).

**Load — missing dependency, must be added.** `DeltaStorage.write()`
(`src/dataenginex/providers/object_store/storage.py`) uses the `deltalake`
(delta-rs) Python package's `write_deltalake()` — genuine Delta Lake format
(real `_delta_log/` transaction log), so Spark's native `.format("delta")`
writer is protocol-compatible for the cross-engine interop this spec requires.
**But** `delta-spark` (the JVM package providing `io.delta:delta-spark` /
`DeltaSparkSessionExtension`) is **not currently a dependency anywhere** —
`pyproject.toml`'s `delta` group only has `deltalake>=1.6.2` (the Python-native
library the DuckDB path uses, no JVM involved). Without it, Spark's
`df.write.format("delta")` fails with `ClassNotFoundException` at runtime —
`engine.py`'s `spark.sql.extensions`/`spark.sql.catalog.spark_catalog` config
values reference classes that come from that missing package. **Add
`delta-spark>=4.0.0` to the `spark` dependency group**, and in
`SparkConnectClient.connect()` use `delta.pip_utils.configure_spark_with_delta_pip(builder)`
before `.getOrCreate()` so the JAR resolves automatically (first run needs
network access to fetch from Maven; cached under `~/.ivy2` after).

**Resource accounting — small, in-scope addition.** Since `runner.py` is being
edited anyway: after each Spark stage, call
`eng.spark_metrics.record_job_metrics(run_id, {...})` using
`spark.sparkContext.statusTracker()` for job/stage/executor metrics
(`ResourceAccounting.collect_spark_metrics()` already has the
`statusTracker().getExecutorInfos()` call — reuse it) so Spark pipeline runs
get the same cost/perf visibility DuckDB runs already have via existing
metrics. This is the one place in this spec where a previously-orphaned
subsystem gets a real caller, because the cost of wiring it is near-zero given
the runner is already being touched.

Everything else in the existing runner spec (architecture diagram, error
handling contract, testing plan, `local[*]`-only scope) stands unchanged.

### 2. Catalog adapter — real fix, because it's live in production today

`routers/data.py:2021` calls `eng.spark_catalog.list_databases()` /
`.list_tables(db)` directly on every catalog-page load, unconditionally, inside
a `contextlib.suppress(Exception)` block. Three real bugs, confirmed by
reading the code:

1. **`.connect()` is never called.** `SparkCatalogAdapter._client` starts
   `None` and only a call to `.connect()` populates it. The catalog page never
   calls `.connect()`, so `_client` is always `None`, so every call falls
   through to the hardcoded mock branch (`{"default", "bronze", "silver",
   "gold"}` fake databases, `{"employees", "departments", "sales"}` fake
   tables) — **silently, forever**, mixed into the real catalog listing next to
   genuine DuckDB tables with no visual distinction. This is actively
   misleading, worse than showing nothing.
2. **Type bug**: `spark_catalog.list_tables(db)` passes the whole database
   *dict* (`{"name": ..., "description": ...}`) where `list_tables(database:
   str)` expects a string — should be `db["name"]`. Currently masked because
   the mock-path branch never uses its `database` argument (returns the same 3
   fake tables regardless), so no exception occurs — but it means the loop
   also runs once per fake database, injecting the *same 3 fake tables 4
   times* (12 fake rows total) into the catalog page today, uncommitted.
3. **`catalog_type="file"` (the actual default — `engine.py:1318`,
   `getattr(self.config, "spark_catalog_type", "file")`) has no real
   implementation at all** — `_connect_file_catalog()` just logs and leaves
   `_client = None`. The Hive Metastore and Unity Catalog code paths are real
   (Hive) or explicitly stubbed (Unity), but **neither applies to this
   platform** — this is local-first, single-node, no Hive Metastore or Unity
   Catalog service exists anywhere in the stack (confirmed: no
   `hive-site.xml`, no metastore service in any compose/deployment file). The
   catalog type that's actually in play (`"file"`) is the one with zero real
   implementation.

**Fix**: (a) call `spark_catalog.connect()` once at catalog-page load, before
the list calls; (b) fix the `list_tables(db["name"])` bug; (c) implement
`_connect_file_catalog()` for real — scan `.dex/lakehouse/{layer}/*/`
(mirrors `DeltaStorage`'s layout) for `_delta_log/` directories, and for each,
read table metadata via `deltalake.DeltaTable(path).schema()` /
`.metadata()` — no Spark session or JVM needed for this, it's the same
Python-native `deltalake` package the DuckDB path already depends on. This
makes the catalog page's "Spark" column genuinely show the real Delta tables
DuckDB/Spark pipelines have written, instead of 12 rows of fake data.
Hive/Unity code paths are left as-is (real for Hive, stub for Unity) —
`pyhive[hive]` fails to build on Python 3.13 in this environment (`sasl`
extension needs `longintrepr.h`, removed from CPython's public headers in
3.11+ — a known upstream `sasl`/Python-version incompatibility, not something
fixable from this repo), so the Hive path stays best-effort/untested,
documented as a known limitation rather than chased further — there's no Hive
Metastore server in this stack to test against even if it built.

### 3. Streaming, ML, lineage, data-source registry — fix tests, don't build new integration

For `StreamingQueryManager`, `MLlibProvider`, `OpenLineageProjection`, and
`DataSourceRegistry`: the implementations are real (not stubs) and internally
consistent, but each requires either infrastructure that doesn't exist here
(OpenLineage server) or has no current call site by explicit prior decision
(streaming pipelines stay off the Spark engine) or by absence of a use case
(no custom Spark data sources registered anywhere, ML training UI already
works via a separate sklearn path). Building real UI/pipeline integration for
any of these — a streaming query dashboard, a "train via Spark" option in the
finetune page, a lineage graph viewer, a data-source registration UI — is new
feature work with no current requirement driving it, not a "fix what's
broken." Scope for this spec: **fix the 3 failing tests for these classes
(`DataSourceRegistry`, `MLlibProvider`, `OpenLineageProjection`) to call the
real APIs**, verify each still behaves correctly against real `pyspark` (smoke
test, not new integration), and leave them unwired. `ResourceAccounting`
follows the same "fix + verify" treatment plus the one wiring point described
in §1.

### 4. Delete orphaned scaffolding

Delete: `spark/session/` (3 files + `__init__.py`), `spark/local/` (2 files +
`__init__.py`), `spark/pipelines/` (3 files + `__init__.py`),
`spark/catalog/resource_projection.py`, `spark/lineage/spark_to_dex_lineage.py`,
`spark/metrics/metrics_correlation.py`. Update `spark/__init__.py` and the
per-package `__init__.py` files that re-export these to drop the removed
names. Rewrite `tests/unit/test_spark_module.py`: remove the test classes for
deleted code, fix the 5 failing tests' API calls to match real signatures,
keep the rest.

## Error handling

Same contract as the existing runner spec: per-stage failure capture, no
silent fake-success, `PipelineResult(success=False, error=...)`. For the
catalog fix: if `.connect()` fails (e.g. `.dex/lakehouse/` doesn't exist yet
for a fresh project), fall through to an **empty** list, not the mock data —
"no Spark tables yet" is honest, "12 fake rows" is not.

## Testing

Now possible end-to-end with Java 17 + `pyspark` 4.2.0 installed:

- **Unit**: one test per transform-type translation (12 types, small in-memory
  DataFrames), asserting Spark output matches DuckDB output for identical
  config + input (property test, per existing spec). Rewritten
  `test_spark_module.py` for the classes that remain.
- **Integration**: real pipeline run with `engine: spark` against a live
  source (moviedex `bronze_wiki_edits` or similar), asserting real row counts.
- **Cross-engine**: write with `engine: duckdb`, read via a `engine: spark`
  pipeline's extract step and vice versa — proves the `deltalake`/`delta-spark`
  interop claim holds in practice, not just in theory.
- **Catalog**: point `SparkCatalogAdapter` (file mode) at a `.dex/lakehouse/`
  with real Delta tables (produced by the integration test above), assert
  `list_databases()`/`list_tables()` return real data, assert catalog page no
  longer shows the 12 fake rows.
- **Failure path**: forced transform error, forced quality-gate failure —
  assert `success=False` propagates, no silent success.
- **Deletion safety**: confirm `ruff`/`mypy`/full test suite are clean after
  removing the orphaned files (no stray imports left dangling).

## Open questions for implementer

- `SparkConnectClient.execute_sql()` logs
  `self._session.sparkSession.sessionId` (`client.py:84`) — this attribute
  path looks wrong for a plain embedded `SparkSession` (`.sparkSession` isn't
  a real attribute on it; that shape belongs to Spark Connect's remote client
  object). Verify empirically once implementing — likely needs to become
  `self._session.sparkContext.applicationId` or simply dropped from the log
  line for local mode.
- Delta JAR resolution (`configure_spark_with_delta_pip`) needs network access
  on first run in any given environment (CI included) to pull from Maven —
  confirm CI has network egress for this, or pre-warm the `~/.ivy2` cache in
  the CI image, or the first Spark-engine pipeline run in CI will be slow/flaky.
