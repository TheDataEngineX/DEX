# Spark Pipeline Execution — Design

**Date:** 2026-08-14
**Status:** Draft — scoped, not yet implemented
**Author:** Claude (session with jaymyaka, dex-studio repo)

## Problem

`PipelineConfig.engine` supports `"duckdb"` (default) and `"spark"`. The DuckDB
path (`PipelineRunner.run()` in
`src/dataenginex/domains/data/pipeline/runner.py`) is a real, production-grade
implementation: extract → transform chain → quality gate → Delta write, with
careful on-disk/spill-directory handling to avoid OOM on 100M+ row tables.

The Spark path (`PipelineRunner._run_spark()`, same file, line ~296) is not
functional:

1. **Wrong client API.** It calls `spark_client.initialize()` and
   `spark_client.stop()`. `SparkConnectClient`
   (`src/dataenginex/spark/connect/client.py`) only exposes `.connect()` and
   `.disconnect()`. Every `engine: spark` pipeline run crashes immediately
   with `AttributeError` before doing anything.
2. **No real stages.** Even with the method names fixed, the four "stages"
   (`spark_extract`, `spark_transform`, `spark_quality`, `spark_load`) are
   checkpoint/progress markers only — no extract, no transform, no quality
   check, no write. The function unconditionally returns
   `PipelineResult(success=True, rows_input=0, rows_output=0)`. Fixing #1
   alone would make broken pipelines report fake success while moving zero
   rows, which is worse than the current crash.

Confirmed empirically: set `bronze_wiki_edits` (real IMDB/Wikipedia live
source) to `engine: spark` in a moviedex `dex.yaml`, ran it via dex-studio —
immediate crash, exact trace above.

## Goal

Full parity with the DuckDB engine: every pipeline that runs correctly with
`engine: duckdb` must run correctly — same transform semantics, same quality
gate behavior, same Delta output — with `engine: spark`, differing only in
the execution engine underneath.

## Scope

**In scope:**

- Fix `SparkConnectClient` method calls in `_run_spark()`.
- Implement all 9 transform types from `TransformStepConfig`
  (`src/dataenginex/config/schema.py`) as Spark DataFrame operations:
  `filter`, `derive`, `cast`, `deduplicate`, `sql`, `rename`,
  `drop_columns`, `fill_null`, `aggregate`, `window` (10 listed in the
  schema comment, `window` and `aggregate` share `partition_by`/`order_by`
  and `group_by`/`agg_exprs` respectively — treat as 10 distinct step
  kinds).
- Source extraction into a Spark DataFrame (reuse existing source registry /
  connector config — do not duplicate source-reading logic).
- Quality gate reusing the same `QualityCheckConfig` semantics
  (`completeness`, `uniqueness`, custom SQL) the DuckDB path already
  enforces, evaluated against the Spark DataFrame.
- Delta-lake write via Spark's Delta connector, targeting the **same
  lakehouse path** the DuckDB engine writes to
  (`.dex/lakehouse/{layer}/{destination}`), so tables are cross-engine
  readable/writable (a `spark`-engine pipeline's output must be queryable by
  a downstream `duckdb`-engine pipeline, and vice versa).
- Local `local[*]` Spark mode only (matches current single-node DuckDB
  model). `sc://` remote Spark Connect config path already exists in
  `SparkConnectClient.connect()` — do not need to build it, just don't
  break it.
- Dev/test environment: add `pyspark` as an actually-installed dependency
  for the group that needs it (`uv sync --group spark`) and document the
  Java requirement. Neither PySpark nor Java is present in the current dev
  environment — confirmed via `import pyspark` failing.

**Out of scope (explicitly not this spec):**

- Distributed/remote Spark cluster execution — `local[*]` only.
- Converting streaming pipelines (`bronze_wiki_edits_stream` and similar,
  driven by Kafka consumers) to the Spark engine — they stay on their
  current path.
- Spark Structured Streaming.
- Performance tuning / partitioning strategy beyond "correct and works."
- Any dex-studio (UI) changes — the SQL console's `_run_sql_spark()` and its
  `eng.spark_sql` (`SparkSQLExecutor`) path were already fixed separately
  (dex-studio session, 2026-08-14) to call the correct API; this spec is
  pipeline execution only, a different code path (`eng.spark_client` /
  `SparkConnectClient`, not `eng.spark_sql`).

## Architecture

Mirror the DuckDB path's stage structure, swap the engine:

```
extract (source → DataFrame)
  → transform chain (TransformStepConfig[] → chained DataFrame ops)
  → quality gate (QualityCheckConfig → pass/fail, same semantics as DuckDB path)
  → load (DataFrame → Delta write at .dex/lakehouse/{layer}/{destination})
```

**New component: `SparkTransformApplier`**
(`src/dataenginex/spark/transforms/applier.py`, new file). One method per
transform kind, translating `TransformStepConfig` → DataFrame op:

| type | config fields | Spark op |
|---|---|---|
| `filter` | `condition` | `df.filter(expr(condition))` |
| `derive` | `expression`, `name` | `df.withColumn(name, expr(expression))` |
| `cast` | `columns: {col: type}` | `df.withColumn(col, col(col).cast(type))` per entry |
| `deduplicate` | `key: str \| list[str]` | `df.dropDuplicates(key)` |
| `sql` | `sql` | `df.createOrReplaceTempView("_data")` then `spark.sql(sql)` |
| `rename` | `mapping: {old: new}` | `df.withColumnRenamed(old, new)` per entry |
| `drop_columns` | `columns: list[str]` | `df.drop(*columns)` |
| `fill_null` | `defaults: {col: value}` | `df.fillna(defaults)` |
| `aggregate` | `group_by`, `agg_exprs: {out_col: expr}` | `df.groupBy(*group_by).agg(...)` |
| `window` | `expression`, `name`, `partition_by`, `order_by` | `df.withColumn(name, expr(expression).over(Window.partitionBy(...).orderBy(...)))` |

Same `_data` view-name convention the DuckDB `sql` transform already uses
(confirm exact name against `runner.py`'s DuckDB `sql` transform before
implementing — keep them identical so pipeline authors don't need
engine-specific SQL).

**Extract**: reuse whatever the DuckDB path uses to resolve `source →
read config` (connector registry) — read the same way, just materializing
into `spark.read` instead of a DuckDB relation. Do not duplicate
source-config parsing.

**Quality gate**: apply the same `QualityCheckConfig` fields
(`completeness`, `uniqueness`, `custom_sql`) against the Spark DataFrame —
`completeness` via null-ratio aggregation, `uniqueness` via
groupBy-count-filter, `custom_sql` via the same temp-view pattern as the
`sql` transform.

**Load**: Spark's native Delta writer
(`df.write.format("delta").mode(...).save(path)`), targeting the identical
path DuckDB's `DeltaStorage` writes to, so both engines' outputs are
interchangeable.

## Error handling

- `SparkConnectClient.connect()` already raises `ImportError` with a clear
  message when PySpark isn't installed — keep this, it's correct.
- `_run_spark()` must catch failures **per stage**, not swallow them, and
  return `PipelineResult(success=False, error=str(exc))` — matching the
  DuckDB path's contract that `execute_sql`/router code already depends on
  (dex-studio's pipeline detail page reads `error` from run records).
- No stage should report success while producing zero rows unless the
  source genuinely had zero rows (that's a legitimate empty-source case,
  distinguishable from "we didn't actually run" by checking `rows_input`).

## Testing

Cannot be tested in the current dev environment as-is — needs:

1. `uv sync --group spark` (installs `pyspark>=4.2.0`).
2. A JVM (Java 17+, matching Spark 4.2 requirements) available on the dev
   machine / CI runner.

Test plan once the environment is ready:

- **Unit**: one test per transform-type translation, small in-memory
  DataFrames, asserting output matches the DuckDB engine's output for the
  same `TransformStepConfig` + input rows (property: same config, same
  input, same output, regardless of engine).
- **Integration**: run a real pipeline end-to-end against a live source
  (e.g. `bronze_wiki_edits` against the live Wikipedia recent-changes feed,
  the same manual test done during scoping) with `engine: spark`, assert
  real row counts, not zeros.
- **Cross-engine**: write a table with `engine: duckdb`, read it via a
  Spark-engine pipeline's extract step (and the reverse) — assert both
  succeed, proving Delta-format interop holds.
- **Failure path**: force a transform error (bad SQL, missing column),
  assert `PipelineResult(success=False, error=...)` propagates correctly,
  no silent fake-success.

## Open questions for implementer

- Exact `_data` temp-view name and whether the DuckDB `sql` transform's SQL
  dialect (DuckDB SQL) is close enough to Spark SQL that pipeline authors'
  existing `sql`-type transform configs work unmodified, or whether this
  needs a translation/compatibility note in pipeline docs.
- Whether `aggregate`/`window` step configs used anywhere in the existing
  moviedex example pipelines exercise edge cases (multi-column `group_by`,
  ordered window functions) worth prioritizing in initial implementation.
