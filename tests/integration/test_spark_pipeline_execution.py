"""Integration tests for the Spark engine path in PipelineRunner._run_spark().

Requires: Java 17+, pyspark>=4.2.0, delta-spark>=4.0.0 installed
(``uv sync --group data`` + a working JAVA_HOME). Automatically skipped
otherwise via the ``requires_pyspark`` marker (see tests/conftest.py) — same
skip machinery used by tests/unit/test_spark_*.py.

Fixture pattern mirrors tests/unit/test_pipeline_runner.py's `sample_config`:
a real CSV file on disk registered as a "csv" source in a real dex.yaml,
loaded through `load_config()` — no mock connector, no `PipelineRunner(...)`
placeholder args.
"""

from __future__ import annotations

from pathlib import Path

from dataenginex.config import load_config
from dataenginex.domains.data.pipeline.runner import PipelineRunner
from tests.conftest import requires_pyspark


def _write_config(
    tmp_path: Path,
    *,
    filter_condition: str,
    pipeline_name: str = "spark_test_pipeline",
    with_quality: bool = False,
) -> Path:
    csv_file = tmp_path / "rows.csv"
    csv_file.write_text("id,value\n1,1\n2,5\n3,10\n4,0\n")

    quality_block = (
        """
      quality:
        row_count_min: 1
"""
        if with_quality
        else ""
    )

    config_file = tmp_path / "dex.yaml"
    config_file.write_text(f"""
project:
  name: spark-test-project

data:
  sources:
    fixture_source:
      type: csv
      connection:
        path: "{tmp_path}"
        default_file: "rows.csv"
  pipelines:
    {pipeline_name}:
      source: fixture_source
      engine: spark
      destination: spark_test_output
      transforms:
        - type: filter
          condition: "{filter_condition}"{quality_block}
""")
    return config_file


@requires_pyspark
def test_spark_pipeline_moves_real_rows(tmp_path: Path) -> None:
    config_file = _write_config(tmp_path, filter_condition="value > 1", with_quality=True)
    config = load_config(config_file)
    data_dir = tmp_path / ".dex" / "lakehouse"
    runner = PipelineRunner(config, data_dir=data_dir, project_dir=tmp_path)

    result = runner.run("spark_test_pipeline")

    assert result.success is True, result.error
    assert result.rows_input == 4
    assert result.rows_output > 0
    assert result.rows_output <= result.rows_input
    # filter "value > 1" drops rows with value 1 and 0 (ids 1, 4) -> 2 survive.
    assert result.rows_output == 2

    out_path = data_dir / "silver" / "spark_test_output"
    assert (out_path / "_delta_log").exists(), "spark load stage did not write a Delta table"


@requires_pyspark
def test_spark_pipeline_failure_path(tmp_path: Path) -> None:
    """A transform referencing a nonexistent column must fail loudly.

    Spark's analyzer resolves column references eagerly inside df.filter(),
    so SparkTransformApplier._apply_filter raises immediately — this must
    surface as success=False + error set, never a silent zero-row success.
    """
    config_file = _write_config(
        tmp_path,
        filter_condition="not_a_real_column > 1",
        pipeline_name="spark_broken_pipeline",
    )
    config = load_config(config_file)
    data_dir = tmp_path / ".dex" / "lakehouse"
    runner = PipelineRunner(config, data_dir=data_dir, project_dir=tmp_path)

    result = runner.run("spark_broken_pipeline")

    assert result.success is False
    assert result.error is not None and result.error != ""
    # Extract succeeded before the transform blew up.
    assert result.rows_input == 4
    assert result.rows_output == 0
    assert not (data_dir / "silver" / "spark_test_output" / "_delta_log").exists()


@requires_pyspark
def test_spark_pipeline_honors_target_layer_override(tmp_path: Path) -> None:
    """target.layer must win over destination-prefix inference — same contract
    as DuckDB's _load() (``cfg.target.get("layer", _infer_layer(name))``).

    destination is prefixed "bronze_" so the buggy version of this code —
    which inferred the layer from `destination` instead of consulting
    `target.layer` — would have written to bronze/ instead. Asserts the
    table actually lands under gold/, and nothing gets written to bronze/.
    """
    csv_file = tmp_path / "rows.csv"
    csv_file.write_text("id,value\n1,1\n2,5\n3,10\n4,0\n")

    config_file = tmp_path / "dex.yaml"
    config_file.write_text(f"""
project:
  name: spark-layer-override-test

data:
  sources:
    fixture_source:
      type: csv
      connection:
        path: "{tmp_path}"
        default_file: "rows.csv"
  pipelines:
    spark_layer_override_pipeline:
      source: fixture_source
      engine: spark
      destination: bronze_layer_override_test
      target:
        layer: gold
""")
    config = load_config(config_file)
    data_dir = tmp_path / ".dex" / "lakehouse"
    runner = PipelineRunner(config, data_dir=data_dir, project_dir=tmp_path)

    result = runner.run("spark_layer_override_pipeline")

    assert result.success is True, result.error
    gold_path = data_dir / "gold" / "bronze_layer_override_test"
    bronze_path = data_dir / "bronze" / "bronze_layer_override_test"
    assert (gold_path / "_delta_log").exists(), "target.layer override was not honored"
    assert not bronze_path.exists(), (
        "wrote to the destination-inferred layer instead of target.layer"
    )


@requires_pyspark
def test_duckdb_and_spark_engines_share_delta_tables(tmp_path: Path) -> None:
    """Write a Delta table via engine=duckdb, read it via a real Spark session,
    and write one via engine=spark, read it via deltalake (delta-rs) — the
    same library DuckDB's DeltaStorage uses. Proves on-disk Delta protocol
    interop, not just that both engines import cleanly.
    """
    csv_file = tmp_path / "rows.csv"
    csv_file.write_text("id,value\n1,10\n2,20\n3,30\n")

    config_file = tmp_path / "dex.yaml"
    config_file.write_text(f"""
project:
  name: interop-test-project

data:
  sources:
    fixture_source:
      type: csv
      connection:
        path: "{tmp_path}"
        default_file: "rows.csv"
  pipelines:
    duckdb_interop:
      source: fixture_source
      engine: duckdb
      destination: interop_from_duckdb
      target:
        layer: silver
        format: delta
    spark_interop:
      source: fixture_source
      engine: spark
      destination: interop_from_spark
""")
    config = load_config(config_file)
    data_dir = tmp_path / ".dex" / "lakehouse"
    runner = PipelineRunner(config, data_dir=data_dir, project_dir=tmp_path)

    duckdb_result = runner.run("duckdb_interop")
    assert duckdb_result.success is True, duckdb_result.error
    spark_result = runner.run("spark_interop")
    assert spark_result.success is True, spark_result.error

    # Direction 1: duckdb-written delta table, read via a real Spark session.
    from dataenginex.spark.connect.client import SparkConnectClient

    client = SparkConnectClient(project_id="interop-test-project")
    try:
        client.connect()
        spark = client.get_spark_session()
        duckdb_table_path = data_dir / "silver" / "interop_from_duckdb"
        spark_read = spark.read.format("delta").load(str(duckdb_table_path))
        assert spark_read.count() == 3
        assert set(spark_read.columns) >= {"id", "value"}
    finally:
        client.disconnect()

    # Direction 2: spark-written delta table, read via deltalake (delta-rs).
    from deltalake import DeltaTable

    spark_table_path = data_dir / "silver" / "interop_from_spark"
    dt = DeltaTable(str(spark_table_path))
    arrow_table = dt.to_pyarrow_table()
    assert arrow_table.num_rows == 3
    assert set(arrow_table.column_names) >= {"id", "value"}
