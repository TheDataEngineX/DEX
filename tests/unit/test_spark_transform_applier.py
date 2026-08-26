import pytest
from pyspark.sql import Row, SparkSession
from pyspark.sql.types import DoubleType, StructField, StructType

from dataenginex.config.schema import TransformStepConfig
from dataenginex.spark.transforms.applier import SparkTransformApplier


@pytest.fixture(scope="module")
def spark():
    session = SparkSession.builder.master("local[1]").appName("test-transforms").getOrCreate()
    yield session
    session.stop()


@pytest.fixture
def applier():
    return SparkTransformApplier()


def test_filter(spark, applier):
    df = spark.createDataFrame([(1, 5.0), (2, 8.0)], ["id", "rating"])
    step = TransformStepConfig(type="filter", condition="rating > 6.0")
    out = applier.apply(df, step, "bronze")
    assert out.count() == 1
    assert out.collect()[0]["id"] == 2


def test_derive(spark, applier):
    df = spark.createDataFrame([(10.0,)], ["rating"])
    step = TransformStepConfig(type="derive", name="rating_pct", expression="rating / 10.0 * 100")
    out = applier.apply(df, step, "bronze")
    assert out.collect()[0]["rating_pct"] == 100.0


def test_cast(spark, applier):
    df = spark.createDataFrame([("5",)], ["rating"])
    step = TransformStepConfig(type="cast", columns={"rating": "double"})
    out = applier.apply(df, step, "bronze")
    assert dict(out.dtypes)["rating"] == "double"


def test_deduplicate(spark, applier):
    df = spark.createDataFrame([(1, "a"), (1, "b")], ["id", "val"])
    step = TransformStepConfig(type="deduplicate", key=["id"])
    out = applier.apply(df, step, "bronze")
    assert out.count() == 1


def test_sql(spark, applier):
    df = spark.createDataFrame([(1,), (2,)], ["id"])
    step = TransformStepConfig(type="sql", sql="SELECT * FROM _data WHERE id > 1")
    out = applier.apply(df, step, "bronze")
    assert out.count() == 1
    assert out.collect()[0]["id"] == 2


def test_rename(spark, applier):
    df = spark.createDataFrame([(1,)], ["old_name"])
    step = TransformStepConfig(type="rename", mapping={"old_name": "new_name"})
    out = applier.apply(df, step, "bronze")
    assert "new_name" in out.columns
    assert "old_name" not in out.columns


def test_drop_columns(spark, applier):
    df = spark.createDataFrame([(1, 2)], ["keep", "drop_me"])
    step = TransformStepConfig(type="drop_columns", columns=["drop_me"])
    out = applier.apply(df, step, "bronze")
    assert out.columns == ["keep"]


def test_fill_null(spark, applier):
    # An all-None column has no data Spark can infer a type from
    # (PySparkValueError: CANNOT_DETERMINE_TYPE) — unlike DuckDB, which infers
    # NULL as a nullable column with no complaint. An explicit schema sidesteps
    # inference and keeps the intent (nullable double column) unchanged.
    schema = StructType([StructField("rating", DoubleType(), True)])
    df = spark.createDataFrame([(None,)], schema=schema)
    step = TransformStepConfig(type="fill_null", defaults={"rating": 0.0})
    out = applier.apply(df, step, "bronze")
    assert out.collect()[0]["rating"] == 0.0


def test_aggregate(spark, applier):
    df = spark.createDataFrame([("a", 1), ("a", 2), ("b", 5)], ["grp", "val"])
    step = TransformStepConfig(type="aggregate", group_by=["grp"], agg_exprs={"total": "SUM(val)"})
    out = applier.apply(df, step, "bronze")
    rows = {r["grp"]: r["total"] for r in out.collect()}
    assert rows["a"] == 3
    assert rows["b"] == 5


def test_window(spark, applier):
    df = spark.createDataFrame([("a", 1.0), ("a", 2.0)], ["grp", "score"])
    step = TransformStepConfig(
        type="window", name="rnk", expression="RANK()", partition_by=["grp"], order_by="score DESC"
    )
    out = applier.apply(df, step, "bronze")
    assert "rnk" in out.columns


def test_explode(spark, applier):
    df = spark.createDataFrame([(1, ["a", "b"])], ["id", "genres"])
    # column/alias are not top-level TransformStepConfig fields (pydantic silently
    # drops unknown kwargs) — mirrors the DuckDB ExplodeTransform config convention
    # of passing them via `options`, merged in by runner._build_transform_kwargs.
    step = TransformStepConfig(type="explode", options={"column": "genres", "alias": "genre"})
    out = applier.apply(df, step, "bronze")
    assert out.count() == 2
    assert "genres" not in out.columns
    assert "genre" in out.columns


def test_json_normalize(spark, applier):
    # A plain Python dict infers as MapType in Spark, not StructType — DuckDB's
    # json_normalize flattens a STRUCT (fixed named fields), which is Spark's
    # StructType, not a MAP. A Row value infers as StructType, matching the
    # transform's real target type.
    df = spark.createDataFrame([(1, Row(a=10, b=20))], ["id", "info"])
    step = TransformStepConfig(type="json_normalize", options={"column": "info", "prefix": "info_"})
    out = applier.apply(df, step, "bronze")
    assert "info_a" in out.columns and "info_b" in out.columns
    assert "info" not in out.columns
