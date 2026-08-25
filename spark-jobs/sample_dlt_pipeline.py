"""Sample Delta Live Tables pipeline for local development.

This demonstrates a simple ETL pipeline using Spark DLT.
Run with: docker compose exec spark-master spark-submit /opt/spark/jobs/sample_dlt_pipeline.py
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp, lit

# Initialize Spark session
spark = SparkSession.builder \
    .appName("DEX Sample DLT Pipeline") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000") \
    .config("spark.hadoop.fs.s3a.access.key", "minioadmin") \
    .config("spark.hadoop.fs.s3a.secret.key", "minioadmin") \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .getOrCreate()

# Bronze layer: Raw data ingestion
def ingest_raw_data():
    """Ingest raw data from source to bronze layer."""
    # Sample data
    data = [
        ("Alice", 30, "Engineering", "2024-01-15"),
        ("Bob", 25, "Marketing", "2024-01-16"),
        ("Charlie", 35, "Engineering", "2024-01-17"),
        ("Diana", 28, "Sales", "2024-01-18"),
        ("Eve", 32, "Engineering", "2024-01-19"),
    ]
    
    df = spark.createDataFrame(data, ["name", "age", "department", "hire_date"])
    df = df.withColumn("ingested_at", current_timestamp())
    df = df.withColumn("source", lit("sample_data"))
    
    # Write to bronze layer (Delta format on MinIO)
    df.write \
        .format("delta") \
        .mode("overwrite") \
        .save("s3a://dex-lake/bronze/employees")
    
    print(f"Bronze layer: {df.count()} records written")
    return df

# Silver layer: Cleaned and validated data
def clean_and_validate(bronze_df):
    """Clean and validate data for silver layer."""
    silver_df = bronze_df \
        .filter(col("name").isNotNull()) \
        .filter(col("age") > 0) \
        .filter(col("age") < 100) \
        .withColumn("name_upper", col("name").upper()) \
        .withColumn("processed_at", current_timestamp())
    
    # Write to silver layer
    silver_df.write \
        .format("delta") \
        .mode("overwrite") \
        .save("s3a://dex-lake/silver/employees")
    
    print(f"Silver layer: {silver_df.count()} records written")
    return silver_df

# Gold layer: Business-ready aggregations
def create_aggregations(silver_df):
    """Create business-ready aggregations for gold layer."""
    # Department summary
    dept_summary = silver_df.groupBy("department") \
        .count() \
        .withColumnRenamed("count", "employee_count") \
        .withColumn("summary_date", current_timestamp())
    
    dept_summary.write \
        .format("delta") \
        .mode("overwrite") \
        .save("s3a://dex-lake/gold/department_summary")
    
    print(f"Gold layer: department summary written")
    
    # Age statistics
    age_stats = silver_df.groupBy("department") \
        .avg("age") \
        .withColumnRenamed("avg(age)", "avg_age") \
        .withColumn("stats_date", current_timestamp())
    
    age_stats.write \
        .format("delta") \
        .mode("overwrite") \
        .save("s3a://dex-lake/gold/age_statistics")
    
    print(f"Gold layer: age statistics written")

if __name__ == "__main__":
    print("Starting DLT pipeline...")
    
    # Execute pipeline
    bronze_df = ingest_raw_data()
    silver_df = clean_and_validate(bronze_df)
    create_aggregations(silver_df)
    
    print("Pipeline completed successfully!")
    spark.stop()
