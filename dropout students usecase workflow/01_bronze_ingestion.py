# Databricks notebook source
# MAGIC
# MAGIC %run /Users/saitwins777@gmail.com/dropout/00_shared_config

# COMMAND ----------


ensure_schemas()
materialize_reason_mapping()

print_header("1. BRONZE INGESTION")

raw_df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(SOURCE_PATH)
)

normalized_cols = [normalize_column_name(c) for c in raw_df.columns]

bronze_df = raw_df
for old_col, new_col in zip(raw_df.columns, normalized_cols):
    bronze_df = bronze_df.withColumnRenamed(old_col, new_col)

bronze_df = bronze_df.withColumn(
    "student_id",
    F.row_number().over(Window.orderBy(F.monotonically_increasing_id())).cast("bigint")
)

business_cols = [c for c in bronze_df.columns if c != "student_id"]

bronze_df = bronze_df.withColumn(
    "record_hash",
    F.sha2(
        F.concat_ws(
            "||",
            *[F.coalesce(F.col(c).cast("string"), F.lit("NULL")) for c in business_cols]
        ),
        256
    )
)

bronze_df = (
    bronze_df
    .withColumn("ingestion_ts", F.current_timestamp())
    .withColumn("source_file", F.lit(SOURCE_PATH))
)

write_delta(bronze_df, BRONZE_TABLE)

print("Bronze row count:", spark.table(BRONZE_TABLE).count())
display(spark.table(BRONZE_TABLE).limit(5))

# COMMAND ----------

