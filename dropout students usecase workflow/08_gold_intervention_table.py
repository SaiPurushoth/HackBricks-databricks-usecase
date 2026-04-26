# Databricks notebook source
# MAGIC %run /Users/saitwins777@gmail.com/dropout/00_shared_config

# COMMAND ----------

# Databricks notebook source
# MAGIC %run ./00_shared_config

ensure_schemas()

print_header("8. FINAL INTERVENTION QUEUE")

spark.sql(f"DROP TABLE IF EXISTS {INTERVENTION_TABLE}")

score_sdf = spark.table(MODEL_SCORE_TABLE).select(
    F.col("student_id").cast("bigint").alias("student_id"),
    "actual_dropout",
    "risk_score",
    "predicted_dropout",
)

explanations_sdf = (
    spark.table(EXPLANATIONS_TABLE)
    .select(
        F.col("student_id").cast("bigint").alias("student_id"),
        F.coalesce(F.col("raw_feature_1"), F.lit("other_model_signal")).alias("raw_feature_1"),
        F.coalesce(F.col("raw_feature_2"), F.lit("other_model_signal")).alias("raw_feature_2"),
        F.coalesce(F.col("raw_feature_3"), F.lit("other_model_signal")).alias("raw_feature_3"),
        F.coalesce(F.col("factor_1"), F.lit("Other model risk signal")).alias("factor_1"),
        F.coalesce(F.col("factor_2"), F.lit("Other model risk signal")).alias("factor_2"),
        F.coalesce(F.col("factor_3"), F.lit("Other model risk signal")).alias("factor_3"),
    )
)

fairness_flag_sdf = (
    spark.table(FAIRNESS_TABLE)
    .groupBy()
    .agg(F.max(F.coalesce(F.col("fairness_review_flag"), F.lit(0))).alias("global_review_required"))
)

decision_df = (
    score_sdf.alias("s")
    .join(explanations_sdf.alias("e"), on="student_id", how="left")
    .crossJoin(fairness_flag_sdf)
    .withColumn(
        "confidence_band",
        F.when(F.col("s.risk_score") >= 0.80, "high_confidence")
        .when(F.col("s.risk_score") >= 0.60, "medium_confidence")
        .otherwise("low_confidence")
    )
    .withColumn(
        "intervention_tier",
        F.when(F.col("s.risk_score") >= 0.70, "high")
        .when(F.col("s.risk_score") >= 0.40, "medium")
        .otherwise("low")
    )
    .withColumn(
        "recommended_action",
        map_action_udf(F.col("factor_1"), F.col("factor_2"), F.col("factor_3"), F.col("s.risk_score"))
    )
    .withColumn(
        "priority_rank",
        F.row_number().over(Window.orderBy(F.desc("s.risk_score"), F.desc("s.predicted_dropout")))
    )
    .withColumn("queue_ts", F.current_timestamp())
    .select(
        F.col("student_id"),
        F.col("s.risk_score").alias("risk_score"),
        F.col("s.predicted_dropout").alias("predicted_dropout"),
        F.col("s.actual_dropout").alias("actual_dropout"),
        F.col("confidence_band"),
        F.col("intervention_tier"),
        F.col("recommended_action"),
        F.col("factor_1").alias("top_risk_factor_1"),
        F.col("factor_2").alias("top_risk_factor_2"),
        F.col("factor_3").alias("top_risk_factor_3"),
        F.col("global_review_required").alias("fairness_review_flag"),
        F.col("priority_rank"),
        F.col("queue_ts"),
    )
)

(
    decision_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(INTERVENTION_TABLE)
)

print("Intervention queue created.")
display(spark.table(INTERVENTION_TABLE))