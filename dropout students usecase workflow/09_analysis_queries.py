# Databricks notebook source
# MAGIC %run /Users/saitwins777@gmail.com/dropout/00_shared_config

# COMMAND ----------

# Databricks notebook source
# MAGIC %run ./00_shared_config

print_header("9. OPTIONAL ANALYSIS QUERIES")

display(spark.table(FEATURE_TABLE).groupBy("target", "target_dropout").count().orderBy("target"))
display(spark.table(FEATURE_TABLE).groupBy("financial_segment").count().orderBy("financial_segment"))
display(spark.table(MODEL_METRICS_TABLE))
display(spark.table(FAIRNESS_TABLE))
display(spark.table(INTERVENTION_TABLE).orderBy(F.desc("risk_score")).limit(100))
