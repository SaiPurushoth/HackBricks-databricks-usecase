# Databricks notebook source
# DBTITLE 1,Bronze Ingestion
# MAGIC %run ./01_bronze_ingestion

# COMMAND ----------

# DBTITLE 1,Silver Validation
# MAGIC %run ./02_silver_validation

# COMMAND ----------

# DBTITLE 1,Silver Tables
# MAGIC %run ./03_silver_tables

# COMMAND ----------

# DBTITLE 1,Feature Table
# MAGIC %run ./04_feature_table

# COMMAND ----------

# DBTITLE 1,Train Model
# MAGIC %run ./05_train_model

# COMMAND ----------

# DBTITLE 1,Fairness Audit
# MAGIC %run ./06_fairness_audit

# COMMAND ----------

# DBTITLE 1,Explanations
# MAGIC %run ./07_explanations

# COMMAND ----------

# DBTITLE 1,Intervention Queue
# MAGIC %run ./08_intervention_queue

# COMMAND ----------

# DBTITLE 1,Analysis Queries
# MAGIC %run ./09_analysis_queries