# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql import types as T
import re
import mlflow



BRONZE_SCHEMA = "dropout_bronze"
SILVER_SCHEMA = "dropout_silver"
FEATURE_SCHEMA = "dropout_feature"
GOLD_SCHEMA = "dropout_gold"

BRONZE_TABLE = f"{BRONZE_SCHEMA}.student_dropout_raw"

PROFILE_TABLE = f"{SILVER_SCHEMA}.student_profile"
FINANCIAL_TABLE = f"{SILVER_SCHEMA}.student_financial_status"
SEM1_TABLE = f"{SILVER_SCHEMA}.student_academic_sem1"
SEM2_TABLE = f"{SILVER_SCHEMA}.student_academic_sem2"
MACRO_TABLE = f"{SILVER_SCHEMA}.student_macro_context"

FEATURE_TABLE = f"{FEATURE_SCHEMA}.student_dropout_risk_features"

MODEL_METRICS_TABLE = f"{GOLD_SCHEMA}.model_metrics"
FAIRNESS_TABLE = f"{GOLD_SCHEMA}.fairness_audit"
EXPLANATIONS_TABLE = f"{GOLD_SCHEMA}.student_explanations"
INTERVENTION_TABLE = f"{GOLD_SCHEMA}.student_intervention_queue"

SOURCE_PATH = "/Volumes/workspace/default/hackbricks/students_dropout_academic_success.csv"


spark.sql(f"CREATE DATABASE IF NOT EXISTS {BRONZE_SCHEMA}")
spark.sql(f"CREATE DATABASE IF NOT EXISTS {SILVER_SCHEMA}")
spark.sql(f"CREATE DATABASE IF NOT EXISTS {FEATURE_SCHEMA}")
spark.sql(f"CREATE DATABASE IF NOT EXISTS {GOLD_SCHEMA}")

# COMMAND ----------

def normalize_column_name(col_name: str) -> str:
    col = col_name.strip().lower()
    col = col.replace("'", "")
    col = col.replace("/", "_")
    col = col.replace("(", "_")
    col = col.replace(")", "_")
    col = re.sub(r"[^a-z0-9]+", "_", col)
    col = re.sub(r"_+", "_", col).strip("_")
    return col

# COMMAND ----------

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

bronze_df = (
    bronze_df
    .withColumn("student_id", F.monotonically_increasing_id())
    .withColumn("ingestion_ts", F.current_timestamp())
    .withColumn("source_file", F.lit(SOURCE_PATH))
)

bronze_df.write.format("delta").mode("overwrite").saveAsTable(BRONZE_TABLE)

display(spark.table(BRONZE_TABLE).limit(5))
print(spark.table(BRONZE_TABLE).count())

# COMMAND ----------

bronze = spark.table(BRONZE_TABLE)

student_profile_df = bronze.select(
    "student_id",
    "marital_status",
    "application_mode",
    "application_order",
    "course",
    "daytime_evening_attendance",
    "previous_qualification",
    "previous_qualification_grade",
    "nacionality",
    "mothers_qualification",
    "fathers_qualification",
    "mothers_occupation",
    "fathers_occupation",
    "admission_grade",
    "displaced",
    "educational_special_needs",
    "gender",
    "age_at_enrollment",
    "international"
)

student_profile_df.write.format("delta").mode("overwrite").saveAsTable(PROFILE_TABLE)

# COMMAND ----------

# Databricks notebook source

from pyspark.sql import functions as F
from pyspark.sql import types as T

# =========================================================
# CONFIG
# =========================================================

BRONZE_SCHEMA = "dropout_bronze"
SILVER_SCHEMA = "dropout_silver"

BRONZE_TABLE = f"{BRONZE_SCHEMA}.student_dropout_raw"

PROFILE_TABLE = f"{SILVER_SCHEMA}.student_profile"
DEMOGRAPHIC_TABLE = f"{SILVER_SCHEMA}.student_demographic_features"
ACADEMIC_BG_TABLE = f"{SILVER_SCHEMA}.student_academic_background"
FAMILY_BG_TABLE = f"{SILVER_SCHEMA}.student_family_background"
FINANCIAL_TABLE = f"{SILVER_SCHEMA}.student_financial_status"

spark.sql(f"CREATE DATABASE IF NOT EXISTS {SILVER_SCHEMA}")

# =========================================================
# LOAD BRONZE
# =========================================================

bronze = spark.table(BRONZE_TABLE)

print("Bronze row count:", bronze.count())
print("Bronze columns:")
print(bronze.columns)

# =========================================================
# REQUIRED COLUMN CHECK
# =========================================================

required_columns = [
    "student_id",
    "marital_status",
    "application_mode",
    "application_order",
    "course",
    "daytime_evening_attendance",
    "previous_qualification",
    "previous_qualification_grade",
    "nacionality",
    "mothers_qualification",
    "fathers_qualification",
    "mothers_occupation",
    "fathers_occupation",
    "admission_grade",
    "displaced",
    "educational_special_needs",
    "gender",
    "age_at_enrollment",
    "international",
    "debtor",
    "tuition_fees_up_to_date",
    "scholarship_holder"
]

missing_columns = [c for c in required_columns if c not in bronze.columns]
if missing_columns:
    raise ValueError(f"Missing required Bronze columns: {missing_columns}")

# =========================================================
# BASIC VALIDATION / CLEAN FILTER
# =========================================================
# We exclude only clearly invalid rows.
# Review-type anomalies can still be kept in hackathon workflows.
# =========================================================

silver_source = (
    bronze
    .withColumn("invalid_binary_flag",
        (
            ~F.col("debtor").isin(0, 1) |
            ~F.col("tuition_fees_up_to_date").isin(0, 1) |
            ~F.col("scholarship_holder").isin(0, 1) |
            ~F.col("displaced").isin(0, 1) |
            ~F.col("educational_special_needs").isin(0, 1) |
            ~F.col("international").isin(0, 1)
        )
    )
    .withColumn("negative_numeric_flag",
        (
            (F.col("age_at_enrollment") < 0) |
            (F.col("admission_grade") < 0) |
            (F.col("previous_qualification_grade") < 0)
        )
    )
    .withColumn("invalid_age_flag",
        (F.col("age_at_enrollment") < 15) | (F.col("age_at_enrollment") > 100)
    )
    .filter(
        ~(F.col("invalid_binary_flag") | F.col("negative_numeric_flag"))
    )
)

print("Rows after basic Silver filtering:", silver_source.count())

# =========================================================
# 1. STUDENT PROFILE / IDENTIFIERS
# =========================================================
# This is the main anchor table.
# Keep only the key identifiers / student-level core fields here.
# =========================================================

student_profile_df = (
    silver_source
    .select(
        F.col("student_id").cast(T.LongType()).alias("student_id"),
        F.col("course").cast(T.IntegerType()).alias("course"),
        F.col("application_mode").cast(T.IntegerType()).alias("application_mode"),
        F.col("application_order").cast(T.IntegerType()).alias("application_order"),
        F.col("daytime_evening_attendance").cast(T.IntegerType()).alias("daytime_evening_attendance")
    )
    .withColumn("silver_created_ts", F.current_timestamp())
)

student_profile_df.write.format("delta").mode("overwrite").saveAsTable(PROFILE_TABLE)

# =========================================================
# 2. DEMOGRAPHIC FEATURES
# =========================================================

student_demographic_df = (
    silver_source
    .select(
        F.col("student_id").cast(T.LongType()).alias("student_id"),
        F.col("marital_status").cast(T.IntegerType()).alias("marital_status"),
        F.col("nacionality").cast(T.IntegerType()).alias("nacionality"),
        F.col("gender").cast(T.IntegerType()).alias("gender"),
        F.col("age_at_enrollment").cast(T.IntegerType()).alias("age_at_enrollment"),
        F.col("international").cast(T.IntegerType()).alias("international"),
        F.col("displaced").cast(T.IntegerType()).alias("displaced"),
        F.col("educational_special_needs").cast(T.IntegerType()).alias("educational_special_needs")
    )
    .withColumn("silver_created_ts", F.current_timestamp())
)

student_demographic_df.write.format("delta").mode("overwrite").saveAsTable(DEMOGRAPHIC_TABLE)

# =========================================================
# 3. ACADEMIC BACKGROUND
# =========================================================
# This is pre-admission / background academic info,
# not semester performance.
# =========================================================

student_academic_background_df = (
    silver_source
    .select(
        F.col("student_id").cast(T.LongType()).alias("student_id"),
        F.col("previous_qualification").cast(T.IntegerType()).alias("previous_qualification"),
        F.col("previous_qualification_grade").cast(T.DoubleType()).alias("previous_qualification_grade"),
        F.col("admission_grade").cast(T.DoubleType()).alias("admission_grade")
    )
    .withColumn("silver_created_ts", F.current_timestamp())
)

student_academic_background_df.write.format("delta").mode("overwrite").saveAsTable(ACADEMIC_BG_TABLE)

# =========================================================
# 4. FAMILY BACKGROUND
# =========================================================

student_family_background_df = (
    silver_source
    .select(
        F.col("student_id").cast(T.LongType()).alias("student_id"),
        F.col("mothers_qualification").cast(T.IntegerType()).alias("mothers_qualification"),
        F.col("fathers_qualification").cast(T.IntegerType()).alias("fathers_qualification"),
        F.col("mothers_occupation").cast(T.IntegerType()).alias("mothers_occupation"),
        F.col("fathers_occupation").cast(T.IntegerType()).alias("fathers_occupation")
    )
    .withColumn("silver_created_ts", F.current_timestamp())
)

student_family_background_df.write.format("delta").mode("overwrite").saveAsTable(FAMILY_BG_TABLE)

# =========================================================
# 5. FINANCIAL STATUS
# =========================================================

student_financial_status_df = (
    silver_source
    .select(
        F.col("student_id").cast(T.LongType()).alias("student_id"),
        F.col("debtor").cast(T.IntegerType()).alias("debtor"),
        F.col("tuition_fees_up_to_date").cast(T.IntegerType()).alias("tuition_fees_up_to_date"),
        F.col("scholarship_holder").cast(T.IntegerType()).alias("scholarship_holder")
    )
    .withColumn(
        "financial_risk_band",
        F.when(
            (F.col("debtor") == 1) & (F.col("tuition_fees_up_to_date") == 0),
            "high"
        ).when(
            (F.col("debtor") == 1) |
            (F.col("tuition_fees_up_to_date") == 0) |
            (F.col("scholarship_holder") == 0),
            "medium"
        ).otherwise("low")
    )
    .withColumn("silver_created_ts", F.current_timestamp())
)

student_financial_status_df.write.format("delta").mode("overwrite").saveAsTable(FINANCIAL_TABLE)

# =========================================================
# VALIDATION OUTPUT
# =========================================================

print("Silver tables created successfully.\n")

print("student_profile:", spark.table(PROFILE_TABLE).count())
print("student_demographic_features:", spark.table(DEMOGRAPHIC_TABLE).count())
print("student_academic_background:", spark.table(ACADEMIC_BG_TABLE).count())
print("student_family_background:", spark.table(FAMILY_BG_TABLE).count())
print("student_financial_status:", spark.table(FINANCIAL_TABLE).count())

print("\nSample: student_profile")
display(spark.table(PROFILE_TABLE).limit(10))

print("\nSample: student_demographic_features")
display(spark.table(DEMOGRAPHIC_TABLE).limit(10))

print("\nSample: student_academic_background")
display(spark.table(ACADEMIC_BG_TABLE).limit(10))

print("\nSample: student_family_background")
display(spark.table(FAMILY_BG_TABLE).limit(10))

print("\nSample: student_financial_status")
display(spark.table(FINANCIAL_TABLE).limit(10))

print("\nFinancial risk band distribution")
display(
    spark.table(FINANCIAL_TABLE)
    .groupBy("financial_risk_band")
    .count()
    .orderBy("financial_risk_band")
)

# COMMAND ----------

# MAGIC %pip install mlflow scikit-learn pandas numpy shap
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

for tbl in [
    "dropout_gold.student_intervention_queue",
    "dropout_gold.student_explanations",
    "dropout_gold.fairness_audit",
    "dropout_gold.model_metrics",
    "dropout_feature.student_dropout_risk_features",
    "dropout_silver.student_context",
    "dropout_silver.student_academic_performance",
    "dropout_silver.student_financial_status",
    "dropout_silver.student_family_background",
    "dropout_silver.student_academic_background",
    "dropout_silver.student_demographic_features",
    "dropout_silver.student_profile",
    "dropout_bronze.student_dropout_raw"
]:
    spark.sql(f"DROP TABLE IF EXISTS {tbl}")

# COMMAND ----------

# Databricks notebook source

from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window

import re
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


# =========================================================
# CONFIG
# =========================================================

SOURCE_PATH = "/Volumes/workspace/default/hackbricks/students_dropout_academic_success.csv"

BRONZE_SCHEMA = "dropout_bronze"
SILVER_SCHEMA = "dropout_silver"
FEATURE_SCHEMA = "dropout_feature"
GOLD_SCHEMA = "dropout_gold"

BRONZE_TABLE = f"{BRONZE_SCHEMA}.student_dropout_raw"

PROFILE_TABLE = f"{SILVER_SCHEMA}.student_profile"
DEMOGRAPHIC_TABLE = f"{SILVER_SCHEMA}.student_demographic_features"
ACADEMIC_BG_TABLE = f"{SILVER_SCHEMA}.student_academic_background"
FAMILY_BG_TABLE = f"{SILVER_SCHEMA}.student_family_background"
FINANCIAL_TABLE = f"{SILVER_SCHEMA}.student_financial_status"
ACADEMIC_PERF_TABLE = f"{SILVER_SCHEMA}.student_academic_performance"
CONTEXT_TABLE = f"{SILVER_SCHEMA}.student_context"

FEATURE_TABLE = f"{FEATURE_SCHEMA}.student_dropout_risk_features"

MODEL_METRICS_TABLE = f"{GOLD_SCHEMA}.model_metrics"
FAIRNESS_TABLE = f"{GOLD_SCHEMA}.fairness_audit"
EXPLANATIONS_TABLE = f"{GOLD_SCHEMA}.student_explanations"
INTERVENTION_TABLE = f"{GOLD_SCHEMA}.student_intervention_queue"
REASON_MAPPING_TABLE = f"{GOLD_SCHEMA}.feature_reason_mapping"

RESET_TABLES = False

for schema_name in [BRONZE_SCHEMA, SILVER_SCHEMA, FEATURE_SCHEMA, GOLD_SCHEMA]:
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {schema_name}")


# =========================================================
# REASON MAPPING CONFIG
# =========================================================

FEATURE_REASON_MAPPING = [
    ("sem_grade_delta", "Semester grade decline"),
    ("overall_grade_avg", "Low overall grades"),
    ("curricular_units_1st_sem_grade", "Weak first-semester grades"),
    ("curricular_units_2nd_sem_grade", "Weak second-semester grades"),
    ("previous_qualification_grade", "Weak prior academic record"),
    ("admission_grade", "Lower admission grade"),

    ("sem_approval_delta", "Drop in subject approvals"),
    ("overall_approval_ratio", "Low approval ratio"),
    ("curricular_units_1st_sem_approved", "Low first-semester approvals"),
    ("curricular_units_2nd_sem_approved", "Low second-semester approvals"),
    ("grade_drop_flag", "Recent decline in grades"),
    ("approval_drop_flag", "Recent decline in approvals"),
    ("academic_momentum_band_declining", "Declining academic momentum"),
    ("academic_momentum_band_stable", "Stagnant academic momentum"),

    ("engagement_risk_proxy", "Low learning engagement"),
    ("evaluation_delta", "Drop in evaluation participation"),
    ("engagement_drop_flag", "Recent decline in engagement"),
    ("curricular_units_1st_sem_evaluations", "Low first-semester evaluation activity"),
    ("curricular_units_2nd_sem_evaluations", "Low second-semester evaluation activity"),
    ("curricular_units_1st_sem_without_evaluations", "Missed first-semester evaluations"),
    ("curricular_units_2nd_sem_without_evaluations", "Missed second-semester evaluations"),
    ("sem1_non_evaluated_ratio", "High first-semester non-evaluation ratio"),
    ("sem2_non_evaluated_ratio", "High second-semester non-evaluation ratio"),
    ("absenteeism_index", "High overall absenteeism"),
    ("is_ghosting", "Student disengagement pattern"),

    ("financial_stress_index", "High financial stress"),
    ("financial_segment_high_financial_stress", "Severe financial pressure"),
    ("financial_segment_moderate_financial_stress", "Moderate financial pressure"),
    ("financial_segment_low_financial_stress", "Low financial pressure"),
    ("debtor", "Outstanding financial dues"),
    ("tuition_fees_up_to_date", "Tuition payments not up to date"),
    ("scholarship_holder", "No scholarship support"),
    ("financial_risk_band_high", "High financial risk profile"),
    ("financial_risk_band_medium", "Moderate financial risk profile"),

    ("academic_load_delta", "Change in academic workload"),
    ("curricular_units_1st_sem_enrolled", "First-semester course load pattern"),
    ("curricular_units_2nd_sem_enrolled", "Second-semester course load pattern"),
    ("total_approved_units", "Low total approved units"),

    ("age_at_enrollment", "Age-related study risk pattern"),
    ("course", "Course-level dropout pattern"),
    ("application_mode", "Admission pathway risk pattern"),
    ("application_order", "Admission preference pattern"),
    ("is_primary_choice", "Not primary choice application"),
    ("daytime_evening_attendance", "Attendance schedule pattern"),
    ("marital_status", "Personal background pattern"),
    ("gender", "Gender-linked risk pattern"),
    ("nacionality", "Nationality-linked risk pattern"),
    ("international", "International student adjustment pattern"),
    ("displaced", "Student displacement background"),
    ("educational_special_needs", "Special educational support need"),
    ("previous_qualification", "Prior qualification pathway"),
    ("mothers_qualification", "Mother education background"),
    ("fathers_qualification", "Father education background"),
    ("mothers_occupation", "Mother occupation background"),
    ("fathers_occupation", "Father occupation background"),

    ("course_rigor_score", "Enrolled in high-rigor course"),
    ("grade_deflation_flag", "Course grade deflation pattern"),
    ("admission_gap", "Performance below course average"),
    ("competitive_density", "High-competition course environment"),

    ("unemployment_rate", "High unemployment environment"),
    ("inflation_rate", "Inflation-related stress environment"),
    ("gdp", "Economic environment pattern"),
]


# =========================================================
# HELPERS
# =========================================================

def normalize_column_name(col_name: str) -> str:
    col = col_name.strip().lower()
    col = col.replace("'", "")
    col = col.replace("/", "_")
    col = col.replace("(", "_")
    col = col.replace(")", "_")
    col = col.replace("%", "pct")
    col = re.sub(r"[^a-z0-9]+", "_", col)
    col = re.sub(r"_+", "_", col).strip("_")
    return col

def safe_ratio(numerator_col: str, denominator_col: str):
    return (
        F.when(F.col(denominator_col).isNull(), F.lit(0.0))
         .when(F.col(denominator_col) <= 0, F.lit(0.0))
         .otherwise(F.col(numerator_col).cast("double") / F.col(denominator_col).cast("double"))
    )

def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def cast_student_id(df):
    return df.withColumn("student_id", F.col("student_id").cast("bigint"))

def write_delta(df, table_name):
    (
        cast_student_id(df)
        .write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(table_name)
    )

def normalize_raw_feature_name(raw_feature: str) -> str:
    """Normalize feature name by removing prefixes and extracting base feature."""
    if raw_feature is None or raw_feature == "other_model_signal":
        return None
    
    r = raw_feature.lower()
    
    # Remove 'num__' or 'cat__' prefix
    for prefix in ["num__", "cat__"]:
        if r.startswith(prefix):
            r = r[len(prefix):]
            break
    
    # Handle one-hot encoded categoricals: 'course_17' -> 'course'
    if '_' in r:
        parts = r.split('_')
        # If last part is a number, remove it (one-hot encoding)
        if len(parts) > 1 and parts[-1].isdigit():
            r = '_'.join(parts[:-1])
    
    return r

def build_reason_mapping_dict():
    return {k.lower(): v for k, v in FEATURE_REASON_MAPPING}

REASON_MAP_DICT = build_reason_mapping_dict()

def map_raw_feature_to_reason(raw_feature: str) -> str:
    """Map raw feature name to human-readable reason."""
    if raw_feature is None or raw_feature == "other_model_signal":
        return "Other model risk signal"

    normalized = normalize_raw_feature_name(raw_feature)
    if normalized is None:
        return "Other model risk signal"

    # Try exact match first
    if normalized in REASON_MAP_DICT:
        return REASON_MAP_DICT[normalized]
    
    # Try partial match
    for key, label in REASON_MAP_DICT.items():
        if normalized == key or key in normalized or normalized in key:
            return label

    return "Other model risk signal"

def get_scored_test_dataframe(model_pipeline, X_test_df, y_test_series, full_pdf, idx_test):
    scored_pdf = X_test_df.copy()
    scored_pdf["student_id"] = full_pdf.loc[idx_test, "student_id"].astype("int64").values
    scored_pdf["actual_dropout"] = y_test_series.values
    scored_pdf["risk_score"] = model_pipeline.predict_proba(X_test_df)[:, 1]
    scored_pdf["predicted_dropout"] = (scored_pdf["risk_score"] >= 0.5).astype(int)
    return scored_pdf

def map_action_from_reasons(r1, r2, r3, risk_score):
    reasons = {r1, r2, r3}

    if risk_score >= 0.70 and any(x in reasons for x in [
        "High financial stress", "Severe financial pressure", "Outstanding financial dues",
        "Tuition payments not up to date", "No scholarship support"
    ]):
        return "financial_aid_counseling"

    if risk_score >= 0.70 and any(x in reasons for x in [
        "Semester grade decline", "Low overall grades", "Weak first-semester grades",
        "Weak second-semester grades", "Drop in subject approvals", "Low approval ratio",
        "Low first-semester approvals", "Low second-semester approvals",
        "Declining academic momentum"
    ]):
        return "academic_mentoring"

    if risk_score >= 0.60 and any(x in reasons for x in [
        "Low learning engagement", "Drop in evaluation participation",
        "Recent decline in engagement", "Missed first-semester evaluations",
        "Missed second-semester evaluations", "High overall absenteeism",
        "Student disengagement pattern"
    ]):
        return "counselor_outreach"

    if risk_score >= 0.60:
        return "counselor_review"

    return "monitor"

map_action_udf = F.udf(map_action_from_reasons, T.StringType())


# =========================================================
# OPTIONAL RESET DURING DEVELOPMENT
# =========================================================

if RESET_TABLES:
    for tbl in [
        INTERVENTION_TABLE,
        EXPLANATIONS_TABLE,
        FAIRNESS_TABLE,
        MODEL_METRICS_TABLE,
        REASON_MAPPING_TABLE,
        FEATURE_TABLE,
        CONTEXT_TABLE,
        ACADEMIC_PERF_TABLE,
        FINANCIAL_TABLE,
        FAMILY_BG_TABLE,
        ACADEMIC_BG_TABLE,
        DEMOGRAPHIC_TABLE,
        PROFILE_TABLE,
        BRONZE_TABLE
    ]:
        spark.sql(f"DROP TABLE IF EXISTS {tbl}")


# =========================================================
# MATERIALIZE REASON MAPPING TABLE
# =========================================================

mapping_pdf = pd.DataFrame(FEATURE_REASON_MAPPING, columns=["raw_feature_key", "mapped_reason"])
mapping_sdf = spark.createDataFrame(mapping_pdf)

(
    mapping_sdf.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(REASON_MAPPING_TABLE)
)

display(spark.table(REASON_MAPPING_TABLE))


# =========================================================
# 1. BRONZE INGESTION
# =========================================================

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


# =========================================================
# 2. SILVER SOURCE VALIDATION
# =========================================================

print_header("2. SILVER SOURCE VALIDATION")

bronze = spark.table(BRONZE_TABLE)

required_columns = [
    "student_id",
    "marital_status",
    "application_mode",
    "application_order",
    "course",
    "daytime_evening_attendance",
    "previous_qualification",
    "previous_qualification_grade",
    "nacionality",
    "mothers_qualification",
    "fathers_qualification",
    "mothers_occupation",
    "fathers_occupation",
    "admission_grade",
    "displaced",
    "educational_special_needs",
    "gender",
    "age_at_enrollment",
    "international",
    "debtor",
    "tuition_fees_up_to_date",
    "scholarship_holder",
    "curricular_units_1st_sem_credited",
    "curricular_units_1st_sem_enrolled",
    "curricular_units_1st_sem_evaluations",
    "curricular_units_1st_sem_approved",
    "curricular_units_1st_sem_grade",
    "curricular_units_1st_sem_without_evaluations",
    "curricular_units_2nd_sem_credited",
    "curricular_units_2nd_sem_enrolled",
    "curricular_units_2nd_sem_evaluations",
    "curricular_units_2nd_sem_approved",
    "curricular_units_2nd_sem_grade",
    "curricular_units_2nd_sem_without_evaluations",
    "unemployment_rate",
    "inflation_rate",
    "gdp",
    "target"
]

missing_columns = [c for c in required_columns if c not in bronze.columns]
if missing_columns:
    raise ValueError(f"Missing required Bronze columns: {missing_columns}")

validation_df = (
    bronze
    .withColumn(
        "invalid_binary_flag",
        (
            ~F.col("debtor").isin(0, 1) |
            ~F.col("tuition_fees_up_to_date").isin(0, 1) |
            ~F.col("scholarship_holder").isin(0, 1) |
            ~F.col("displaced").isin(0, 1) |
            ~F.col("educational_special_needs").isin(0, 1) |
            ~F.col("international").isin(0, 1)
        )
    )
    .withColumn(
        "negative_numeric_flag",
        (
            (F.col("age_at_enrollment") < 0) |
            (F.col("admission_grade") < 0) |
            (F.col("previous_qualification_grade") < 0) |
            (F.col("curricular_units_1st_sem_credited") < 0) |
            (F.col("curricular_units_1st_sem_enrolled") < 0) |
            (F.col("curricular_units_1st_sem_evaluations") < 0) |
            (F.col("curricular_units_1st_sem_approved") < 0) |
            (F.col("curricular_units_1st_sem_without_evaluations") < 0) |
            (F.col("curricular_units_2nd_sem_credited") < 0) |
            (F.col("curricular_units_2nd_sem_enrolled") < 0) |
            (F.col("curricular_units_2nd_sem_evaluations") < 0) |
            (F.col("curricular_units_2nd_sem_approved") < 0) |
            (F.col("curricular_units_2nd_sem_without_evaluations") < 0)
        )
    )
    .withColumn(
        "sem1_inconsistent_flag",
        (
            (F.col("curricular_units_1st_sem_approved") > F.col("curricular_units_1st_sem_enrolled")) |
            (F.col("curricular_units_1st_sem_without_evaluations") > F.col("curricular_units_1st_sem_enrolled"))
        )
    )
    .withColumn(
        "sem2_inconsistent_flag",
        (
            (F.col("curricular_units_2nd_sem_approved") > F.col("curricular_units_2nd_sem_enrolled")) |
            (F.col("curricular_units_2nd_sem_without_evaluations") > F.col("curricular_units_2nd_sem_enrolled"))
        )
    )
    .withColumn(
        "invalid_target_flag",
        ~F.col("target").isin("Dropout", "Graduate", "Enrolled")
    )
)

silver_source = validation_df.filter(
    ~(
        F.col("invalid_binary_flag") |
        F.col("negative_numeric_flag") |
        F.col("invalid_target_flag") |
        F.col("sem1_inconsistent_flag") |
        F.col("sem2_inconsistent_flag")
    )
)

print("Rows after Silver validation:", silver_source.count())

display(
    validation_df.select(
        F.sum(F.col("invalid_binary_flag").cast("int")).alias("invalid_binary_rows"),
        F.sum(F.col("negative_numeric_flag").cast("int")).alias("negative_numeric_rows"),
        F.sum(F.col("sem1_inconsistent_flag").cast("int")).alias("sem1_inconsistent_rows"),
        F.sum(F.col("sem2_inconsistent_flag").cast("int")).alias("sem2_inconsistent_rows"),
        F.sum(F.col("invalid_target_flag").cast("int")).alias("invalid_target_rows")
    )
)


# =========================================================
# 3. SILVER TABLES
# =========================================================

print_header("3. SILVER TABLES")

student_profile_df = (
    silver_source
    .select(
        F.col("student_id").cast(T.LongType()).alias("student_id"),
        F.col("course").cast(T.IntegerType()).alias("course"),
        F.col("application_mode").cast(T.IntegerType()).alias("application_mode"),
        F.col("application_order").cast(T.IntegerType()).alias("application_order"),
        F.col("daytime_evening_attendance").cast(T.IntegerType()).alias("daytime_evening_attendance")
    )
    .withColumn("silver_created_ts", F.current_timestamp())
)

student_demographic_df = (
    silver_source
    .select(
        F.col("student_id").cast(T.LongType()).alias("student_id"),
        F.col("marital_status").cast(T.IntegerType()).alias("marital_status"),
        F.col("nacionality").cast(T.IntegerType()).alias("nacionality"),
        F.col("gender").cast(T.IntegerType()).alias("gender"),
        F.col("age_at_enrollment").cast(T.IntegerType()).alias("age_at_enrollment"),
        F.col("international").cast(T.IntegerType()).alias("international"),
        F.col("displaced").cast(T.IntegerType()).alias("displaced"),
        F.col("educational_special_needs").cast(T.IntegerType()).alias("educational_special_needs")
    )
    .withColumn("silver_created_ts", F.current_timestamp())
)

student_academic_background_df = (
    silver_source
    .select(
        F.col("student_id").cast(T.LongType()).alias("student_id"),
        F.col("previous_qualification").cast(T.IntegerType()).alias("previous_qualification"),
        F.col("previous_qualification_grade").cast(T.DoubleType()).alias("previous_qualification_grade"),
        F.col("admission_grade").cast(T.DoubleType()).alias("admission_grade")
    )
    .withColumn("silver_created_ts", F.current_timestamp())
)

student_family_background_df = (
    silver_source
    .select(
        F.col("student_id").cast(T.LongType()).alias("student_id"),
        F.col("mothers_qualification").cast(T.IntegerType()).alias("mothers_qualification"),
        F.col("fathers_qualification").cast(T.IntegerType()).alias("fathers_qualification"),
        F.col("mothers_occupation").cast(T.IntegerType()).alias("mothers_occupation"),
        F.col("fathers_occupation").cast(T.IntegerType()).alias("fathers_occupation")
    )
    .withColumn("silver_created_ts", F.current_timestamp())
)

student_financial_status_df = (
    silver_source
    .select(
        F.col("student_id").cast(T.LongType()).alias("student_id"),
        F.col("debtor").cast(T.IntegerType()).alias("debtor"),
        F.col("tuition_fees_up_to_date").cast(T.IntegerType()).alias("tuition_fees_up_to_date"),
        F.col("scholarship_holder").cast(T.IntegerType()).alias("scholarship_holder")
    )
    .withColumn(
        "financial_risk_band",
        F.when(
            (F.col("debtor") == 1) & (F.col("tuition_fees_up_to_date") == 0),
            "high"
        ).when(
            (F.col("debtor") == 1) |
            (F.col("tuition_fees_up_to_date") == 0) |
            (F.col("scholarship_holder") == 0),
            "medium"
        ).otherwise("low")
    )
    .withColumn("silver_created_ts", F.current_timestamp())
)

student_academic_performance_df = (
    silver_source
    .select(
        F.col("student_id").cast(T.LongType()).alias("student_id"),
        F.col("curricular_units_1st_sem_credited").cast(T.IntegerType()).alias("curricular_units_1st_sem_credited"),
        F.col("curricular_units_1st_sem_enrolled").cast(T.IntegerType()).alias("curricular_units_1st_sem_enrolled"),
        F.col("curricular_units_1st_sem_evaluations").cast(T.IntegerType()).alias("curricular_units_1st_sem_evaluations"),
        F.col("curricular_units_1st_sem_approved").cast(T.IntegerType()).alias("curricular_units_1st_sem_approved"),
        F.col("curricular_units_1st_sem_grade").cast(T.DoubleType()).alias("curricular_units_1st_sem_grade"),
        F.col("curricular_units_1st_sem_without_evaluations").cast(T.IntegerType()).alias("curricular_units_1st_sem_without_evaluations"),
        F.col("curricular_units_2nd_sem_credited").cast(T.IntegerType()).alias("curricular_units_2nd_sem_credited"),
        F.col("curricular_units_2nd_sem_enrolled").cast(T.IntegerType()).alias("curricular_units_2nd_sem_enrolled"),
        F.col("curricular_units_2nd_sem_evaluations").cast(T.IntegerType()).alias("curricular_units_2nd_sem_evaluations"),
        F.col("curricular_units_2nd_sem_approved").cast(T.IntegerType()).alias("curricular_units_2nd_sem_approved"),
        F.col("curricular_units_2nd_sem_grade").cast(T.DoubleType()).alias("curricular_units_2nd_sem_grade"),
        F.col("curricular_units_2nd_sem_without_evaluations").cast(T.IntegerType()).alias("curricular_units_2nd_sem_without_evaluations")
    )
    .withColumn("sem1_approval_ratio", safe_ratio("curricular_units_1st_sem_approved", "curricular_units_1st_sem_enrolled"))
    .withColumn("sem2_approval_ratio", safe_ratio("curricular_units_2nd_sem_approved", "curricular_units_2nd_sem_enrolled"))
    .withColumn("sem1_evaluation_ratio", safe_ratio("curricular_units_1st_sem_evaluations", "curricular_units_1st_sem_enrolled"))
    .withColumn("sem2_evaluation_ratio", safe_ratio("curricular_units_2nd_sem_evaluations", "curricular_units_2nd_sem_enrolled"))
    .withColumn("sem1_non_evaluated_ratio", safe_ratio("curricular_units_1st_sem_without_evaluations", "curricular_units_1st_sem_enrolled"))
    .withColumn("sem2_non_evaluated_ratio", safe_ratio("curricular_units_2nd_sem_without_evaluations", "curricular_units_2nd_sem_enrolled"))
    .withColumn("silver_created_ts", F.current_timestamp())
)

student_context_df = (
    silver_source
    .select(
        F.col("student_id").cast(T.LongType()).alias("student_id"),
        F.col("unemployment_rate").cast(T.DoubleType()).alias("unemployment_rate"),
        F.col("inflation_rate").cast(T.DoubleType()).alias("inflation_rate"),
        F.col("gdp").cast(T.DoubleType()).alias("gdp"),
        F.col("target").cast(T.StringType()).alias("target")
    )
    .withColumn("target_dropout", F.when(F.col("target") == "Dropout", 1).otherwise(0))
    .withColumn("silver_created_ts", F.current_timestamp())
)

write_delta(student_profile_df, PROFILE_TABLE)
write_delta(student_demographic_df, DEMOGRAPHIC_TABLE)
write_delta(student_academic_background_df, ACADEMIC_BG_TABLE)
write_delta(student_family_background_df, FAMILY_BG_TABLE)
write_delta(student_financial_status_df, FINANCIAL_TABLE)
write_delta(student_academic_performance_df, ACADEMIC_PERF_TABLE)
write_delta(student_context_df, CONTEXT_TABLE)

print("Silver tables created.")


# =========================================================
# 4. FEATURE TABLE
# =========================================================

print_header("4. FEATURE TABLE")

profile_df = spark.table(PROFILE_TABLE).drop("silver_created_ts")
demographic_df = spark.table(DEMOGRAPHIC_TABLE).drop("silver_created_ts")
academic_bg_df = spark.table(ACADEMIC_BG_TABLE).drop("silver_created_ts")
family_bg_df = spark.table(FAMILY_BG_TABLE).drop("silver_created_ts")
financial_df = spark.table(FINANCIAL_TABLE).drop("silver_created_ts")
academic_perf_df = spark.table(ACADEMIC_PERF_TABLE).drop("silver_created_ts")
context_df = spark.table(CONTEXT_TABLE).drop("silver_created_ts")

# Calculate course-level aggregate features using window functions
course_window = Window.partitionBy("course")

academic_perf_with_course_stats = (
    academic_perf_df
    .join(profile_df.select("student_id", "course"), "student_id")
    .join(academic_bg_df.select("student_id", "admission_grade"), "student_id")
    .withColumn(
        "course_rigor_score",
        F.avg((F.col("curricular_units_1st_sem_grade") + F.col("curricular_units_2nd_sem_grade")) / 2.0).over(course_window)
    )
    .withColumn(
        "competitive_density",
        F.avg(F.col("admission_grade")).over(course_window)
    )
    .withColumn(
        "course_avg_grade",
        (F.col("curricular_units_1st_sem_grade") + F.col("curricular_units_2nd_sem_grade")) / 2.0
    )
)

# Calculate university-wide 25th percentile for grade deflation flag
university_25th_percentile = academic_perf_with_course_stats.approxQuantile("course_rigor_score", [0.25], 0.01)[0]

feature_df = (
    profile_df
    .join(demographic_df, "student_id")
    .join(academic_bg_df, "student_id")
    .join(family_bg_df, "student_id")
    .join(financial_df, "student_id")
    .join(
        academic_perf_with_course_stats.select(
            "student_id",
            "curricular_units_1st_sem_credited",
            "curricular_units_1st_sem_enrolled",
            "curricular_units_1st_sem_evaluations",
            "curricular_units_1st_sem_approved",
            "curricular_units_1st_sem_grade",
            "curricular_units_1st_sem_without_evaluations",
            "curricular_units_2nd_sem_credited",
            "curricular_units_2nd_sem_enrolled",
            "curricular_units_2nd_sem_evaluations",
            "curricular_units_2nd_sem_approved",
            "curricular_units_2nd_sem_grade",
            "curricular_units_2nd_sem_without_evaluations",
            "sem1_approval_ratio",
            "sem2_approval_ratio",
            "sem1_evaluation_ratio",
            "sem2_evaluation_ratio",
            "sem1_non_evaluated_ratio",
            "sem2_non_evaluated_ratio",
            "course_rigor_score",
            "competitive_density",
            "course_avg_grade"
        ),
        "student_id"
    )
    .join(context_df, "student_id")

    # Existing features
    .withColumn("sem_grade_delta", F.col("curricular_units_2nd_sem_grade") - F.col("curricular_units_1st_sem_grade"))
    .withColumn("sem_approval_delta", F.col("sem2_approval_ratio") - F.col("sem1_approval_ratio"))
    .withColumn("evaluation_delta", F.col("curricular_units_2nd_sem_evaluations") - F.col("curricular_units_1st_sem_evaluations"))
    .withColumn("non_evaluated_delta", F.col("curricular_units_2nd_sem_without_evaluations") - F.col("curricular_units_1st_sem_without_evaluations"))
    .withColumn("academic_load_delta", F.col("curricular_units_2nd_sem_enrolled") - F.col("curricular_units_1st_sem_enrolled"))

    .withColumn("overall_grade_avg", (F.col("curricular_units_1st_sem_grade") + F.col("curricular_units_2nd_sem_grade")) / 2.0)
    .withColumn(
        "overall_approval_ratio",
        F.when(
            (F.col("curricular_units_1st_sem_enrolled") + F.col("curricular_units_2nd_sem_enrolled")) > 0,
            (F.col("curricular_units_1st_sem_approved") + F.col("curricular_units_2nd_sem_approved")) /
            (F.col("curricular_units_1st_sem_enrolled") + F.col("curricular_units_2nd_sem_enrolled"))
        ).otherwise(F.lit(0.0))
    )
    .withColumn(
        "financial_stress_index",
        (
            F.when(F.col("debtor") == 1, 1).otherwise(0) +
            F.when(F.col("tuition_fees_up_to_date") == 0, 1).otherwise(0) +
            F.when(F.col("scholarship_holder") == 0, 1).otherwise(0)
        )
    )
    .withColumn(
        "financial_segment",
        F.when(F.col("financial_stress_index") >= 2, "high_financial_stress")
         .when(F.col("financial_stress_index") == 1, "moderate_financial_stress")
         .otherwise("low_financial_stress")
    )
    .withColumn(
        "engagement_risk_proxy",
        (
            F.when(F.col("curricular_units_1st_sem_evaluations") == 0, 1).otherwise(0) +
            F.when(F.col("curricular_units_2nd_sem_evaluations") == 0, 1).otherwise(0) +
            F.when(F.col("curricular_units_1st_sem_without_evaluations") > 0, 1).otherwise(0) +
            F.when(F.col("curricular_units_2nd_sem_without_evaluations") > 0, 1).otherwise(0)
        )
    )
    .withColumn("grade_drop_flag", F.when(F.col("sem_grade_delta") < 0, 1).otherwise(0))
    .withColumn("approval_drop_flag", F.when(F.col("sem_approval_delta") < 0, 1).otherwise(0))
    .withColumn("engagement_drop_flag", F.when(F.col("evaluation_delta") < 0, 1).otherwise(0))
    .withColumn(
        "academic_momentum_band",
        F.when((F.col("sem_grade_delta") < 0) & (F.col("sem_approval_delta") < 0), "declining")
         .when((F.abs(F.col("sem_grade_delta")) < 0.25) & (F.abs(F.col("sem_approval_delta")) < 0.05), "stable")
         .otherwise("improving_or_mixed")
    )
    
    # NEW FEATURES for improved prediction
    # 1. Absenteeism Index - total missed evaluations across both semesters
    .withColumn(
        "absenteeism_index",
        F.col("curricular_units_1st_sem_without_evaluations") + F.col("curricular_units_2nd_sem_without_evaluations")
    )
    
    # 2. Ghosting Flag - student enrolled but did not participate in 2nd semester evaluations
    .withColumn(
        "is_ghosting",
        F.when(
            (F.col("curricular_units_2nd_sem_evaluations") == 0) & (F.col("curricular_units_2nd_sem_enrolled") > 0),
            1
        ).otherwise(0)
    )
    
    # 3. Total Approved Units - overall academic success metric
    .withColumn(
        "total_approved_units",
        F.col("curricular_units_1st_sem_approved") + F.col("curricular_units_2nd_sem_approved")
    )
    
    # 4. Is Primary Choice - indicates if this was the student's first choice
    .withColumn(
        "is_primary_choice",
        F.when(F.col("application_order") == 1, 1).otherwise(0)
    )
    
    # 5. Admission Gap - how student's admission grade compares to course average
    .withColumn(
        "admission_gap",
        F.col("admission_grade") - F.col("course_rigor_score")
    )
    
    # 6. Grade Deflation Flag - indicates if enrolled in a low-performing course
    .withColumn(
        "grade_deflation_flag",
        F.when(F.col("course_rigor_score") < F.lit(university_25th_percentile), 1).otherwise(0)
    )
    
    .withColumn("feature_ts", F.current_timestamp())
)

# Drop intermediate columns not needed for modeling
feature_df = feature_df.drop("course_avg_grade")

write_delta(feature_df, FEATURE_TABLE)

print("Feature table row count:", spark.table(FEATURE_TABLE).count())
display(spark.table(FEATURE_TABLE).limit(10))


# =========================================================
# 5. FEATURE VALIDATION
# =========================================================

print_header("5. FEATURE VALIDATION")

feature_tbl = spark.table(FEATURE_TABLE)

display(feature_tbl.groupBy("target", "target_dropout").count().orderBy("target"))
display(feature_tbl.groupBy("financial_segment").count().orderBy("financial_segment"))

# Validate new features
print("\n=== NEW FEATURES VALIDATION ===")
display(
    feature_tbl.select(
        F.count("*").alias("total_rows"),
        F.sum(F.col("is_ghosting")).alias("ghosting_students"),
        F.sum(F.col("is_primary_choice")).alias("primary_choice_students"),
        F.sum(F.col("grade_deflation_flag")).alias("deflated_course_students"),
        F.avg(F.col("absenteeism_index")).alias("avg_absenteeism"),
        F.avg(F.col("total_approved_units")).alias("avg_total_approved"),
        F.avg(F.col("course_rigor_score")).alias("avg_course_rigor"),
        F.avg(F.col("competitive_density")).alias("avg_competitive_density"),
        F.avg(F.col("admission_gap")).alias("avg_admission_gap")
    )
)

display(
    feature_tbl.select(
        *[
            F.sum(F.col(c).isNull().cast("int")).alias(c)
            for c in [
                "sem_grade_delta",
                "sem_approval_delta",
                "overall_grade_avg",
                "overall_approval_ratio",
                "financial_stress_index",
                "engagement_risk_proxy",
                "absenteeism_index",
                "is_ghosting",
                "total_approved_units",
                "course_rigor_score",
                "competitive_density",
                "admission_gap"
            ]
        ]
    )
)


# =========================================================
# 6. MODEL TRAINING
# =========================================================

print_header("6. MODEL TRAINING")

pdf = spark.table(FEATURE_TABLE).toPandas()

categorical_cols = [
    "course",
    "application_mode",
    "application_order",
    "daytime_evening_attendance",
    "marital_status",
    "nacionality",
    "gender",
    "international",
    "displaced",
    "educational_special_needs",
    "previous_qualification",
    "mothers_qualification",
    "fathers_qualification",
    "mothers_occupation",
    "fathers_occupation",
    "debtor",
    "tuition_fees_up_to_date",
    "scholarship_holder",
    "financial_risk_band",
    "financial_segment",
    "academic_momentum_band",
    "grade_drop_flag",
    "approval_drop_flag",
    "engagement_drop_flag",
    "is_ghosting",
    "is_primary_choice",
    "grade_deflation_flag"
]

drop_cols = [c for c in ["target", "target_dropout", "feature_ts", "student_id"] if c in pdf.columns]

X = pdf.drop(columns=drop_cols)
y = pdf["target_dropout"]

categorical_cols = [c for c in categorical_cols if c in X.columns]
numeric_cols = [c for c in X.columns if c not in categorical_cols]

for c in categorical_cols:
    X[c] = X[c].astype(str)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric_cols
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_cols
        )
    ]
)

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, pdf.index, test_size=0.2, random_state=42, stratify=y
)

def log_and_train(model, model_name):
    with mlflow.start_run(run_name=model_name):
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        probs = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "f1": float(f1_score(y_test, preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probs))
        }

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("feature_count", X.shape[1])
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, name="model")

        return pipeline, metrics, mlflow.active_run().info.run_id

lr_pipeline, lr_metrics, lr_run_id = log_and_train(
    LogisticRegression(max_iter=1000, class_weight="balanced"),
    "logistic_regression_baseline"
)

rf_pipeline, rf_metrics, rf_run_id = log_and_train(
    RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced"
    ),
    "random_forest_baseline"
)

best_model = rf_pipeline if rf_metrics["roc_auc"] >= lr_metrics["roc_auc"] else lr_pipeline
best_model_name = "random_forest_baseline" if rf_metrics["roc_auc"] >= lr_metrics["roc_auc"] else "logistic_regression_baseline"

metrics_pdf = pd.DataFrame([
    {"model_name": "logistic_regression_baseline", "run_id": lr_run_id, **lr_metrics},
    {"model_name": "random_forest_baseline", "run_id": rf_run_id, **rf_metrics}
])
metrics_pdf["training_ts"] = pd.Timestamp.utcnow()

metrics_sdf = spark.createDataFrame(metrics_pdf)
(
    metrics_sdf.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(MODEL_METRICS_TABLE)
)

print("Best model:", best_model_name)
print(f"Feature count: {X.shape[1]} (including new engagement and contextual features)")
display(spark.table(MODEL_METRICS_TABLE))


# =========================================================
# 7. FAIRNESS AUDIT
# =========================================================

print_header("7. FAIRNESS AUDIT")

scored = get_scored_test_dataframe(
    model_pipeline=best_model,
    X_test_df=X_test,
    y_test_series=y_test,
    full_pdf=pdf,
    idx_test=idx_test
)

demo_pdf = spark.table(DEMOGRAPHIC_TABLE).select("student_id", "gender").toPandas()
demo_pdf["student_id"] = demo_pdf["student_id"].astype("int64")

feature_attr_pdf = spark.table(FEATURE_TABLE).select("student_id", "financial_segment").toPandas()
feature_attr_pdf["student_id"] = feature_attr_pdf["student_id"].astype("int64")

audit_pdf = (
    scored[["student_id", "actual_dropout", "predicted_dropout", "risk_score"]]
    .merge(demo_pdf, on="student_id", how="left")
    .merge(feature_attr_pdf, on="student_id", how="left")
)

audit_pdf["gender"] = audit_pdf["gender"].fillna(-1).astype(str)
audit_pdf["financial_segment"] = audit_pdf["financial_segment"].fillna("unknown").astype(str)

def build_fairness_report(df: pd.DataFrame, protected_col: str, model_name: str) -> pd.DataFrame:
    overall_flag_rate = df["predicted_dropout"].mean()
    overall_tpr = (
        df.loc[df["actual_dropout"] == 1, "predicted_dropout"].mean()
        if (df["actual_dropout"] == 1).sum() > 0 else 0.0
    )

    rows = []
    for group_name, g in df.groupby(protected_col, dropna=False):
        positive_rate = g["predicted_dropout"].mean()
        actual_positive = g[g["actual_dropout"] == 1]
        tpr = actual_positive["predicted_dropout"].mean() if len(actual_positive) > 0 else 0.0

        rows.append({
            "model_name": model_name,
            "protected_attribute": protected_col,
            "group_name": str(group_name),
            "population_count": int(len(g)),
            "positive_prediction_rate": float(positive_rate),
            "true_positive_rate": float(tpr),
            "demographic_parity_gap": float(positive_rate - overall_flag_rate),
            "equal_opportunity_gap": float(tpr - overall_tpr),
            "audit_ts": pd.Timestamp.utcnow().isoformat()
        })

    return pd.DataFrame(rows)

gender_report = build_fairness_report(audit_pdf, "gender", best_model_name)
finance_report = build_fairness_report(audit_pdf, "financial_segment", best_model_name)

fairness_report = pd.concat([gender_report, finance_report], ignore_index=True)
fairness_sdf = spark.createDataFrame(fairness_report)

(
    fairness_sdf.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(FAIRNESS_TABLE)
)

display(spark.table(FAIRNESS_TABLE))


# =========================================================
# 8. EXPLANATIONS (FIXED)
# =========================================================

print_header("8. EXPLANATIONS")

# Optional cleanup if older bad schema exists
spark.sql(f"DROP TABLE IF EXISTS {EXPLANATIONS_TABLE}")

scored = get_scored_test_dataframe(
    model_pipeline=best_model,
    X_test_df=X_test,
    y_test_series=y_test,
    full_pdf=pdf,
    idx_test=idx_test
)

positive_student_ids = scored.loc[scored["predicted_dropout"] == 1, "student_id"].astype("int64").tolist()

if len(positive_student_ids) == 0:
    explanations_pdf = pd.DataFrame(columns=[
        "student_id", "risk_score",
        "raw_feature_1", "raw_feature_2", "raw_feature_3",
        "factor_1", "factor_2", "factor_3"
    ])
else:
    explain_pdf = pdf[pdf["student_id"].isin(positive_student_ids)].copy()

    drop_cols = [c for c in ["target", "target_dropout", "feature_ts", "student_id"] if c in explain_pdf.columns]
    explain_X = explain_pdf.drop(columns=drop_cols)

    for c in categorical_cols:
        if c in explain_X.columns:
            explain_X[c] = explain_X[c].astype(str)

    pre = best_model.named_steps["preprocessor"]
    model = best_model.named_steps["model"]
    feature_names = pre.get_feature_names_out()

    explain_transformed = pre.transform(explain_X)

    if hasattr(explain_transformed, "toarray"):
        explain_transformed = explain_transformed.toarray()

    explain_transformed = np.asarray(explain_transformed, dtype=np.float64)

    if not SHAP_AVAILABLE:
        raise RuntimeError("SHAP is required but not available")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(explain_transformed)

    if isinstance(shap_values, list):
        sv = np.asarray(shap_values[1], dtype=np.float64)
    else:
        sv = np.asarray(shap_values, dtype=np.float64)

    rows = []
    scored_lookup = scored.set_index("student_id")["risk_score"].to_dict()

    # FIXED: Properly extract individual features
    for i in range(len(explain_pdf)):
        sid = int(explain_pdf.iloc[i]["student_id"])
        risk_score = float(scored_lookup.get(sid, 0.0))

        contrib = sv[i]
        ranked_idx = np.argsort(np.abs(contrib))[::-1]

        raw_selected = []
        mapped_selected = []
        seen_mapped = set()

        for j in ranked_idx:
            # FIX: Ensure we get a single feature name as string
            try:
                raw_feature = str(feature_names[j])
                
                # Skip if this looks like an array or invalid
                if '[' in raw_feature or ']' in raw_feature:
                    continue
                if raw_feature == 'other_model_signal':
                    continue
                    
            except (IndexError, TypeError):
                continue
            
            mapped_reason = map_raw_feature_to_reason(raw_feature)

            # Only add if we haven't seen this mapped reason yet
            if mapped_reason not in seen_mapped:
                raw_selected.append(raw_feature)
                mapped_selected.append(mapped_reason)
                seen_mapped.add(mapped_reason)

            if len(mapped_selected) >= 3:
                break

        # Pad with defaults if needed
        while len(raw_selected) < 3:
            raw_selected.append("other_model_signal")
        while len(mapped_selected) < 3:
            mapped_selected.append("Other model risk signal")

        rows.append({
            "student_id": sid,
            "risk_score": risk_score,
            "raw_feature_1": raw_selected[0],
            "raw_feature_2": raw_selected[1],
            "raw_feature_3": raw_selected[2],
            "factor_1": mapped_selected[0],
            "factor_2": mapped_selected[1],
            "factor_3": mapped_selected[2]
        })

    explanations_pdf = pd.DataFrame(rows)

explanation_schema = T.StructType([
    T.StructField("student_id", T.LongType(), True),
    T.StructField("risk_score", T.DoubleType(), True),
    T.StructField("raw_feature_1", T.StringType(), True),
    T.StructField("raw_feature_2", T.StringType(), True),
    T.StructField("raw_feature_3", T.StringType(), True),
    T.StructField("factor_1", T.StringType(), True),
    T.StructField("factor_2", T.StringType(), True),
    T.StructField("factor_3", T.StringType(), True),
])

if explanations_pdf.empty:
    explanations_pdf = pd.DataFrame(columns=[
        "student_id", "risk_score",
        "raw_feature_1", "raw_feature_2", "raw_feature_3",
        "factor_1", "factor_2", "factor_3"
    ])

for col in ["raw_feature_1", "raw_feature_2", "raw_feature_3", "factor_1", "factor_2", "factor_3"]:
    if col in explanations_pdf.columns:
        fill_value = "other_model_signal" if col.startswith("raw_feature_") else "Other model risk signal"
        explanations_pdf[col] = explanations_pdf[col].fillna(fill_value).astype(str)

if "student_id" in explanations_pdf.columns and len(explanations_pdf) > 0:
    explanations_pdf["student_id"] = explanations_pdf["student_id"].astype("int64")
if "risk_score" in explanations_pdf.columns and len(explanations_pdf) > 0:
    explanations_pdf["risk_score"] = explanations_pdf["risk_score"].astype("float64")

explanations_sdf = spark.createDataFrame(explanations_pdf, schema=explanation_schema)

(
    explanations_sdf.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(EXPLANATIONS_TABLE)
)

print("Explanation rows written:", explanations_sdf.count())
print("\n✅ SHAP feature extraction FIXED - now properly extracting individual features!")


# =========================================================
# 9. FINAL INTERVENTION QUEUE
# =========================================================

print_header("9. FINAL INTERVENTION QUEUE")

spark.sql(f"DROP TABLE IF EXISTS {INTERVENTION_TABLE}")

scored = get_scored_test_dataframe(
    model_pipeline=best_model,
    X_test_df=X_test,
    y_test_series=y_test,
    full_pdf=pdf,
    idx_test=idx_test
)

score_sdf = (
    spark.createDataFrame(scored[["student_id", "actual_dropout", "risk_score", "predicted_dropout"]])
    .withColumn("student_id", F.col("student_id").cast("bigint"))
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

fairness_sdf = spark.table(FAIRNESS_TABLE)

fairness_flag_sdf = (
    fairness_sdf
    .withColumn(
        "review_required_flag",
        F.when(
            (F.abs(F.col("demographic_parity_gap")) > 0.10) |
            (F.abs(F.col("equal_opportunity_gap")) > 0.10),
            True
        ).otherwise(False)
    )
    .groupBy()
    .agg(F.max(F.col("review_required_flag").cast("int")).alias("global_review_required"))
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
        map_action_udf(
            F.col("factor_1"),
            F.col("factor_2"),
            F.col("factor_3"),
            F.col("s.risk_score")
        )
    )
    .withColumn(
        "priority_rank",
        F.row_number().over(
            Window.orderBy(
                F.desc("s.risk_score"),
                F.desc("s.predicted_dropout")
            )
        )
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
        F.col("queue_ts")
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
print("Queue row count:", spark.table(INTERVENTION_TABLE).count())
print("High-risk students (>= 0.7):", spark.table(INTERVENTION_TABLE).filter("risk_score >= 0.70").count())
print("\n✅ Check the intervention distribution - you should now see diverse actions!")
print("Run the SQL analysis cells (11-18) to explore the different intervention patterns.")

# COMMAND ----------

display(
    spark.table(INTERVENTION_TABLE)
    .orderBy("priority_rank")
    .limit(100)
)

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Understanding SHAP Predictions
# MAGIC %md
# MAGIC # 🔍 How SHAP Predictions Work in This Pipeline
# MAGIC
# MAGIC ## Step-by-Step Workflow:
# MAGIC
# MAGIC ### 1️⃣ **Model Training** (Cell 8 - Section 6)
# MAGIC - Random Forest trained on **65 features** (numeric + categorical after preprocessing)
# MAGIC - Model learns patterns distinguishing dropouts from graduates  
# MAGIC - Outputs **risk_score** (0-1) = probability student will dropout
# MAGIC
# MAGIC ### 2️⃣ **SHAP Explanation** (Cell 8 - Section 8)
# MAGIC For EACH student predicted as dropout (risk_score ≥ 0.5):
# MAGIC - **SHAP TreeExplainer** computes how much each feature contributed to the prediction
# MAGIC - Each feature gets a **SHAP value** = impact on pushing prediction higher/lower
# MAGIC - Features sorted by absolute SHAP value (biggest impact first)
# MAGIC - Top 3 features selected with **distinct mapped reasons**
# MAGIC
# MAGIC **Example mapping:**
# MAGIC - Raw SHAP feature: `num__previous_qualification_grade`  
# MAGIC - Mapped reason: `"Weak prior academic record"`
# MAGIC
# MAGIC ### 3️⃣ **Risk Score Calculation**
# MAGIC - `risk_score` = Random Forest's probability output (0 to 1)
# MAGIC - **NOT based on SHAP** - SHAP only **explains** why the score is high
# MAGIC - Score comes from model's internal voting across 300 decision trees
# MAGIC
# MAGIC ### 4️⃣ **Intervention Decision** (Cell 8 - Section 9)
# MAGIC Based on `risk_score` + top 3 reasons:
# MAGIC - **Financial factors** + high risk → `financial_aid_counseling`
# MAGIC - **Academic factors** + high risk → `academic_mentoring`  
# MAGIC - **Engagement factors** + medium risk → `counselor_outreach`
# MAGIC - Otherwise → `counselor_review` or `monitor`
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ⚠️ Current Issue:
# MAGIC Many students show `"Other model risk signal"` because:
# MAGIC 1. SHAP identifies one-hot encoded categorical features (e.g., `cat__course_17`)
# MAGIC 2. These don't match entries in `FEATURE_REASON_MAPPING` dictionary
# MAGIC 3. Fall back to default "Other model risk signal"
# MAGIC
# MAGIC **Next cells show queries to explore diverse patterns...**

# COMMAND ----------

# DBTITLE 1,Query 1: Action Distribution
# MAGIC %sql
# MAGIC -- =========================================================
# MAGIC -- 1. RECOMMENDED ACTIONS DISTRIBUTION
# MAGIC -- =========================================================
# MAGIC
# MAGIC SELECT 
# MAGIC   recommended_action,
# MAGIC   COUNT(*) as student_count,
# MAGIC   ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) as pct_of_total,
# MAGIC   ROUND(AVG(risk_score), 4) as avg_risk_score,
# MAGIC   ROUND(MIN(risk_score), 4) as min_risk,
# MAGIC   ROUND(MAX(risk_score), 4) as max_risk
# MAGIC FROM dropout_gold.student_intervention_queue
# MAGIC GROUP BY recommended_action
# MAGIC ORDER BY student_count DESC;

# COMMAND ----------

# DBTITLE 1,Query 2: Top Risk Factors
# MAGIC %sql
# MAGIC -- =========================================================
# MAGIC -- 2. MOST COMMON RISK FACTORS
# MAGIC -- =========================================================
# MAGIC
# MAGIC -- Top Primary Factor
# MAGIC SELECT 
# MAGIC   'Factor 1' as factor_position,
# MAGIC   top_risk_factor_1 as risk_factor,
# MAGIC   COUNT(*) as student_count,
# MAGIC   ROUND(AVG(risk_score), 3) as avg_risk_score
# MAGIC FROM dropout_gold.student_intervention_queue
# MAGIC GROUP BY top_risk_factor_1
# MAGIC
# MAGIC UNION ALL
# MAGIC
# MAGIC -- Top Secondary Factor  
# MAGIC SELECT 
# MAGIC   'Factor 2' as factor_position,
# MAGIC   top_risk_factor_2 as risk_factor,
# MAGIC   COUNT(*) as student_count,
# MAGIC   ROUND(AVG(risk_score), 3) as avg_risk_score
# MAGIC FROM dropout_gold.student_intervention_queue
# MAGIC GROUP BY top_risk_factor_2
# MAGIC
# MAGIC UNION ALL
# MAGIC
# MAGIC -- Top Tertiary Factor
# MAGIC SELECT 
# MAGIC   'Factor 3' as factor_position,
# MAGIC   top_risk_factor_3 as risk_factor,
# MAGIC   COUNT(*) as student_count,
# MAGIC   ROUND(AVG(risk_score), 3) as avg_risk_score
# MAGIC FROM dropout_gold.student_intervention_queue
# MAGIC GROUP BY top_risk_factor_3
# MAGIC
# MAGIC ORDER BY factor_position, student_count DESC;

# COMMAND ----------

# DBTITLE 1,Query 3: Unique Risk Combinations
# MAGIC %sql
# MAGIC -- =========================================================
# MAGIC -- 3. UNIQUE RISK FACTOR COMBINATIONS (Top 20)
# MAGIC -- =========================================================
# MAGIC
# MAGIC SELECT 
# MAGIC   top_risk_factor_1,
# MAGIC   top_risk_factor_2,
# MAGIC   top_risk_factor_3,
# MAGIC   COUNT(*) as student_count,
# MAGIC   ROUND(AVG(risk_score), 4) as avg_risk_score,
# MAGIC   FIRST(recommended_action) as typical_action,
# MAGIC   ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) as pct_of_total
# MAGIC FROM dropout_gold.student_intervention_queue
# MAGIC GROUP BY top_risk_factor_1, top_risk_factor_2, top_risk_factor_3
# MAGIC ORDER BY student_count DESC
# MAGIC LIMIT 20;

# COMMAND ----------

# DBTITLE 1,Query 4: Financial Aid Cases
# MAGIC %sql
# MAGIC -- =========================================================
# MAGIC -- 4. FINANCIAL AID COUNSELING CANDIDATES
# MAGIC -- =========================================================
# MAGIC
# MAGIC SELECT 
# MAGIC   student_id,
# MAGIC   ROUND(risk_score, 4) as risk_score,
# MAGIC   confidence_band,
# MAGIC   top_risk_factor_1,
# MAGIC   top_risk_factor_2,
# MAGIC   top_risk_factor_3,
# MAGIC   priority_rank
# MAGIC FROM dropout_gold.student_intervention_queue
# MAGIC WHERE recommended_action = 'financial_aid_counseling'
# MAGIC ORDER BY risk_score DESC
# MAGIC LIMIT 20;

# COMMAND ----------

# DBTITLE 1,Query 5: Academic Mentoring Cases
# MAGIC %sql
# MAGIC -- =========================================================
# MAGIC -- 5. ACADEMIC MENTORING CANDIDATES
# MAGIC -- =========================================================
# MAGIC
# MAGIC SELECT 
# MAGIC   student_id,
# MAGIC   ROUND(risk_score, 4) as risk_score,
# MAGIC   confidence_band,
# MAGIC   top_risk_factor_1,
# MAGIC   top_risk_factor_2,
# MAGIC   top_risk_factor_3,
# MAGIC   priority_rank
# MAGIC FROM dropout_gold.student_intervention_queue
# MAGIC WHERE recommended_action = 'academic_mentoring'
# MAGIC ORDER BY risk_score DESC
# MAGIC LIMIT 20;

# COMMAND ----------

# DBTITLE 1,Query 6: Diverse High-Risk Students
# MAGIC %sql
# MAGIC -- =========================================================
# MAGIC -- 6. HIGH-RISK STUDENTS WITH DIVERSE FACTORS
# MAGIC -- (Excluding most common 'Weak prior academic record')
# MAGIC -- =========================================================
# MAGIC
# MAGIC SELECT 
# MAGIC   student_id,
# MAGIC   ROUND(risk_score, 4) as risk_score,
# MAGIC   recommended_action,
# MAGIC   top_risk_factor_1,
# MAGIC   top_risk_factor_2,
# MAGIC   top_risk_factor_3,
# MAGIC   priority_rank
# MAGIC FROM dropout_gold.student_intervention_queue
# MAGIC WHERE 
# MAGIC   risk_score >= 0.7
# MAGIC   AND top_risk_factor_1 NOT IN ('Weak prior academic record', 'Other model risk signal')
# MAGIC ORDER BY risk_score DESC
# MAGIC LIMIT 30;

# COMMAND ----------

# DBTITLE 1,Query 7: Intervention Tier Analysis
# MAGIC %sql
# MAGIC -- =========================================================
# MAGIC -- 7. INTERVENTION TIER vs ACTION TYPE
# MAGIC -- =========================================================
# MAGIC
# MAGIC SELECT 
# MAGIC   intervention_tier,
# MAGIC   recommended_action,
# MAGIC   COUNT(*) as student_count,
# MAGIC   ROUND(AVG(risk_score), 4) as avg_risk_score,
# MAGIC   ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(PARTITION BY intervention_tier), 2) as pct_within_tier
# MAGIC FROM dropout_gold.student_intervention_queue
# MAGIC GROUP BY intervention_tier, recommended_action
# MAGIC ORDER BY intervention_tier, student_count DESC;

# COMMAND ----------

# DBTITLE 1,SHAP Feature Debugging & Mapping Improvement
# =========================================================
# DEBUG: SEE ACTUAL RAW SHAP FEATURES
# =========================================================

print("\n" + "="*80)
print("DIAGNOSING RAW SHAP FEATURES")
print("="*80)

# Get the explanation table
explanations = spark.table(EXPLANATIONS_TABLE).toPandas()

print(f"\nTotal explained students: {len(explanations)}")
print(f"\nSample of RAW features SHAP identified (before mapping):")

# Show sample of raw features
sample = explanations.head(10)[["student_id", "risk_score", "raw_feature_1", "raw_feature_2", "raw_feature_3"]]
print(sample.to_string())

print("\n" + "="*80)
print("RAW FEATURE FREQUENCY ANALYSIS")
print("="*80)

# Count most common raw features
print("\n🔵 Most common RAW features in position 1:")
raw1_counts = explanations['raw_feature_1'].value_counts().head(15)
for feat, count in raw1_counts.items():
    print(f"  {feat}: {count} students")

print("\n🔵 Most common RAW features in position 2:")
raw2_counts = explanations['raw_feature_2'].value_counts().head(15)
for feat, count in raw2_counts.items():
    print(f"  {feat}: {count} students")

print("\n🔵 Most common RAW features in position 3:")
raw3_counts = explanations['raw_feature_3'].value_counts().head(15)
for feat, count in raw3_counts.items():
    print(f"  {feat}: {count} students")

print("\n" + "="*80)
print("MAPPED REASON DISTRIBUTION")
print("="*80)

print("\n🟢 Mapped reasons for position 1:")
mapped1_counts = explanations['factor_1'].value_counts().head(10)
for reason, count in mapped1_counts.items():
    print(f"  {reason}: {count} students")

print("\n" + "="*80)
print("⚠️ WHY MAPPING FAILS")
print("="*80)

print("""
The problem is clear:

1. SHAP identifies ONE-HOT ENCODED features like:
   - 'cat__course_17' (course ID 17)
   - 'cat__application_mode_5' (application mode 5) 
   - 'num__age_at_enrollment' (numeric age)
   
2. Your FEATURE_REASON_MAPPING only has:
   - 'course' (not 'cat__course_17')
   - 'application_mode' (not 'cat__application_mode_5')
   - 'age_at_enrollment' (this might match!)

3. The normalize_raw_feature_name() function:
   - Removes 'num__' and 'cat__' prefixes
   - But 'course_17' still doesn't match 'course' exactly
   - Falls back to "Other model risk signal"

SOLUTION: Improve the mapping function to handle:
- One-hot encoded categoricals: extract base feature name
- Example: 'cat__course_17' → 'course'
""")

print("\n" + "="*80)
print("✅ IMPROVED MAPPING FUNCTION")
print("="*80)

# Create improved mapping function
def improved_map_raw_feature_to_reason(raw_feature: str) -> str:
    """Enhanced feature mapping that handles one-hot encoded features"""
    if raw_feature is None or raw_feature == "other_model_signal":
        return "Other model risk signal"
    
    # Normalize: remove prefix and lowercase
    normalized = raw_feature.lower()
    for prefix in ["num__", "cat__"]:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break
    
    # Extract base feature name (remove one-hot encoding suffix)
    # Example: 'course_17' -> 'course', 'application_mode_5' -> 'application_mode'
    base_feature = normalized
    if '_' in normalized:
        parts = normalized.split('_')
        # Try removing last part if it's a number
        if len(parts) > 1 and parts[-1].isdigit():
            base_feature = '_'.join(parts[:-1])
    
    # Try to match against FEATURE_REASON_MAPPING
    for key, label in REASON_MAP_DICT.items():
        if base_feature == key or normalized == key or key in normalized or normalized in key:
            return label
    
    return "Other model risk signal"

# Test the improved mapping
print("\nTesting improved mapping on sample raw features:")
test_features = [
    "cat__course_17",
    "cat__application_mode_5", 
    "num__age_at_enrollment",
    "num__curricular_units_1st_sem_grade",
    "cat__financial_segment_high_financial_stress",
    "num__financial_stress_index"
]

for feat in test_features:
    old_result = map_raw_feature_to_reason(feat)
    new_result = improved_map_raw_feature_to_reason(feat)
    print(f"\n  {feat}")
    print(f"    OLD: {old_result}")
    print(f"    NEW: {new_result}")

print("\n" + "="*80)
print("🛠️ NEXT STEPS TO GET DIVERSE INTERVENTIONS")
print("="*80)

print("""
TO FIX THE ISSUE:

1. ✅ Update Cell 8 - Replace map_raw_feature_to_reason with improved version
   
2. ✅ Re-run Cell 8 - This will:
   - Regenerate explanations with better mappings
   - Create more diverse factor_1, factor_2, factor_3
   - Trigger different intervention logic
   
3. ✅ Expected improvements:
   - Financial factors will trigger 'financial_aid_counseling'
   - Academic factors will trigger 'academic_mentoring' 
   - Engagement factors will trigger 'counselor_outreach'
   - More diversity in recommendations!

4. 🔍 Alternative: Add more specific patterns to map_action_from_reasons
   - Currently it looks for exact strings like "High financial stress"
   - Could be more flexible: check if 'financial' in any factor
""")

print("\n💡 Want me to update Cell 8 with the improved mapping? Let me know!")

# COMMAND ----------

# DBTITLE 1,Query 8: Check EXPLANATIONS Table
# MAGIC %sql
# MAGIC -- =========================================================
# MAGIC -- 8. INSPECT RAW SHAP FEATURES IN EXPLANATIONS TABLE
# MAGIC -- =========================================================
# MAGIC
# MAGIC SELECT 
# MAGIC   student_id,
# MAGIC   ROUND(risk_score, 4) as risk_score,
# MAGIC   raw_feature_1,
# MAGIC   raw_feature_2, 
# MAGIC   raw_feature_3,
# MAGIC   factor_1 as mapped_1,
# MAGIC   factor_2 as mapped_2,
# MAGIC   factor_3 as mapped_3
# MAGIC FROM dropout_gold.student_explanations
# MAGIC ORDER BY risk_score DESC
# MAGIC LIMIT 30;

# COMMAND ----------

# DBTITLE 1,Fix Instructions
# MAGIC %md
# MAGIC # 🔧 **How to Fix the SHAP Feature Selection Bug**
# MAGIC
# MAGIC ## ❌ **The Problem:**
# MAGIC
# MAGIC In **Cell 8, Section 8 (EXPLANATIONS)**, the code has a bug where:
# MAGIC - `raw_feature_1` stores `"['feature1' 'feature2']"` (an array as string)
# MAGIC - Should store just `"feature1"` (single feature)
# MAGIC
# MAGIC ## ✅ **The Solution:**
# MAGIC
# MAGIC Replace the SHAP explanation loop in Cell 8 with this corrected version:
# MAGIC
# MAGIC ```python
# MAGIC # CORRECTED: Extract individual features, not arrays
# MAGIC for i in range(len(explain_pdf)):
# MAGIC     sid = int(explain_pdf.iloc[i]["student_id"])
# MAGIC     risk_score = float(scored_lookup.get(sid, 0.0))
# MAGIC
# MAGIC     contrib = sv[i]
# MAGIC     ranked_idx = np.argsort(np.abs(contrib))[::-1]
# MAGIC
# MAGIC     raw_selected = []
# MAGIC     mapped_selected = []
# MAGIC     seen_mapped = set()
# MAGIC
# MAGIC     for j in ranked_idx:
# MAGIC         # FIX: Ensure feature_names[j] returns a single string
# MAGIC         raw_feature = str(feature_names[j])
# MAGIC         
# MAGIC         # Skip if this is somehow still an array
# MAGIC         if '[' in raw_feature or raw_feature == 'other_model_signal':
# MAGIC             continue
# MAGIC             
# MAGIC         mapped_reason = map_raw_feature_to_reason(raw_feature)
# MAGIC
# MAGIC         # Only add if we haven't seen this mapped reason yet
# MAGIC         if mapped_reason not in seen_mapped:
# MAGIC             raw_selected.append(raw_feature)
# MAGIC             mapped_selected.append(mapped_reason)
# MAGIC             seen_mapped.add(mapped_reason)
# MAGIC
# MAGIC         if len(mapped_selected) >= 3:
# MAGIC             break
# MAGIC
# MAGIC     # Pad with defaults if needed
# MAGIC     while len(raw_selected) < 3:
# MAGIC         raw_selected.append("other_model_signal")
# MAGIC     while len(mapped_selected) < 3:
# MAGIC         mapped_selected.append("Other model risk signal")
# MAGIC
# MAGIC     rows.append({
# MAGIC         "student_id": sid,
# MAGIC         "risk_score": risk_score,
# MAGIC         "raw_feature_1": raw_selected[0],  # Single string
# MAGIC         "raw_feature_2": raw_selected[1],  # Single string
# MAGIC         "raw_feature_3": raw_selected[2],  # Single string
# MAGIC         "factor_1": mapped_selected[0],
# MAGIC         "factor_2": mapped_selected[1],
# MAGIC         "factor_3": mapped_selected[2]
# MAGIC     })
# MAGIC ```
# MAGIC
# MAGIC ## 🚀 **After the Fix:**
# MAGIC
# MAGIC You'll see diverse interventions:
# MAGIC - 💰 `financial_aid_counseling` - for students with financial stress
# MAGIC - 📚 `academic_mentoring` - for students with grade/approval issues
# MAGIC - 👥 `counselor_outreach` - for students with engagement problems
# MAGIC - 🔍 `counselor_review` - for complex cases
# MAGIC - 📊 `monitor` - for low-risk students
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Would you like me to update Cell 8 with this fix?**

# COMMAND ----------

# DBTITLE 1,Alternative: Debug feature_names
# =========================================================
# ALTERNATIVE DIAGNOSTIC: Check what feature_names contains
# =========================================================

print("\n" + "="*80)
print("DEBUGGING: What does feature_names actually contain?")
print("="*80)

# Re-extract the feature names from the trained model
preprocessor = best_model.named_steps['preprocessor']
feature_names_out = preprocessor.get_feature_names_out()

print(f"\nTotal features after preprocessing: {len(feature_names_out)}")
print(f"\nFirst 20 feature names:")
for i, fname in enumerate(feature_names_out[:20]):
    print(f"  [{i}] {fname} (type: {type(fname)})")

print(f"\nLast 10 feature names:")
for i, fname in enumerate(feature_names_out[-10:], start=len(feature_names_out)-10):
    print(f"  [{i}] {fname} (type: {type(fname)})")

# Check for any anomalies
print(f"\n\n🔍 Checking for array-type features:")
array_features = []
for i, fname in enumerate(feature_names_out):
    fname_str = str(fname)
    if '[' in fname_str or ']' in fname_str:
        array_features.append((i, fname_str))

if array_features:
    print(f"\n⚠️ Found {len(array_features)} array-type features:")
    for idx, feat in array_features[:10]:
        print(f"  [{idx}] {feat}")
else:
    print("\n✅ No array-type features found - all look normal!")
    print("\nThis means the bug is in how features are being selected/saved,")
    print("not in the feature_names itself.")

print("\n" + "="*80)
print("💡 RECOMMENDATION")
print("="*80)
print("""
The most likely cause is in the SHAP explanation loop where features
are being accumulated. The code might be grouping features that have
the same mapped reason instead of picking them individually.

Solution: Update Cell 8 Section 8 with the corrected loop shown in
the previous markdown cell.
""")