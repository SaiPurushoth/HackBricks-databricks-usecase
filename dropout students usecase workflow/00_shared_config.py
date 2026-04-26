# Databricks notebook source
# Databricks notebook source

from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window

import re
import smtplib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from email.mime.text import MIMEText

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
    roc_auc_score,
)

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


SOURCE_PATH = "/Volumes/workspace/default/hackbricks/students_dropout_academic_success.csv"

BRONZE_SCHEMA = "dropout_bronze"
SILVER_SCHEMA = "dropout_silver"
FEATURE_SCHEMA = "dropout_feature"
GOLD_SCHEMA = "dropout_gold"

BRONZE_TABLE = f"{BRONZE_SCHEMA}.student_dropout_raw"
VALIDATED_SOURCE_TABLE = f"{SILVER_SCHEMA}.student_source_validated"

PROFILE_TABLE = f"{SILVER_SCHEMA}.student_profile"
DEMOGRAPHIC_TABLE = f"{SILVER_SCHEMA}.student_demographic_features"
ACADEMIC_BG_TABLE = f"{SILVER_SCHEMA}.student_academic_background"
FAMILY_BG_TABLE = f"{SILVER_SCHEMA}.student_family_background"
FINANCIAL_TABLE = f"{SILVER_SCHEMA}.student_financial_status"
ACADEMIC_PERF_TABLE = f"{SILVER_SCHEMA}.student_academic_performance"
CONTEXT_TABLE = f"{SILVER_SCHEMA}.student_context"
COUNSELOR_ASSIGNMENT_TABLE = f"{SILVER_SCHEMA}.student_counselor_assignment"

FEATURE_TABLE = f"{FEATURE_SCHEMA}.student_dropout_risk_features"

MODEL_METRICS_TABLE = f"{GOLD_SCHEMA}.model_metrics"
MODEL_SCORE_TABLE = f"{GOLD_SCHEMA}.model_test_scores"
FAIRNESS_TABLE = f"{GOLD_SCHEMA}.fairness_audit"
EXPLANATIONS_TABLE = f"{GOLD_SCHEMA}.student_explanations"
INTERVENTION_TABLE = f"{GOLD_SCHEMA}.student_intervention_queue"
REASON_MAPPING_TABLE = f"{GOLD_SCHEMA}.feature_reason_mapping"
EMAIL_ALERT_LOG_TABLE = f"{GOLD_SCHEMA}.counselor_email_alert_log"
REGISTERED_MODEL_NAME = "student_dropout_risk_model"

RESET_TABLES = False
RANDOM_STATE = 42
TEST_SIZE = 0.2

AUDIT_ONLY_FEATURES = [
    "gender",
    "nacionality",
    "marital_status",
    "age_at_enrollment",
    "international",
    "displaced",
    "educational_special_needs",
    "mothers_qualification",
    "fathers_qualification",
    "mothers_occupation",
    "fathers_occupation",
]


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


def ensure_schemas():
    for schema_name in [BRONZE_SCHEMA, SILVER_SCHEMA, FEATURE_SCHEMA, GOLD_SCHEMA]:
        spark.sql(f"CREATE DATABASE IF NOT EXISTS {schema_name}")


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


def materialize_reason_mapping():
    mapping_pdf = pd.DataFrame(FEATURE_REASON_MAPPING, columns=["raw_feature_key", "mapped_reason"])
    mapping_sdf = spark.createDataFrame(mapping_pdf)
    (
        mapping_sdf.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(REASON_MAPPING_TABLE)
    )


def normalize_raw_feature_name(raw_feature: str) -> str:
    if raw_feature is None or raw_feature == "other_model_signal":
        return None
    r = raw_feature.lower()
    for prefix in ["num__", "cat__"]:
        if r.startswith(prefix):
            r = r[len(prefix):]
            break
    if "_" in r:
        parts = r.split("_")
        if len(parts) > 1 and parts[-1].isdigit():
            r = "_".join(parts[:-1])
    return r


def build_reason_mapping_dict():
    return {k.lower(): v for k, v in FEATURE_REASON_MAPPING}


REASON_MAP_DICT = build_reason_mapping_dict()
KNOWN_REASON_KEYS = sorted(REASON_MAP_DICT.keys(), key=len, reverse=True)
SENSITIVE_BASE_FEATURES = {
    "gender",
    "nacionality",
    "marital_status",
    "age_at_enrollment",
    "international",
    "displaced",
    "educational_special_needs",
    "mothers_qualification",
    "fathers_qualification",
    "mothers_occupation",
    "fathers_occupation",
}

FINANCIAL_REASON_SET = {
    "High financial stress",
    "Severe financial pressure",
    "Moderate financial pressure",
    "Outstanding financial dues",
    "Tuition payments not up to date",
    "No scholarship support",
    "High financial risk profile",
    "Moderate financial risk profile",
}

ACADEMIC_REASON_SET = {
    "Semester grade decline",
    "Low overall grades",
    "Weak first-semester grades",
    "Weak second-semester grades",
    "Weak prior academic record",
    "Lower admission grade",
    "Drop in subject approvals",
    "Low approval ratio",
    "Low first-semester approvals",
    "Low second-semester approvals",
    "Recent decline in grades",
    "Recent decline in approvals",
    "Declining academic momentum",
    "Stagnant academic momentum",
    "Low total approved units",
    "Performance below course average",
}

ENGAGEMENT_REASON_SET = {
    "Low learning engagement",
    "Drop in evaluation participation",
    "Recent decline in engagement",
    "Low first-semester evaluation activity",
    "Low second-semester evaluation activity",
    "Missed first-semester evaluations",
    "Missed second-semester evaluations",
    "High first-semester non-evaluation ratio",
    "High second-semester non-evaluation ratio",
    "High overall absenteeism",
    "Student disengagement pattern",
}

CONTEXT_REASON_SET = {
    "Enrolled in high-rigor course",
    "Course grade deflation pattern",
    "High-competition course environment",
    "High unemployment environment",
    "Inflation-related stress environment",
    "Economic environment pattern",
    "Change in academic workload",
    "First-semester course load pattern",
    "Second-semester course load pattern",
}


def map_raw_feature_to_reason(raw_feature: str) -> str:
    if raw_feature is None or raw_feature == "other_model_signal":
        return "Other model risk signal"
    normalized = normalize_raw_feature_name(raw_feature)
    if normalized is None:
        return "Other model risk signal"
    if normalized in REASON_MAP_DICT:
        return REASON_MAP_DICT[normalized]
    for key, label in REASON_MAP_DICT.items():
        if normalized == key or key in normalized or normalized in key:
            return label
    return "Other model risk signal"


def extract_base_feature_name(raw_feature: str) -> str:
    normalized = normalize_raw_feature_name(raw_feature)
    if normalized is None:
        return None
    if normalized in REASON_MAP_DICT or normalized in SENSITIVE_BASE_FEATURES:
        return normalized
    for key in KNOWN_REASON_KEYS:
        if normalized == key or normalized.startswith(f"{key}_"):
            return key
    for key in SENSITIVE_BASE_FEATURES:
        if normalized == key or normalized.startswith(f"{key}_"):
            return key
    if "_" in normalized:
        parts = normalized.split("_")
        if len(parts) > 1 and parts[-1].isdigit():
            collapsed = "_".join(parts[:-1])
            if collapsed in REASON_MAP_DICT or collapsed in SENSITIVE_BASE_FEATURES:
                return collapsed
    return normalized


def is_sensitive_feature(raw_feature: str) -> bool:
    return extract_base_feature_name(raw_feature) in SENSITIVE_BASE_FEATURES


def extract_positive_class_shap_values(shap_values, n_samples: int, n_features: int) -> np.ndarray:
    values = shap_values.values if hasattr(shap_values, "values") else shap_values
    if isinstance(values, list):
        arr = np.asarray(values[1] if len(values) > 1 else values[0], dtype=np.float64)
    else:
        arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 3:
        if arr.shape[-1] == 2:
            arr = arr[:, :, 1]
        elif arr.shape[0] == 2:
            arr = arr[1]
        else:
            arr = arr[:, :, 0]
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim == 2 and arr.shape == (n_features, n_samples):
        arr = arr.T
    if arr.ndim != 2:
        raise ValueError(f"Unexpected SHAP output shape: {arr.shape}")
    if arr.shape[0] != n_samples or arr.shape[1] != n_features:
        raise ValueError(f"SHAP shape mismatch. Expected ({n_samples}, {n_features}), got {arr.shape}")
    return arr


def extract_fallback_contributions(model, transformed_matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(transformed_matrix, dtype=np.float64)
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=np.float64)
        return arr * importances.reshape(1, -1)
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=np.float64)
        if coef.ndim == 2:
            coef = coef[1] if coef.shape[0] > 1 else coef[0]
        return arr * coef.reshape(1, -1)
    raise RuntimeError(f"No explanation method available for model type: {type(model).__name__}")


def get_scored_test_dataframe(model_pipeline, X_test_df, y_test_series, full_pdf, idx_test):
    scored_pdf = X_test_df.copy()
    scored_pdf["student_id"] = full_pdf.loc[idx_test, "student_id"].astype("int64").values
    scored_pdf["actual_dropout"] = y_test_series.values
    scored_pdf["risk_score"] = model_pipeline.predict_proba(X_test_df)[:, 1]
    scored_pdf["predicted_dropout"] = (scored_pdf["risk_score"] >= 0.5).astype(int)
    return scored_pdf


def map_action_from_reasons(r1, r2, r3, risk_score):
    reasons = {r for r in [r1, r2, r3] if r and r != "Other model risk signal"}

    financial_hits = len(reasons & FINANCIAL_REASON_SET)
    academic_hits = len(reasons & ACADEMIC_REASON_SET)
    engagement_hits = len(reasons & ENGAGEMENT_REASON_SET)
    context_hits = len(reasons & CONTEXT_REASON_SET)

    domain_hits = sum([
        financial_hits > 0,
        academic_hits > 0,
        engagement_hits > 0,
        context_hits > 0
    ])

    first_gen_hits = len(reasons & {
        "Mother education background",
        "Father education background",
        "Mother occupation background",
        "Father occupation background",
        "Prior qualification pathway"
    })

    tough_course_hits = len(reasons & {
        "Enrolled in high-rigor course",
        "Course grade deflation pattern",
        "High-competition course environment",
        "Course-level dropout pattern",
        "Performance below course average"
    })

    # Highest priority: very high risk across multiple domains
    if risk_score >= 0.85 and domain_hits >= 2:
        return "high_priority_case_review"

    # High risk with combined issues
    if risk_score >= 0.75 and domain_hits >= 2:
        return "coordinated_support_plan"

    # Family educational background + academic support need
    if risk_score >= 0.60 and first_gen_hits > 0 and academic_hits > 0:
        return "mentorship_program_referral"

    # Course difficulty / course mismatch
    if risk_score >= 0.70 and tough_course_hits > 0 and academic_hits > 0:
        return "course_pathway_review"

    # Financial stress
    if risk_score >= 0.70 and financial_hits > 0:
        return "financial_support_review"

    # Academic weakness
    if risk_score >= 0.70 and academic_hits > 0:
        return "academic_tutoring_required"

    # Attendance / disengagement
    if risk_score >= 0.60 and engagement_hits > 0:
        return "attendance_reengagement"

    # Medium risk general support
    if risk_score >= 0.55:
        return "student_success_coaching"

    # Light-touch human follow-up
    if risk_score >= 0.40:
        return "advisor_follow_up"

    # Low risk
    return "no_immediate_action"


map_action_udf = F.udf(map_action_from_reasons, T.StringType())


def get_feature_training_inputs():
    pdf = spark.table(FEATURE_TABLE).toPandas()
    categorical_cols = [
        "course", "application_mode", "application_order", "daytime_evening_attendance",
        "previous_qualification", "debtor",
        "tuition_fees_up_to_date", "scholarship_holder", "financial_risk_band",
        "financial_segment", "academic_momentum_band", "grade_drop_flag",
        "approval_drop_flag", "engagement_drop_flag", "is_ghosting", "is_primary_choice",
        "grade_deflation_flag",
    ]
    drop_cols = [c for c in ["target", "target_dropout", "feature_ts", "student_id"] if c in pdf.columns]
    X = pdf.drop(columns=drop_cols)
    audit_only_cols = [c for c in AUDIT_ONLY_FEATURES if c in X.columns]
    if audit_only_cols:
        X = X.drop(columns=audit_only_cols)
    y = pdf["target_dropout"]
    categorical_cols = [c for c in categorical_cols if c in X.columns]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    for c in categorical_cols:
        X[c] = X[c].astype(str)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols),
        ]
    )
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, pdf.index, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return {
        "pdf": pdf,
        "X": X,
        "y": y,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "preprocessor": preprocessor,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "idx_train": idx_train,
        "idx_test": idx_test,
    }


def load_best_model_from_mlflow():
    metrics_pdf = spark.table(MODEL_METRICS_TABLE).toPandas()
    best_row = metrics_pdf.sort_values(["is_best_model", "roc_auc"], ascending=[False, False]).iloc[0]
    if (
        "registered_model_version" in best_row.index
        and pd.notnull(best_row["registered_model_version"])
        and int(best_row["registered_model_version"]) != -1
    ):
        model_uri = f"models:/{REGISTERED_MODEL_NAME}/{int(best_row['registered_model_version'])}"
    else:
        model_uri = f"runs:/{best_row['run_id']}/model"
    return mlflow.sklearn.load_model(model_uri), best_row


def register_run_model(run_id: str, registered_model_name: str = REGISTERED_MODEL_NAME):
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/model"
    registered_model = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
    return int(registered_model.version)


def get_secret_or_none(scope: str, key: str):
    try:
        return dbutils.secrets.get(scope=scope, key=key)
    except Exception:
        return None


def send_email_via_smtp(
    smtp_host: str,
    smtp_port: int,
    smtp_username: str,
    smtp_password: str,
    sender_email: str,
    recipient_email: str,
    subject: str,
    body: str,
):
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = recipient_email

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(smtp_username, smtp_password)
        server.sendmail(sender_email, [recipient_email], msg.as_string())


# COMMAND ----------

