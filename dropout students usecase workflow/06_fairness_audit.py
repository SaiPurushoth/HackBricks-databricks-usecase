# Databricks notebook source
# MAGIC %pip install fairlearn
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run /Users/saitwins777@gmail.com/dropout/00_shared_config

# COMMAND ----------

# Databricks notebook source
# MAGIC %run ./00_shared_config

ensure_schemas()

print_header("6. FAIRNESS AUDIT WITH FAIRLEARN")

# If needed once per cluster:
# %pip install fairlearn

try:
    from fairlearn.metrics import (
        MetricFrame,
        selection_rate,
        true_positive_rate,
        false_positive_rate,
        demographic_parity_difference,
        demographic_parity_ratio,
        equal_opportunity_difference,
        equal_opportunity_ratio,
        equalized_odds_difference,
        equalized_odds_ratio,
    )
except Exception as exc:
    raise RuntimeError(
        "fairlearn is required for 06_fairness_audit.py. Install it with `%pip install fairlearn`."
    ) from exc

scored = spark.table(MODEL_SCORE_TABLE).toPandas()

demo_pdf = (
    spark.table(DEMOGRAPHIC_TABLE)
    .select("student_id", "gender")
    .toPandas()
)
demo_pdf["student_id"] = demo_pdf["student_id"].astype("int64")

socio_pdf = (
    spark.table(FEATURE_TABLE)
    .select("student_id", "financial_segment", "scholarship_holder")
    .toPandas()
)
socio_pdf["student_id"] = socio_pdf["student_id"].astype("int64")

audit_pdf = (
    scored[["student_id", "actual_dropout", "predicted_dropout", "risk_score"]]
    .merge(demo_pdf, on="student_id", how="left")
    .merge(socio_pdf, on="student_id", how="left")
)

audit_pdf["gender"] = audit_pdf["gender"].fillna("unknown").astype(str)
audit_pdf["financial_segment"] = audit_pdf["financial_segment"].fillna("unknown").astype(str)
audit_pdf["scholarship_holder"] = audit_pdf["scholarship_holder"].fillna("unknown").astype(str)

audit_pdf["gender_x_financial_segment"] = (
    audit_pdf["gender"].astype(str) + "|" + audit_pdf["financial_segment"].astype(str)
)

def build_fairlearn_report(df: pd.DataFrame, sensitive_col: str, model_name: str) -> pd.DataFrame:
    y_true = df["actual_dropout"]
    y_pred = df["predicted_dropout"]
    sensitive = df[sensitive_col]

    mf = MetricFrame(
        metrics={
            "selection_rate": selection_rate,
            "true_positive_rate": true_positive_rate,
            "false_positive_rate": false_positive_rate,
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive,
    )

    by_group = mf.by_group.reset_index()
    by_group.columns = [sensitive_col if c == "index" else c for c in by_group.columns]

    summary = {
        "model_name": model_name,
        "protected_attribute": sensitive_col,
        "demographic_parity_difference": float(
            demographic_parity_difference(
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive,
            )
        ),
        "demographic_parity_ratio": float(
            demographic_parity_ratio(
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive,
            )
        ),
        "equal_opportunity_difference": float(
            equal_opportunity_difference(
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive,
            )
        ),
        "equal_opportunity_ratio": float(
            equal_opportunity_ratio(
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive,
            )
        ),
        "equalized_odds_difference": float(
            equalized_odds_difference(
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive,
            )
        ),
        "equalized_odds_ratio": float(
            equalized_odds_ratio(
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive,
            )
        ),
        "audit_ts": pd.Timestamp.utcnow().isoformat(),
    }

    rows = []
    for _, r in by_group.iterrows():
        rows.append({
            "model_name": model_name,
            "protected_attribute": sensitive_col,
            "group_name": str(r[sensitive_col]),
            "population_count": int((df[sensitive_col] == r[sensitive_col]).sum()),
            "selection_rate": float(r["selection_rate"]) if pd.notnull(r["selection_rate"]) else None,
            "true_positive_rate": float(r["true_positive_rate"]) if pd.notnull(r["true_positive_rate"]) else None,
            "false_positive_rate": float(r["false_positive_rate"]) if pd.notnull(r["false_positive_rate"]) else None,
            "demographic_parity_difference": summary["demographic_parity_difference"],
            "demographic_parity_ratio": summary["demographic_parity_ratio"],
            "equal_opportunity_difference": summary["equal_opportunity_difference"],
            "equal_opportunity_ratio": summary["equal_opportunity_ratio"],
            "equalized_odds_difference": summary["equalized_odds_difference"],
            "equalized_odds_ratio": summary["equalized_odds_ratio"],
            "fairness_review_flag": int(
                abs(summary["demographic_parity_difference"]) > 0.10
                or abs(summary["equal_opportunity_difference"]) > 0.10
                or abs(summary["equalized_odds_difference"]) > 0.10
            ),
            "audit_ts": summary["audit_ts"],
        })

    return pd.DataFrame(rows)


model_name = (
    spark.table(MODEL_METRICS_TABLE)
    .filter("is_best_model = 1")
    .select("model_name")
    .first()[0]
)

gender_report = build_fairlearn_report(audit_pdf, "gender", model_name)
financial_report = build_fairlearn_report(audit_pdf, "financial_segment", model_name)
scholarship_report = build_fairlearn_report(audit_pdf, "scholarship_holder", model_name)
intersection_report = build_fairlearn_report(audit_pdf, "gender_x_financial_segment", model_name)

fairness_report = pd.concat(
    [gender_report, financial_report, scholarship_report, intersection_report],
    ignore_index=True,
)

fairness_sdf = spark.createDataFrame(fairness_report)
(
    fairness_sdf.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(FAIRNESS_TABLE)
)

display(spark.table(FAIRNESS_TABLE))

# COMMAND ----------

