# Databricks notebook source
# MAGIC %run /Users/saitwins777@gmail.com/dropout/00_shared_config

# COMMAND ----------

# Databricks notebook source
# MAGIC %run ./00_shared_config

ensure_schemas()

print_header("7. EXPLANATIONS")

spark.sql(f"DROP TABLE IF EXISTS {EXPLANATIONS_TABLE}")

inputs = get_feature_training_inputs()
pdf = inputs["pdf"]
X = inputs["X"]
categorical_cols = inputs["categorical_cols"]

best_model, _ = load_best_model_from_mlflow()
scored = spark.table(MODEL_SCORE_TABLE).toPandas()
positive_student_ids = scored.loc[scored["predicted_dropout"] == 1, "student_id"].astype("int64").tolist()

if len(positive_student_ids) == 0:
    explanations_pdf = pd.DataFrame(columns=[
        "student_id", "risk_score", "raw_feature_1", "raw_feature_2", "raw_feature_3",
        "factor_1", "factor_2", "factor_3",
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

    if SHAP_AVAILABLE and isinstance(model, RandomForestClassifier):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(explain_transformed)
        sv = extract_positive_class_shap_values(shap_values, explain_transformed.shape[0], explain_transformed.shape[1])
    elif SHAP_AVAILABLE and isinstance(model, LogisticRegression):
        explainer = shap.LinearExplainer(model, explain_transformed)
        shap_values = explainer.shap_values(explain_transformed)
        sv = extract_positive_class_shap_values(shap_values, explain_transformed.shape[0], explain_transformed.shape[1])
    elif SHAP_AVAILABLE:
        explainer = shap.Explainer(model, explain_transformed)
        shap_values = explainer(explain_transformed)
        sv = extract_positive_class_shap_values(shap_values, explain_transformed.shape[0], explain_transformed.shape[1])
    else:
        print("SHAP not available; using model-based fallback explanations.")
        sv = extract_fallback_contributions(model, explain_transformed)

    rows = []
    scored_lookup = scored.set_index("student_id")["risk_score"].to_dict()

    for i in range(len(explain_pdf)):
        sid = int(explain_pdf.iloc[i]["student_id"])
        risk_score = float(scored_lookup.get(sid, 0.0))
        contrib = sv[i]
        positive_ranked_idx = np.where(contrib > 0)[0]
        positive_ranked_idx = positive_ranked_idx[np.argsort(contrib[positive_ranked_idx])[::-1]]
        fallback_ranked_idx = np.argsort(np.abs(contrib))[::-1]
        ranked_idx = np.concatenate([positive_ranked_idx, fallback_ranked_idx])

        raw_selected = []
        mapped_selected = []
        seen_raw = set()
        seen_mapped = set()

        for j in ranked_idx:
            try:
                raw_feature = str(feature_names[int(j)])
            except (IndexError, TypeError):
                continue
            if "[" in raw_feature or "]" in raw_feature or raw_feature == "other_model_signal":
                continue
            if raw_feature in seen_raw or is_sensitive_feature(raw_feature):
                continue
            mapped_reason = map_raw_feature_to_reason(raw_feature)
            if mapped_reason == "Other model risk signal":
                continue
            if mapped_reason not in seen_mapped:
                raw_selected.append(raw_feature)
                mapped_selected.append(mapped_reason)
                seen_raw.add(raw_feature)
                seen_mapped.add(mapped_reason)
            if len(mapped_selected) >= 3:
                break

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
            "factor_3": mapped_selected[2],
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
        "student_id", "risk_score", "raw_feature_1", "raw_feature_2", "raw_feature_3",
        "factor_1", "factor_2", "factor_3",
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
