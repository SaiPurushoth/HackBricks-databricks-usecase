# Databricks notebook source
# MAGIC %run /Users/saitwins777@gmail.com/dropout/00_shared_config

# COMMAND ----------

# Databricks notebook source
# MAGIC %run ./00_shared_config

ensure_schemas()

print_header("5. MODEL TRAINING")

inputs = get_feature_training_inputs()
pdf = inputs["pdf"]
X = inputs["X"]
y = inputs["y"]
preprocessor = inputs["preprocessor"]
X_train = inputs["X_train"]
X_test = inputs["X_test"]
y_train = inputs["y_train"]
y_test = inputs["y_test"]
idx_test = inputs["idx_test"]


def log_and_train(model, model_name):
    with mlflow.start_run(run_name=model_name):
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        probs = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "f1": float(f1_score(y_test, preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probs)),
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
    "logistic_regression_baseline",
)

rf_pipeline, rf_metrics, rf_run_id = log_and_train(
    RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    ),
    "random_forest_baseline",
)

rf_is_best = rf_metrics["roc_auc"] >= lr_metrics["roc_auc"]
best_model = rf_pipeline if rf_is_best else lr_pipeline
best_model_name = "random_forest_baseline" if rf_is_best else "logistic_regression_baseline"
best_run_id = rf_run_id if rf_is_best else lr_run_id
registered_model_version = None

try:
    registered_model_version = register_run_model(best_run_id)
    print(
        f"Registered best model in Databricks Model Registry as "
        f"{REGISTERED_MODEL_NAME} version {registered_model_version}"
    )
except Exception as exc:
    print(f"Model registration skipped or failed: {exc}")

metrics_pdf = pd.DataFrame([
    {
        "model_name": "logistic_regression_baseline",
        "run_id": lr_run_id,
        "is_best_model": int(not rf_is_best),
        "registered_model_name": REGISTERED_MODEL_NAME if (not rf_is_best and registered_model_version is not None) else "",
        "registered_model_version": int(registered_model_version) if (not rf_is_best and registered_model_version is not None) else -1,
        **lr_metrics,
    },
    {
        "model_name": "random_forest_baseline",
        "run_id": rf_run_id,
        "is_best_model": int(rf_is_best),
        "registered_model_name": REGISTERED_MODEL_NAME if (rf_is_best and registered_model_version is not None) else "",
        "registered_model_version": int(registered_model_version) if (rf_is_best and registered_model_version is not None) else -1,
        **rf_metrics,
    },
])
metrics_pdf["training_ts"] = pd.Timestamp.utcnow()

metrics_schema = T.StructType([
    T.StructField("model_name", T.StringType(), True),
    T.StructField("run_id", T.StringType(), True),
    T.StructField("is_best_model", T.LongType(), True),
    T.StructField("registered_model_name", T.StringType(), True),
    T.StructField("registered_model_version", T.LongType(), True),
    T.StructField("accuracy", T.DoubleType(), True),
    T.StructField("precision", T.DoubleType(), True),
    T.StructField("recall", T.DoubleType(), True),
    T.StructField("f1", T.DoubleType(), True),
    T.StructField("roc_auc", T.DoubleType(), True),
    T.StructField("training_ts", T.TimestampType(), True),
])

spark.sql(f"DROP TABLE IF EXISTS {MODEL_METRICS_TABLE}")
metrics_sdf = spark.createDataFrame(metrics_pdf, schema=metrics_schema)
(
    metrics_sdf.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(MODEL_METRICS_TABLE)
)


scored = get_scored_test_dataframe(
    model_pipeline=best_model,
    X_test_df=X_test,
    y_test_series=y_test,
    full_pdf=pdf,
    idx_test=idx_test,
)

score_sdf = (
    spark.createDataFrame(scored[["student_id", "actual_dropout", "risk_score", "predicted_dropout"]])
    .withColumn("student_id", F.col("student_id").cast("bigint"))
    .withColumn("model_name", F.lit(best_model_name))
    .withColumn("score_ts", F.current_timestamp())
)

(
    score_sdf.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(MODEL_SCORE_TABLE)
)

print("Best model:", best_model_name)
print("Feature count:", X.shape[1])
display(
    spark.table(MODEL_METRICS_TABLE).select(
        "model_name",
        "run_id",
        "is_best_model",
        "registered_model_name",
        "registered_model_version",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "training_ts",
    )
)