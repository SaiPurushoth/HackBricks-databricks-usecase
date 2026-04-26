# Databricks notebook source
# MAGIC %run /Users/saitwins777@gmail.com/dropout/00_shared_config

# COMMAND ----------

# Databricks notebook source
# MAGIC %run ./00_shared_config

ensure_schemas()

print_header("10. COUNSELOR EMAIL ALERTS")

# Configure these in Databricks before running:
# 1. Create table dropout_silver.student_counselor_assignment with:
#    student_id BIGINT, counselor_name STRING, counselor_email STRING, active_flag INT
# 2. Create secret scope + keys for SMTP credentials.
# 3. Schedule this notebook as a Workflow task every 48 hours.



DEMO_COUNSELOR_NAME = "Vishali"
DEMO_COUNSELOR_EMAIL = "sakthivelvishali@gmail.com"

smtp_host = dbutils.secrets.get("dropout-alerts", "smtp-host")
smtp_port = dbutils.secrets.get("dropout-alerts", "smtp-port")
smtp_username = dbutils.secrets.get("dropout-alerts", "smtp-username")
smtp_password = dbutils.secrets.get("dropout-alerts", "smtp-password")
sender_email = dbutils.secrets.get("dropout-alerts", "smtp-sender")






smtp_host    = 'smtp.gmail.com'
smtp_port  = 587
smtp_username = 'saitwins777@gmail.com'
smtp_password = 'xxmi nzsl rtxy cyqq'
sender_email   = 'saitwins777@gmail.com'


test_subject = "Databricks SMTP Test"
test_body = "This is a test email from Databricks."

send_email_via_smtp(
    smtp_host=smtp_host,
    smtp_port=int(smtp_port),
    smtp_username=smtp_username,
    smtp_password=smtp_password,
    sender_email=sender_email,
    recipient_email="sakthivelvishali@gmail.com",
    subject=test_subject,
    body=test_body,
)

print("single test email attempt completed")


ALERT_LOOKBACK_HOURS = 48

spark.sql(
    f"""
    CREATE TABLE IF NOT EXISTS {EMAIL_ALERT_LOG_TABLE} (
        student_id BIGINT,
        counselor_email STRING,
        risk_score DOUBLE,
        recommended_action STRING,
        top_risk_factor_1 STRING,
        top_risk_factor_2 STRING,
        top_risk_factor_3 STRING,
        alert_subject STRING,
        alert_body STRING,
        sent_status STRING,
        sent_error STRING,
        sent_ts TIMESTAMP
    ) USING DELTA
    """
)

if not spark.catalog.tableExists(COUNSELOR_ASSIGNMENT_TABLE):
    print(
        f"Missing {COUNSELOR_ASSIGNMENT_TABLE}. Creating demo counselor assignment "
        f"for all students using {DEMO_COUNSELOR_NAME} <{DEMO_COUNSELOR_EMAIL}>."
    )
    demo_assignment_df = (
        spark.table(INTERVENTION_TABLE)
        .select(F.col("student_id").cast("bigint").alias("student_id"))
        .dropDuplicates()
        .withColumn("counselor_name", F.lit(DEMO_COUNSELOR_NAME))
        .withColumn("counselor_email", F.lit(DEMO_COUNSELOR_EMAIL))
        .withColumn("active_flag", F.lit(1))
    )
    (
        demo_assignment_df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(COUNSELOR_ASSIGNMENT_TABLE)
    )


missing_smtp = [
    name for name, value in [
        ("smtp-host", smtp_host),
        ("smtp-port", smtp_port),
        ("smtp-username", smtp_username),
        ("smtp-password", smtp_password),
        ("smtp-sender", sender_email),
    ] if value in [None, ""]
]
if missing_smtp:
    raise ValueError(f"Missing SMTP secrets in scope {SMTP_SECRET_SCOPE}: {missing_smtp}")

queue_df = (
    spark.table(INTERVENTION_TABLE)
    .filter(F.col("predicted_dropout") == 1)
    .filter(F.col("risk_score") >= 0.70)
)

counselor_df = (
    spark.table(COUNSELOR_ASSIGNMENT_TABLE)
    .filter(F.coalesce(F.col("active_flag"), F.lit(1)) == 1)
    .select("student_id", "counselor_name", "counselor_email")
)

recent_log_df = (
    spark.table(EMAIL_ALERT_LOG_TABLE)
    .filter(F.col("sent_status") == "sent")
    .filter(F.col("sent_ts") >= F.current_timestamp() - F.expr(f"INTERVAL {ALERT_LOOKBACK_HOURS} HOURS"))
    .select("student_id", "counselor_email")
    .dropDuplicates()
)

pending_alerts_df = (
    queue_df.alias("q")
    .join(counselor_df.alias("c"), on="student_id", how="inner")
    .join(recent_log_df.alias("l"), on=["student_id", "counselor_email"], how="left_anti")
    .select(
        "student_id",
        "counselor_name",
        "counselor_email",
        "risk_score",
        "recommended_action",
        F.col("top_risk_factor_1").alias("reason_1"),
        F.col("top_risk_factor_2").alias("reason_2"),
        F.col("top_risk_factor_3").alias("reason_3"),
        "queue_ts",
    )
)

pending_alerts_pdf = pending_alerts_df.toPandas()

if pending_alerts_pdf.empty:
    print("No pending counselor alerts to send.")
else:
    log_rows = []
    for row in pending_alerts_pdf.itertuples(index=False):
        subject = f"Student Risk Alert: Student {row.student_id} score {row.risk_score:.3f}"
        body = (
            f"Hello {row.counselor_name},\n\n"
            f"This is an automated student risk alert generated from the dropout monitoring workflow.\n\n"
            f"Student ID: {row.student_id}\n"
            f"Risk score: {row.risk_score:.3f}\n"
            f"Recommended action: {row.recommended_action}\n"
            f"Top reasons:\n"
            f"1. {row.reason_1}\n"
            f"2. {row.reason_2}\n"
            f"3. {row.reason_3}\n\n"
            f"Queue timestamp: {row.queue_ts}\n\n"
            f"Please review the student record and take the recommended next step.\n"
        )

        status = "sent"
        error_message = None
        try:
            send_email_via_smtp(
                smtp_host=smtp_host,
                smtp_port=int(smtp_port),
                smtp_username=smtp_username,
                smtp_password=smtp_password,
                sender_email=sender_email,
                recipient_email=row.counselor_email,
                subject=subject,
                body=body,
            )
        except Exception as exc:
            status = "failed"
            error_message = str(exc)[:2000]

        log_rows.append({
            "student_id": int(row.student_id),
            "counselor_email": row.counselor_email,
            "risk_score": float(row.risk_score),
            "recommended_action": row.recommended_action,
            "top_risk_factor_1": row.reason_1,
            "top_risk_factor_2": row.reason_2,
            "top_risk_factor_3": row.reason_3,
            "alert_subject": subject,
            "alert_body": body,
            "sent_status": status,
            "sent_error": error_message,
            "sent_ts": pd.Timestamp.utcnow(),
        })

    log_sdf = spark.createDataFrame(pd.DataFrame(log_rows))
    (
        log_sdf.write
        .format("delta")
        .mode("append")
        .saveAsTable(EMAIL_ALERT_LOG_TABLE)
    )

    print(f"Processed {len(log_rows)} counselor alerts.")
    display(spark.table(EMAIL_ALERT_LOG_TABLE).orderBy(F.desc("sent_ts")).limit(50))


# COMMAND ----------

