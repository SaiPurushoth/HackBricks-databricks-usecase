# Databricks notebook source
# MAGIC %run /Users/saitwins777@gmail.com/dropout/00_shared_config
# MAGIC

# COMMAND ----------

# MAGIC %run /Users/saitwins777@gmail.com/dropout/02_silver_validation

# COMMAND ----------

# Databricks notebook source
# MAGIC %run ./00_shared_config

ensure_schemas()

print_header("3. SILVER TABLES")

silver_source = spark.table(VALIDATED_SOURCE_TABLE)

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
        F.when((F.col("debtor") == 1) & (F.col("tuition_fees_up_to_date") == 0), "high")
        .when(
            (F.col("debtor") == 1) |
            (F.col("tuition_fees_up_to_date") == 0) |
            (F.col("scholarship_holder") == 0),
            "medium"
        )
        .otherwise("low")
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
