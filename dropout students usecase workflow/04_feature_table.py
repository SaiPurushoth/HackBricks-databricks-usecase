# Databricks notebook source
# MAGIC %run /Users/saitwins777@gmail.com/dropout/00_shared_config

# COMMAND ----------

# Databricks notebook source
# MAGIC %run ./00_shared_config

ensure_schemas()

print_header("4. FEATURE TABLE")

profile_df = spark.table(PROFILE_TABLE).drop("silver_created_ts")
demographic_df = spark.table(DEMOGRAPHIC_TABLE).drop("silver_created_ts")
academic_bg_df = spark.table(ACADEMIC_BG_TABLE).drop("silver_created_ts")
family_bg_df = spark.table(FAMILY_BG_TABLE).drop("silver_created_ts")
financial_df = spark.table(FINANCIAL_TABLE).drop("silver_created_ts")
academic_perf_df = spark.table(ACADEMIC_PERF_TABLE).drop("silver_created_ts")
context_df = spark.table(CONTEXT_TABLE).drop("silver_created_ts")

course_window = Window.partitionBy("course")

academic_perf_with_course_stats = (
    academic_perf_df
    .join(profile_df.select("student_id", "course"), "student_id")
    .join(academic_bg_df.select("student_id", "admission_grade"), "student_id")
    .withColumn(
        "course_rigor_score",
        F.avg((F.col("curricular_units_1st_sem_grade") + F.col("curricular_units_2nd_sem_grade")) / 2.0).over(course_window)
    )
    .withColumn("competitive_density", F.avg(F.col("admission_grade")).over(course_window))
    .withColumn("course_avg_grade", (F.col("curricular_units_1st_sem_grade") + F.col("curricular_units_2nd_sem_grade")) / 2.0)
)

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
        F.when(F.col("debtor") == 1, 1).otherwise(0) +
        F.when(F.col("tuition_fees_up_to_date") == 0, 1).otherwise(0) +
        F.when(F.col("scholarship_holder") == 0, 1).otherwise(0)
    )
    .withColumn(
        "financial_segment",
        F.when(F.col("financial_stress_index") >= 2, "high_financial_stress")
        .when(F.col("financial_stress_index") == 1, "moderate_financial_stress")
        .otherwise("low_financial_stress")
    )
    .withColumn(
        "engagement_risk_proxy",
        F.when(F.col("curricular_units_1st_sem_evaluations") == 0, 1).otherwise(0) +
        F.when(F.col("curricular_units_2nd_sem_evaluations") == 0, 1).otherwise(0) +
        F.when(F.col("curricular_units_1st_sem_without_evaluations") > 0, 1).otherwise(0) +
        F.when(F.col("curricular_units_2nd_sem_without_evaluations") > 0, 1).otherwise(0)
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
    .withColumn("absenteeism_index", F.col("curricular_units_1st_sem_without_evaluations") + F.col("curricular_units_2nd_sem_without_evaluations"))
    .withColumn("is_ghosting", F.when((F.col("curricular_units_2nd_sem_evaluations") == 0) & (F.col("curricular_units_2nd_sem_enrolled") > 0), 1).otherwise(0))
    .withColumn("total_approved_units", F.col("curricular_units_1st_sem_approved") + F.col("curricular_units_2nd_sem_approved"))
    .withColumn("is_primary_choice", F.when(F.col("application_order") == 1, 1).otherwise(0))
    .withColumn("admission_gap", F.col("admission_grade") - F.col("course_rigor_score"))
    .withColumn("grade_deflation_flag", F.when(F.col("course_rigor_score") < F.lit(university_25th_percentile), 1).otherwise(0))
    .withColumn("feature_ts", F.current_timestamp())
)

feature_df = feature_df.drop("course_avg_grade")
write_delta(feature_df, FEATURE_TABLE)

print("Feature table row count:", spark.table(FEATURE_TABLE).count())
display(spark.table(FEATURE_TABLE).limit(10))
