# Databricks notebook source
# MAGIC %run /Users/saitwins777@gmail.com/dropout/00_shared_config

# COMMAND ----------

# Databricks notebook source
# MAGIC %run ./00_shared_config

ensure_schemas()

print_header("2. SILVER SOURCE VALIDATION")

bronze = spark.table(BRONZE_TABLE)

required_columns = [
    "student_id", "marital_status", "application_mode", "application_order", "course",
    "daytime_evening_attendance", "previous_qualification", "previous_qualification_grade",
    "nacionality", "mothers_qualification", "fathers_qualification", "mothers_occupation",
    "fathers_occupation", "admission_grade", "displaced", "educational_special_needs",
    "gender", "age_at_enrollment", "international", "debtor", "tuition_fees_up_to_date",
    "scholarship_holder", "curricular_units_1st_sem_credited", "curricular_units_1st_sem_enrolled",
    "curricular_units_1st_sem_evaluations", "curricular_units_1st_sem_approved",
    "curricular_units_1st_sem_grade", "curricular_units_1st_sem_without_evaluations",
    "curricular_units_2nd_sem_credited", "curricular_units_2nd_sem_enrolled",
    "curricular_units_2nd_sem_evaluations", "curricular_units_2nd_sem_approved",
    "curricular_units_2nd_sem_grade", "curricular_units_2nd_sem_without_evaluations",
    "unemployment_rate", "inflation_rate", "gdp", "target"
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

write_delta(silver_source, VALIDATED_SOURCE_TABLE)

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
