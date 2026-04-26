# 🎓 Student Dropout Risk Prediction System

[![HackBricks](https://img.shields.io/badge/HackBricks-First%20Runner--Up-orange?style=for-the-badge)](https://www.databricks.com/)
[![Databricks](https://img.shields.io/badge/Databricks-Powered-red?style=for-the-badge&logo=databricks)](https://www.databricks.com/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)](https://www.python.org/)

> **🏆 First Runner-Up** at HackBricks - A Databricks-focused Hackathon

An end-to-end ML solution built on Databricks for predicting student dropout risk and enabling proactive intervention strategies to improve retention rates.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Technical Stack](#technical-stack)
- [Project Structure](#project-structure)
- [Workflow Pipeline](#workflow-pipeline)
- [Setup & Installation](#setup--installation)
- [Results & Outputs](#results--outputs)
- [Future Enhancements](#future-enhancements)
- [Acknowledgments](#acknowledgments)

---

## 🎯 Overview

Student dropout is a critical challenge in educational institutions. This solution leverages **machine learning**, **Delta Lake**, and **MLflow** on the Databricks platform to:

- **Predict** students at risk of dropping out
- **Identify** key factors contributing to dropout risk
- **Recommend** targeted interventions
- **Monitor** fairness and model performance
- **Automate** alerts and reporting

The system processes student demographic, academic, and financial data through a **medallion architecture** (Bronze → Silver → Gold), trains ML models, generates explainable predictions, and delivers actionable insights via dashboards and automated alerts.

---

## 🏗️ Architecture

The solution follows Databricks best practices with a medallion architecture:

![Architecture Diagram](docs/architecture_diagram.png)

### Pipeline Flow:

```
Student Dataset
    ↓
🥉 Bronze Layer (Raw Data Storage)
    ↓
🥈 Silver Layer (Business-Friendly Views)
    ├─ Student Profile
    ├─ Financial Status
    ├─ Academic Performance
    └─ Student Context
    ↓
🧠 Feature Table (Risk Signals)
    ↓
🤖 ML Model Training & Comparison
    ├─ Logistic Regression
    └─ Random Forest
    ↓
✅ Best Model Selection
    ├─ Fairness Audit
    ├─ SHAP Explanations
    └─ Risk Scores
    ↓
🎯 Final Output
    ├─ Priority Rank
    ├─ Risk Score
    ├─ Top 3 Reasons
    └─ Recommended Action
```

---

## ✨ Key Features

### 1. **End-to-End ML Pipeline**
- Automated data ingestion and validation
- Feature engineering with Delta Lake
- Model training with MLflow experiment tracking
- Model comparison and selection

### 2. **Explainable AI**
- SHAP (SHapley Additive exPlanations) values for each prediction
- Top 3 risk factors identified per student
- Interpretable risk scores (0-1 scale)

### 3. **Fairness & Ethics**
- Automated fairness audits across demographic groups
- Bias detection in model predictions
- Responsible AI reporting

### 4. **Actionable Insights**
- Intervention tier assignment (Low/Medium/High)
- Specific action recommendations (counseling, tutoring, financial aid)
- Priority ranking for resource allocation

### 5. **Real-Time Monitoring**
- Interactive Lakeview dashboards
- Automated email alerts for high-risk students
- Genie AI-powered natural language queries

### 6. **Production-Ready**
- Delta Lake for ACID transactions
- Unity Catalog for governance
- Scalable Spark processing
- Version-controlled assets

---

## 🛠️ Technical Stack

| Component | Technology |
|-----------|-----------|
| **Platform** | Databricks (Unity Catalog enabled) |
| **Data Storage** | Delta Lake (Medallion Architecture) |
| **Processing** | Apache Spark (PySpark) |
| **ML Framework** | scikit-learn, MLflow |
| **Explainability** | SHAP (SHapley Additive exPlanations) |
| **Visualization** | Lakeview Dashboards, Plotly |
| **AI Analytics** | Genie Data Rooms |
| **Orchestration** | Databricks Workflows |
| **Alerting** | SMTP Email Integration |
| **Language** | Python 3.9+ |

---

## 📁 Project Structure

```
HackBricks-databricks-usecase/
│
├── dropout students usecase workflow/
│   ├── 00_shared_config.ipynb              # Configuration and parameters
│   ├── 00_pipeline_orchestrator.ipynb      # Pipeline setup
│   ├── 01_bronze_ingestion.ipynb           # Raw data ingestion
│   ├── 02_silver_validation.ipynb          # Data quality checks
│   ├── 03_silver_tables.ipynb              # Business views creation
│   ├── 04_feature_table.ipynb              # Feature engineering
│   ├── 05_train_model.ipynb                # ML model training
│   ├── 06_fairness_audit.ipynb             # Bias detection
│   ├── 07_explanations.ipynb               # SHAP analysis
│   ├── 08_gold_intervention_table.ipynb    # Final output table
│   ├── 09_analysis_queries.ipynb           # Analytics queries
│   ├── 10_email_alerts.ipynb               # Automated alerts
│   └── 99_full_pipeline_driver.ipynb       # End-to-end runner
│
├── dashboards/
│   ├── Student Retention Dashboard         # Lakeview dashboard
│   ├── Student Dropout Risk Management     # Genie space
│   ├── student_retention_dashboard_export.json
│   └── student_dropout_risk_management_export.json
│
└── sample scripts/
    ├── Hackbricks212.ipynb                 # Demo scripts
    └── bronze_injestion.ipynb              # Sample ingestion
```

---

## 🔄 Workflow Pipeline

### Stage 1: Bronze Layer (Data Ingestion)
- **Purpose**: Ingest raw student data
- **Output**: `workspace.dropout_bronze.raw_students`
- **Process**: Load CSV data into Delta tables with minimal transformation

### Stage 2: Silver Layer (Data Transformation)
- **Purpose**: Create clean, business-friendly views
- **Outputs**:
  - `student_profile`: Demographics and enrollment info
  - `financial_status`: Tuition, scholarships, financial stress indicators
  - `academic_performance`: Semester-wise grades and approvals
  - `student_context`: External factors and ground truth labels
- **Process**: Data cleaning, validation, and domain modeling

### Stage 3: Feature Engineering
- **Purpose**: Generate ML-ready features
- **Output**: `workspace.dropout_features.student_features`
- **Features**:
  - Academic trends (grade progression, approval rates)
  - Financial indicators (debt ratio, scholarship coverage)
  - Engagement metrics (course load changes)
  - Risk signals (consecutive failures, low attendance)

### Stage 4: Model Training
- **Purpose**: Train and compare ML models
- **Models**: Logistic Regression, Random Forest
- **Tracking**: MLflow experiments with metrics (AUC, F1, Precision, Recall)
- **Output**: Best model registered in MLflow Model Registry

### Stage 5: Model Evaluation
- **Fairness Audit**: Check for demographic bias
- **SHAP Explanations**: Generate feature importance per prediction
- **Output**: `workspace.dropout_gold.ml_explanations`

### Stage 6: Gold Layer (Intervention Queue)
- **Purpose**: Generate actionable insights
- **Output**: `workspace.dropout_gold.student_intervention_queue`
- **Columns**:
  - `student_id`, `risk_score`, `intervention_tier`
  - `top_risk_factors` (Top 3 reasons)
  - `recommended_action`
  - `priority_rank`

### Stage 7: Monitoring & Alerting
- **Dashboards**: Real-time visualization of at-risk students
- **Email Alerts**: Automated notifications for high-risk cases
- **Genie**: Natural language queries on intervention data

---

## 🚀 Setup & Installation

### Prerequisites
- Databricks workspace (Unity Catalog enabled)
- Cluster with DBR 14.3 LTS or higher
- Python 3.9+
- Required libraries: `scikit-learn`, `mlflow`, `shap`, `plotly`

### Installation Steps

1. **Clone or Import Repository**
   ```bash
   # Import into Databricks workspace
   # Use Repos or upload notebooks manually
   ```

2. **Configure Unity Catalog**
   ```python
   # In 00_shared_config.ipynb
   CATALOG = "workspace"
   BRONZE_SCHEMA = "dropout_bronze"
   SILVER_SCHEMA = "dropout_silver"
   FEATURES_SCHEMA = "dropout_features"
   GOLD_SCHEMA = "dropout_gold"
   ```

3. **Upload Student Data**
   ```python
   # Place CSV in DBFS or external storage
   # Update path in 01_bronze_ingestion.ipynb
   ```

4. **Run Pipeline**
   ```python
   # Option 1: Run full pipeline
   %run ./99_full_pipeline_driver

   # Option 2: Run individual notebooks in sequence
   ```

5. **Configure Email Alerts** (Optional)
   ```python
   # In 10_email_alerts.ipynb
   SMTP_SERVER = "smtp.gmail.com"
   SMTP_PORT = 587
   FROM_EMAIL = "your-email@gmail.com"
   TO_EMAIL = "recipient@example.com"
   ```

6. **Access Dashboards**
   - Open `Student Retention Dashboard` in `/dashboards`
   - Query via `Student Dropout Risk Management` Genie space

---

## 📊 Results & Outputs

### Dashboard Insights

![Student Retention Dashboard](docs/dashboard_screenshot.png)

**Key Metrics:**
- **294** students needing immediate support
- **885** cases flagged for manual review
- **61.38%** of students in low-risk category
- **25.81%** in high-risk requiring intervention

**Intervention Breakdown:**
- Academic tutoring: **458** students
- Financial counseling: **180** students
- Success coaching: **120** students
- Mental health support: **95** students

### Automated Alert System

![Email Alert Example](docs/email_alert_screenshot.png)

**Alert Details:**
- Student ID: 4216
- Risk Score: 0.906 (High Priority)
- Top Reasons:
  1. Low first-semester approvals
  2. Low total approved units
  3. High financial stress
- Recommended Action: High-priority case review

### Model Performance

| Model | AUC | F1 Score | Precision | Recall |
|-------|-----|----------|-----------|--------|
| Random Forest | 0.89 | 0.84 | 0.82 | 0.87 |
| Logistic Regression | 0.85 | 0.79 | 0.81 | 0.78 |

**Selected Model**: Random Forest (better recall for at-risk detection)

### Data Pipeline Status

![Pipeline Stages](docs/pipeline_screenshot.png)

All 9 pipeline stages completed successfully:
✅ Bronze Ingestion → Silver Tables → Feature Table → Model Training → Fairness Audit → SHAP Explanations → Gold Intervention Table → Analysis Queries

---

## 🔮 Future Enhancements

- [ ] **Real-time Streaming**: Ingest live data from student information systems
- [ ] **Advanced Models**: Experiment with XGBoost, LightGBM, Neural Networks
- [ ] **Time-Series Analysis**: Predict dropout risk progression over semesters
- [ ] **A/B Testing**: Measure intervention effectiveness
- [ ] **Mobile Dashboard**: Native app for counselors
- [ ] **Natural Language Reports**: Auto-generate intervention summaries using LLMs
- [ ] **Integration**: Connect with CRM systems for case management
- [ ] **Multi-tenancy**: Scale to multiple institutions

---

## 🙏 Acknowledgments

- **HackBricks Organizers**: For hosting an amazing Databricks-focused hackathon
- **Databricks Community**: For excellent documentation and support
- **Team Members**: [Add your team members here]
- **Dataset**: Synthetic student data generated for demonstration purposes

---

## 📜 License

This project was developed for the HackBricks Hackathon. Please contact the maintainers for usage permissions.

---

## 📧 Contact

**Project Maintainer**: gsaipurushoth7@gmail.com

**GitHub**: https://github.com/SaiPurushoth

**LinkedIn**: https://www.linkedin.com/in/sai-purushoth-0642871b7/

---

<div align="center">

**Built with ❤️ on Databricks**

🏆 **First Runner-Up - HackBricks Hackathon**

</div>
