# 🎓 Student Dropout Risk Prediction System

[![HackBricks](https://img.shields.io/badge/HackBricks-First%20Runner--Up-orange?style=for-the-badge)](https://www.databricks.com/)
[![Databricks](https://img.shields.io/badge/Databricks-Powered-red?style=for-the-badge\&logo=databricks)](https://www.databricks.com/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge\&logo=python)](https://www.python.org/)

> 🏆 **First Runner-Up at HackBricks** – A Databricks-focused Hackathon

An end-to-end ML solution built on Databricks for predicting student dropout risk and enabling proactive intervention strategies.

---

## 📋 Table of Contents

* Overview
* Architecture
* Key Features
* Technical Stack
* Project Structure
* Workflow Pipeline
* Setup & Installation
* Results & Outputs
* Future Enhancements
* Acknowledgments

---

## 🎯 Overview

Student dropout is a critical challenge in educational institutions. This solution leverages:

* Machine Learning
* Delta Lake
* MLflow
* Databricks

### What this system does:

* Predict at-risk students
* Identify contributing factors
* Recommend interventions
* Monitor fairness
* Automate alerts & dashboards

---

## 🏗️ Architecture

> ⚠️ Make sure images exist in `/docs` folder in your repo

![Architecture Diagram](./docs/architecture_diagram.png)

### Pipeline Flow

```
Student Dataset
    ↓
🥉 Bronze Layer (Raw Data)
    ↓
🥈 Silver Layer (Clean Views)
    ↓
🧠 Feature Table
    ↓
🤖 ML Models
    ↓
✅ Model Selection + Explainability
    ↓
🎯 Final Output (Risk + Action)
```

---

## ✨ Key Features

### 🚀 End-to-End ML Pipeline

* Automated ingestion
* Feature engineering
* MLflow tracking
* Model comparison

### 🔍 Explainable AI

* SHAP values per prediction
* Top 3 risk factors
* Interpretable scores

### ⚖️ Fairness & Ethics

* Bias detection
* Demographic audits
* Responsible AI reporting

### 📊 Actionable Insights

* Risk tiers (Low/Medium/High)
* Recommended actions
* Priority ranking

### 📡 Real-Time Monitoring

* Lakeview dashboards
* Email alerts
* Natural language queries (Genie)

### 🏭 Production Ready

* Delta Lake ACID
* Unity Catalog governance
* Scalable Spark pipelines

---

## 🛠️ Technical Stack

| Component      | Technology           |
| -------------- | -------------------- |
| Platform       | Databricks           |
| Storage        | Delta Lake           |
| Processing     | Apache Spark         |
| ML             | scikit-learn, MLflow |
| Explainability | SHAP                 |
| Visualization  | Lakeview, Plotly     |
| Orchestration  | Databricks Workflows |
| Language       | Python 3.9+          |

---

## 📁 Project Structure

```
HackBricks-databricks-usecase/
│
├── dropout-students-usecase-workflow/
│   ├── 00_shared_config.ipynb
│   ├── 01_bronze_ingestion.ipynb
│   ├── 02_silver_validation.ipynb
│   ├── 03_silver_tables.ipynb
│   ├── 04_feature_table.ipynb
│   ├── 05_train_model.ipynb
│   ├── 06_fairness_audit.ipynb
│   ├── 07_explanations.ipynb
│   ├── 08_gold_intervention_table.ipynb
│   ├── 09_analysis_queries.ipynb
│   ├── 10_email_alerts.ipynb
│   └── 99_full_pipeline_driver.ipynb
│
├── dashboards/
├── docs/
└── sample-scripts/
```

---

## 🔄 Workflow Pipeline

### 🥉 Bronze

* Raw ingestion into Delta tables

### 🥈 Silver

* Cleaned business views:

  * student_profile
  * financial_status
  * academic_performance

### 🧠 Feature Engineering

* Academic trends
* Financial indicators
* Engagement metrics

### 🤖 Model Training

* Logistic Regression
* Random Forest
* MLflow tracking

### 📊 Evaluation

* Fairness audit
* SHAP explanations

### 🥇 Gold Layer

* Final intervention table
* Risk scores + recommendations

### 📡 Monitoring

* Dashboards
* Email alerts

---

## 🚀 Setup & Installation

### Prerequisites

* Databricks workspace
* DBR 14.3+
* Python 3.9+

### Steps

1. Import repo into Databricks
2. Configure catalog in `00_shared_config.ipynb`
3. Upload dataset
4. Run:

```python
%run ./99_full_pipeline_driver
```

5. (Optional) Configure email alerts

---

## 📊 Results & Outputs

![Dashboard](./docs/dashboard_screenshot.png)

### Key Metrics

* 294 high-risk students
* 885 review cases
* 61% low-risk population

### Model Performance

| Model               | AUC  | F1   |
| ------------------- | ---- | ---- |
| Random Forest       | 0.89 | 0.84 |
| Logistic Regression | 0.85 | 0.79 |

---

## 🔮 Future Enhancements

* Real-time streaming
* Advanced ML models (XGBoost)
* Time-series predictions
* A/B testing
* Mobile dashboards
* LLM-based reports

---

## 🙏 Acknowledgments

* HackBricks organizers
* Databricks community
* Team members

---

## 📧 Contact

* GitHub: https://github.com/SaiPurushoth
* LinkedIn: https://www.linkedin.com/in/sai-purushoth-0642871b7/

---

<div align="center">

**Built with ❤️ on Databricks**

🏆 HackBricks First Runner-Up

</div>
