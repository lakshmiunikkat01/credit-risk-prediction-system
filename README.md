#  Credit Risk Prediction System (FinTech ML Project)

##  Overview

This project builds an end-to-end Credit Risk Prediction System using machine learning techniques.

The goal is to predict whether a loan applicant is likely to default, enabling financial institutions to make informed lending decisions.

This project simulates how fintech companies perform risk scoring and loan approval evaluation.

---

##  Business Problem

Lending institutions face two major risks:

- Approving high-risk customers → Financial Loss
- Rejecting low-risk customers → Lost Revenue

The objective is to build a predictive model that:
- Identifies high-risk applicants
- Minimizes false approvals
- Supports risk-based decision making
- Generates interpretable risk scores

---

##  Dataset

Loan Approval Dataset

Features include:
- Gender
- Marital Status
- Dependents
- Education
- Self Employment
- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Term
- Credit History
- Property Area

Target:
- Loan_Status (0 = Rejected / Default Risk, 1 = Approved)

Total records: 614

---

##  Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib / Seaborn
- SQLite (optional logging)
- Streamlit (Web App Deployment)

---

##  Project Structure
credit_risk_system/
│
├── data/
│ └── raw.csv
│
├── models/
│ └── credit_model.pkl
│
├── src/
│ ├── preprocess.py
│ ├── train.py
│ ├── risk.py
│
├── app.py (Streamlit Web App)
├── main.py
├── requirements.txt

---

## 🔍 Methodology

### 1️ Data Cleaning
- Handling missing values (median/mode strategy)
- Encoding categorical variables
- Feature preprocessing

### 2️ Exploratory Data Analysis
- Default rate by income
- Default rate by employment
- Correlation heatmap
- Credit history impact analysis

### 3️ Model Training
Compared multiple models:

- Logistic Regression
- Random Forest
- XGBoost

Evaluation Metric:
- ROC-AUC Score
- Precision
- Recall
- Confusion Matrix

---

##  Model Performance Example

| Model | ROC-AUC |
|--------|----------|
| Logistic Regression | 0.74–0.76 |
| Random Forest | ~0.75 |
| XGBoost | ~0.73 |

Logistic Regression provided strong interpretability.
Random Forest offered better non-linear modeling.

---

##  Risk Scoring System

Predicted probability is converted into Risk Bands:

- Low Risk → Probability < 0.35
- Medium Risk → 0.35 – 0.65
- High Risk → > 0.65

This makes model output business-friendly and actionable.

Example Output:

Risk Score: 0.41  
Risk Category: Medium Credit Risk

---

##  Business-Oriented Evaluation

Focus was placed on:

- False Positive Rate (approving risky customers)
- Precision for defaulters
- Cost-sensitive decision evaluation

This simulates real-world lending risk assessment used in fintech.

---

##  Streamlit Deployment

Interactive web app allows:

- Manual input of applicant details
- Real-time risk prediction
- Risk category display
- Score interpretation



