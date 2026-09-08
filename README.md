# Customer Churn Prediction & Retention Intelligence System

A full-stack, enterprise-grade Machine Learning web application developed using **Python Flask**, **Scikit-Learn**, **SQLite**, and modern responsive **Bootstrap 5 & Chart.js**. Designed for university ML capstone presentations, viva demonstrations, and production-oriented subscriber attrition analysis.

---

## 1. System Overview

Customer retention is significantly more cost-effective than acquisition in subscription and recurring-revenue businesses. **ChurnShield AI** is an end-to-end intelligence platform that:
1. Ingests customer demographic, service subscription, and billing data.
2. Applies a leak-free Scikit-Learn preprocessing pipeline (`ColumnTransformer`, `StandardScaler`, `OneHotEncoder`).
3. Computes churn probabilities via calibrated machine learning models (`Random Forest`, `Gradient Boosting`, `Logistic Regression`).
4. Maps probabilities into calibrated Risk Categories (**Low Risk <30%**, **Medium Risk 30–70%**, **High Risk ≥70%**) and a 0–100 Risk Score.
5. Surfaces transparent, non-causal observable risk factors associated with the prediction.
6. Recommends tailored retention plays (e.g. contract migration incentives, bundled tech support, autopay bill credits).
7. Enables interactive **What-If Scenario Simulation** to explore policy impacts before execution.
8. Supports **Batch CSV Vectorized Scoring** and prioritized **Retention Campaign Exports**.

---

## 2. Key Features

- **🏠 Interactive Analytics Dashboard:** Real KPI metric cards (Total Customers: 7,043, Baseline Churn Rate: 26.54%, Average Monthly Charge: $64.76) and responsive Chart.js visualizations.
- **👤 Multi-Section Individual Prediction:** Demographics, services, and billing inputs with **1-Click Demo Presets** (High-Risk, Medium-Risk, Low-Risk) for live viva presentations.
- **🔬 What-If Churn Simulator:** Side-by-side comparison of baseline customer profiles against proposed contract/pricing changes with real-time probability delta calculations.
- **📁 High-Throughput Batch Prediction:** Upload CSV files, validate schemas, preview results, and download complete scored CSV files.
- **🎯 Retention Campaign Queue:** Filter and prioritize accounts with churn probabilities ≥ 70%, complete with targeted business playbooks and 1-click CSV export.
- **📜 Prediction Audit History:** Full SQLite persistence of every prediction made through the UI or API, complete with timestamps and search/filter capabilities.
- **🤖 Transparent Model Governance:** Live model comparison scorecard, Confusion Matrix heatmap, ROC Curve, Feature Importance ranking, and plain-English metric explanations for examiners.
- **🌓 Dark & Light Mode:** Seamless theme switching with persistent client-side `localStorage`.

---

## 3. Technology Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.9+ (tested on Python 3.13) |
| **Web Framework** | Flask 3.0+ |
| **Machine Learning** | Scikit-Learn 1.3+, Joblib 1.3+ |
| **Data Processing** | Pandas 2.0+, NumPy 1.24+ |
| **Visualization** | Matplotlib 3.7+, Seaborn 0.12+, Chart.js 4.4+ |
| **Database** | SQLite 3 via Flask-SQLAlchemy 3.1+ |
| **Frontend** | HTML5, CSS3, JavaScript (ES6+), Bootstrap 5.3, Bootstrap Icons |

---

## 4. Project Structure

```
customer-churn-prediction/
│
├── app.py                     # Main application entry point (Starts Flask server)
├── requirements.txt           # Python dependency specifications
├── README.md                  # Comprehensive documentation and viva guide
├── .gitignore                 # Version control exclusions
├── start.bat                  # Automated 1-click Windows starter script
│
├── backend/
│   ├── __init__.py            # Flask application factory (create_app)
│   ├── routes.py              # Web page controllers & REST API endpoints
│   ├── prediction.py          # Single & batch inference service
│   ├── database.py            # SQLAlchemy models (PredictionHistory)
│   └── utils.py               # Dataset statistics, CSV exports & validations
│
├── ml/
│   ├── __init__.py
│   ├── train_model.py         # Standalone model training & comparison script
│   ├── evaluate_model.py      # Accuracy, Recall, Precision, ROC-AUC, CM metrics
│   ├── preprocessing.py       # Data cleaning & ColumnTransformer pipeline
│   ├── model_utils.py         # Model loader, risk categorizer, retention advice
│   └── models/
│       ├── churn_model.pkl    # Serialized winning model
│       ├── preprocessor.pkl   # Serialized fitted ColumnTransformer
│       └── model_metadata.json# Dynamic metrics, thresholds & comparison table
│
├── data/
│   └── churn.csv              # Authentic Telco Customer Churn dataset (7,043 rows)
│
├── instance/
│   └── churn.db               # SQLite database file
│
├── templates/
│   ├── base.html              # Core layout (sidebar, top navbar, theme toggle)
│   ├── index.html             # Landing page
│   ├── dashboard.html         # Main dashboard with KPI cards & Chart.js
│   ├── predict.html           # Multi-section prediction form with demo presets
│   ├── result.html            # Prediction risk card, gauge, factors & retention
│   ├── batch.html             # CSV batch upload, validation & batch analytics
│   ├── analytics.html         # Deep demographic & service cohort analytics
│   ├── model.html             # Model performance dashboard & viva metric guide
│   ├── simulator.html         # What-If churn scenario simulator
│   ├── history.html           # SQLite prediction history with search/filter
│   ├── retention.html         # High-risk retention campaign list & export
│   ├── about.html             # System architecture & technical documentation
│   ├── 404.html               # Custom 404 error page
│   └── 500.html               # Custom 500 error page
│
├── static/
│   ├── css/
│   │   └── style.css          # Custom SaaS dashboard styling & theme variables
│   ├── js/
│   │   ├── charts.js          # Chart.js color palette & theme helper
│   │   ├── dashboard.js       # Dynamic dashboard charts
│   │   ├── prediction.js      # Form calculation & demo preset loader
│   │   └── simulator.js       # Live simulation execution
│   └── images/
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       └── feature_importance.png
│
└── uploads/
    ├── .gitkeep
    └── sample_customers.csv   # Template CSV for testing batch predictions
```

---

## 5. Machine Learning Pipeline

### Data Preprocessing (`ml/preprocessing.py`)
- **Missing Value Handling:** Replaced whitespace values in `TotalCharges` (occurring on new accounts with `tenure == 0`) and imputed with `0.0`.
- **Numerical Pipeline:** `SimpleImputer(strategy='median')` followed by `StandardScaler()`.
- **Categorical Pipeline:** `SimpleImputer(strategy='most_frequent')` followed by `OneHotEncoder(drop='first', handle_unknown='ignore')`.
- **Leakage Prevention:** Transformers are fitted strictly on `X_train` and applied to `X_test` and production inference inputs.

### Models Evaluated (`ml/train_model.py`)
1. **Logistic Regression:** Linear probabilistic model with balanced class weighting.
2. **Random Forest Classifier (Selected):** Ensemble of 150 bagged decision trees with depth limiting and balanced weights.
3. **Gradient Boosting Classifier:** Sequential boosting minimizing binary deviance.

### Evaluation Metrics (Evaluated on Unseen 1,409 Test Rows)
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest (Selected)** | **75.73%** | **52.88%** | **78.61%** | **0.6323** | **0.8440** |
| Logistic Regression | 73.81% | 50.43% | 78.34% | 0.6136 | 0.8417 |
| Gradient Boosting | 80.20% | 65.99% | 52.41% | 0.5842 | 0.8436 |

> **Selection Rationale:** Random Forest delivers the highest ROC-AUC (0.8440) combined with an exceptional Recall of **78.61%**, ensuring that over 3 out of 4 churning customers are identified early.

---

## 6. Installation & Quick Start (VS Code / Windows)

### Prerequisites
- Python 3.9 or higher installed and added to your `PATH`.

### Option A: Automatic 1-Click Startup (Recommended)
Double-click `start.bat` or run in terminal:
```cmd
start.bat
```

### Option B: Manual Setup in VS Code Terminal

1. **Open the project in VS Code:**
   ```powershell
   cd C:\Users\SHRADDHA\.gemini\antigravity\scratch\customer-churn-prediction
   ```

2. **Create and activate a virtual environment:**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
   *(If script execution is restricted, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

3. **Install required dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

4. **(Optional) Retrain or evaluate models:**
   ```powershell
   python ml/train_model.py
   ```

5. **Start the Application:**
   ```powershell
   python app.py
   ```

6. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

---

## 7. REST API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` or `/api/health` | `GET` | Health check returning `{ "status": "healthy" }` |
| `/api/model-info` | `GET` | Serialized model metrics, confusion matrix & comparisons |
| `/api/dashboard-stats` | `GET` | Aggregated dataset analytics for Chart.js |
| `/api/predict` | `POST` | JSON endpoint accepting customer features and returning probability, risk score, factors, and recommendations |
| `/api/simulate` | `POST` | JSON endpoint comparing baseline vs modified scenario |
| `/api/history` | `GET` | JSON endpoint returning recent logged predictions |
| `/download/<filename>` | `GET` | Secure file download for batch results and retention lists |

### Sample JSON Prediction Request:
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customerID": "API-TEST-1",
    "gender": "Female",
    "SeniorCitizen": "0",
    "Partner": "No",
    "Dependents": "No",
    "tenure": 3,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 92.50,
    "TotalCharges": 277.50
  }'
```

---

## 8. University Viva & Presentation Defense Guide

### Q1: Why is Recall prioritized over Accuracy in Customer Churn Prediction?
**Answer:** The dataset exhibits class imbalance (~73.5% retain vs ~26.5% churn). A naive model predicting "No Churn" for everyone achieves 73.5% accuracy but is practically useless. In customer retention, a **False Negative** (missing a customer who will churn) means losing that customer's entire lifetime revenue. A **False Positive** (reaching out to a customer who might stay) only costs a discount or support call. Therefore, capturing the highest possible proportion of churners (high recall) is the primary commercial objective.

### Q2: How is data leakage prevented in this pipeline?
**Answer:** Data leakage is prevented by:
1. Executing stratified train-test split *before* fitting any preprocessing transformations.
2. Fitting the `ColumnTransformer` (imputation, scaling, and one-hot encoding) strictly on the 80% training partition (`X_train`).
3. Transforming `X_test` and live production inputs using only the learned training parameters.

### Q3: Why does the system report "factors associated with this prediction" instead of "causes"?
**Answer:** Supervised machine learning identifies statistical correlations and conditional probability distributions (`P(Churn | X)`). Correlation does not imply direct causation. High monthly charges and month-to-month contracts are observable indicators of churn risk, but modifying them in isolation without understanding customer satisfaction may not deterministically prevent churn.

---

## 9. Genuine Limitations & Future Improvements

### Limitations
1. **Data Stationarity Assumption:** The trained model assumes subscriber behavioral patterns remain stationary over time. Macroeconomic shifts or competitor pricing changes can cause data drift.
2. **Lack of Behavioral Telemetry:** The dataset does not include real-time app logins, service tickets, network latency events, or NPS satisfaction survey scores.

### Future Scope
1. **Automated CI/CD Retraining:** Scheduled cron drift monitoring and automatic retraining with shadow model verification.
2. **Direct CRM Integration:** Webhook triggers to push high-risk accounts into Salesforce or Zendesk queues for instant outreach.
3. **Customer Lifetime Value (CLV) Optimization:** Weighting churn probabilities by net subscriber lifetime value to optimize retention budget allocation.
