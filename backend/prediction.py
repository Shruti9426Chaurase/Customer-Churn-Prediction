"""
Prediction service integrating Scikit-Learn pipeline with Flask business logic.
Supports single-customer prediction, batch CSV processing, and What-If simulation.
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure project root is in python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.preprocessing import clean_dataset, ALL_FEATURE_COLS
from ml.model_utils import (
    load_artifacts,
    classify_risk,
    explain_prediction_factors,
    generate_retention_recommendations,
    DEFAULT_THRESHOLDS
)


def predict_single_customer(customer_data: dict) -> dict:
    """
    Executes end-to-end prediction for a single customer.
    Returns probability, risk category, risk score, factors, and recommendations.
    """
    model, preprocessor, metadata = load_artifacts()
    if model is None or preprocessor is None:
        raise RuntimeError("ML model or preprocessor artifact not found. Please train model first.")

    thresholds = metadata.get('risk_thresholds', DEFAULT_THRESHOLDS) if metadata else DEFAULT_THRESHOLDS

    # Convert customer_data into a single-row DataFrame
    df_raw = pd.DataFrame([customer_data])

    # Ensure all required columns exist with defaults if missing
    for col in ALL_FEATURE_COLS:
        if col not in df_raw.columns:
            if col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
                df_raw[col] = 0.0
            elif col == 'SeniorCitizen':
                df_raw[col] = '0'
            else:
                df_raw[col] = 'No'

    df_clean = clean_dataset(df_raw)
    X_features = df_clean[ALL_FEATURE_COLS]

    # Transform through fitted ColumnTransformer
    X_proc = preprocessor.transform(X_features)

    # Inference
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X_proc)[0, 1])
    else:
        # Logistic / linear decision score
        dec = float(model.decision_function(X_proc)[0])
        prob = float(1.0 / (1.0 + np.exp(-dec)))

    prob = round(prob, 4)
    pred_label = "Churn" if prob >= 0.50 else "Retain"
    risk_score = int(round(prob * 100))

    risk_category, badge_class, color_hex = classify_risk(prob, thresholds)
    factors = explain_prediction_factors(customer_data, top_n=4)
    retention_recs = generate_retention_recommendations(customer_data, risk_category, prob)

    return {
        'customer_id': customer_data.get('customerID', 'CUST-DEMO'),
        'prediction': pred_label,
        'probability': prob,
        'probability_percent': round(prob * 100, 1),
        'risk_category': risk_category,
        'risk_score': risk_score,
        'badge_class': badge_class,
        'color_hex': color_hex,
        'factors': factors,
        'recommendations': retention_recs,
        'model_name': metadata.get('model_name', 'Random Forest') if metadata else 'Random Forest'
    }


def predict_batch_dataframe(df: pd.DataFrame) -> tuple:
    """
    Executes vectorized prediction over a batch DataFrame.
    Returns (annotated_df, summary_stats).
    """
    model, preprocessor, metadata = load_artifacts()
    if model is None or preprocessor is None:
        raise RuntimeError("ML model or preprocessor artifact not found.")

    thresholds = metadata.get('risk_thresholds', DEFAULT_THRESHOLDS) if metadata else DEFAULT_THRESHOLDS

    # Validate required columns
    missing_cols = [c for c in ALL_FEATURE_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {', '.join(missing_cols)}")

    df_clean = clean_dataset(df)
    X_features = df_clean[ALL_FEATURE_COLS]
    X_proc = preprocessor.transform(X_features)

    # Vectorized probabilities
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_proc)[:, 1]
    else:
        dec = model.decision_function(X_proc)
        probs = 1.0 / (1.0 + np.exp(-dec))

    probs = np.round(probs, 4)

    # Classifications
    low_th = thresholds.get('low', 0.30)
    med_th = thresholds.get('medium', 0.70)

    categories = []
    badges = []
    for p in probs:
        if p < low_th:
            categories.append("Low Risk")
            badges.append("success")
        elif p < med_th:
            categories.append("Medium Risk")
            badges.append("warning")
        else:
            categories.append("High Risk")
            badges.append("danger")

    predictions = ["Churn" if p >= 0.50 else "Retain" for p in probs]
    risk_scores = [int(round(p * 100)) for p in probs]

    # Annotate original DataFrame
    result_df = df.copy()
    result_df['prediction'] = predictions
    result_df['churn_probability'] = probs
    result_df['risk_category'] = categories
    result_df['risk_score'] = risk_scores

    total_customers = len(result_df)
    churn_count = sum(1 for pred in predictions if pred == "Churn")
    high_risk_count = sum(1 for cat in categories if cat == "High Risk")
    med_risk_count = sum(1 for cat in categories if cat == "Medium Risk")
    low_risk_count = sum(1 for cat in categories if cat == "Low Risk")
    avg_prob = round(float(np.mean(probs)) * 100, 1)

    summary = {
        'total_customers': total_customers,
        'churn_count': churn_count,
        'retain_count': total_customers - churn_count,
        'high_risk_count': high_risk_count,
        'medium_risk_count': med_risk_count,
        'low_risk_count': low_risk_count,
        'avg_churn_probability': avg_prob,
        'churn_rate': round((churn_count / total_customers) * 100, 1) if total_customers > 0 else 0
    }

    return result_df, summary
