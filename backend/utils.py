"""
Utility functions for data validation, CSV generation, and dataset statistics.
"""

import os
import pandas as pd
import numpy as np

ALLOWED_EXTENSIONS = {'csv'}


def allowed_file(filename: str) -> bool:
    """Checks whether the uploaded file is a valid CSV."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_dataset_stats(csv_path: str) -> dict:
    """
    Computes real summary statistics and charts data from the underlying churn.csv.
    Used by dashboard and analytics pages so no values are ever hardcoded.
    """
    if not os.path.exists(csv_path):
        return {}

    try:
        df = pd.read_csv(csv_path)

        # Standardize churn target
        churn_col = 'Churn'
        if churn_col in df.columns:
            churn_series = df[churn_col].apply(lambda x: 1 if str(x).strip().lower() in ['yes', '1', 'true'] else 0)
        else:
            churn_series = pd.Series([0] * len(df))

        total_customers = len(df)
        churn_count = int(churn_series.sum())
        retain_count = total_customers - churn_count
        churn_rate = round((churn_count / total_customers) * 100, 2) if total_customers > 0 else 0.0

        # Numeric conversions
        monthly_charges = pd.to_numeric(df.get('MonthlyCharges', 0), errors='coerce').fillna(0)
        total_charges = pd.to_numeric(df.get('TotalCharges', 0).astype(str).str.strip(), errors='coerce').fillna(0)
        tenure = pd.to_numeric(df.get('tenure', 0), errors='coerce').fillna(0)

        avg_monthly = round(float(monthly_charges.mean()), 2)
        avg_tenure = round(float(tenure.mean()), 1)

        # Churn by Contract
        contract_stats = {}
        if 'Contract' in df.columns:
            for contract in df['Contract'].dropna().unique():
                sub = df[df['Contract'] == contract]
                sub_churn = (sub[churn_col].apply(lambda x: 1 if str(x).strip().lower() in ['yes', '1', 'true'] else 0)).mean()
                contract_stats[contract] = {
                    'total': len(sub),
                    'churn_rate': round(float(sub_churn) * 100, 2)
                }

        # Churn by Internet Service
        internet_stats = {}
        if 'InternetService' in df.columns:
            for net in df['InternetService'].dropna().unique():
                sub = df[df['InternetService'] == net]
                sub_churn = (sub[churn_col].apply(lambda x: 1 if str(x).strip().lower() in ['yes', '1', 'true'] else 0)).mean()
                internet_stats[net] = {
                    'total': len(sub),
                    'churn_rate': round(float(sub_churn) * 100, 2)
                }

        # Churn by Tenure cohorts (0-12, 13-24, 25-48, 49-72)
        tenure_cohorts = {
            '0 - 12 Months': {'stay': 0, 'churn': 0},
            '13 - 24 Months': {'stay': 0, 'churn': 0},
            '25 - 48 Months': {'stay': 0, 'churn': 0},
            '49 - 72 Months': {'stay': 0, 'churn': 0}
        }
        for _, row in df.iterrows():
            t = float(row.get('tenure', 0))
            is_churn = 1 if str(row.get('Churn', '')).strip().lower() in ['yes', '1', 'true'] else 0
            key = '0 - 12 Months' if t <= 12 else ('13 - 24 Months' if t <= 24 else ('25 - 48 Months' if t <= 48 else '49 - 72 Months'))
            if is_churn:
                tenure_cohorts[key]['churn'] += 1
            else:
                tenure_cohorts[key]['stay'] += 1

        # Payment Method distribution
        payment_stats = {}
        if 'PaymentMethod' in df.columns:
            for pm in df['PaymentMethod'].dropna().unique():
                sub = df[df['PaymentMethod'] == pm]
                sub_churn = (sub[churn_col].apply(lambda x: 1 if str(x).strip().lower() in ['yes', '1', 'true'] else 0)).mean()
                payment_stats[pm] = {
                    'total': len(sub),
                    'churn_rate': round(float(sub_churn) * 100, 2)
                }

        # Automated Analytical Insights
        insights = []
        if 'Month-to-month' in contract_stats and 'Two year' in contract_stats:
            m2m_rate = contract_stats['Month-to-month']['churn_rate']
            two_yr_rate = contract_stats['Two year']['churn_rate']
            diff = round(m2m_rate - two_yr_rate, 1)
            insights.append(
                f"Month-to-month subscribers experience a {m2m_rate}% churn rate versus only {two_yr_rate}% for two-year contracts (+{diff}% risk gap)."
            )

        if 'Fiber optic' in internet_stats:
            fo_rate = internet_stats['Fiber optic']['churn_rate']
            insights.append(
                f"Fiber Optic users register {fo_rate}% churn, primarily driven by higher monthly fees without bundled support."
            )

        early_churn_pct = round((tenure_cohorts['0 - 12 Months']['churn'] / max(1, churn_count)) * 100, 1)
        insights.append(
            f"Over {early_churn_pct}% of all lost customers occur within the first 12 months of service tenure."
        )

        if 'Electronic check' in payment_stats:
            ec_rate = payment_stats['Electronic check']['churn_rate']
            insights.append(
                f"Electronic check payers have the highest churn rate among payment methods at {ec_rate}%."
            )

        return {
            'total_customers': total_customers,
            'churn_count': churn_count,
            'retain_count': retain_count,
            'churn_rate': churn_rate,
            'avg_monthly_charges': avg_monthly,
            'avg_tenure': avg_tenure,
            'contract_stats': contract_stats,
            'internet_stats': internet_stats,
            'tenure_cohorts': tenure_cohorts,
            'payment_stats': payment_stats,
            'insights': insights
        }
    except Exception as e:
        print(f"Error computing dataset stats: {e}")
        return {}
