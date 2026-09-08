"""
Data preprocessing pipeline for Customer Churn Prediction.
Handles data cleaning, imputation, encoding, and scaling with zero data leakage.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Define canonical features
NUMERICAL_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']

CATEGORICAL_COLS = [
    'gender',
    'SeniorCitizen',
    'Partner',
    'Dependents',
    'PhoneService',
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaperlessBilling',
    'PaymentMethod'
]

ALL_FEATURE_COLS = NUMERICAL_COLS + CATEGORICAL_COLS


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw customer data.
    - Handles spaces and missing values in TotalCharges
    - Normalizes SeniorCitizen
    - Encodes target Churn if present
    """
    df = df.copy()

    # Drop duplicate records if any
    df = df.drop_duplicates()

    # Clean TotalCharges: replace spaces with NaN, convert to float, impute with 0.0 or median
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce')
        # Where tenure is 0, TotalCharges is typically 0.0
        df['TotalCharges'] = df['TotalCharges'].fillna(0.0).astype(float)

    # Clean tenure and MonthlyCharges
    if 'tenure' in df.columns:
        df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce').fillna(0).astype(float)
    if 'MonthlyCharges' in df.columns:
        df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce').fillna(0.0).astype(float)

    # Normalize SeniorCitizen (convert 0/1 or Yes/No to string)
    if 'SeniorCitizen' in df.columns:
        df['SeniorCitizen'] = df['SeniorCitizen'].apply(
            lambda x: '1' if str(x).strip().lower() in ['1', 'yes', 'true'] else '0'
        )

    # Clean categorical string columns
    for col in CATEGORICAL_COLS:
        if col in df.columns and col != 'SeniorCitizen':
            df[col] = df[col].astype(str).str.strip()

    # Encode target if present
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].apply(
            lambda x: 1 if str(x).strip().lower() in ['yes', '1', 'true'] else 0
        ).astype(int)

    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Builds a ColumnTransformer with imputation, scaling, and one-hot encoding.
    """
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, NUMERICAL_COLS),
            ('cat', cat_pipeline, CATEGORICAL_COLS)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )

    return preprocessor


def get_feature_names(fitted_preprocessor: ColumnTransformer) -> list:
    """
    Retrieves human-readable output feature names from the fitted preprocessor.
    """
    try:
        return list(fitted_preprocessor.get_feature_names_out())
    except Exception:
        # Fallback calculation
        names = list(NUMERICAL_COLS)
        try:
            cat_encoder = fitted_preprocessor.named_transformers_['cat'].named_steps['encoder']
            cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_COLS)
            names.extend(list(cat_names))
        except Exception:
            pass
        return names


def prepare_training_data(csv_path: str, test_size: float = 0.2, random_state: int = 42):
    """
    Loads dataset from CSV, cleans it, performs stratified train-test split,
    fits the preprocessor on training data, and transforms both splits.
    """
    raw_df = pd.read_csv(csv_path)
    cleaned_df = clean_dataset(raw_df)

    if 'Churn' not in cleaned_df.columns:
        raise ValueError("Target column 'Churn' not found in dataset.")

    X = cleaned_df[ALL_FEATURE_COLS]
    y = cleaned_df['Churn']

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Fit preprocessor on X_train only (prevents data leakage)
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    feature_names = get_feature_names(preprocessor)

    return {
        'X_train_raw': X_train,
        'X_test_raw': X_test,
        'X_train_proc': X_train_proc,
        'X_test_proc': X_test_proc,
        'y_train': y_train,
        'y_test': y_test,
        'preprocessor': preprocessor,
        'feature_names': feature_names,
        'total_rows': len(cleaned_df),
        'churn_rate': float(y.mean())
    }
