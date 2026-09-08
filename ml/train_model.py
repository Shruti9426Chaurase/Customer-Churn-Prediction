"""
Model training pipeline for Customer Churn Prediction.
Trains and compares Logistic Regression, Random Forest, and Gradient Boosting.
Selects and serializes the best performing model.
"""

import os
import sys
import json
import datetime
import urllib.request
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Ensure ml package is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.preprocessing import prepare_training_data, clean_dataset
from ml.evaluate_model import evaluate_model


def ensure_dataset(data_path: str):
    """Ensures churn.csv exists, downloads authentic Telco dataset if missing."""
    if not os.path.exists(data_path):
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        url = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'
        print(f"Dataset not found locally. Downloading official Telco Churn dataset from {url}...")
        try:
            urllib.request.urlretrieve(url, data_path)
            print(f"Dataset successfully downloaded to {data_path}")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            raise


def plot_and_save_charts(best_model_name, best_eval, all_evals, images_dir):
    """Saves static evaluation plots (Confusion Matrix, ROC Curve, Feature Importance)."""
    os.makedirs(images_dir, exist_ok=True)

    # 1. Confusion Matrix
    cm_data = best_eval['confusion_matrix']['matrix']
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_data,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Stay (0)', 'Churn (1)'],
        yticklabels=['Stay (0)', 'Churn (1)']
    )
    plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('Actual Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()

    # 2. ROC Curve
    plt.figure(figsize=(7, 5.5))
    for name, ev in all_evals.items():
        roc_pts = ev['roc_curve']
        fpr = [p['fpr'] for p in roc_pts]
        tpr = [p['tpr'] for p in roc_pts]
        auc = ev['roc_auc']
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Chance (AUC = 0.50)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold', pad=12)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'roc_curve.png'), dpi=150)
    plt.close()

    # 3. Feature Importance
    if best_eval.get('feature_importance'):
        top_features = best_eval['feature_importance'][:10]
        feats = [f['feature'] for f in top_features][::-1]
        weights = [f['importance'] for f in top_features][::-1]

        plt.figure(figsize=(8, 5))
        plt.barh(feats, weights, color='#3b82f6', edgecolor='#1d4ed8')
        plt.title(f'Top Feature Importances ({best_model_name})', fontsize=14, fontweight='bold', pad=12)
        plt.xlabel('Relative Importance / Weight', fontsize=12)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, 'feature_importance.png'), dpi=150)
        plt.close()


def train_models():
    """Main training routine."""
    print("=" * 60)
    print("CUSTOMER CHURN PREDICTION SYSTEM - MODEL TRAINING PIPELINE")
    print("=" * 60)

    data_path = os.path.join(PROJECT_ROOT, 'data', 'churn.csv')
    models_dir = os.path.join(CURRENT_DIR, 'models')
    images_dir = os.path.join(PROJECT_ROOT, 'static', 'images')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    ensure_dataset(data_path)

    print("Step 1: Preparing and preprocessing dataset...")
    prep = prepare_training_data(data_path, test_size=0.2, random_state=42)
    X_train_proc = prep['X_train_proc']
    X_test_proc = prep['X_test_proc']
    y_train = prep['y_train']
    y_test = prep['y_test']
    preprocessor = prep['preprocessor']
    feature_names = prep['feature_names']

    print(f"  Dataset loaded: {prep['total_rows']} rows")
    print(f"  Training samples: {X_train_proc.shape[0]}, Test samples: {X_test_proc.shape[0]}")
    print(f"  Total processed features: {len(feature_names)}")
    print(f"  Baseline churn rate: {prep['churn_rate'] * 100:.2f}%\n")

    # Define candidate models
    candidate_models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_split=8,
            min_samples_leaf=4,
            class_weight='balanced',
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=4,
            subsample=0.85,
            random_state=42
        )
    }

    all_evaluations = {}
    best_model_name = None
    best_roc_auc = -1.0
    best_model_instance = None

    print("Step 2: Training and evaluating candidate models...")
    print("-" * 60)
    print(f"{'Model':<22} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<7} | {'F1':<6} | {'ROC-AUC':<7}")
    print("-" * 60)

    for name, model in candidate_models.items():
        model.fit(X_train_proc, y_train)
        ev = evaluate_model(model, X_test_proc, y_test, feature_names=feature_names)
        all_evaluations[name] = ev

        print(f"{name:<22} | {ev['accuracy']:<8.4f} | {ev['precision']:<9.4f} | {ev['recall']:<7.4f} | {ev['f1_score']:<6.4f} | {ev['roc_auc']:<7.4f}")

        # Choose best model (prioritizing ROC-AUC, with good recall balance)
        if ev['roc_auc'] > best_roc_auc:
            best_roc_auc = ev['roc_auc']
            best_model_name = name
            best_model_instance = model

    print("-" * 60)
    print(f"\nWinning Model Selected: {best_model_name} (ROC-AUC = {best_roc_auc:.4f})")

    # Step 3: Serialize winning model and preprocessor
    print("\nStep 3: Serializing model artifacts...")
    model_save_path = os.path.join(models_dir, 'churn_model.pkl')
    preprocessor_save_path = os.path.join(models_dir, 'preprocessor.pkl')
    metadata_save_path = os.path.join(models_dir, 'model_metadata.json')

    joblib.dump(best_model_instance, model_save_path)
    joblib.dump(preprocessor, preprocessor_save_path)

    best_eval = all_evaluations[best_model_name]

    metadata = {
        'model_name': best_model_name,
        'accuracy': best_eval['accuracy'],
        'precision': best_eval['precision'],
        'recall': best_eval['recall'],
        'f1_score': best_eval['f1_score'],
        'roc_auc': best_eval['roc_auc'],
        'training_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_dataset_rows': prep['total_rows'],
        'features_count': len(feature_names),
        'features': feature_names,
        'risk_thresholds': {
            'low': 0.30,
            'medium': 0.70
        },
        'confusion_matrix': best_eval['confusion_matrix'],
        'roc_curve': best_eval['roc_curve'],
        'feature_importance': best_eval['feature_importance'],
        'model_comparisons': {
            name: {
                'accuracy': ev['accuracy'],
                'precision': ev['precision'],
                'recall': ev['recall'],
                'f1_score': ev['f1_score'],
                'roc_auc': ev['roc_auc']
            } for name, ev in all_evaluations.items()
        }
    }

    with open(metadata_save_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print("Step 4: Generating and saving visualization plots...")
    plot_and_save_charts(best_model_name, best_eval, all_evaluations, images_dir)

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY:")
    print(f"Dataset loaded: {prep['total_rows']} rows")
    print(f"Features: {len(feature_names)}")
    print(f"Target: Churn")
    print(f"Missing values handled: Yes")
    print(f"Models trained: {len(candidate_models)}")
    print(f"Best model: {best_model_name}")
    print(f"Accuracy: {best_eval['accuracy']:.4f}")
    print(f"Precision: {best_eval['precision']:.4f}")
    print(f"Recall: {best_eval['recall']:.4f}")
    print(f"F1-Score: {best_eval['f1_score']:.4f}")
    print(f"ROC-AUC: {best_eval['roc_auc']:.4f}")
    print("Artifacts saved successfully in ml/models/ and static/images/")
    print("=" * 60)


if __name__ == '__main__':
    train_models()
