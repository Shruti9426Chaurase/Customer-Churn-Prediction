"""
Model evaluation module for Customer Churn Prediction.
Calculates Accuracy, Precision, Recall, F1-score, ROC-AUC,
Confusion Matrix, and ROC Curve points.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)


def evaluate_model(model, X_test, y_test, feature_names=None):
    """
    Evaluates a trained classifier on test data and returns a structured dictionary.
    """
    y_pred = model.predict(X_test)
    
    # Probability of churn (class 1)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_test, y_prob))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    confusion_dict = {
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'matrix': cm.tolist()
    }

    # ROC curve (sample down to ~30 points for smooth Chart.js rendering)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    step = max(1, len(fpr) // 30)
    roc_points = [
        {'fpr': round(float(fpr[i]), 4), 'tpr': round(float(tpr[i]), 4)}
        for i in range(0, len(fpr), step)
    ]
    if roc_points[-1]['fpr'] < 1.0:
        roc_points.append({'fpr': 1.0, 'tpr': 1.0})

    # Feature Importance extraction
    importances = []
    if feature_names:
        if hasattr(model, "feature_importances_"):
            raw_weights = model.feature_importances_
        elif hasattr(model, "coef_"):
            raw_weights = np.abs(model.coef_[0])
        else:
            raw_weights = np.ones(len(feature_names))

        feature_imp_pairs = sorted(
            zip(feature_names, raw_weights),
            key=lambda x: x[1],
            reverse=True
        )

        importances = [
            {'feature': f, 'importance': round(float(w), 4)}
            for f, w in feature_imp_pairs[:15]
        ]

    return {
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1_score': round(f1, 4),
        'roc_auc': round(auc, 4),
        'confusion_matrix': confusion_dict,
        'roc_curve': roc_points,
        'feature_importance': importances
    }
