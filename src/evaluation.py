"""
evaluation.py
--------------
Compute and display regression & classification evaluation metrics.
"""

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

def evaluate_regression(model, X_test, y_test) -> dict:
    """Return R², MAE, and RMSE for a regression model."""
    y_pred = model.predict(X_test)
    metrics = {
        "R²": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
    }
    return metrics, y_pred


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def evaluate_classification(model, X_test, y_test) -> dict:
    """
    Return accuracy, precision, recall, F1, confusion matrix, and ROC-AUC
    for a classification model.
    """
    y_pred = model.predict(X_test)

    # For ROC-AUC we need probability estimates
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(
            (y_test == "Pass").astype(int), y_prob
        )
    except Exception:
        roc_auc = None

    cm = confusion_matrix(y_test, y_pred, labels=["Fail", "Pass"])

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, pos_label="Pass", zero_division=0),
        "Recall": recall_score(y_test, y_pred, pos_label="Pass", zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, pos_label="Pass", zero_division=0),
        "ROC-AUC": roc_auc,
        "Confusion Matrix": cm,
    }
    return metrics, y_pred


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_report(name: str, metrics: dict):
    """Print metrics in a readable format."""
    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    for key, val in metrics.items():
        if key == "Confusion Matrix":
            print(f"  {key}:")
            print(f"    {val}")
        elif val is None:
            print(f"  {key:<12}: N/A")
        elif isinstance(val, float):
            print(f"  {key:<12}: {val:.4f}")
        else:
            print(f"  {key:<12}: {val}")
    print()
