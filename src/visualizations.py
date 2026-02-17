"""
visualizations.py
------------------
Matplotlib / Seaborn charts for EDA and model evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Consistent style
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# ---------------------------------------------------------------------------
# EDA plots
# ---------------------------------------------------------------------------

def plot_score_distribution(df: pd.DataFrame, col: str = "final_exam_score"):
    """Histogram + KDE of a numeric column."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[col], kde=True, bins=30, color="#5B8FF9", edgecolor="white", ax=ax)
    ax.set_title(f"Distribution of {col.replace('_', ' ').title()}", fontsize=14, weight="bold")
    ax.set_xlabel(col.replace("_", " ").title())
    ax.set_ylabel("Count")
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame):
    """Annotated correlation heatmap of numeric features."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap=cmap,
        vmin=-1, vmax=1, center=0, square=True,
        linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, weight="bold")
    plt.tight_layout()
    return fig


def plot_scatter_with_regression(df: pd.DataFrame, x: str, y: str = "final_exam_score"):
    """Scatter plot with a regression line overlay."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.regplot(
        data=df, x=x, y=y,
        scatter_kws={"alpha": 0.4, "s": 20, "color": "#5B8FF9"},
        line_kws={"color": "#E8684A", "linewidth": 2},
        ax=ax,
    )
    ax.set_title(
        f"{x.replace('_', ' ').title()} vs {y.replace('_', ' ').title()}",
        fontsize=14, weight="bold",
    )
    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel(y.replace("_", " ").title())
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Model evaluation plots
# ---------------------------------------------------------------------------

def plot_feature_importance(model, feature_names: list, top_n: int = 15):
    """Horizontal bar chart of feature importances from a tree-based model."""
    # Navigate through pipeline if needed
    estimator = model
    if hasattr(estimator, "named_steps"):
        estimator = estimator.named_steps.get("model", estimator)

    if not hasattr(estimator, "feature_importances_"):
        print("[WARNING] Model does not have feature_importances_.")
        return None

    importances = estimator.feature_importances_
    indices = np.argsort(importances)[-top_n:]

    names = [feature_names[i] if i < len(feature_names) else f"feat_{i}" for i in indices]
    values = importances[indices]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(names, values, color="#5B8FF9", edgecolor="white")
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance (Top Features)", fontsize=14, weight="bold")

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    return fig


def plot_predicted_vs_actual(y_true, y_pred):
    """Scatter of predicted vs actual with the ideal 45° line."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.4, s=20, color="#5B8FF9", label="Predictions")

    mn, mx = min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=2, label="Ideal (y = x)")

    ax.set_xlabel("Actual Score", fontsize=12)
    ax.set_ylabel("Predicted Score", fontsize=12)
    ax.set_title("Predicted vs Actual", fontsize=14, weight="bold")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm, labels=None):
    """Seaborn heatmap of a confusion matrix."""
    if labels is None:
        labels = ["Fail", "Pass"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, weight="bold")
    plt.tight_layout()
    return fig
