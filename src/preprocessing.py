"""
preprocessing.py
-----------------
Data cleaning, feature engineering, and sklearn pipeline construction.
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "study_hours_per_week",
    "attendance_rate",
    "previous_exam_score",
    "internal_marks",
    "assignment_completion",
]

CATEGORICAL_FEATURES = [
    "gender",
    "parental_education",
    "test_preparation",
    "extracurricular",
]

TARGET_REGRESSION = "final_exam_score"
TARGET_CLASSIFICATION = "pass_fail"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def add_pass_fail(df: pd.DataFrame, threshold: float = 50.0) -> pd.DataFrame:
    """Derive a binary pass/fail column from the final exam score."""
    df = df.copy()
    df[TARGET_CLASSIFICATION] = np.where(
        df[TARGET_REGRESSION] >= threshold, "Pass", "Fail"
    )
    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Return a ColumnTransformer that:
      • Imputes + scales numeric features
      • Imputes + one-hot-encodes categorical features
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def prepare_data(
    df: pd.DataFrame,
    target: str = TARGET_REGRESSION,
    test_size: float = 0.20,
    random_state: int = 42,
):
    """
    Split features / target and return (X_train, X_test, y_train, y_test, preprocessor).

    Parameters
    ----------
    target : str
        Column name of the target variable.
    """
    if target == TARGET_CLASSIFICATION and TARGET_CLASSIFICATION not in df.columns:
        df = add_pass_fail(df)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    preprocessor = build_preprocessor()

    return X_train, X_test, y_train, y_test, preprocessor


def get_feature_names(preprocessor, fitted: bool = True):
    """Retrieve feature names from a fitted ColumnTransformer."""
    if not fitted:
        return NUMERIC_FEATURES + ["cat_feature"]  # placeholder
    num_names = NUMERIC_FEATURES
    cat_names = list(
        preprocessor.named_transformers_["cat"]
        .named_steps["encoder"]
        .get_feature_names_out(CATEGORICAL_FEATURES)
    )
    return num_names + cat_names
