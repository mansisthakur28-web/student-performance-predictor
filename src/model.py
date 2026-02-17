"""
model.py
--------
Train, tune, save, and load regression & classification models.
"""

import os
import joblib
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_regression_models(X_train, y_train, preprocessor):
    """
    Train multiple regression models and return a dict of {name: fitted_pipeline}.
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=100, random_state=42
        ),
    }

    fitted = {}
    for name, estimator in models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])
        pipe.fit(X_train, y_train)
        print(f"  ✓ {name} trained")
        fitted[name] = pipe

    return fitted


def train_classification_models(X_train, y_train, preprocessor):
    """
    Train classification models for pass/fail prediction.
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
    }

    fitted = {}
    for name, estimator in models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])
        pipe.fit(X_train, y_train)
        print(f"  ✓ {name} trained")
        fitted[name] = pipe

    return fitted


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

def tune_model(pipeline, param_grid: dict, X, y, cv: int = 5, scoring=None):
    """
    Run GridSearchCV on an sklearn Pipeline and return the best estimator.

    `param_grid` keys should use the 'model__<param>' convention, e.g.
        {"model__n_estimators": [50, 100, 200]}
    """
    search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X, y)
    print(f"  ✓ Best params: {search.best_params_}")
    print(f"  ✓ Best score : {search.best_score_:.4f}")
    return search.best_estimator_


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(model, path: str = "models/model.pkl"):
    """Persist a fitted model / pipeline to disk."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    joblib.dump(model, path)
    print(f"[INFO] Model saved → {path}")


def load_model(path: str = "models/model.pkl"):
    """Load a previously saved model / pipeline."""
    model = joblib.load(path)
    print(f"[INFO] Model loaded ← {path}")
    return model
