"""
data_loader.py
--------------
Load student performance data from CSV, or generate a synthetic dataset
for development / demo purposes.
"""

import os
import numpy as np
import pandas as pd


EXPECTED_COLUMNS = [
    "gender",
    "parental_education",
    "study_hours_per_week",
    "attendance_rate",
    "previous_exam_score",
    "internal_marks",
    "assignment_completion",
    "test_preparation",
    "extracurricular",
    "final_exam_score",
]


def load_data(path: str = "data/student_performance.csv") -> pd.DataFrame:
    """Load the student performance CSV and run basic validation."""
    if not os.path.exists(path):
        print(f"[INFO] Dataset not found at '{path}'. Generating synthetic data …")
        generate_synthetic_data(n=1000, path=path)

    df = pd.read_csv(path)
    print(f"[INFO] Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        print(f"[WARNING] Missing expected columns: {missing}")

    return df


def generate_synthetic_data(n: int = 1000, path: str = "data/student_performance.csv") -> pd.DataFrame:
    """
    Create a realistic synthetic student-performance dataset.

    Relationships baked in:
      • Higher study hours → higher scores
      • Higher attendance → higher scores
      • Internal marks correlated with final score
      • Random noise keeps it realistic
    """
    rng = np.random.default_rng(42)

    gender = rng.choice(["Male", "Female"], size=n)
    parental_education = rng.choice(
        ["High School", "Some College", "Bachelor's", "Master's", "Associate's"],
        size=n,
        p=[0.30, 0.25, 0.22, 0.10, 0.13],
    )

    study_hours = rng.uniform(0, 25, size=n).round(1)          # hrs / week
    attendance = rng.uniform(40, 100, size=n).round(1)          # percent
    previous_score = rng.uniform(30, 100, size=n).round(1)
    assignment_completion = rng.uniform(30, 100, size=n).round(1)
    test_preparation = rng.choice(["None", "Completed"], size=n, p=[0.45, 0.55])
    extracurricular = rng.choice(["Yes", "No"], size=n, p=[0.40, 0.60])

    # ---------- target: final_exam_score (0-100) ----------
    base = (
        0.25 * study_hours * 4          # max contribution ~25
        + 0.20 * attendance              # max ~20
        + 0.25 * previous_score          # max ~25
        + 0.15 * assignment_completion   # max ~15
    )
    # bonus for test prep
    prep_bonus = np.where(np.array(test_preparation) == "Completed", 5, 0)
    noise = rng.normal(0, 5, size=n)
    final_score = np.clip(base + prep_bonus + noise, 0, 100).round(1)

    # internal marks ~ correlated with final score
    internal_marks = np.clip(final_score + rng.normal(0, 4, size=n), 0, 100).round(1)

    df = pd.DataFrame({
        "gender": gender,
        "parental_education": parental_education,
        "study_hours_per_week": study_hours,
        "attendance_rate": attendance,
        "previous_exam_score": previous_score,
        "internal_marks": internal_marks,
        "assignment_completion": assignment_completion,
        "test_preparation": test_preparation,
        "extracurricular": extracurricular,
        "final_exam_score": final_score,
    })

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[INFO] Synthetic dataset saved to '{path}' ({n} rows)")
    return df


if __name__ == "__main__":
    df = load_data()
    print(df.head())
    print(df.describe())
