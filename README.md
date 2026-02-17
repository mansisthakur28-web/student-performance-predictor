# 🎓 Student Performance Predictor

An end-to-end Machine Learning project that predicts student exam scores (**regression**) and pass/fail status (**classification**) from study habits, attendance, and prior performance.

---

## ✨ Features

| Capability | Details |
|---|---|
| **Regression** | Linear Regression, Random Forest, Gradient Boosting |
| **Classification** | Logistic Regression, Random Forest (pass/fail @ 50) |
| **Hyperparameter Tuning** | GridSearchCV with 5-fold cross-validation |
| **Visualizations** | Score distribution, correlation heatmap, scatter plots, feature importance, predicted vs actual, confusion matrix |
| **Interactive App** | Streamlit dashboard with sliders, live predictions, and charts |
| **Feedback System** | Anonymous thumbs-up/down + comments logged to CSV |

---

## 📁 Project Structure

```
├── data/                        # Dataset (auto-generated on first run)
│   └── student_performance.csv
├── models/                      # Saved model pipelines (.pkl)
├── src/
│   ├── data_loader.py           # Load data / generate synthetic dataset
│   ├── preprocessing.py         # Sklearn pipeline (impute, scale, encode)
│   ├── model.py                 # Train, tune, save/load models
│   ├── evaluation.py            # Regression & classification metrics
│   └── visualizations.py        # Matplotlib / Seaborn charts
├── notebooks/
│   └── student_performance_eda.ipynb   # Full EDA & modeling notebook
├── app/
│   └── streamlit_app.py         # Interactive web predictor
├── feedback/
│   └── feedback_log.csv         # User feedback (auto-created)
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### 2 — Run the Jupyter Notebook

```bash
cd notebooks
jupyter notebook student_performance_eda.ipynb
```

Run all cells to:
- Generate the synthetic dataset (if `data/student_performance.csv` doesn't exist)
- Explore 5+ visualizations
- Train & evaluate regression and classification models
- Save the best models to `models/`

### 3 — Launch the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

Adjust the sidebar sliders to get live predictions, explore charts, and submit feedback.

---

## 📊 Evaluation Metrics

| Task | Metrics |
|---|---|
| Regression (score) | R², MAE, RMSE |
| Classification (pass/fail) | Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix |

---

## ⚖️ Fairness & Ethics

- Demographic features are included **for analysis purposes only**; the core predictors are study hours, attendance, prior scores, and assignments.
- The model should **complement, not replace**, educator judgment.
- No personal identifiers are stored in feedback logs.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| Pandas / NumPy | Data wrangling |
| scikit-learn | ML pipelines, models, metrics |
| Matplotlib / Seaborn | Visualizations |
| Streamlit | Interactive web app |
| Jupyter | Notebook exploration |

---

## 📝 License

This project is provided for educational purposes. Feel free to extend and adapt it.
# Student Performance Predictor

Student Performance Predictor is a Python-based machine learning project that predicts a student's exam score or pass/fail status using features like study hours, attendance, internal/previous exam scores, and assignment completion. It uses a public student-performance dataset, Pandas/NumPy for preprocessing, scikit-learn models for regression and classification, and visualizations (distributions, correlation heatmap, scatter plots, feature importance, predicted vs actual) to explain the results. An optional Streamlit app lets users enter student details and see predictions along with basic insights.
