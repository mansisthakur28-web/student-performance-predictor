"""
streamlit_app.py
-----------------
Interactive Student Performance Predictor with feedback collection.

Run:  streamlit run app/streamlit_app.py
"""

import sys, os

# ── project root on path ──────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import datetime
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.data_loader import load_data
from src.preprocessing import (
    prepare_data, add_pass_fail, build_preprocessor, get_feature_names,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    TARGET_REGRESSION, TARGET_CLASSIFICATION,
)
from src.model import (
    train_regression_models, train_classification_models, save_model, load_model,
)
from src.evaluation import evaluate_regression, evaluate_classification
from src.visualizations import (
    plot_score_distribution, plot_correlation_heatmap,
    plot_scatter_with_regression, plot_feature_importance,
    plot_predicted_vs_actual, plot_confusion_matrix,
)

# ── page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎓 Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e0e0e0;
    }
    /* Cards */
    .metric-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        backdrop-filter: blur(12px);
    }
    .metric-card h2 { margin:0; font-size:2.4rem; }
    .metric-card p  { margin:4px 0 0; opacity:0.7; }
    .pass-badge {
        display: inline-block;
        padding: 6px 20px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .pass { background: #2ecc71; color: #fff; }
    .fail { background: #e74c3c; color: #fff; }
    /* Fairness banner */
    .fairness-banner {
        background: rgba(46,204,113,0.12);
        border-left: 4px solid #2ecc71;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 20px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# DATA & MODEL CACHING
# ══════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading dataset …")
def get_data():
    data_path = os.path.join(PROJECT_ROOT, "data", "student_performance.csv")
    df = load_data(data_path)
    df = add_pass_fail(df)
    return df


@st.cache_resource(show_spinner="Training models (one-time) …")
def get_models(df):
    # Regression
    X_tr, X_te, y_tr, y_te, pre_r = prepare_data(df, target=TARGET_REGRESSION)
    reg_models = train_regression_models(X_tr, y_tr, pre_r)
    reg_metrics = {}
    reg_preds = {}
    for name, m in reg_models.items():
        met, yp = evaluate_regression(m, X_te, y_te)
        reg_metrics[name] = met
        reg_preds[name] = yp

    # Classification
    X_tr_c, X_te_c, y_tr_c, y_te_c, pre_c = prepare_data(df, target=TARGET_CLASSIFICATION)
    cls_models = train_classification_models(X_tr_c, y_tr_c, pre_c)
    cls_metrics = {}
    cls_preds = {}
    for name, m in cls_models.items():
        met, yp = evaluate_classification(m, X_te_c, y_te_c)
        cls_metrics[name] = met
        cls_preds[name] = yp

    # Save best models
    models_dir = os.path.join(PROJECT_ROOT, "models")
    save_model(reg_models["Random Forest Regressor"], os.path.join(models_dir, "best_regressor.pkl"))
    save_model(cls_models["Random Forest Classifier"], os.path.join(models_dir, "best_classifier.pkl"))

    # Feature names
    feat_names = get_feature_names(reg_models["Random Forest Regressor"].named_steps["preprocessor"])

    return (
        reg_models, reg_metrics, reg_preds,
        cls_models, cls_metrics, cls_preds,
        X_te, y_te, X_te_c, y_te_c,
        feat_names,
    )


# ── load once ─────────────────────────────────────────────────────────
df = get_data()
(
    reg_models, reg_metrics, reg_preds,
    cls_models, cls_metrics, cls_preds,
    X_te, y_te, X_te_c, y_te_c,
    feat_names,
) = get_models(df)

best_reg = reg_models["Random Forest Regressor"]
best_cls = cls_models["Random Forest Classifier"]


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR — Input Features
# ══════════════════════════════════════════════════════════════════════

st.sidebar.title("🎓 Student Inputs")
st.sidebar.markdown("Adjust the sliders to predict performance.")

study_hours = st.sidebar.slider("📚 Study Hours / Week", 0.0, 25.0, 10.0, 0.5)
attendance = st.sidebar.slider("📅 Attendance Rate (%)", 40.0, 100.0, 75.0, 1.0)
prev_score = st.sidebar.slider("📝 Previous Exam Score", 30.0, 100.0, 65.0, 1.0)
internal = st.sidebar.slider("📊 Internal Marks", 0.0, 100.0, 60.0, 1.0)
assignment = st.sidebar.slider("✅ Assignment Completion (%)", 30.0, 100.0, 70.0, 1.0)

gender = st.sidebar.selectbox("👤 Gender", ["Male", "Female"])
parental_edu = st.sidebar.selectbox(
    "🎓 Parental Education",
    ["High School", "Some College", "Associate's", "Bachelor's", "Master's"],
)
test_prep = st.sidebar.selectbox("📖 Test Preparation", ["None", "Completed"])
extra = st.sidebar.selectbox("⚽ Extracurricular", ["Yes", "No"])

# Build input DataFrame matching training column order
input_df = pd.DataFrame([{
    "study_hours_per_week": study_hours,
    "attendance_rate": attendance,
    "previous_exam_score": prev_score,
    "internal_marks": internal,
    "assignment_completion": assignment,
    "gender": gender,
    "parental_education": parental_edu,
    "test_preparation": test_prep,
    "extracurricular": extra,
}])


# ══════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════

# Fairness banner
st.markdown(
    '<div class="fairness-banner">'
    "⚖️ <strong>Fairness Note:</strong> Demographic features (gender, parental education) "
    "are used for analysis only and are not the primary drivers of predictions. "
    "This tool should complement — not replace — human judgment."
    "</div>",
    unsafe_allow_html=True,
)

st.title("🎓 Student Performance Predictor")
st.caption("Predict exam scores and pass/fail status to support timely academic interventions.")

# ── Predictions ───────────────────────────────────────────────────────
pred_score = best_reg.predict(input_df)[0]
pred_label = best_cls.predict(input_df)[0]

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f'<div class="metric-card"><h2>{pred_score:.1f}</h2>'
        f'<p>Predicted Exam Score</p></div>',
        unsafe_allow_html=True,
    )

with col2:
    badge = "pass" if pred_label == "Pass" else "fail"
    st.markdown(
        f'<div class="metric-card">'
        f'<h2><span class="pass-badge {badge}">{pred_label}</span></h2>'
        f'<p>Pass / Fail Prediction</p></div>',
        unsafe_allow_html=True,
    )

with col3:
    r2 = reg_metrics["Random Forest Regressor"]["R²"]
    st.markdown(
        f'<div class="metric-card"><h2>{r2:.3f}</h2>'
        f'<p>Model R² Score</p></div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────
tab_eda, tab_reg, tab_cls, tab_feat, tab_fb = st.tabs([
    "📊 EDA", "📈 Regression", "✅ Classification", "🌲 Feature Importance", "💬 Feedback",
])

# ── EDA Tab ───────────────────────────────────────────────────────────
with tab_eda:
    st.subheader("Score Distribution")
    fig = plot_score_distribution(df)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Correlation Heatmap")
    fig = plot_correlation_heatmap(df)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Scatter Plots")
    scatter_col = st.selectbox(
        "Select feature",
        ["study_hours_per_week", "attendance_rate", "previous_exam_score", "internal_marks"],
    )
    fig = plot_scatter_with_regression(df, x=scatter_col)
    st.pyplot(fig)
    plt.close(fig)

# ── Regression Tab ────────────────────────────────────────────────────
with tab_reg:
    st.subheader("Regression Model Comparison")
    metrics_df = pd.DataFrame(reg_metrics).T
    st.dataframe(metrics_df.style.format("{:.4f}").highlight_min(axis=0, subset=["MAE", "RMSE"]).highlight_max(axis=0, subset=["R²"]))

    st.subheader("Predicted vs Actual")
    sel_reg = st.selectbox("Model", list(reg_models.keys()), key="reg_sel")
    fig = plot_predicted_vs_actual(y_te, reg_preds[sel_reg])
    st.pyplot(fig)
    plt.close(fig)

# ── Classification Tab ───────────────────────────────────────────────
with tab_cls:
    st.subheader("Classification Model Comparison")
    cls_display = {}
    for name, met in cls_metrics.items():
        cls_display[name] = {k: v for k, v in met.items() if k != "Confusion Matrix"}
    cls_df = pd.DataFrame(cls_display).T
    st.dataframe(cls_df.style.format("{:.4f}"))

    st.subheader("Confusion Matrix")
    sel_cls = st.selectbox("Model", list(cls_models.keys()), key="cls_sel")
    cm = cls_metrics[sel_cls]["Confusion Matrix"]
    fig = plot_confusion_matrix(cm)
    st.pyplot(fig)
    plt.close(fig)

# ── Feature Importance Tab ───────────────────────────────────────────
with tab_feat:
    st.subheader("Regression — Feature Importance")
    fig = plot_feature_importance(best_reg, feat_names)
    if fig:
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Classification — Feature Importance")
    fig = plot_feature_importance(best_cls, feat_names)
    if fig:
        st.pyplot(fig)
        plt.close(fig)

# ── Feedback Tab ─────────────────────────────────────────────────────
with tab_fb:
    st.subheader("💬 Was This Prediction Helpful?")
    st.markdown("Your anonymous feedback helps us improve the model.")

    with st.form("feedback_form", clear_on_submit=True):
        helpful = st.radio("Did the prediction feel accurate?", ["👍 Yes", "👎 No", "🤷 Not sure"])
        comments = st.text_area("Any additional comments? (optional)", max_chars=500)
        submitted = st.form_submit_button("Submit Feedback")

    if submitted:
        fb_dir = os.path.join(PROJECT_ROOT, "feedback")
        os.makedirs(fb_dir, exist_ok=True)
        fb_path = os.path.join(fb_dir, "feedback_log.csv")

        row = {
            "timestamp": datetime.datetime.now().isoformat(),
            "predicted_score": round(pred_score, 1),
            "predicted_label": pred_label,
            "helpful": helpful,
            "comments": comments,
        }
        fb_df = pd.DataFrame([row])

        if os.path.exists(fb_path):
            fb_df.to_csv(fb_path, mode="a", header=False, index=False)
        else:
            fb_df.to_csv(fb_path, index=False)

        st.success("✅ Thank you! Your feedback has been recorded anonymously.")
