"""DriftAware — Enterprise Drift-Aware AutoML & Model Lifecycle Platform.

A domain-independent web application for automated machine learning, statistical drift detection,
model performance monitoring, controlled candidate retraining, and versioned registry management.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    BASE_DIR,
    DEFAULT_CATEGORICAL_TVD_THRESHOLD,
    DEFAULT_MINIMUM_DRIFT_SAMPLES,
    DEFAULT_MINIMUM_IMPROVEMENT_FOR_PROMOTION,
    DEFAULT_NUMERICAL_PSI_THRESHOLD,
    DEFAULT_P_VALUE_ALPHA,
    DEFAULT_PERFORMANCE_DEGRADATION_THRESHOLD,
    SAMPLE_DIR,
)
from modules.activity_logger import get_recent_activity, log_activity
from modules.data_processing import (
    build_preprocessor,
    clean_dataframe,
    get_dataset_recommendations,
    load_csv,
    profile_dataframe,
    split_features_target,
)
from modules.data_validation import detect_identifier_columns, safe_read_csv
from modules.drift_detection import (
    build_distribution_comparison_plot,
    compare_feature_drift,
    generate_demo_monitoring_batch,
    validate_incoming_batch,
)
from modules.explainability import (
    compute_permutation_importance,
    explain_local_prediction,
    get_native_feature_importance,
)
from modules.model_evaluation import (
    build_confusion_matrix_plot,
    build_per_class_metrics_table,
    build_regression_residual_plot,
    build_roc_curve_plot,
    compute_classification_metrics,
    compute_regression_metrics,
)
from modules.model_registry import (
    get_approved_model,
    get_model_by_version,
    list_all_versions,
    promote_version,
    rollback_to_version,
    save_model_version,
)
from modules.model_training import TrainingResult, train_models
from modules.performance_monitoring import (
    evaluate_batch_performance,
    get_monitoring_history,
)
from modules.reporting import generate_executive_html_report
from modules.retraining import (
    check_retraining_trigger,
    execute_retraining_workflow,
    validate_and_promote_candidate,
)
from modules.task_detection import TaskType, detect_task_type
from modules.ui_theme import ICONS, apply_theme, render_page_header

# Set Streamlit Page Config
st.set_page_config(
    page_title="DriftAware — MLOps Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply Enterprise SaaS Design System
apply_theme()


def _init_session_state() -> None:
    """Initialize state variables for application persistence across runs."""
    defaults: dict[str, Any] = {
        "current_page": "Overview",
        "dataset": None,
        "dataset_name": None,
        "target_column": None,
        "selected_features": None,
        "task_type": None,
        "task_detection_result": None,
        "cleaning_config": None,
        "cleaned_dataset": None,
        "training_result": None,
        "selected_evaluation_model": None,
        "reference_data": None,
        "incoming_monitoring_data": None,
        "monitoring_batch_name": None,
        "drift_table": None,
        "drift_summary": None,
        "performance_record": None,
        "latest_retraining_result": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def clear_experiment_results() -> None:
    """Invalidate stale training, evaluation, and drift results when dataset or target configuration changes."""
    st.session_state.training_result = None
    st.session_state.cleaned_dataset = None
    st.session_state.reference_data = None
    st.session_state.drift_table = None
    st.session_state.drift_summary = None
    st.session_state.performance_record = None
    st.session_state.latest_retraining_result = None
    st.session_state.selected_evaluation_model = None


_init_session_state()

# Navigation Definitions
NAV_ITEMS = [
    "Overview",
    "Datasets",
    "Data preparation",
    "AutoML experiments",
    "Model evaluation",
    "Explainability",
    "Drift monitoring",
    "Model registry",
    "Reports",
    "Settings",
    "Help & documentation",
]


# ==============================================================================
# Helper State Getters
# ==============================================================================

def _get_active_dataset() -> pd.DataFrame | None:
    if st.session_state.cleaned_dataset is not None:
        return st.session_state.cleaned_dataset
    return st.session_state.dataset


def _require_dataset() -> pd.DataFrame | None:
    df = _get_active_dataset()
    if df is None:
        st.info("No dataset selected. Navigate to **Datasets** to upload a CSV or load a sample benchmark.")
        return None
    return df


def _require_trained_result() -> TrainingResult | None:
    res = st.session_state.training_result
    if res is None:
        st.info("No trained AutoML model available in this session. Run an experiment in **AutoML experiments**.")
        return None
    return res


def _switch_page(target_page: str) -> None:
    st.session_state.current_page = target_page
    st.rerun()


# ==============================================================================
# Top Header Bar
# ==============================================================================

def render_top_bar() -> None:
    """Render top application utility bar with dataset shortcut and user chip."""
    col_left, col_right = st.columns([6, 2])
    with col_left:
        active_name = st.session_state.dataset_name or "None"
        target_name = st.session_state.target_column or "None"
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 16px; padding: 4px 0 16px 0;">
                <span style="font-size: 13px; color: #64748b;">
                    Active Dataset: <strong style="color: #0f172a;">{active_name}</strong>
                </span>
                <span style="color: #cbd5e1;">•</span>
                <span style="font-size: 13px; color: #64748b;">
                    Target: <strong style="color: #0f172a;">{target_name}</strong>
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_right:
        c_btn, c_avatar = st.columns([3, 1])
        with c_btn:
            if st.button("↑ Upload dataset", use_container_width=True, key="top_upload_btn"):
                _switch_page("Datasets")
        with c_avatar:
            st.markdown(
                """
                <div style="display: flex; justify-content: flex-end;">
                    <div class="user-avatar-chip" title="User Session">MR</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ==============================================================================
# PAGE 1: Overview
# ==============================================================================

def show_overview_page() -> None:
    render_page_header("Overview", "Monitor models, detect drift, and manage retraining.")

    # 1. Four Summary Metric Cards
    df = _get_active_dataset()
    app_model, app_meta = get_approved_model()
    drift_summary = st.session_state.drift_summary
    retrain_res = st.session_state.latest_retraining_result

    # Data Card Values
    if df is not None:
        dataset_val = f"{len(df):,} rows"
        dataset_sub = st.session_state.dataset_name or "Active dataset"
    else:
        dataset_val = "Not selected"
        dataset_sub = "Upload a dataset to get started."

    # Model Card Values
    if app_meta:
        model_val = app_meta.get("model_name", "Trained Model")
        model_sub = f"Version: {app_meta.get('approved_version_id', 'v1')[:12]}"
    else:
        model_val = "No model yet"
        model_sub = "Train and promote a model."

    # Drift Card Values
    if drift_summary:
        drift_cnt = drift_summary.get("drifted_features", 0)
        drift_val = f"{drift_cnt} Drifted" if drift_cnt > 0 else "Stable"
        drift_sub = f"{drift_summary.get('compared_features', 0)} features monitored."
    else:
        drift_val = "Not run"
        drift_sub = "Upload a new batch to check drift."

    # Retraining Card Values
    if retrain_res:
        val_res = retrain_res.get("validation_result", {})
        retrain_val = "Promoted" if val_res.get("promoted") else "Preserved"
        retrain_sub = f"{val_res.get('status', 'Completed')}"
    else:
        retrain_val = "Idle"
        retrain_sub = "No retraining in progress."

    # Render Summary Cards Grid
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="summary-card">
                <div>
                    <div class="summary-card-top">
                        <div class="summary-icon-box">{ICONS['database']}</div>
                        <div class="summary-label">Active Dataset</div>
                    </div>
                    <div class="summary-value">{dataset_val}</div>
                </div>
                <div class="summary-subtext">{dataset_sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="summary-card">
                <div>
                    <div class="summary-card-top">
                        <div class="summary-icon-box">{ICONS['cube']}</div>
                        <div class="summary-label">Approved Model</div>
                    </div>
                    <div class="summary-value">{model_val}</div>
                </div>
                <div class="summary-subtext">{model_sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="summary-card">
                <div>
                    <div class="summary-card-top">
                        <div class="summary-icon-box">{ICONS['activity']}</div>
                        <div class="summary-label">Latest Drift Check</div>
                    </div>
                    <div class="summary-value">{drift_val}</div>
                </div>
                <div class="summary-subtext">{drift_sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="summary-card">
                <div>
                    <div class="summary-card-top">
                        <div class="summary-icon-box">{ICONS['clock']}</div>
                        <div class="summary-label">Retraining Status</div>
                    </div>
                    <div class="summary-value">{retrain_val}</div>
                </div>
                <div class="summary-subtext">{retrain_sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # 2. Middle Row: Model Lifecycle (2/3) & Quick Start (1/3)
    col_mid_left, col_mid_right = st.columns([2.1, 1.0])

    with col_mid_left:
        st.markdown(
            f"""
            <div class="saas-card">
                <div class="card-header-flex">
                    <div>
                        <h3 class="card-title">Model lifecycle</h3>
                        <p class="card-subtitle">Complete the end-to-end workflow from data to deployment with automated monitoring.</p>
                    </div>
                </div>
                <div class="stepper-container">
                    <div class="stepper-line"></div>
                    <div class="stepper-step">
                        <div class="stepper-icon-circle">{ICONS['database']}</div>
                        <div class="stepper-title">1. Prepare data</div>
                        <div class="stepper-desc">Upload, profile and clean data</div>
                    </div>
                    <div class="stepper-step">
                        <div class="stepper-icon-circle">{ICONS['settings']}</div>
                        <div class="stepper-title">2. Train models</div>
                        <div class="stepper-desc">Run AutoML with multiple algorithms</div>
                    </div>
                    <div class="stepper-step">
                        <div class="stepper-icon-circle">{ICONS['chart']}</div>
                        <div class="stepper-title">3. Evaluate</div>
                        <div class="stepper-desc">Compare performance and select best model</div>
                    </div>
                    <div class="stepper-step">
                        <div class="stepper-icon-circle">{ICONS['activity']}</div>
                        <div class="stepper-title">4. Monitor drift</div>
                        <div class="stepper-desc">Detect data and concept drift</div>
                    </div>
                    <div class="stepper-step">
                        <div class="stepper-icon-circle">{ICONS['shield']}</div>
                        <div class="stepper-title">5. Validate & promote</div>
                        <div class="stepper-desc">Retrain, validate and manage versions</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_mid_right:
        with st.container(border=True):
            st.markdown(
                """
                <div class="card-header-flex">
                    <div>
                        <h3 class="card-title">Quick start</h3>
                        <p class="card-subtitle">Get started with DriftAware in a few simple steps.</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("↑  Upload a dataset                 >", type="primary", use_container_width=True, key="qs_upload"):
                _switch_page("Datasets")

            if st.button("🗄️  Try a sample dataset            >", use_container_width=True, key="qs_sample"):
                sample_path = SAMPLE_DIR / "manufacturing_reference.csv"
                if sample_path.exists():
                    raw = pd.read_csv(sample_path)
                    st.session_state.dataset = raw
                    st.session_state.dataset_name = "manufacturing_reference.csv"
                    st.session_state.target_column = "defect_detected"
                    st.session_state.task_detection_result = detect_task_type(raw["defect_detected"])
                    st.session_state.task_type = "classification"
                    recs = get_dataset_recommendations(raw, "manufacturing_reference.csv")
                    st.session_state.selected_features = recs["recommended_features"]
                    clear_experiment_results()
                    st.session_state.reference_data = raw.drop(columns=["defect_detected"]).copy()
                    log_activity("DATASET", "manufacturing_reference.csv", "Loaded manufacturing sensor benchmark dataset", "Active")
                    st.success("Loaded manufacturing sample dataset!")
                    st.rerun()

            if st.button("📄  View workflow guide            >", use_container_width=True, key="qs_guide"):
                _switch_page("Help & documentation")

    # 3. Bottom Row: Recent Activity (2/3) & Getting Started (1/3)
    col_bot_left, col_bot_right = st.columns([2.1, 1.0])

    with col_bot_left:
        with st.container(border=True):
            st.markdown(
                """
                <div class="card-header-flex">
                    <div>
                        <h3 class="card-title">Recent activity</h3>
                        <p class="card-subtitle">Your latest datasets, experiments, drift checks and retraining events.</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            activities = get_recent_activity(limit=6)
            if activities:
                act_df = pd.DataFrame(activities)[["type", "name", "description", "status", "date"]]
                act_df.columns = ["TYPE", "NAME", "DESCRIPTION", "STATUS", "DATE"]
                st.dataframe(act_df, use_container_width=True, hide_index=True)
            else:
                st.markdown(
                    f"""
                    <div class="empty-state-box">
                        <div style="display: flex; justify-content: center; margin-bottom: 8px;">{ICONS['document']}</div>
                        <div class="empty-state-title">No activity yet</div>
                        <div class="empty-state-text">Start by uploading a dataset or try a sample dataset to see your recent activity here.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with col_bot_right:
        st.markdown(
            """
            <div class="saas-card">
                <div class="card-header-flex">
                    <div>
                        <h3 class="card-title">Getting started</h3>
                        <p class="card-subtitle">Follow these steps to explore the platform.</p>
                    </div>
                </div>
                <div class="checklist-item">
                    <div class="checklist-num">1</div>
                    <div>
                        <div class="checklist-text-title">Upload a dataset</div>
                        <div class="checklist-text-desc">Use your data or try a sample dataset.</div>
                    </div>
                </div>
                <div class="checklist-item">
                    <div class="checklist-num">2</div>
                    <div>
                        <div class="checklist-text-title">Run an AutoML experiment</div>
                        <div class="checklist-text-desc">Train and compare multiple models.</div>
                    </div>
                </div>
                <div class="checklist-item">
                    <div class="checklist-num">3</div>
                    <div>
                        <div class="checklist-text-title">Monitor for drift</div>
                        <div class="checklist-text-desc">Upload new data to check for drift.</div>
                    </div>
                </div>
                <div class="checklist-item" style="margin-bottom: 0;">
                    <div class="checklist-num">4</div>
                    <div>
                        <div class="checklist-text-title">Manage model lifecycle</div>
                        <div class="checklist-text-desc">Validate and promote the best model.</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# PAGE 2: Datasets
# ==============================================================================

def show_datasets_page() -> None:
    render_page_header("Datasets", "Upload, inspect, and select tabular datasets for automated learning.")

    col1, col2 = st.columns([1.6, 1.0])
    with col1:
        with st.container(border=True):
            st.subheader("Upload Dataset")
            uploaded_file = st.file_uploader("Upload CSV file (max 50 MB)", type=["csv"], key="csv_upload")
            if uploaded_file is not None:
                if st.session_state.dataset_name != uploaded_file.name or st.session_state.dataset is None:
                    try:
                        raw_bytes = uploaded_file.getvalue()
                        df = safe_read_csv(raw_bytes)
                        recs = get_dataset_recommendations(df, uploaded_file.name)
                        st.session_state.dataset = df
                        st.session_state.dataset_name = uploaded_file.name
                        rec_target = recs.get("recommended_target")
                        if rec_target and rec_target in df.columns:
                            st.session_state.target_column = rec_target
                            task_res = detect_task_type(df[rec_target])
                            st.session_state.task_detection_result = task_res
                            st.session_state.task_type = task_res["task_type"]
                            all_c = [c for c in df.columns if c != rec_target]
                            rec_f = [c for c in recs.get("recommended_features", all_c) if c in all_c]
                            st.session_state.selected_features = rec_f or all_c
                        else:
                            st.session_state.target_column = None
                            st.session_state.task_detection_result = None
                            st.session_state.task_type = None
                            st.session_state.selected_features = [c for c in df.columns]

                        clear_experiment_results()
                        log_activity("DATASET", uploaded_file.name, f"Uploaded CSV ({len(df):,} rows, {len(df.columns)} cols)", "Uploaded")
                        st.success(f"Loaded **{uploaded_file.name}** successfully.")
                    except Exception as exc:
                        st.error(f"Upload error: {exc}")

            st.markdown("---")
            st.caption("Need a sample file to test the upload feature?")
            c_dl1, c_dl2 = st.columns(2)
            with c_dl1:
                test_path = BASE_DIR / "test_upload_dataset.csv"
                if test_path.exists():
                    with open(test_path, "rb") as f:
                        st.download_button(
                            "📥 Loan Risk CSV (1.2k rows)",
                            data=f.read(),
                            file_name="test_upload_dataset.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
            with c_dl2:
                churn_path = SAMPLE_DIR / "customer_churn_sample.csv"
                if churn_path.exists():
                    with open(churn_path, "rb") as f:
                        st.download_button(
                            "📥 Churn CSV (800 rows)",
                            data=f.read(),
                            file_name="customer_churn_sample.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )

    with col2:
        with st.container(border=True):
            st.subheader("Sample Benchmarks")
            st.caption("Quickly test the platform with realistic pre-configured datasets:")

            if st.button("⚙️ AI4I 2020 Predictive Maintenance (Classification)", use_container_width=True):
                ai4i_sample = SAMPLE_DIR / "ai4i2020.csv"
                if not ai4i_sample.exists():
                    desk_ai4i = Path(r"C:\Users\ADMIN\Desktop\ai4i2020.csv")
                    if desk_ai4i.exists():
                        import shutil
                        shutil.copy(desk_ai4i, ai4i_sample)
                if ai4i_sample.exists():
                    df = pd.read_csv(ai4i_sample)
                    recs = get_dataset_recommendations(df, "ai4i2020.csv")
                    st.session_state.dataset = df
                    st.session_state.dataset_name = "ai4i2020.csv"
                    st.session_state.target_column = "Machine failure"
                    task_res = detect_task_type(df["Machine failure"])
                    st.session_state.task_detection_result = task_res
                    st.session_state.task_type = "classification"
                    st.session_state.selected_features = recs["recommended_features"]
                    clear_experiment_results()
                    log_activity("DATASET", "ai4i2020.csv", "Loaded AI4I 2020 Predictive Maintenance benchmark", "Ready")
                    st.success("Loaded AI4I 2020 Predictive Maintenance Dataset.")
                    st.rerun()

            if st.button("🏭 Manufacturing Sensor Dataset (Classification)", use_container_width=True):
                df = pd.read_csv(SAMPLE_DIR / "manufacturing_reference.csv")
                recs = get_dataset_recommendations(df, "manufacturing_reference.csv")
                st.session_state.dataset = df
                st.session_state.dataset_name = "manufacturing_reference.csv"
                st.session_state.target_column = "defect_detected"
                st.session_state.task_detection_result = detect_task_type(df["defect_detected"])
                st.session_state.task_type = "classification"
                st.session_state.selected_features = recs["recommended_features"]
                clear_experiment_results()
                log_activity("DATASET", "manufacturing_reference.csv", "Loaded manufacturing sensor dataset", "Ready")
                st.success("Loaded Manufacturing Sensor Dataset.")
                st.rerun()

            if st.button("👥 Customer Churn Dataset (Classification)", use_container_width=True):
                df = pd.read_csv(SAMPLE_DIR / "customer_churn_sample.csv")
                recs = get_dataset_recommendations(df, "customer_churn_sample.csv")
                st.session_state.dataset = df
                st.session_state.dataset_name = "customer_churn_sample.csv"
                st.session_state.target_column = "churn"
                st.session_state.task_detection_result = detect_task_type(df["churn"])
                st.session_state.task_type = "classification"
                st.session_state.selected_features = recs["recommended_features"]
                clear_experiment_results()
                log_activity("DATASET", "customer_churn_sample.csv", "Loaded customer churn dataset", "Ready")
                st.success("Loaded Customer Churn Dataset.")
                st.rerun()

            if st.button("🏡 Housing Valuation Dataset (Regression)", use_container_width=True):
                df = pd.read_csv(SAMPLE_DIR / "housing_price_sample.csv")
                recs = get_dataset_recommendations(df, "housing_price_sample.csv")
                st.session_state.dataset = df
                st.session_state.dataset_name = "housing_price_sample.csv"
                st.session_state.target_column = "price_usd"
                st.session_state.task_detection_result = detect_task_type(df["price_usd"])
                st.session_state.task_type = "regression"
                st.session_state.selected_features = recs["recommended_features"]
                clear_experiment_results()
                log_activity("DATASET", "housing_price_sample.csv", "Loaded housing valuation dataset", "Ready")
                st.success("Loaded Housing Valuation Dataset.")
                st.rerun()

    df = st.session_state.dataset
    if df is not None:
        with st.container(border=True):
            st.subheader("Dataset Summary & Profiling")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", f"{len(df):,}")
            c2.metric("Columns", f"{len(df.columns)}")
            c3.metric("Missing Values", f"{df.isna().sum().sum():,}")
            c4.metric("Duplicates", f"{df.duplicated().sum():,}")

            # ID detection warnings
            id_detected = detect_identifier_columns(df)
            if id_detected:
                st.warning("⚠️ High-cardinality ID columns detected. Recommended to exclude from predictive features:")
                st.dataframe(pd.DataFrame(id_detected), use_container_width=True, hide_index=True)

            st.subheader("Data Preview")
            st.dataframe(df.head(10), use_container_width=True)

            # 1. Target Column Selection (Never silently defaults to first column)
            st.markdown("---")
            st.subheader("1. Target Variable Selection")
            st.caption("Select the prediction target (dependent variable y). Never silently defaults to an identifier column.")

            target_options = ["— None (Select a target variable) —"] + df.columns.tolist()
            current_target = st.session_state.target_column
            current_target_idx = 0
            if current_target and current_target in df.columns:
                current_target_idx = 1 + df.columns.tolist().index(current_target)

            selected_target_option = st.selectbox(
                "Prediction Target Column",
                target_options,
                index=current_target_idx,
                help="The column to predict. For ai4i2020.csv, select 'Machine failure'.",
            )

            if selected_target_option == "— None (Select a target variable) —":
                if st.session_state.target_column is not None:
                    st.session_state.target_column = None
                    st.session_state.task_detection_result = None
                    st.session_state.task_type = None
                    clear_experiment_results()
                    st.rerun()
                st.warning(
                    "⚠️ **No target column selected.** Please choose a prediction target column from the dropdown above to enable task detection and AutoML training."
                )
            else:
                if st.session_state.target_column != selected_target_option:
                    st.session_state.target_column = selected_target_option
                    task_res = detect_task_type(df[selected_target_option])
                    st.session_state.task_detection_result = task_res
                    st.session_state.task_type = task_res["task_type"]
                    # Clean selected features
                    if st.session_state.selected_features:
                        st.session_state.selected_features = [
                            c for c in st.session_state.selected_features if c != selected_target_option
                        ]
                    clear_experiment_results()
                    st.rerun()

                task_res = st.session_state.task_detection_result or detect_task_type(df[selected_target_option])
                col_task1, col_task2 = st.columns([1.6, 1.0])
                with col_task1:
                    st.info(f"💡 Suggested Task: **{task_res['task_type'].upper()}** ({task_res['sub_type']}) — {task_res['rationale']}")
                with col_task2:
                    current_task = st.session_state.task_type or task_res["task_type"]
                    override_task = st.selectbox(
                        "Task Type Override",
                        ["classification", "regression"],
                        index=0 if current_task == "classification" else 1,
                        help="Override suggested task type if appropriate for your use case.",
                    )
                    if override_task != st.session_state.task_type:
                        st.session_state.task_type = override_task
                        clear_experiment_results()

                if (st.session_state.task_type or task_res["task_type"]) == "classification" and task_res.get("class_distribution"):
                    st.markdown("#### Target Class Distribution")
                    c_dist1, c_dist2 = st.columns([1.1, 1.4])
                    with c_dist1:
                        dist_df = pd.DataFrame(task_res["class_distribution"])
                        st.dataframe(dist_df, use_container_width=True, hide_index=True)
                    with c_dist2:
                        if task_res.get("is_imbalanced"):
                            st.warning(
                                f"⚠️ **Class Imbalance Detected** (Ratio: **{task_res.get('imbalance_ratio', 1.0):.1f}:1**)\n\n"
                                f"- **Minority Class**: `{task_res.get('minority_class')}` ({task_res.get('minority_percentage', 0.0):.1f}%)\n"
                                f"- **Majority Class**: `{task_res.get('majority_class')}`\n\n"
                                "AutoML will automatically apply balanced class weights (`class_weight='balanced'`) and stratified train/validation splitting. We recommend selecting **Macro F1** or **Recall (Class 1)** as the primary evaluation metric."
                            )
                        else:
                            st.success("✅ **Balanced Classes**: Target classes are evenly distributed. Standard stratified splitting and evaluation are active.")

            # 2. Predictive Feature Selection & Leakage Prevention
            if st.session_state.target_column and st.session_state.target_column in df.columns:
                st.markdown("---")
                st.subheader("2. Predictive Feature Selection")
                st.caption("Select which input features (X) the models will use. Identifiers and target-leaking columns should be excluded.")

                recs = get_dataset_recommendations(df, st.session_state.dataset_name)
                candidate_features = [c for c in df.columns if c != st.session_state.target_column]

                # Initialize selected features if None
                if st.session_state.selected_features is None:
                    rec_f = [c for c in recs.get("recommended_features", candidate_features) if c in candidate_features]
                    st.session_state.selected_features = rec_f or candidate_features
                else:
                    # Filter existing to valid candidate features
                    st.session_state.selected_features = [
                        c for c in st.session_state.selected_features if c in candidate_features
                    ]

                c_fbtn1, c_fbtn2 = st.columns([1.8, 3.2])
                with c_fbtn1:
                    if st.button("↺ Apply Recommended Features", use_container_width=True):
                        rec_f = [c for c in recs.get("recommended_features", candidate_features) if c in candidate_features]
                        st.session_state.selected_features = rec_f or candidate_features
                        clear_experiment_results()
                        st.rerun()

                selected_feats = st.multiselect(
                    "Active Predictive Features",
                    options=candidate_features,
                    default=st.session_state.selected_features,
                    help="Select at least one feature. Exclude IDs and leakage indicators.",
                )

                if selected_feats != st.session_state.selected_features:
                    st.session_state.selected_features = selected_feats
                    clear_experiment_results()

                if not selected_feats:
                    st.error("❌ **No features selected.** Please select at least one predictive feature for AutoML training.")
                else:
                    st.success(f"✓ **{len(selected_feats)} features selected** out of {len(candidate_features)} available columns.")

                # Warnings for identifiers & leakage in selected features
                id_cols_detected = {d["column"] for d in detect_identifier_columns(df)}
                leakage_candidates = set(recs.get("excluded_features", []))

                warn_ids = [c for c in selected_feats if c in id_cols_detected]
                warn_leak = [c for c in selected_feats if c in leakage_candidates and c not in id_cols_detected]

                if warn_ids:
                    st.warning(f"⚠️ **Identifier Column Selected**: {warn_ids}. Identifiers cause memorization and poor test generalization.")
                if warn_leak:
                    st.error(
                        f"🚨 **Target Leakage Warning**: {warn_leak} are failure-mode indicators that directly leak the target outcome. "
                        "Including these produces artificially inflated scores that fail in production."
                    )

                # Recommended exclusion summary
                if recs.get("excluded_features"):
                    with st.expander("ℹ️ Domain Recommendations & Exclusion Reference", expanded=False):
                        if recs.get("benchmark_name"):
                            st.markdown(f"**Benchmark**: {recs.get('benchmark_name')}")
                        st.markdown(recs.get("notes", ""))
                        ex_df = pd.DataFrame([
                            {"Excluded Column": col, "Reason": recs.get("exclusion_reasons", {}).get(col, "Recommended for exclusion")}
                            for col in recs.get("excluded_features", [])
                            if col in df.columns
                        ])
                        if not ex_df.empty:
                            st.dataframe(ex_df, use_container_width=True, hide_index=True)
    else:
        with st.container(border=True):
            st.markdown(
                f"""
                <div class="empty-state-box">
                    <div style="display: flex; justify-content: center; margin-bottom: 8px;">{ICONS['database']}</div>
                    <div class="empty-state-title">No dataset loaded yet</div>
                    <div class="empty-state-text">Upload a custom CSV file or pick one of the sample benchmarks above to inspect columns, detect data types, and start training.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ==============================================================================
# PAGE 3: Data preparation
# ==============================================================================

def show_data_prep_page() -> None:
    render_page_header("Data preparation", "Profile distributions, inspect missing values, and apply cleaning rules.")
    df = _require_dataset()
    if df is None:
        return

    tab_profile, tab_clean = st.tabs(["📊 Data Profiling", "🧹 Data Cleaning"])

    with tab_profile:
        with st.container(border=True):
            profile = profile_dataframe(df)

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Rows", f"{profile['n_rows']:,}")
            c2.metric("Total Features", f"{profile['n_cols']:,}")
            c3.metric("Memory Footprint", f"{profile['memory_mb']} MB")

            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.subheader("Column Types & Cardinality")
                st.dataframe(profile["dtypes_table"], use_container_width=True, hide_index=True)
            with col_b:
                st.subheader("Summary Statistics")
                st.dataframe(profile["summary"], use_container_width=True)

            num_cols = profile["numerical_columns"]
            if num_cols:
                st.subheader("Feature Distributions")
                chosen_num = st.selectbox("Select Numerical Feature", num_cols)
                fig_hist = px.histogram(
                    df, x=chosen_num, nbins=30, marginal="box",
                    title=f"Distribution of {chosen_num}",
                    template="plotly_white",
                    color_discrete_sequence=["#2563eb"],
                )
                fig_hist.update_layout(margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_hist, use_container_width=True)

    with tab_clean:
        with st.container(border=True):
            st.subheader("Cleaning Rules")
            target_col = st.session_state.target_column
            st.caption(f"Target column protected: **{target_col or 'None selected'}**")

            c1, c2 = st.columns(2)
            with c1:
                drop_dups = st.checkbox("Remove duplicate rows", value=True)
                drop_const = st.checkbox("Drop constant columns (single unique value)", value=True)
                max_miss = st.slider("Drop columns exceeding missing % threshold", 10, 100, 80)
            with c2:
                id_detected = detect_identifier_columns(df)
                default_drops = [d["column"] for d in id_detected if d["column"] != target_col]
                cols_to_drop = st.multiselect(
                    "Exclude Identifier Columns",
                    [c for c in df.columns if c != target_col],
                    default=default_drops,
                )

            if st.button("Apply Cleaning Rules", type="primary"):
                try:
                    cleaned = clean_dataframe(
                        df,
                        drop_duplicates=drop_dups,
                        target_column=target_col,
                        drop_columns=cols_to_drop,
                        drop_constant_columns=drop_const,
                        max_missing_col_pct=max_miss,
                    )
                    st.session_state.cleaned_dataset = cleaned
                    st.session_state.training_result = None
                    log_activity("DATA_CLEAN", st.session_state.dataset_name or "dataset", f"Applied cleaning ({cleaned.shape[0]:,} rows remaining)", "Success")
                    st.success(f"Cleaning complete: {cleaned.shape[0]:,} rows × {cleaned.shape[1]} columns.")
                except Exception as exc:
                    st.error(f"Cleaning error: {exc}")

            active_df = _get_active_dataset()
            if active_df is not None:
                st.subheader("Cleaned Dataset Preview")
                st.dataframe(active_df.head(10), use_container_width=True)


# ==============================================================================
# PAGE 4: AutoML experiments
# ==============================================================================

def show_automl_page() -> None:
    render_page_header("AutoML experiments", "Benchmark and compare candidate machine learning pipelines.")
    df = _require_dataset()
    if df is None:
        return

    target_col = st.session_state.target_column
    if not target_col or target_col not in df.columns:
        st.warning("⚠️ **No target variable selected.** Please configure a prediction target in **Datasets** before running AutoML training.")
        if st.button("Go to Datasets", type="primary", key="goto_datasets_automl"):
            _switch_page("Datasets")
        return

    candidate_cols = [c for c in df.columns if c != target_col]
    active_features = [c for c in (st.session_state.selected_features or candidate_cols) if c in candidate_cols]
    if not active_features:
        st.error("❌ **No predictive features selected.** Please configure at least one input feature in **Datasets**.")
        if st.button("Go to Datasets", type="primary", key="goto_datasets_automl_feats"):
            _switch_page("Datasets")
        return

    task_info = st.session_state.task_detection_result or detect_task_type(df[target_col])
    task_type_default = st.session_state.task_type or task_info["task_type"]

    with st.container(border=True):
        st.subheader("AutoML Experiment Configuration")

        # Display active dataset, target, and features overview
        cfg_col1, cfg_col2, cfg_col3 = st.columns(3)
        cfg_col1.markdown(f"**Target Variable**: `{target_col}`")
        cfg_col2.markdown(f"**Task Suggestion**: `{task_info['task_type'].upper()}` ({task_info['sub_type']})")
        cfg_col3.markdown(f"**Active Features**: `{len(active_features)} selected`")

        with st.expander("🔍 View Selected Predictive Features (X)", expanded=False):
            st.caption(", ".join([f"`{c}`" for c in active_features]))

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            task_choice = st.selectbox(
                "Task Type",
                ["classification", "regression"],
                index=0 if task_type_default == "classification" else 1,
                help="Automatically suggested from target data type and cardinality. You can manually override.",
            )
        with c2:
            test_split = st.slider("Holdout Validation Split", 10, 40, 20, step=5) / 100.0
        with c3:
            if task_choice == "classification":
                pri_metric = st.selectbox(
                    "Primary Selection Metric",
                    ["f1_macro", "recall_class_1", "f1_weighted", "accuracy", "roc_auc", "precision_weighted", "recall_weighted"],
                    index=0,
                    help="Macro F1 and Class 1 Recall are heavily recommended for imbalanced datasets to prevent majority-class bias.",
                )
            else:
                pri_metric = st.selectbox("Primary Selection Metric", ["r2", "rmse", "mae"])

        if task_choice == "classification" and task_info.get("is_imbalanced"):
            st.info(
                f"⚖️ **Class Imbalance Active**: Target `{target_col}` has a {task_info.get('imbalance_ratio', 1.0):.1f}:1 class ratio. "
                "Candidate models are trained with `class_weight='balanced'` and regularized leaves to prevent zero-defect predictions."
            )

        if st.button("Run AutoML Experiment", type="primary", use_container_width=True):
            # Validate task choice against actual target values
            y_check = df[target_col].dropna()
            if task_choice == "classification" and y_check.nunique() < 2:
                st.error(f"❌ Classification requires at least 2 distinct target classes. Target '{target_col}' has only {y_check.nunique()} unique value.")
            elif task_choice == "regression" and not pd.api.types.is_numeric_dtype(y_check):
                st.error(f"❌ Regression target '{target_col}' must be numeric (found '{y_check.dtype}'). Please select Classification.")
            else:
                with st.spinner("Training candidate algorithms on selected features..."):
                    try:
                        res = train_models(
                            df,
                            target_column=target_col,
                            feature_columns=active_features,
                            task_type=task_choice,
                            primary_metric=pri_metric,
                            test_size=test_split,
                        )
                        st.session_state.training_result = res
                        st.session_state.reference_data = res.X_train.copy()
                        st.session_state.task_type = task_choice
                        st.session_state.drift_table = None
                        st.session_state.drift_summary = None

                        score = res.metrics[res.best_model_name].get(pri_metric, 0.0)
                        log_activity("EXPERIMENT", res.best_model_name, f"Benchmark completed ({pri_metric}: {score:.4f})", "Completed")
                        st.success(f"Benchmark completed! Best candidate: **{res.best_model_name}** ({pri_metric}: {score:.4f})")
                    except Exception as exc:
                        st.error(f"Experiment failed: {exc}")

    res = st.session_state.training_result
    if res is not None:
        with st.container(border=True):
            st.subheader("Leaderboard Results")

            # Champion summary metrics
            best_m = res.metrics.get(res.best_model_name, {})
            b_c1, b_c2, b_c3, b_c4 = st.columns(4)
            b_c1.metric("Top Model", res.best_model_name)
            if "f1_macro" in best_m:
                b_c2.metric("Macro F1", f"{best_m.get('f1_macro', 0.0):.4f}")
            else:
                b_c2.metric("R² Score", f"{best_m.get('r2', 0.0):.4f}")

            if "recall_class_1" in best_m:
                b_c3.metric("Defect (Class 1) Recall", f"{best_m.get('recall_class_1', 0.0):.1%}")
            elif "f1_weighted" in best_m:
                b_c3.metric("Weighted F1", f"{best_m.get('f1_weighted', 0.0):.4f}")
            else:
                b_c3.metric("RMSE", f"{best_m.get('rmse', 0.0):.4f}")

            if "accuracy" in best_m:
                b_c4.metric("Validation Accuracy", f"{best_m.get('accuracy', 0.0):.2%}")
            else:
                b_c4.metric("MAE", f"{best_m.get('mae', 0.0):.4f}")

            st.dataframe(res.comparison, use_container_width=True, hide_index=True)

            success_df = res.comparison[res.comparison["status"] == "success"]
            if not success_df.empty:
                fig_bar = px.bar(
                    success_df,
                    x="model",
                    y=pri_metric,
                    title=f"Holdout Performance by {pri_metric}",
                    template="plotly_white",
                    color="model",
                    color_discrete_sequence=["#2563eb", "#3b82f6", "#60a5fa", "#93c5fd"],
                )
                fig_bar.update_layout(margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)


# ==============================================================================
# PAGE 5: Model evaluation
# ==============================================================================

def show_evaluation_page() -> None:
    render_page_header("Model evaluation", "Inspect validation performance, classification matrices, and residuals.")
    res = _require_trained_result()
    if res is None:
        return

    with st.container(border=True):
        model_keys = list(res.models.keys())
        default_idx = (
            model_keys.index(st.session_state.selected_evaluation_model)
            if st.session_state.selected_evaluation_model in model_keys
            else model_keys.index(res.best_model_name)
        )
        model_name = st.selectbox("Select Model to Evaluate", model_keys, index=default_idx)
        st.session_state.selected_evaluation_model = model_name
        pipeline = res.models[model_name]
        metrics = res.metrics[model_name]

        preds = pipeline.predict(res.X_test)

        if res.task_type == "classification":
            # Prominent Defect Detection Highlight Card if binary classification
            if "recall_class_1" in metrics:
                tp = int(metrics.get("true_positives", 0))
                fn = int(metrics.get("false_negatives", 0))
                fp = int(metrics.get("false_positives", 0))
                tn = int(metrics.get("true_negatives", 0))
                total_defects = tp + fn
                rec_1 = metrics.get("recall_class_1", 0.0)
                prec_1 = metrics.get("precision_class_1", 0.0)
                f1_1 = metrics.get("f1_class_1", 0.0)
                macro_f1 = metrics.get("f1_macro", 0.0)

                st.markdown(
                    f"""
                    <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-left: 4px solid #2563eb; border-radius: 8px; padding: 16px 20px; margin-bottom: 20px;">
                        <div style="font-size: 0.95rem; font-weight: 700; color: #1e293b; margin-bottom: 8px;">
                            🎯 Class 1 (Defect) Detection Performance
                        </div>
                        <div style="display: flex; gap: 32px; flex-wrap: wrap; font-size: 0.9rem; color: #334155;">
                            <div><strong>Defect Recall:</strong> <span style="color: #2563eb; font-weight: 700; font-size: 1.1rem;">{rec_1:.1%}</span> ({tp} / {total_defects} defects caught)</div>
                            <div><strong>Defect Precision:</strong> <span style="font-weight: 600;">{prec_1:.1%}</span></div>
                            <div><strong>Defect F1-Score:</strong> <span style="font-weight: 600;">{f1_1:.4f}</span></div>
                            <div><strong>Macro F1:</strong> <span style="font-weight: 600;">{macro_f1:.4f}</span></div>
                            <div><strong>Missed Defects (FN):</strong> <span style="color: {'#ef4444' if fn > 0 else '#10b981'}; font-weight: 600;">{fn}</span></div>
                            <div><strong>False Alarms (FP):</strong> <span style="color: #f59e0b; font-weight: 600;">{fp}</span></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Key Summary Metrics
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Macro F1", f"{metrics.get('f1_macro', 0.0):.4f}")
            if "recall_class_1" in metrics:
                c2.metric("Defect Recall (Class 1)", f"{metrics.get('recall_class_1', 0.0):.1%}")
                c3.metric("Defect Precision (Class 1)", f"{metrics.get('precision_class_1', 0.0):.1%}")
            else:
                c2.metric("Macro Recall", f"{metrics.get('recall_macro', 0.0):.4f}")
                c3.metric("Macro Precision", f"{metrics.get('precision_macro', 0.0):.4f}")
            c4.metric("Weighted F1", f"{metrics.get('f1_weighted', 0.0):.4f}")
            c5.metric("Accuracy", f"{metrics.get('accuracy', 0.0):.2%}")

            tab1, tab2, tab3 = st.tabs(["📊 Confusion Matrix", "📋 Per-Class Breakdown", "📈 ROC Curve"])
            with tab1:
                cm_fig = build_confusion_matrix_plot(res.y_test, preds)
                st.plotly_chart(cm_fig, use_container_width=True)

                if "recall_class_1" in metrics:
                    cm_c1, cm_c2, cm_c3, cm_c4 = st.columns(4)
                    cm_c1.metric("True Negatives (TN)", f"{int(metrics.get('true_negatives', 0))}")
                    cm_c2.metric("False Positives (FP)", f"{int(metrics.get('false_positives', 0))}")
                    cm_c3.metric("False Negatives (FN)", f"{int(metrics.get('false_negatives', 0))}")
                    cm_c4.metric("True Positives (TP)", f"{int(metrics.get('true_positives', 0))}")
            with tab2:
                st.subheader("Per-Class Metrics & Support")
                class_table = build_per_class_metrics_table(res.y_test, preds)
                st.dataframe(class_table, use_container_width=True, hide_index=True)
            with tab3:
                if hasattr(pipeline, "predict_proba"):
                    probs = pipeline.predict_proba(res.X_test)
                    roc_fig = build_roc_curve_plot(res.y_test, probs)
                    if roc_fig:
                        st.plotly_chart(roc_fig, use_container_width=True)
                    else:
                        st.info("ROC Curve available for binary classification.")
                else:
                    st.info("Selected algorithm does not output probability estimates.")
        else:
            metric_cols = st.columns(len(metrics))
            for col, (m_key, m_val) in zip(metric_cols, metrics.items()):
                col.metric(m_key.replace("_", " ").title(), f"{m_val:.4f}" if np.isfinite(m_val) else "N/A")

            st.subheader("Residual Scatter")
            res_fig = build_regression_residual_plot(res.y_test, preds)
            st.plotly_chart(res_fig, use_container_width=True)


# ==============================================================================
# PAGE 6: Explainability
# ==============================================================================

def show_explainability_page() -> None:
    render_page_header("Explainability", "Understand model predictions using global feature importance and local breakdowns.")
    res = _require_trained_result()
    if res is None:
        return

    with st.container(border=True):
        model_keys = list(res.models.keys())
        default_idx = (
            model_keys.index(st.session_state.selected_evaluation_model)
            if st.session_state.selected_evaluation_model in model_keys
            else model_keys.index(res.best_model_name)
        )
        model_name = st.selectbox("Model to Explain", model_keys, index=default_idx)
        st.session_state.selected_evaluation_model = model_name
        pipeline = res.models[model_name]

        tab_perm, tab_local, tab_native = st.tabs(["🌍 Permutation Importance", "🔍 Local Instance Breakdown", "🌲 Native Weights"])

        with tab_perm:
            with st.spinner("Computing permutation importance..."):
                imp_df = compute_permutation_importance(
                    pipeline,
                    res.X_test,
                    res.y_test,
                    task_type=res.task_type,
                    scoring_metric=res.primary_metric,
                )
            if not imp_df.empty:
                fig_imp = px.bar(
                    imp_df.sort_values("importance_mean"),
                    x="importance_mean",
                    y="feature",
                    error_x="importance_std",
                    orientation="h",
                    title=f"Global Permutation Importance ({res.primary_metric})",
                    template="plotly_white",
                    color_discrete_sequence=["#2563eb"],
                )
                fig_imp.update_layout(margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_imp, use_container_width=True)
                st.dataframe(imp_df, use_container_width=True, hide_index=True)
            else:
                st.warning("Feature importance calculation unavailable.")

        with tab_local:
            sample_idx = st.slider("Select Sample Row from Validation Split", 0, len(res.X_test) - 1, 0)
            sample_row = res.X_test.iloc[[sample_idx]]
            true_label = res.y_test.iloc[sample_idx]
            pred_val = pipeline.predict(sample_row)[0]

            st.metric("Model Prediction", f"{pred_val}", f"True Label: {true_label}")
            baseline_median = res.X_train.median(numeric_only=True).to_frame().T
            local_df = explain_local_prediction(pipeline, sample_row, baseline_median, res.task_type)
            if not local_df.empty:
                fig_loc = px.bar(
                    local_df.sort_values("local_impact"),
                    x="local_impact",
                    y="feature",
                    orientation="h",
                    title="Feature Impact Relative to Baseline Median",
                    template="plotly_white",
                    color_discrete_sequence=["#3b82f6"],
                )
                fig_loc.update_layout(margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_loc, use_container_width=True)
                st.dataframe(local_df, use_container_width=True, hide_index=True)

        with tab_native:
            native_df = get_native_feature_importance(pipeline)
            if native_df is not None:
                fig_nat = px.bar(
                    native_df.head(15).sort_values("importance"),
                    x="importance",
                    y="transformed_feature",
                    orientation="h",
                    title="Top Transformed Features (Native Estimator Weights)",
                    template="plotly_white",
                    color_discrete_sequence=["#60a5fa"],
                )
                fig_nat.update_layout(margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_nat, use_container_width=True)
            else:
                st.info("Native feature importances are not exposed by this algorithm pipeline.")


# ==============================================================================
# PAGE 7: Drift monitoring
# ==============================================================================

def show_drift_page() -> None:
    render_page_header("Drift monitoring", "Detect data distribution changes between reference baselines and incoming batches.")
    res = st.session_state.training_result

    if st.session_state.reference_data is None:
        if res is not None:
            st.session_state.reference_data = res.X_train.copy()
        elif _get_active_dataset() is not None:
            active_df = _get_active_dataset()
            target_col = st.session_state.target_column
            cols = [c for c in active_df.columns if c != target_col]
            id_cols = detect_identifier_columns(active_df)
            cols = [c for c in cols if c not in id_cols]

            ai4i_core = [
                "Type",
                "Air temperature [K]",
                "Process temperature [K]",
                "Rotational speed [rpm]",
                "Torque [Nm]",
                "Tool wear [min]",
            ]
            if all(c in active_df.columns for c in ai4i_core):
                cols = ai4i_core

            st.session_state.reference_data = active_df[cols].copy()
        else:
            st.info("Establish a reference baseline first by training a model or uploading a reference CSV.")
            return

    ref_data = st.session_state.reference_data

    # Display baseline details
    with st.container(border=True):
        st.caption("Active Reference Baseline")
        b1, b2, b3 = st.columns(3)
        b1.metric("Baseline Samples", len(ref_data))
        b2.metric("Baseline Features", len(ref_data.columns))
        b3.metric("Target Variable", st.session_state.target_column or "Machine failure")

    with st.container(border=True):
        c1, c2 = st.columns([1.5, 1.1])
        with c1:
            st.subheader("Incoming Monitoring Batch")
            incoming_file = st.file_uploader("Upload CSV for Drift Check", type=["csv"], key="drift_csv")
            if incoming_file is not None and st.button("Run Drift Assessment", type="primary"):
                loaded = load_csv(incoming_file.getvalue())
                if loaded is not None:
                    st.session_state.incoming_monitoring_data = loaded
                    st.session_state.monitoring_batch_name = incoming_file.name

        with c2:
            st.subheader("Test Demonstrations")
            ref_norm = {str(c).strip() for c in ref_data.columns}
            ai4i_drifted_p = SAMPLE_DIR / "ai4i_monitoring_drifted.csv"
            ai4i_stable_p = SAMPLE_DIR / "ai4i_monitoring_stable.csv"
            mfg_drifted_p = SAMPLE_DIR / "manufacturing_monitoring_drifted.csv"
            mfg_stable_p = SAMPLE_DIR / "manufacturing_monitoring_stable.csv"

            if st.button("⚡ Test Drifted Batch (Manufacturing Sensors)", use_container_width=True):
                if ai4i_drifted_p.exists() and "Air temperature [K]" in ref_norm:
                    batch = pd.read_csv(ai4i_drifted_p)
                    name = "ai4i_monitoring_drifted.csv"
                elif mfg_drifted_p.exists() and "vibration_hz" in ref_norm:
                    batch = pd.read_csv(mfg_drifted_p)
                    name = "manufacturing_monitoring_drifted.csv"
                else:
                    batch = generate_demo_monitoring_batch(ref_data, drift=True, n_samples=350)
                    name = "manufacturing_monitoring_drifted.csv"

                st.session_state.incoming_monitoring_data = batch
                st.session_state.monitoring_batch_name = name
                st.success(f"Loaded Drifted Batch ({len(batch)} samples, {len(batch.columns)} features)")
                st.rerun()

            if st.button("✅ Test Stable Batch (Manufacturing Sensors)", use_container_width=True):
                if ai4i_stable_p.exists() and "Air temperature [K]" in ref_norm:
                    batch = pd.read_csv(ai4i_stable_p)
                    name = "ai4i_monitoring_stable.csv"
                elif mfg_stable_p.exists() and "vibration_hz" in ref_norm:
                    batch = pd.read_csv(mfg_stable_p)
                    name = "manufacturing_monitoring_stable.csv"
                else:
                    batch = generate_demo_monitoring_batch(ref_data, drift=False, n_samples=350)
                    name = "manufacturing_monitoring_stable.csv"

                st.session_state.incoming_monitoring_data = batch
                st.session_state.monitoring_batch_name = name
                st.success(f"Loaded Stable Batch ({len(batch)} samples, {len(batch.columns)} features)")
                st.rerun()

    incoming_df = st.session_state.incoming_monitoring_data
    if incoming_df is not None:
        if incoming_df.empty or len(incoming_df) == 0:
            st.error("⚠️ **Empty Incoming Batch**: The incoming batch contains 0 rows. Please provide a batch with valid sample observations.")
            return

        target_col = st.session_state.target_column
        incoming_feats = incoming_df.drop(columns=[target_col]) if (target_col and target_col in incoming_df.columns) else incoming_df

        drift_table, drift_summary = compare_feature_drift(ref_data, incoming_feats)
        st.session_state.drift_table = drift_table
        st.session_state.drift_summary = drift_summary

        status = drift_summary["status"]
        log_activity(
            "DRIFT_CHECK",
            st.session_state.monitoring_batch_name or "batch",
            f"{drift_summary['drifted_features']} of {drift_summary['compared_features']} features drifted",
            "Drift Detected" if status == "drift_detected" else ("Schema Issue" if status in ("schema_mismatch", "needs_attention", "empty_batch") else "Stable"),
        )

        with st.container(border=True):
            if status == "drift_detected":
                st.error(f"🚨 **Distribution Shift Detected**: {drift_summary['drifted_features']} of {drift_summary['compared_features']} monitored feature(s) drifted beyond threshold.")
            elif status in ("schema_mismatch", "needs_attention", "empty_batch"):
                st.warning(f"⚠️ **Batch Validation / Schema Issue**: {drift_summary['message']}")
            elif status == "no_drift_detected":
                st.success(f"✅ **Stable**: No significant distribution drift detected across all {drift_summary['compared_features']} monitored features.")
            else:
                st.info(drift_summary["message"])

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Incoming Batch Rows", len(incoming_df))
            m2.metric("Features Monitored", f"{drift_summary['compared_features']} / {len(ref_data.columns)}")
            m3.metric("Features Drifted", drift_summary["drifted_features"])
            missing_count = len(drift_summary.get("missing_features", []))
            m4.metric("Missing Features", missing_count)

            if drift_summary.get("schema_issues"):
                with st.expander("Schema and Validation Notes", expanded=True):
                    for issue in drift_summary["schema_issues"]:
                        st.write(f"- {issue}")

            st.subheader("Feature-by-Feature Results")
            st.dataframe(drift_table, use_container_width=True, hide_index=True)

            # Educational Note on Feature Drift vs Model Performance
            with st.container(border=True):
                st.markdown(
                    r"""
                    #### 💡 MLOps Note: Feature Drift vs. Model Performance
                    - **Input Feature Drift (\(P(X)\))**: Measures shifts in the distribution of incoming features compared to training baselines (e.g., sensor wear, operational temperature changes).
                    - **Model Performance Degradation (\(P(Y|X)\))**: Occurs when the relationship between input features and true targets changes.
                    - **Key Principle**: Feature drift serves as an early-warning signal, but **does not by itself prove that model accuracy has declined**. If shifts occur in regions where model predictions remain reliable, accuracy remains intact. To verify true accuracy or F1 degradation, evaluate ground-truth labels in **Model Evaluation** or the **Model Registry**.
                    """
                )

            # Distribution Overlay Plot
            cur_norm_map = {str(c).strip(): c for c in incoming_feats.columns}
            common_features = [c for c in ref_data.columns if str(c).strip() in cur_norm_map]
            if common_features:
                st.subheader("Distribution Overlay")
                sel_feat = st.selectbox("Select Feature to Compare", common_features)
                fig_overlay = build_distribution_comparison_plot(ref_data, incoming_feats, sel_feat)
                st.plotly_chart(fig_overlay, use_container_width=True)


# ==============================================================================
# PAGE 8: Model registry
# ==============================================================================

def show_registry_page() -> None:
    render_page_header("Model registry", "Audit model versions, run validation-gated candidate retraining, and serve inference.")

    tab_ver, tab_retrain, tab_inf = st.tabs(["📜 Versions & Promotion", "🔄 Retraining Gate", "🔮 Inference & Predictions"])

    with tab_ver:
        with st.container(border=True):
            st.subheader("Approved Production Model")
            app_model, app_meta = get_approved_model()
            if app_meta:
                st.success(f"Active Version: **{app_meta.get('approved_version_id')}** — {app_meta.get('model_name')} ({app_meta.get('task_type')})")
                st.json(app_meta)
            else:
                st.info("No approved model yet. You can promote the candidate from AutoML experiments.")
                res = st.session_state.training_result
                if res is not None and st.button("Promote Current Best Candidate to Approved"):
                    val_res = validate_and_promote_candidate(
                        res.best_model,
                        res.best_model_name,
                        res.X_test,
                        res.y_test,
                        res.task_type,
                        res.target_column,
                        primary_metric=res.primary_metric,
                        trigger_reason="initial_baseline",
                    )
                    log_activity("MODEL_REGISTRY", val_res["version_id"], f"Promoted {res.best_model_name} to Approved", "Approved")
                    st.success(f"Candidate promoted as {val_res['version_id']}!")
                    st.rerun()

            st.markdown("---")
            st.subheader("Version History")
            versions = list_all_versions()
            if versions:
                ver_df = pd.DataFrame([
                    {
                        "version_id": v.get("version_id"),
                        "model_name": v.get("model_name"),
                        "status": v.get("status"),
                        "created_at": v.get("created_at")[:19],
                        "task_type": v.get("task_type"),
                        "primary_metric": v.get("primary_metric"),
                        "score": v.get("candidate_score", v.get("metrics", {}).get(v.get("primary_metric"))),
                    }
                    for v in versions
                ])
                st.dataframe(ver_df, use_container_width=True, hide_index=True)

                st.subheader("Version Rollback")
                sel_ver = st.selectbox("Rollback Approved Pointer to", [v["version_id"] for v in versions])
                if st.button("Execute Rollback"):
                    rollback_to_version(sel_ver)
                    log_activity("MODEL_REGISTRY", sel_ver, "Rolled back approved model pointer", "Rolled Back")
                    st.success(f"Rolled back to {sel_ver}!")
                    st.rerun()
            else:
                st.info("No registered model versions.")

    with tab_retrain:
        with st.container(border=True):
            st.subheader("Retraining Trigger Diagnostics")
            drift_sum = st.session_state.drift_summary
            perf_rec = st.session_state.performance_record
            trigger = check_retraining_trigger(drift_sum, perf_rec)

            if trigger["triggered"]:
                st.warning(f"🔔 Retraining trigger active: {'; '.join(trigger['reasons'])}")
            else:
                st.info("Normal operations: no active retraining trigger condition.")

            st.subheader("Train & Validate Candidate Model")
            target_col = st.session_state.target_column
            res = st.session_state.training_result

            retrain_up = st.file_uploader("Upload New Labeled CSV for Retraining", type=["csv"], key="retrain_file")
            use_drift_as_labeled = st.checkbox("Or use loaded monitoring batch as retraining data")

            if st.button("Execute Retraining & Validation Gate", type="primary"):
                data_to_use = None
                if retrain_up is not None:
                    data_to_use = load_csv(retrain_up.getvalue())
                elif use_drift_as_labeled and st.session_state.incoming_monitoring_data is not None:
                    data_to_use = st.session_state.incoming_monitoring_data

                if data_to_use is None or target_col not in data_to_use.columns:
                    st.error("Labeled dataset with target column is required.")
                else:
                    with st.spinner("Training candidates and running validation gate..."):
                        try:
                            task_t = res.task_type if res else "classification"
                            pri_m = res.primary_metric if res else "f1_weighted"
                            out = execute_retraining_workflow(
                                data_to_use,
                                target_column=target_col,
                                task_type=task_t,
                                primary_metric=pri_m,
                            )
                            st.session_state.latest_retraining_result = out
                            v_res = out["validation_result"]
                            if v_res["promoted"]:
                                log_activity("RETRAINING", v_res["version_id"], "Candidate passed validation and was PROMOTED", "Promoted")
                                st.success(f"🎉 Candidate passed validation and was PROMOTED as `{v_res['version_id']}`!")
                            else:
                                log_activity("RETRAINING", v_res["version_id"], f"Candidate failed validation: {v_res['rationale']}", "Preserved")
                                st.warning(f"⚠️ Candidate failed validation ({v_res['rationale']}). Approved model was PRESERVED.")
                        except Exception as exc:
                            st.error(f"Retraining error: {exc}")

            latest_r = st.session_state.latest_retraining_result
            if latest_r:
                st.subheader("Latest Validation Gate Result")
                st.json(latest_r["validation_result"])

    with tab_inf:
        with st.container(border=True):
            app_model, app_meta = get_approved_model()
            if app_model is None:
                st.warning("Please approve a model version first in Model Registry.")
                return

            st.caption(f"Serving model: **{app_meta.get('model_name')}** ({app_meta.get('approved_version_id')})")
            inf_mode = st.radio("Mode", ["Single Observation Input", "Batch CSV Prediction"])

            if inf_mode == "Single Observation Input":
                ref_df = st.session_state.reference_data
                if ref_df is not None:
                    form_vals = {}
                    cols = st.columns(min(3, len(ref_df.columns)))
                    for i, c_name in enumerate(ref_df.columns):
                        c = cols[i % len(cols)]
                        series = ref_df[c_name]
                        if pd.api.types.is_numeric_dtype(series):
                            med_v = float(series.median()) if not series.empty else 0.0
                            form_vals[c_name] = c.number_input(c_name, value=med_v)
                        else:
                            cats = series.dropna().unique().tolist()
                            form_vals[c_name] = c.selectbox(c_name, cats if cats else ["default"])

                    if st.button("Generate Prediction", type="primary"):
                        inp_df = pd.DataFrame([form_vals])
                        pred_v = app_model.predict(inp_df)[0]
                        st.metric("Model Output", f"{pred_v}")
                        if hasattr(app_model, "predict_proba"):
                            st.write("Class Probabilities:", app_model.predict_proba(inp_df)[0])
            else:
                b_file = st.file_uploader("Upload CSV for Batch Prediction", type=["csv"], key="inf_batch")
                if b_file is not None and st.button("Generate Batch Predictions", type="primary"):
                    b_df = safe_read_csv(b_file.getvalue())
                    preds = app_model.predict(b_df)
                    out_df = b_df.copy()
                    out_df["prediction"] = preds
                    st.dataframe(out_df.head(15), use_container_width=True)
                    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download Predictions CSV", csv_bytes, "batch_predictions.csv", "text/csv")


# ==============================================================================
# PAGE 9: Reports
# ==============================================================================

def show_reports_page() -> None:
    render_page_header("Reports", "Generate standalone executive audit summaries and export tabular artifacts.")

    df = _get_active_dataset()
    res = st.session_state.training_result
    drift_sum = st.session_state.drift_summary
    drift_tbl = st.session_state.drift_table
    versions = list_all_versions()

    with st.container(border=True):
        st.subheader("Audit Report Generator")
        st.write("Compile full dataset profiling, model leaderboard, feature importances, and drift records into a single standalone HTML report.")

        if st.button("Generate Executive Report", type="primary"):
            html_rep = generate_executive_html_report(
                dataset_name=st.session_state.dataset_name or "Dataset",
                dataset_shape=df.shape if df is not None else (0, 0),
                target_column=st.session_state.target_column or "N/A",
                task_type=res.task_type if res else "Unknown",
                leaderboard_df=res.comparison if res else None,
                best_model_name=res.best_model_name if res else None,
                best_metrics=res.metrics[res.best_model_name] if res else None,
                importance_df=compute_permutation_importance(res.best_model, res.X_test, res.y_test, res.task_type) if res else None,
                drift_summary=drift_sum,
                drift_table=drift_tbl,
                monitoring_record=st.session_state.performance_record,
                registry_versions=versions,
            )
            st.success("Executive report generated!")
            st.download_button("📥 Download HTML Report", data=html_rep, file_name="DriftAware_Executive_Report.html", mime="text/html")

        st.markdown("---")
        st.subheader("Data Exports")
        c1, c2 = st.columns(2)
        with c1:
            if res is not None:
                csv_lb = res.comparison.to_csv(index=False).encode("utf-8")
                st.download_button("Export Leaderboard CSV", csv_lb, "model_leaderboard.csv", "text/csv", use_container_width=True)
        with c2:
            if drift_tbl is not None:
                csv_dr = drift_tbl.to_csv(index=False).encode("utf-8")
                st.download_button("Export Drift Table CSV", csv_dr, "drift_analysis.csv", "text/csv", use_container_width=True)


# ==============================================================================
# PAGE 10: Settings
# ==============================================================================

def show_settings_page() -> None:
    render_page_header("Settings", "Configure platform thresholds, statistical parameters, and random seeds.")

    with st.container(border=True):
        st.subheader("Statistical Drift Detection Thresholds")
        c1, c2 = st.columns(2)
        with c1:
            st.slider("Numerical PSI Drift Threshold", 0.05, 0.50, DEFAULT_NUMERICAL_PSI_THRESHOLD, 0.05)
            st.slider("Categorical TVD Threshold", 0.05, 0.50, DEFAULT_CATEGORICAL_TVD_THRESHOLD, 0.05)
        with c2:
            st.slider("Two-Sample KS P-Value Alpha", 0.01, 0.10, DEFAULT_P_VALUE_ALPHA, 0.01)
            st.number_input("Minimum Sample Count for Drift", value=DEFAULT_MINIMUM_DRIFT_SAMPLES)

        st.subheader("Retraining & Lifecycle Parameters")
        st.slider("Performance Degradation Alert Threshold (%)", 1, 20, int(DEFAULT_PERFORMANCE_DEGRADATION_THRESHOLD * 100))
        st.number_input("Minimum Improvement Margin for Promotion", value=DEFAULT_MINIMUM_IMPROVEMENT_FOR_PROMOTION)

        st.markdown("---")
        if st.button("Reset Application Session"):
            st.session_state.clear()
            st.success("Session reset.")
            st.rerun()


# ==============================================================================
# PAGE 11: Help & documentation
# ==============================================================================

def show_help_page() -> None:
    render_page_header("Help & documentation", "Platform architecture, statistical methodologies, and lifecycle guide.")

    with st.container(border=True):
        st.subheader("End-to-End MLOps Lifecycle")
        st.markdown(
            """
            1. **Data Preparation**: Ingest CSV files, auto-detect identifier columns to prevent data leakage, and construct Scikit-learn ColumnTransformers fitted strictly on training data.
            2. **AutoML Benchmark**: Train Linear models, Decision Trees, Random Forests, and HistGradientBoosting in parallel with duration profiling.
            3. **Explainability**: Inspect global permutation importance (model sensitivity) and local prediction breakdowns without relying on blackbox assumptions.
            4. **Drift Monitoring**: Compare production batches against reference baselines using Kolmogorov-Smirnov tests and Population Stability Index (PSI).
            5. **Validation Gate**: Candidate models retrained on drifted batches are evaluated on held-out validation data alongside the currently approved model. If a candidate is inferior, the existing approved model is preserved.
            """
        )

        st.subheader("Mathematical Formulations")
        st.markdown(
            """
            - **Population Stability Index (PSI)**:
              $$\\text{PSI} = \\sum_{b=1}^{B} (P_b - Q_b) \\times \\ln(P_b / Q_b)$$
            - **Total Variation Distance (TVD)**:
              $$\\text{TVD} = \\frac{1}{2} \\sum_{c} |P(c) - Q(c)|$$
            - **Two-Sample Kolmogorov-Smirnov Test**:
              $$D = \\sup_x |F_{\\text{ref}}(x) - F_{\\text{cur}}(x)|$$
            """
        )


# ==============================================================================
# Main Router
# ==============================================================================

def main() -> None:
    # Sidebar Header
    st.sidebar.markdown(
        f"""
        <div style="display: flex; align-items: center; justify-content: space-between; padding: 10px 4px 18px 4px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                {ICONS['logo']}
                <span style="font-size: 19px; font-weight: 700; color: #0f172a; letter-spacing: -0.02em;">DriftAware</span>
            </div>
            <span style="color: #94a3b8; font-size: 14px; cursor: pointer;" title="Collapse Sidebar">&laquo;</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    current_idx = NAV_ITEMS.index(st.session_state.current_page) if st.session_state.current_page in NAV_ITEMS else 0
    selected_page = st.sidebar.radio(
        "Navigation",
        NAV_ITEMS,
        index=current_idx,
        label_visibility="collapsed",
    )
    st.session_state.current_page = selected_page

    # Top Utility Bar
    render_top_bar()

    # Page Routing Table
    page_map = {
        "Overview": show_overview_page,
        "Datasets": show_datasets_page,
        "Data preparation": show_data_prep_page,
        "AutoML experiments": show_automl_page,
        "Model evaluation": show_evaluation_page,
        "Explainability": show_explainability_page,
        "Drift monitoring": show_drift_page,
        "Model registry": show_registry_page,
        "Reports": show_reports_page,
        "Settings": show_settings_page,
        "Help & documentation": show_help_page,
    }

    page_map[selected_page]()


if __name__ == "__main__":
    main()
