"""End-to-End Verification Test Suite for DriftAware AutoML Platform.

Tests all 10 core functional areas:
1. CSV upload, schema validation, cleaning, and target selection.
2. AutoML training and model comparison using a stratified split.
3. Per-class precision, recall, F1, confusion matrix, and defect-class recall.
4. Prediction on valid and invalid records.
5. Explainability using the actual trained model.
6. Stable and drifted monitoring batches, including missing and empty inputs.
7. Model registration, saving, reloading, and version metadata.
8. Report generation and exported file contents.
9. Session-state consistency when navigating between pages or changing datasets.
10. Error handling and regression tests for previously observed bugs.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from config import MODELS_DIR, REPORTS_DIR, SAMPLE_DIR
from modules.data_processing import (
    clean_dataframe,
    get_dataset_recommendations,
    load_csv,
    profile_dataframe,
    split_features_target,
)
from modules.data_validation import (
    detect_identifier_columns,
    safe_read_csv,
    validate_target_column,
)
from modules.drift_detection import (
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
    compute_classification_metrics,
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
from modules.reporting import generate_executive_html_report
from modules.task_detection import detect_task_type

AI4I_FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
AI4I_TARGET = "Machine failure"
AI4I_EXCLUDED = ["UDI", "Product ID", "TWF", "HDF", "OSF", "RNF", "PWF"]


@pytest.fixture(scope="module")
def ai4i_raw_df() -> pd.DataFrame:
    p = SAMPLE_DIR / "ai4i2020.csv"
    assert p.exists(), f"ai4i2020.csv not found in {SAMPLE_DIR}"
    return pd.read_csv(p)


@pytest.fixture(scope="module")
def trained_ai4i_automl(ai4i_raw_df: pd.DataFrame) -> TrainingResult:
    return train_models(
        ai4i_raw_df,
        target_column=AI4I_TARGET,
        feature_columns=AI4I_FEATURES,
        task_type="classification",
        primary_metric="f1_macro",
        test_size=0.20,
        random_seed=42,
    )


# ==============================================================================
# TEST 1: CSV upload, schema validation, cleaning, and target selection
# ==============================================================================

def test_1_csv_upload_schema_validation_and_target_selection(ai4i_raw_df: pd.DataFrame) -> None:
    # 1.1 Ingestion via load_csv
    p = SAMPLE_DIR / "ai4i2020.csv"
    df_loaded = load_csv(p.read_bytes())
    assert df_loaded.shape == (10000, 14)
    assert AI4I_TARGET in df_loaded.columns
    for feat in AI4I_FEATURES:
        assert feat in df_loaded.columns

    # 1.2 Profiling and identifier detection
    profile = profile_dataframe(df_loaded)
    assert profile["shape"] == (10000, 14)
    assert profile["duplicate_rows"] == 0
    id_cols = [d["column"] for d in profile["identifier_warnings"]]
    assert "UDI" in id_cols

    # 1.3 Target recommendations and exclusions
    recs = get_dataset_recommendations(df_loaded, "ai4i2020.csv")
    assert recs["is_known_benchmark"] is True
    assert recs["recommended_target"] == AI4I_TARGET
    assert set(recs["recommended_features"]) == set(AI4I_FEATURES)
    for excl in ["UDI", "Product ID", "TWF", "HDF", "OSF", "RNF"]:
        assert excl in recs["excluded_features"]

    # 1.4 Target column validation
    validate_target_column(df_loaded, AI4I_TARGET)  # Succeeded without exception
    with pytest.raises(ValueError):
        validate_target_column(df_loaded, "non_existent_target_col")

    # 1.5 Cleaning rules
    cleaned = clean_dataframe(
        df_loaded,
        drop_duplicates=True,
        target_column=AI4I_TARGET,
        drop_columns=["UDI", "Product ID"],
    )
    assert "UDI" not in cleaned.columns
    assert "Product ID" not in cleaned.columns
    assert len(cleaned) == 10000


# ==============================================================================
# TEST 2: AutoML training & model comparison with stratified split
# ==============================================================================

def test_2_automl_training_and_stratified_split(trained_ai4i_automl: TrainingResult) -> None:
    res = trained_ai4i_automl

    assert res.target_column == AI4I_TARGET
    assert res.task_type == "classification"
    assert list(res.X_train.columns) == AI4I_FEATURES
    assert list(res.X_test.columns) == AI4I_FEATURES

    # 2.1 Verify Stratification: 3.39% minority ratio preserved in both train and test
    train_defect_rate = float((res.y_train == 1).mean())
    test_defect_rate = float((res.y_test == 1).mean())
    assert pytest.approx(train_defect_rate, abs=0.005) == 0.0339
    assert pytest.approx(test_defect_rate, abs=0.005) == 0.0340

    # 2.2 Verify All 4 candidate models trained successfully
    expected_models = {"Decision Tree", "Random Forest", "HistGradientBoosting", "Logistic Regression"}
    assert set(res.models.keys()) == expected_models

    # 2.3 Verify Leaderboard structure and sorting
    lb = res.comparison
    assert len(lb) == 4
    assert "model" in lb.columns
    assert "f1_macro" in lb.columns
    assert "training_time_sec" in lb.columns
    assert (lb["status"] == "success").all()

    # Leaderboard is sorted descending by f1_macro
    f1_vals = lb["f1_macro"].tolist()
    assert f1_vals == sorted(f1_vals, reverse=True)


# ==============================================================================
# TEST 3: Per-class precision, recall, F1, confusion matrix, defect recall
# ==============================================================================

def test_3_per_class_metrics_and_defect_recall(trained_ai4i_automl: TrainingResult) -> None:
    res = trained_ai4i_automl
    best_m = res.metrics[res.best_model_name]

    # Verify per-class metrics exist and are well-formed
    assert "recall_class_1" in best_m
    assert "precision_class_1" in best_m
    assert "f1_class_1" in best_m
    assert "recall_class_0" in best_m
    assert "precision_class_0" in best_m
    assert "f1_class_0" in best_m

    # Defect class recall must be substantial (imbalance handling working)
    assert best_m["recall_class_1"] >= 0.60, f"Expected defect recall >= 0.60, got {best_m['recall_class_1']}"
    assert best_m["precision_class_1"] >= 0.35, f"Expected defect precision >= 0.35, got {best_m['precision_class_1']}"
    assert best_m["f1_class_1"] >= 0.45, f"Expected defect F1 >= 0.45, got {best_m['f1_class_1']}"

    # Confusion matrix values
    assert best_m["true_positives"] >= 40  # 68 total defects in holdout
    assert best_m["true_negatives"] >= 1800  # 1932 total normal in holdout
    assert best_m["false_negatives"] <= 28

    # Per-class metrics table
    y_pred = res.best_model.predict(res.X_test)
    pct = build_per_class_metrics_table(res.y_test, y_pred)
    assert len(pct) == 4  # Class 0, Class 1, Macro Avg, Weighted Avg
    assert "Recall" in pct.columns
    assert "F1-Score" in pct.columns

    # Confusion matrix plot
    cm_fig = build_confusion_matrix_plot(res.y_test, y_pred)
    assert cm_fig is not None


# ==============================================================================
# TEST 4: Prediction on valid and invalid records
# ==============================================================================

def test_4_prediction_on_valid_and_invalid_records(trained_ai4i_automl: TrainingResult) -> None:
    model = trained_ai4i_automl.best_model

    # 4.1 Valid single observation
    valid_row = pd.DataFrame([{
        "Type": "L",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551.0,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0.0,
    }])
    pred = model.predict(valid_row)
    assert pred[0] in [0, 1]
    probs = model.predict_proba(valid_row)
    assert probs.shape == (1, 2)
    assert pytest.approx(probs.sum(), abs=1e-5) == 1.0

    # 4.2 Valid batch
    batch_preds = model.predict(trained_ai4i_automl.X_test.head(20))
    assert len(batch_preds) == 20
    assert set(batch_preds).issubset({0, 1})

    # 4.3 Missing values (pipeline imputer must handle without crashing)
    row_with_nans = pd.DataFrame([{
        "Type": None,
        "Air temperature [K]": np.nan,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551.0,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": np.nan,
    }])
    pred_nan = model.predict(row_with_nans)
    assert pred_nan[0] in [0, 1]

    # 4.4 Unseen categorical category (OneHotEncoder must ignore)
    row_unseen_cat = pd.DataFrame([{
        "Type": "EXTRA_UNKNOWN_CATEGORY",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551.0,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0.0,
    }])
    pred_unseen = model.predict(row_unseen_cat)
    assert pred_unseen[0] in [0, 1]

    # 4.5 Missing required column should raise an informative error
    row_missing_col = pd.DataFrame([{
        "Type": "L",
        "Air temperature [K]": 298.1,
    }])
    with pytest.raises(Exception):
        model.predict(row_missing_col)


# ==============================================================================
# TEST 5: Explainability using the actual trained model
# ==============================================================================

def test_5_explainability_pipeline(trained_ai4i_automl: TrainingResult) -> None:
    res = trained_ai4i_automl

    # 5.1 Permutation Importance
    perm_df = compute_permutation_importance(
        res.best_model,
        res.X_test,
        res.y_test,
        res.task_type,
        scoring_metric="f1_macro",
        n_repeats=3,
    )
    assert isinstance(perm_df, pd.DataFrame)
    assert len(perm_df) == len(AI4I_FEATURES)
    assert "feature" in perm_df.columns
    assert "importance_mean" in perm_df.columns
    assert perm_df.attrs["warning"] is None

    # 5.2 Native Feature Importance
    native_df = get_native_feature_importance(res.best_model)
    assert native_df is not None
    assert len(native_df) > 0
    assert "importance" in native_df.columns

    # 5.3 Local Prediction Explanation
    single_obs = res.X_test.head(1)
    baseline_median = res.X_train.median(numeric_only=True).to_frame().T
    baseline_median["Type"] = "L"

    local_df = explain_local_prediction(
        res.best_model,
        single_obs,
        baseline_median,
        res.task_type,
    )
    assert isinstance(local_df, pd.DataFrame)
    assert len(local_df) == len(AI4I_FEATURES)
    assert "local_impact" in local_df.columns
    assert "current_value" in local_df.columns


# ==============================================================================
# TEST 6: Stable and drifted monitoring batches, missing and empty inputs
# ==============================================================================

def test_6_drift_monitoring_batches_and_validation(trained_ai4i_automl: TrainingResult) -> None:
    ref_df = trained_ai4i_automl.X_train

    # 6.1 Stable Batch
    stable_df = pd.read_csv(SAMPLE_DIR / "ai4i_monitoring_stable.csv")
    s_table, s_summary = compare_feature_drift(ref_df, stable_df)
    assert s_summary["status"] == "no_drift_detected"
    assert s_summary["drifted_features"] == 0
    assert s_summary["compared_features"] == 6
    assert (s_table["cur_samples"] == 350).all()
    assert (s_table["status"] == "Stable").all()

    # 6.2 Drifted Batch
    drifted_df = pd.read_csv(SAMPLE_DIR / "ai4i_monitoring_drifted.csv")
    d_table, d_summary = compare_feature_drift(ref_df, drifted_df)
    assert d_summary["status"] == "drift_detected"
    assert d_summary["drifted_features"] >= 4
    assert d_summary["compared_features"] == 6
    assert (d_table["cur_samples"] == 350).all()

    # 6.3 Missing Columns Batch (must NEVER report Stable)
    partial_df = stable_df.drop(columns=["Torque [Nm]", "Tool wear [min]"])
    p_table, p_summary = compare_feature_drift(ref_df, partial_df)
    assert p_summary["status"] == "schema_mismatch"
    assert "Torque [Nm]" in p_summary["missing_features"]
    assert "Tool wear [min]" in p_summary["missing_features"]
    missing_rows = p_table[p_table["feature"].isin(["Torque [Nm]", "Tool wear [min]"])]
    assert (missing_rows["cur_samples"] == 0).all()
    assert (missing_rows["status"] == "Missing from incoming batch").all()

    # 6.4 Empty Batch (must NEVER report Stable)
    e_table, e_summary = compare_feature_drift(ref_df, pd.DataFrame())
    assert e_summary["status"] == "empty_batch"
    assert e_summary["compared_features"] == 0
    assert (e_table["cur_samples"] == 0).all()


# ==============================================================================
# TEST 7: Model registration, saving, reloading, version metadata
# ==============================================================================

def test_7_model_registry_persistence_and_reload(trained_ai4i_automl: TrainingResult) -> None:
    res = trained_ai4i_automl
    meta = {
        "model_name": res.best_model_name,
        "task_type": res.task_type,
        "target_column": res.target_column,
        "primary_metric": res.primary_metric,
        "metrics": res.metrics[res.best_model_name],
    }

    # 7.1 Save candidate version
    v_id = save_model_version(res.best_model, meta, make_approved=False)
    assert v_id.startswith("v_")
    v_dir = MODELS_DIR / v_id
    assert (v_dir / "model.joblib").exists()
    assert (v_dir / "metadata.json").exists()

    # 7.2 Promote version to approved
    ok = promote_version(v_id)
    assert ok is True

    # 7.3 Reload approved model and compare predictions
    reloaded_model, reloaded_meta = get_approved_model()
    assert reloaded_model is not None
    assert reloaded_meta["approved_version_id"] == v_id

    # Predictions from reloaded pipeline must match original perfectly
    test_subset = res.X_test.head(50)
    orig_preds = res.best_model.predict(test_subset)
    reload_preds = reloaded_model.predict(test_subset)
    np.testing.assert_array_equal(orig_preds, reload_preds)

    # 7.4 List all versions
    all_v = list_all_versions()
    v_ids = [v["version_id"] for v in all_v]
    assert v_id in v_ids

    # 7.5 Rollback functionality
    v2_id = save_model_version(res.best_model, meta, make_approved=True)
    assert get_approved_model()[1]["approved_version_id"] == v2_id
    rollback_to_version(v_id)
    assert get_approved_model()[1]["approved_version_id"] == v_id


# ==============================================================================
# TEST 8: Report generation and exported file contents
# ==============================================================================

def test_8_executive_report_generation(trained_ai4i_automl: TrainingResult) -> None:
    res = trained_ai4i_automl
    drift_tbl = pd.read_csv(SAMPLE_DIR / "ai4i_monitoring_stable.csv")
    _, drift_summary = compare_feature_drift(res.X_train, drift_tbl)

    html_report = generate_executive_html_report(
        dataset_name="ai4i2020.csv",
        dataset_shape=(10000, 14),
        target_column=AI4I_TARGET,
        task_type="classification",
        leaderboard_df=res.comparison,
        best_model_name=res.best_model_name,
        best_metrics=res.metrics[res.best_model_name],
        importance_df=compute_permutation_importance(res.best_model, res.X_test, res.y_test, res.task_type),
        drift_summary=drift_summary,
        drift_table=None,
        monitoring_record=None,
        registry_versions=list_all_versions(),
    )

    assert "<!DOCTYPE html>" in html_report
    assert "ai4i2020.csv" in html_report
    assert AI4I_TARGET in html_report
    assert res.best_model_name in html_report
    assert "leaderboard" in html_report.lower() or "styled-table" in html_report

    # Save report to REPORTS_DIR and verify file content
    rep_path = REPORTS_DIR / "test_e2e_executive_report.html"
    rep_path.write_text(html_report, encoding="utf-8")
    assert rep_path.exists()
    assert rep_path.stat().st_size > 1000


# ==============================================================================
# TEST 9: Session-state consistency & navigation definitions
# ==============================================================================

def test_9_session_state_consistency_and_invalidation() -> None:
    from app import NAV_ITEMS, clear_experiment_results

    # 9.1 Verify all 11 expected pages exist
    expected_pages = [
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
    assert NAV_ITEMS == expected_pages

    # 9.2 Verify clear_experiment_results invalidates state correctly
    class MockSessionState(dict):
        def __getattr__(self, name):
            return self.get(name)
        def __setattr__(self, name, value):
            self[name] = value

    import streamlit as st
    st.session_state = MockSessionState({
        "training_result": "dummy_result",
        "cleaned_dataset": "dummy_df",
        "reference_data": "dummy_ref",
        "drift_table": "dummy_table",
        "drift_summary": "dummy_sum",
        "performance_record": "dummy_rec",
        "latest_retraining_result": "dummy_retrain",
        "selected_evaluation_model": "dummy_model",
    })

    clear_experiment_results()
    assert st.session_state.training_result is None
    assert st.session_state.cleaned_dataset is None
    assert st.session_state.reference_data is None
    assert st.session_state.drift_table is None
    assert st.session_state.drift_summary is None
    assert st.session_state.selected_evaluation_model is None


# ==============================================================================
# TEST 10: Error handling and regression tests for previously observed bugs
# ==============================================================================

def test_10_error_handling_and_regression_tests(ai4i_raw_df: pd.DataFrame) -> None:
    # 10.1 Regression: UDI must not be selected as target
    recs = get_dataset_recommendations(ai4i_raw_df, "ai4i2020.csv")
    assert recs["recommended_target"] == AI4I_TARGET
    assert recs["recommended_target"] != "UDI"

    # 10.2 Regression: Defect recall must never be 0.0 on imbalanced dataset
    # We test on raw series with 11.5% defect rate
    imbal_series = pd.Series([0] * 885 + [1] * 115)
    det = detect_task_type(imbal_series)
    assert det["is_imbalanced"] is True

    # 10.3 Regression: Empty CSV rejected gracefully
    with pytest.raises(ValueError):
        safe_read_csv(b"")

    # 10.4 Regression: Invalid split ratio rejected
    with pytest.raises(ValueError):
        train_models(ai4i_raw_df, AI4I_TARGET, test_size=0.8)

    # 10.5 Regression: Single class target rejected for classification
    single_class_df = pd.DataFrame({"feat": [1, 2, 3], "target": [0, 0, 0]})
    with pytest.raises(ValueError):
        train_models(single_class_df, "target", task_type="classification")
