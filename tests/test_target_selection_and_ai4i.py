"""Tests for target column selection, dataset recommendations, feature selection, and AI4I 2020 AutoML training."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from config import SAMPLE_DIR
from modules.data_processing import get_dataset_recommendations, split_features_target
from modules.model_training import train_models
from modules.task_detection import detect_task_type


def test_ai4i_dataset_recommendations():
    ai4i_path = SAMPLE_DIR / "ai4i2020.csv"
    if not ai4i_path.exists():
        desk_path = Path(r"C:\Users\ADMIN\Desktop\ai4i2020.csv")
        assert desk_path.exists(), "ai4i2020.csv must exist in SAMPLE_DIR or Desktop"
        df = pd.read_csv(desk_path)
    else:
        df = pd.read_csv(ai4i_path)

    recs = get_dataset_recommendations(df, "ai4i2020.csv")

    assert recs["is_known_benchmark"] is True
    assert recs["recommended_target"] == "Machine failure"
    assert "Type" in recs["recommended_features"]
    assert "Air temperature [K]" in recs["recommended_features"]
    assert "Process temperature [K]" in recs["recommended_features"]
    assert "Rotational speed [rpm]" in recs["recommended_features"]
    assert "Torque [Nm]" in recs["recommended_features"]
    assert "Tool wear [min]" in recs["recommended_features"]

    # Exclusions
    for excluded in ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]:
        assert excluded in recs["excluded_features"]
        assert excluded not in recs["recommended_features"]
        assert excluded in recs["exclusion_reasons"]


def test_ai4i_task_detection_is_binary_classification():
    ai4i_path = SAMPLE_DIR / "ai4i2020.csv"
    df = pd.read_csv(ai4i_path)
    task_res = detect_task_type(df["Machine failure"])

    assert task_res["task_type"] == "classification"
    assert task_res["sub_type"] == "binary"
    assert task_res["is_imbalanced"] is True
    assert task_res["minority_class"] == "1"
    assert task_res["majority_class"] == "0"


def test_split_features_target_with_feature_selection():
    ai4i_path = SAMPLE_DIR / "ai4i2020.csv"
    df = pd.read_csv(ai4i_path)

    features = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]

    X, y = split_features_target(df, "Machine failure", feature_columns=features)

    assert list(X.columns) == features
    assert "UDI" not in X.columns
    assert "Product ID" not in X.columns
    assert "Machine failure" not in X.columns
    assert "TWF" not in X.columns
    assert len(y) == len(X)
    assert y.name == "Machine failure"


def test_ai4i_automl_training_end_to_end():
    ai4i_path = SAMPLE_DIR / "ai4i2020.csv"
    df = pd.read_csv(ai4i_path)

    features = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]

    res = train_models(
        df,
        target_column="Machine failure",
        feature_columns=features,
        task_type="classification",
        primary_metric="f1_macro",
        test_size=0.2,
    )

    assert res.target_column == "Machine failure"
    assert res.task_type == "classification"
    assert list(res.X_train.columns) == features
    assert list(res.X_test.columns) == features

    # Verify candidate models were trained
    assert "Decision Tree" in res.models
    assert "Random Forest" in res.models
    assert "HistGradientBoosting" in res.models
    assert "Logistic Regression" in res.models

    # Champion model achieves strong defect detection without leakage
    best_metrics = res.metrics[res.best_model_name]
    assert best_metrics["f1_macro"] > 0.75
    assert best_metrics["recall_class_1"] > 0.60
    assert best_metrics["accuracy"] > 0.90
    assert best_metrics["true_positives"] >= 40


def test_generic_dataset_recommendations_and_no_silent_default():
    # Synthetic dataset with ID, numerical features, and target
    synthetic_df = pd.DataFrame({
        "row_id": range(100),
        "feat_a": np.random.randn(100),
        "feat_b": np.random.randn(100),
        "target": np.random.choice([0, 1], size=100),
    })

    recs = get_dataset_recommendations(synthetic_df, "unknown_dataset.csv")

    assert recs["is_known_benchmark"] is False
    assert "row_id" in recs["excluded_features"]
    assert "feat_a" in recs["recommended_features"]
    assert "feat_b" in recs["recommended_features"]
