"""Unit and integration tests for class imbalance detection, handling, and metrics."""

import numpy as np
import pandas as pd
import pytest

from config import SAMPLE_DIR
from modules.model_evaluation import (
    build_confusion_matrix_plot,
    build_per_class_metrics_table,
    compute_classification_metrics,
)
from modules.model_training import train_models
from modules.task_detection import detect_task_type


def test_imbalance_detection_on_imbalanced_series():
    # 88.5% class 0, 11.5% class 1
    y = pd.Series([0] * 885 + [1] * 115, name="defect_detected")
    res = detect_task_type(y)

    assert res["task_type"] == "classification"
    assert res["sub_type"] == "binary"
    assert res["is_imbalanced"] is True
    assert res["minority_class"] == "1"
    assert res["majority_class"] == "0"
    assert pytest.approx(res["minority_percentage"], 0.1) == 11.5
    assert len(res["class_distribution"]) == 2


def test_balanced_series_not_flagged_as_imbalanced():
    y = pd.Series([0] * 500 + [1] * 500, name="target")
    res = detect_task_type(y)
    assert res["is_imbalanced"] is False


def test_classification_metrics_and_confusion_matrix_ordering():
    # Ground truth: 177 TNs, 23 Positives (Defects)
    # Suppose model correctly predicts 17 TP, 6 FN, 32 FP, 145 TN
    y_true = np.array([0] * 177 + [1] * 23)
    y_pred = np.array([0] * 145 + [1] * 32 + [0] * 6 + [1] * 17)

    metrics = compute_classification_metrics(y_true, y_pred)

    assert metrics["true_negatives"] == 145
    assert metrics["false_positives"] == 32
    assert metrics["false_negatives"] == 6
    assert metrics["true_positives"] == 17
    assert pytest.approx(metrics["recall_class_1"], 0.01) == (17 / 23)
    assert metrics["recall_class_1"] > 0.70
    assert metrics["f1_macro"] > 0.65

    # Check per-class metrics table
    table = build_per_class_metrics_table(y_true, y_pred)
    assert isinstance(table, pd.DataFrame)
    assert len(table) == 4  # Class 0, Class 1, Macro Avg, Weighted Avg
    assert "Precision" in table.columns
    assert "Recall" in table.columns
    assert "F1-Score" in table.columns
    assert "Support" in table.columns

    # Check confusion matrix plot figure
    fig = build_confusion_matrix_plot(y_true, y_pred)
    assert fig is not None


def test_automl_training_on_manufacturing_reference_dataset():
    data_path = SAMPLE_DIR / "manufacturing_reference.csv"
    assert data_path.exists(), "manufacturing_reference.csv should exist in SAMPLE_DIR"

    df = pd.read_csv(data_path)
    res = train_models(
        df,
        target_column="defect_detected",
        task_type="classification",
        primary_metric="f1_macro",
        test_size=0.20,
    )

    assert res.task_type == "classification"
    assert res.best_model_name in res.models

    # Verify that the previous bug (0 defects detected, TP=0, recall_class_1=0) is resolved
    # Check all successfully trained models
    for model_name, metrics in res.metrics.items():
        assert "recall_class_1" in metrics
        assert "true_positives" in metrics
        assert "false_negatives" in metrics
        # Total defects in 20% holdout of 115 is exactly 23
        assert (metrics["true_positives"] + metrics["false_negatives"]) == 23

    # The champion model must detect defects with strong recall (> 50%)
    best_metrics = res.metrics[res.best_model_name]
    assert best_metrics["true_positives"] >= 15, f"Expected at least 15 TP, got {best_metrics['true_positives']}"
    assert best_metrics["recall_class_1"] >= 0.65, f"Expected defect recall >= 0.65, got {best_metrics['recall_class_1']}"
    assert best_metrics["f1_macro"] >= 0.65, f"Expected macro F1 >= 0.65, got {best_metrics['f1_macro']}"
