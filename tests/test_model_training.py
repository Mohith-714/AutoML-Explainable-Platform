"""Tests for multi-model AutoML training, evaluation metrics, and pipeline execution."""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from modules.model_training import train_models, TrainingResult
from modules.model_evaluation import (
    compute_classification_metrics,
    compute_regression_metrics,
)


@pytest.fixture
def sample_classification_data() -> pd.DataFrame:
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "num_feat": np.random.randn(n),
        "cat_feat": np.random.choice(["type_A", "type_B", "type_C"], n),
        "target": np.random.choice([0, 1], n, p=[0.6, 0.4]),
    })


@pytest.fixture
def sample_regression_data() -> pd.DataFrame:
    np.random.seed(42)
    n = 100
    x1 = np.random.uniform(10, 50, n)
    x2 = np.random.choice(["low", "med", "high"], n)
    y = 2.5 * x1 + np.random.normal(0, 5, n)
    return pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "price": y,
    })


def test_classification_training_pipeline(sample_classification_data: pd.DataFrame) -> None:
    res = train_models(
        sample_classification_data,
        target_column="target",
        task_type="classification",
        test_size=0.25,
    )
    assert isinstance(res, TrainingResult)
    assert res.task_type == "classification"
    assert res.best_model_name in res.models
    assert len(res.models) >= 3
    assert not res.comparison.empty
    assert "accuracy" in res.comparison.columns
    assert "f1_weighted" in res.comparison.columns

    # Test raw inference on unseen observation
    new_obs = pd.DataFrame({
        "num_feat": [0.5],
        "cat_feat": ["type_A"],
    })
    pred = res.best_model.predict(new_obs)
    assert len(pred) == 1
    assert pred[0] in [0, 1]


def test_regression_training_pipeline(sample_regression_data: pd.DataFrame) -> None:
    res = train_models(
        sample_regression_data,
        target_column="price",
        task_type="regression",
        test_size=0.2,
    )
    assert isinstance(res, TrainingResult)
    assert res.task_type == "regression"
    assert "r2" in res.comparison.columns
    assert "mae" in res.comparison.columns
    assert "rmse" in res.comparison.columns

    new_obs = pd.DataFrame({
        "x1": [35.0],
        "x2": ["med"],
    })
    pred = res.best_model.predict(new_obs)
    assert len(pred) == 1
    assert np.isfinite(pred[0])


def test_metrics_calculators() -> None:
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_prob = np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3], [0.6, 0.4], [0.2, 0.8]])

    clf_metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    assert 0.0 <= clf_metrics["accuracy"] <= 1.0
    assert 0.0 <= clf_metrics["f1_weighted"] <= 1.0
    assert 0.0 <= clf_metrics["roc_auc"] <= 1.0

    y_reg_true = np.array([10.0, 20.0, 30.0])
    y_reg_pred = np.array([11.0, 19.0, 32.0])
    reg_metrics = compute_regression_metrics(y_reg_true, y_reg_pred)
    assert reg_metrics["mae"] > 0
    assert reg_metrics["rmse"] > 0
    assert np.isfinite(reg_metrics["r2"])
