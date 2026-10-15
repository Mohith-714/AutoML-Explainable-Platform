"""Tests for heuristic task detection (Classification vs Regression)."""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from modules.task_detection import detect_task_type


def test_binary_classification_strings() -> None:
    target = pd.Series(["Yes", "No", "Yes", "Yes", "No", "No"] * 5)
    res = detect_task_type(target)
    assert res["task_type"] == "classification"
    assert res["sub_type"] == "binary"
    assert res["confidence"] == "high"


def test_binary_classification_numeric() -> None:
    target = pd.Series([0, 1, 1, 0, 1, 0] * 10)
    res = detect_task_type(target)
    assert res["task_type"] == "classification"
    assert res["sub_type"] == "binary"


def test_multiclass_classification_strings() -> None:
    target = pd.Series(["red", "green", "blue", "yellow"] * 10)
    res = detect_task_type(target)
    assert res["task_type"] == "classification"
    assert res["sub_type"] == "multiclass"
    assert res["unique_count"] == 4


def test_continuous_regression() -> None:
    target = pd.Series(np.random.normal(50000, 15000, 100))
    res = detect_task_type(target)
    assert res["task_type"] == "regression"
    assert res["sub_type"] == "continuous"


def test_empty_target_rejection() -> None:
    target = pd.Series([np.nan, np.nan])
    with pytest.raises(ValueError, match="no labeled values"):
        detect_task_type(target)
