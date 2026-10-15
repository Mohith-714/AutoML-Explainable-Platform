"""Tests for statistical drift detection, effect sizes, schema validation, and benchmark batch handlers."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from config import SAMPLE_DIR
from modules.drift_detection import (
    calculate_categorical_distance,
    calculate_psi,
    compare_feature_drift,
    generate_demo_monitoring_batch,
    validate_incoming_batch,
)


def test_psi_calculation_identical_distributions() -> None:
    np.random.seed(42)
    dist1 = np.random.normal(50, 10, 1000)
    dist2 = np.random.normal(50, 10, 1000)
    psi = calculate_psi(dist1, dist2)
    assert psi < 0.1  # Well below 0.1 for identical distributions


def test_psi_calculation_drifted_distribution() -> None:
    np.random.seed(42)
    ref = np.random.normal(50, 10, 1000)
    cur = np.random.normal(85, 12, 1000)  # Significant shift
    psi = calculate_psi(ref, cur)
    assert psi > 0.25


def test_categorical_distance_identical_and_drifted() -> None:
    ref_s = pd.Series(["L"] * 600 + ["M"] * 300 + ["H"] * 100)
    cur_stable = pd.Series(["L"] * 60 + ["M"] * 30 + ["H"] * 10)
    tvd_stable, p_stable = calculate_categorical_distance(ref_s, cur_stable)
    assert tvd_stable < 0.05
    assert p_stable > 0.10

    cur_drifted = pd.Series(["L"] * 10 + ["M"] * 20 + ["H"] * 70)
    tvd_drifted, p_drifted = calculate_categorical_distance(ref_s, cur_drifted)
    assert tvd_drifted > 0.30
    assert p_drifted < 0.001


def test_no_drift_detected() -> None:
    np.random.seed(42)
    ref_df = pd.DataFrame({
        "num": np.random.normal(100, 15, 200),
        "cat": np.random.choice(["X", "Y", "Z"], 200, p=[0.5, 0.3, 0.2]),
    })
    cur_df = pd.DataFrame({
        "num": np.random.normal(100.2, 15.1, 200),
        "cat": np.random.choice(["X", "Y", "Z"], 200, p=[0.49, 0.31, 0.20]),
    })
    table, summary = compare_feature_drift(ref_df, cur_df)
    assert summary["status"] == "no_drift_detected"
    assert summary["drifted_features"] == 0
    assert summary["compared_features"] == 2


def test_significant_drift_detected() -> None:
    np.random.seed(42)
    ref_df = pd.DataFrame({
        "temperature": np.random.normal(65, 5, 250),
        "pressure": np.random.normal(100, 5, 250),
    })
    # Shift temperature by 30 degrees
    cur_df = pd.DataFrame({
        "temperature": np.random.normal(95, 5, 250),
        "pressure": np.random.normal(100, 5, 250),
    })
    table, summary = compare_feature_drift(ref_df, cur_df)
    assert summary["status"] == "drift_detected"
    assert summary["drifted_features"] >= 1

    temp_row = table[table["feature"] == "temperature"].iloc[0]
    assert temp_row["drift_detected"] == True
    assert temp_row["p_value"] < 0.001


def test_insufficient_samples_handling() -> None:
    ref_df = pd.DataFrame({"num": range(100)})
    cur_df = pd.DataFrame({"num": range(5)})  # Only 5 samples (< 20 minimum)
    table, summary = compare_feature_drift(ref_df, cur_df, minimum_samples=20)
    assert summary["status"] == "needs_attention"
    assert "Insufficient" in table.iloc[0]["status"]


def test_schema_drift_missing_columns_never_reports_stable() -> None:
    ref_df = pd.DataFrame({"col_a": [1, 2, 3] * 10, "col_b": [4, 5, 6] * 10})
    cur_df = pd.DataFrame({"col_a": [1, 2, 3] * 10, "col_extra": [7, 8, 9] * 10})
    table, summary = compare_feature_drift(ref_df, cur_df)
    # Must flag schema mismatch and NEVER report 'no_drift_detected'
    assert summary["status"] == "schema_mismatch"
    assert "col_b" in summary["missing_features"]
    assert any("Missing in incoming: col_b" in issue for issue in summary["schema_issues"])

    b_row = table[table["feature"] == "col_b"].iloc[0]
    assert b_row["cur_samples"] == 0
    assert b_row["status"] == "Missing from incoming batch"


def test_empty_incoming_batch_validation() -> None:
    ref_df = pd.DataFrame({"feat_1": [1.0, 2.0, 3.0] * 10, "feat_2": ["A", "B", "C"] * 10})
    cur_empty = pd.DataFrame()
    table, summary = compare_feature_drift(ref_df, cur_empty)

    assert summary["status"] == "empty_batch"
    assert summary["compared_features"] == 0
    assert "empty" in summary["message"].lower()
    assert len(table) == 2
    assert (table["cur_samples"] == 0).all()
    assert (table["status"] == "Batch is empty (0 samples)").all()


def test_column_whitespace_normalization() -> None:
    ref_df = pd.DataFrame({
        "Air temperature [K]": np.random.normal(300, 2, 100),
        "Torque [Nm]": np.random.normal(40, 5, 100),
    })
    # Incoming batch with subtle whitespace variations in column names
    cur_df = pd.DataFrame({
        " Air temperature [K]": np.random.normal(300, 2, 100),
        "Torque [Nm] ": np.random.normal(40, 5, 100),
    })
    table, summary = compare_feature_drift(ref_df, cur_df)
    assert summary["status"] == "no_drift_detected"
    assert summary["compared_features"] == 2
    assert len(summary["missing_features"]) == 0
    assert (table["cur_samples"] == 100).all()


def test_ai4i_manufacturing_stable_batch() -> None:
    ai4i_path = SAMPLE_DIR / "ai4i2020.csv"
    assert ai4i_path.exists(), "ai4i2020.csv must exist in SAMPLE_DIR"
    full_df = pd.read_csv(ai4i_path)

    expected_features = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    ref_df = full_df[expected_features].copy()

    # Test pre-generated stable batch or generate via helper
    stable_path = SAMPLE_DIR / "ai4i_monitoring_stable.csv"
    if stable_path.exists():
        stable_batch = pd.read_csv(stable_path)
    else:
        stable_batch = generate_demo_monitoring_batch(ref_df, drift=False, n_samples=350)

    assert len(stable_batch) == 350
    assert list(stable_batch.columns) == expected_features

    table, summary = compare_feature_drift(ref_df, stable_batch)

    # Assertions on stable comparison
    assert summary["status"] == "no_drift_detected"
    assert summary["compared_features"] == 6
    assert summary["drifted_features"] == 0
    assert len(summary["missing_features"]) == 0

    # Verify per-feature metrics
    for _, row in table.iterrows():
        assert row["ref_samples"] == 10000
        assert row["cur_samples"] == 350
        assert row["cur_null_pct"] == 0.0
        assert row["drift_detected"] is False
        assert row["status"] == "Stable"
        assert row["score"] < 0.10  # Low PSI/TVD


def test_ai4i_manufacturing_drifted_batch() -> None:
    ai4i_path = SAMPLE_DIR / "ai4i2020.csv"
    assert ai4i_path.exists(), "ai4i2020.csv must exist in SAMPLE_DIR"
    full_df = pd.read_csv(ai4i_path)

    expected_features = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    ref_df = full_df[expected_features].copy()

    # Test pre-generated drifted batch or generate via helper
    drifted_path = SAMPLE_DIR / "ai4i_monitoring_drifted.csv"
    if drifted_path.exists():
        drifted_batch = pd.read_csv(drifted_path)
    else:
        drifted_batch = generate_demo_monitoring_batch(ref_df, drift=True, n_samples=350)

    assert len(drifted_batch) == 350
    assert list(drifted_batch.columns) == expected_features

    table, summary = compare_feature_drift(ref_df, drifted_batch)

    # Assertions on drifted comparison
    assert summary["status"] == "drift_detected"
    assert summary["drifted_features"] >= 4
    assert summary["compared_features"] == 6

    # Verify specific shifted features trigger detection
    temp_row = table[table["feature"] == "Air temperature [K]"].iloc[0]
    assert temp_row["drift_detected"] == True
    assert temp_row["score"] > 0.25
    assert temp_row["p_value"] < 0.001

    type_row = table[table["feature"] == "Type"].iloc[0]
    assert type_row["metric"] == "TVD & Chi-Square"
    assert type_row["drift_detected"] == True
    assert type_row["score"] > 0.10


def test_validate_incoming_batch_helper() -> None:
    ref = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

    # Empty batch
    ok, issues = validate_incoming_batch(ref, pd.DataFrame())
    assert ok is False
    assert any("empty" in i.lower() for i in issues)

    # Missing column
    ok, issues = validate_incoming_batch(ref, pd.DataFrame({"a": [1, 2]}))
    assert ok is False
    assert any("Missing" in i for i in issues)

    # Type mismatch
    ok, issues = validate_incoming_batch(ref, pd.DataFrame({"a": ["1", "2"], "b": ["x", "y"]}))
    assert ok is False
    assert any("mismatch" in i.lower() for i in issues)

    # Valid batch with whitespace
    ok, issues = validate_incoming_batch(ref, pd.DataFrame({" a ": [1, 2], "b": ["x", "y"]}))
    assert ok is True
    assert len(issues) == 0
