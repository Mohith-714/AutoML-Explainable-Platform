"""Tests for data validation, ingestion, profiling, and leakage-safe preprocessing."""

from __future__ import annotations

import io
import pytest
import pandas as pd
import numpy as np

from modules.data_validation import (
    safe_read_csv,
    validate_file_size,
    detect_identifier_columns,
    compare_schemas,
)
from modules.data_processing import (
    load_csv,
    profile_dataframe,
    clean_dataframe,
    split_features_target,
    build_preprocessor,
)


def test_valid_csv_loading() -> None:
    csv_bytes = b"feature_a,feature_b,target\n1.0,cat,0\n2.0,dog,1\n3.0,bird,0\n4.0,fish,1\n"
    df = load_csv(csv_bytes)
    assert df.shape == (4, 3)
    assert list(df.columns) == ["feature_a", "feature_b", "target"]


def test_empty_csv_rejection() -> None:
    with pytest.raises(ValueError, match="empty|0 bytes|no data"):
        load_csv(b"")


def test_duplicate_columns_rejection() -> None:
    csv_bytes = b"col1,col1,target\n1,2,0\n3,4,1\n"
    with pytest.raises(ValueError, match="Duplicate column headers"):
        load_csv(csv_bytes)


def test_file_size_limit() -> None:
    large_payload = b"x" * 100
    with pytest.raises(ValueError, match="exceeds limit"):
        validate_file_size(large_payload, max_bytes=50)


def test_identifier_column_detection() -> None:
    df = pd.DataFrame({
        "id": range(50),
        "user_uuid": [f"uuid-{i}" for i in range(50)],
        "normal_num": np.random.randn(50),
        "target": np.random.choice([0, 1], 50),
    })
    detected = detect_identifier_columns(df)
    detected_cols = [d["column"] for d in detected]
    assert "id" in detected_cols
    assert "user_uuid" in detected_cols
    assert "normal_num" not in detected_cols


def test_schema_comparison() -> None:
    ref = ["col_a", "col_b", "col_c"]
    cur = ["col_a", "col_c", "col_extra"]
    diff = compare_schemas(ref, cur)
    assert diff["missing_in_current"] == ["col_b"]
    assert diff["extra_in_current"] == ["col_extra"]
    assert diff["common_columns"] == ["col_a", "col_c"]


def test_dataframe_profiling() -> None:
    df = pd.DataFrame({
        "num": [1.0, 2.0, np.nan, 4.0],
        "cat": ["a", "b", "a", "b"],
        "const": [1, 1, 1, 1],
    })
    prof = profile_dataframe(df)
    assert prof["n_rows"] == 4
    assert prof["n_cols"] == 3
    assert "num" in prof["numerical_columns"]
    assert "cat" in prof["categorical_columns"]
    assert "const" in prof["constant_columns"]
    assert prof["duplicate_rows"] == 0


def test_cleaning_duplicates_and_missing_target() -> None:
    df = pd.DataFrame({
        "feat": [1, 1, 2, 3],
        "target": [0, 0, np.nan, 1],
    })
    # Has 1 duplicate row (1, 0) and 1 row with missing target
    cleaned = clean_dataframe(df, drop_duplicates=True, target_column="target")
    assert len(cleaned) == 2
    assert list(cleaned["feat"]) == [1, 3]


def test_build_preprocessor_and_unseen_categories() -> None:
    X_train = pd.DataFrame({
        "age": [25.0, 30.0, np.nan, 45.0],
        "city": ["NY", "SF", "LA", "NY"],
    })
    preprocessor = build_preprocessor(X_train)
    transformed_train = preprocessor.fit_transform(X_train)
    assert transformed_train.shape[0] == 4

    # New unseen category in test set should not crash
    X_test = pd.DataFrame({
        "age": [35.0],
        "city": ["Tokyo"],  # Unseen
    })
    transformed_test = preprocessor.transform(X_test)
    assert transformed_test.shape[0] == 1
