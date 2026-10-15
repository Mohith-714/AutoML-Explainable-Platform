"""Tests for retraining triggers, candidate validation gates, model preservation, and registry."""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from modules.retraining import (
    check_retraining_trigger,
    validate_and_promote_candidate,
)
from modules.model_registry import (
    get_approved_model,
    list_all_versions,
    promote_version,
    rollback_to_version,
)
from modules.model_training import train_models


@pytest.fixture
def synthetic_train_and_val():
    np.random.seed(42)
    n = 120
    x = np.random.uniform(10, 50, n)
    y = 2.0 * x + np.random.normal(0, 3, n)
    df = pd.DataFrame({"x": x, "target": y})

    # Train initial model
    init_res = train_models(df, "target", task_type="regression")

    # Validation holdout set
    X_val = df[["x"]]
    y_val = df["target"]
    return init_res, X_val, y_val


def test_retraining_trigger_conditions() -> None:
    # No drift, no perf drop
    no_trigger = check_retraining_trigger(
        drift_summary={"drifted_features": 0},
        performance_record={"degradation_detected": False},
    )
    assert no_trigger["triggered"] is False

    # Drift detected
    drift_trigger = check_retraining_trigger(
        drift_summary={"drifted_features": 2},
        performance_record={"degradation_detected": False},
    )
    assert drift_trigger["triggered"] is True
    assert drift_trigger["drift_detected"] is True

    # Performance drop detected
    perf_trigger = check_retraining_trigger(
        drift_summary={"drifted_features": 0},
        performance_record={"degradation_detected": True, "message": "F1 dropped"},
    )
    assert perf_trigger["triggered"] is True
    assert perf_trigger["performance_degraded"] is True


def test_candidate_promotion_and_preservation(synthetic_train_and_val) -> None:
    init_res, X_val, y_val = synthetic_train_and_val

    # 1. Establish initial model as approved v1
    v1_action = validate_and_promote_candidate(
        candidate_model=init_res.best_model,
        candidate_model_name=init_res.best_model_name,
        X_val=X_val,
        y_val=y_val,
        task_type="regression",
        target_column="target",
        primary_metric="r2",
        trigger_reason="initial_baseline",
    )
    assert v1_action["promoted"] is True
    assert v1_action["status"] == "approved"

    app_model_v1, app_meta_v1 = get_approved_model()
    assert app_meta_v1["version_id"] == v1_action["version_id"]

    # 2. Test a superior or identical candidate (should be promoted or approved)
    v2_action = validate_and_promote_candidate(
        candidate_model=init_res.best_model,
        candidate_model_name="Candidate Model Identical",
        X_val=X_val,
        y_val=y_val,
        task_type="regression",
        target_column="target",
        primary_metric="r2",
        trigger_reason="drift_retrain_success",
        minimum_improvement=0.0,
    )
    assert v2_action["promoted"] is True
    assert v2_action["status"] == "approved"

    # 3. Test a flawed / inferior candidate with minimum_improvement required that it cannot meet
    # We require minimum_improvement = 10.0 (impossible on R2)
    v3_action = validate_and_promote_candidate(
        candidate_model=init_res.best_model,
        candidate_model_name="Candidate Failing Improvement",
        X_val=X_val,
        y_val=y_val,
        task_type="regression",
        target_column="target",
        primary_metric="r2",
        trigger_reason="drift_retrain_fail",
        minimum_improvement=10.0,  # Unattainable threshold
    )
    # The inferior candidate MUST fail validation and NOT be promoted
    assert v3_action["promoted"] is False
    assert v3_action["status"] == "candidate_only"

    # Crucial assertion: the currently approved model is preserved!
    current_approved, current_meta = get_approved_model()
    assert current_meta["version_id"] != v3_action["version_id"]
    assert current_meta["version_id"] == v2_action["version_id"]


def test_rollback_functionality(synthetic_train_and_val) -> None:
    init_res, X_val, y_val = synthetic_train_and_val

    # Register version A
    res_a = validate_and_promote_candidate(
        candidate_model=init_res.best_model,
        candidate_model_name="Model A",
        X_val=X_val,
        y_val=y_val,
        task_type="regression",
        target_column="target",
        primary_metric="r2",
        trigger_reason="test_vA",
    )
    # Register version B
    res_b = validate_and_promote_candidate(
        candidate_model=init_res.best_model,
        candidate_model_name="Model B",
        X_val=X_val,
        y_val=y_val,
        task_type="regression",
        target_column="target",
        primary_metric="r2",
        trigger_reason="test_vB",
    )

    curr_model, curr_meta = get_approved_model()
    assert curr_meta["version_id"] == res_b["version_id"]

    # Rollback to version A
    rollback_success = rollback_to_version(res_a["version_id"])
    assert rollback_success is True

    curr_model_after, curr_meta_after = get_approved_model()
    assert curr_meta_after["version_id"] == res_a["version_id"]
