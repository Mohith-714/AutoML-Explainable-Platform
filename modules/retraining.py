"""Drift-aware candidate retraining engine with validation gates and promotion policies."""

from __future__ import annotations

from typing import Any, Literal
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from config import (
    DEFAULT_MINIMUM_IMPROVEMENT_FOR_PROMOTION,
    DEFAULT_MINIMUM_LABELED_RETRAIN_SAMPLES,
)
from modules.model_evaluation import (
    compute_classification_metrics,
    compute_regression_metrics,
)
from modules.model_registry import (
    generate_version_id,
    get_approved_model,
    save_model_version,
)
from modules.model_training import train_models
from modules.task_detection import TaskType

RetrainingMode = Literal["manual_approval", "auto_candidate"]


def check_retraining_trigger(
    drift_summary: dict[str, Any] | None,
    performance_record: dict[str, Any] | None,
    *,
    min_drifted_features: int = 1,
) -> dict[str, Any]:
    """Assess whether monitoring signals justify candidate retraining."""
    reasons: list[str] = []
    drift_detected = False
    perf_degraded = False

    if drift_summary and drift_summary.get("drifted_features", 0) >= min_drifted_features:
        drift_detected = True
        reasons.append(
            f"Feature drift threshold met ({drift_summary.get('drifted_features')} drifted features)."
        )

    if performance_record and performance_record.get("degradation_detected", False):
        perf_degraded = True
        reasons.append(
            f"Performance degradation confirmed: {performance_record.get('message')}."
        )

    triggered = bool(reasons)
    return {
        "triggered": triggered,
        "drift_detected": drift_detected,
        "performance_degraded": perf_degraded,
        "reasons": reasons,
        "advice": (
            "Candidate retraining is recommended. Note that data drift alone does not guarantee "
            "a newly retrained model will outperform the approved model; validation gating is required."
            if triggered
            else "Monitoring signals indicate normal operations; no retraining trigger is currently active."
        ),
    }


def validate_and_promote_candidate(
    candidate_model: Pipeline,
    candidate_model_name: str,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    task_type: TaskType,
    target_column: str,
    *,
    primary_metric: str,
    trigger_reason: str,
    minimum_improvement: float = DEFAULT_MINIMUM_IMPROVEMENT_FOR_PROMOTION,
    mode: RetrainingMode = "manual_approval",
) -> dict[str, Any]:
    """Compare candidate against currently approved model on identical validation data and promote conditionally."""
    if len(X_val) != len(y_val) or len(X_val) < 4:
        raise ValueError("Validation set must have at least 4 labeled samples.")

    # Retrieve current approved model
    current_model, approved_meta = get_approved_model()

    # Calculate candidate metrics
    cand_preds = candidate_model.predict(X_val)
    cand_probs = candidate_model.predict_proba(X_val) if (task_type == "classification" and hasattr(candidate_model, "predict_proba")) else None

    if task_type == "classification":
        cand_metrics = compute_classification_metrics(y_val, cand_preds, cand_probs)
    else:
        cand_metrics = compute_regression_metrics(y_val, cand_preds)

    cand_score = cand_metrics.get(primary_metric, float("nan"))

    # Calculate current model metrics on the exact same validation split if available
    curr_score = None
    curr_metrics = None
    if current_model is not None:
        try:
            curr_preds = current_model.predict(X_val)
            curr_probs = current_model.predict_proba(X_val) if (task_type == "classification" and hasattr(current_model, "predict_proba")) else None
            if task_type == "classification":
                curr_metrics = compute_classification_metrics(y_val, curr_preds, curr_probs)
            else:
                curr_metrics = compute_regression_metrics(y_val, curr_preds)
            curr_score = curr_metrics.get(primary_metric, float("nan"))
        except Exception:
            curr_score = None

    # Decision logic
    passed_validation = False
    is_error_metric = primary_metric in ("mae", "rmse", "mape")

    if curr_score is None or not np.isfinite(curr_score):
        # Initial model baseline or uncomparable
        passed_validation = bool(np.isfinite(cand_score))
        validation_rationale = "First model or baseline unavailable; established as initial approved candidate."
    else:
        if is_error_metric:
            # Lower is better: candidate error must be <= current error - minimum_improvement
            passed_validation = bool(cand_score <= (curr_score - minimum_improvement))
            validation_rationale = (
                f"Candidate error ({cand_score:.4f}) is lower than or equal to current model error ({curr_score:.4f})."
                if passed_validation
                else f"Candidate error ({cand_score:.4f}) did not beat current model error ({curr_score:.4f})."
            )
        else:
            # Higher is better: candidate score must be >= current score + minimum_improvement
            passed_validation = bool(cand_score >= (curr_score + minimum_improvement))
            validation_rationale = (
                f"Candidate score ({cand_score:.4f}) meets or exceeds current model score ({curr_score:.4f})."
                if passed_validation
                else f"Candidate score ({cand_score:.4f}) is below current model score ({curr_score:.4f})."
            )

    # In manual approval mode, if it passed validation we still record it as candidate or promote based on flag
    should_promote = passed_validation

    version_id = generate_version_id()
    metadata: dict[str, Any] = {
        "version_id": version_id,
        "model_name": candidate_model_name,
        "task_type": task_type,
        "target_column": target_column,
        "primary_metric": primary_metric,
        "candidate_score": cand_score,
        "previous_score": curr_score,
        "metrics": cand_metrics,
        "previous_metrics": curr_metrics,
        "trigger_reason": trigger_reason,
        "mode": mode,
        "passed_validation": passed_validation,
        "status": "approved" if should_promote else "candidate_only",
        "validation_rationale": validation_rationale,
    }

    # Save version
    save_model_version(candidate_model, metadata, make_approved=should_promote)

    return {
        "version_id": version_id,
        "promoted": should_promote,
        "passed_validation": passed_validation,
        "candidate_score": cand_score,
        "previous_score": curr_score,
        "candidate_metrics": cand_metrics,
        "current_metrics": curr_metrics,
        "rationale": validation_rationale,
        "status": "approved" if should_promote else "candidate_only",
    }


def execute_retraining_workflow(
    new_data: pd.DataFrame,
    target_column: str,
    task_type: TaskType,
    *,
    primary_metric: str,
    trigger_reason: str = "monitoring_drift_trigger",
    mode: RetrainingMode = "manual_approval",
    min_labeled_samples: int = DEFAULT_MINIMUM_LABELED_RETRAIN_SAMPLES,
) -> dict[str, Any]:
    """Execute complete retraining pipeline on new labeled batch with validation gating."""
    if target_column not in new_data.columns:
        raise ValueError(f"New dataset lacks target column '{target_column}'; supervised retraining is not possible.")

    labeled_data = new_data.dropna(subset=[target_column])
    if len(labeled_data) < min_labeled_samples:
        raise ValueError(
            f"New labeled data has only {len(labeled_data)} rows. Minimum required is {min_labeled_samples}."
        )

    # Train candidates
    train_res = train_models(
        labeled_data,
        target_column,
        task_type=task_type,
        primary_metric=primary_metric,
    )

    # Validate best candidate against current approved model
    validation_res = validate_and_promote_candidate(
        candidate_model=train_res.best_model,
        candidate_model_name=train_res.best_model_name,
        X_val=train_res.X_test,
        y_val=train_res.y_test,
        task_type=task_type,
        target_column=target_column,
        primary_metric=primary_metric,
        trigger_reason=trigger_reason,
        mode=mode,
    )

    return {
        "training_result": train_res,
        "validation_result": validation_res,
    }
