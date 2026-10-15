"""Ground-truth performance tracking, degradation detection, and historical monitoring logs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from config import DEFAULT_PERFORMANCE_DEGRADATION_THRESHOLD, METADATA_DIR
from modules.model_evaluation import (
    compute_classification_metrics,
    compute_regression_metrics,
)
from modules.task_detection import TaskType

MONITORING_LOG_PATH = METADATA_DIR / "monitoring_history.json"


def evaluate_batch_performance(
    pipeline: Pipeline,
    incoming_df: pd.DataFrame,
    target_column: str,
    task_type: TaskType,
    baseline_metrics: dict[str, float],
    primary_metric: str,
    *,
    degradation_threshold: float = DEFAULT_PERFORMANCE_DEGRADATION_THRESHOLD,
    batch_name: str = "monitoring_batch",
) -> dict[str, Any]:
    """Evaluate pipeline on incoming data if ground-truth target column is available.
    
    Distinguishes ground-truth performance evaluation from feature drift.
    """
    if target_column not in incoming_df.columns:
        return {
            "has_ground_truth": False,
            "status": "unlabeled",
            "message": (
                f"Target column '{target_column}' is absent in incoming batch. "
                "True performance cannot be measured without ground-truth labels. "
                "Only input feature drift is currently assessed."
            ),
            "metrics": {},
            "baseline_metrics": baseline_metrics,
            "primary_metric": primary_metric,
            "degradation_detected": False,
            "sample_count": len(incoming_df),
        }

    valid_batch = incoming_df.dropna(subset=[target_column]).copy()
    if len(valid_batch) < 5:
        return {
            "has_ground_truth": True,
            "status": "insufficient_samples",
            "message": f"Only {len(valid_batch)} labeled samples provided; at least 5 needed to evaluate performance.",
            "metrics": {},
            "baseline_metrics": baseline_metrics,
            "primary_metric": primary_metric,
            "degradation_detected": False,
            "sample_count": len(valid_batch),
        }

    X_batch = valid_batch.drop(columns=[target_column])
    y_batch = valid_batch[target_column]

    preds = pipeline.predict(X_batch)
    probs = pipeline.predict_proba(X_batch) if (task_type == "classification" and hasattr(pipeline, "predict_proba")) else None

    if task_type == "classification":
        current_metrics = compute_classification_metrics(y_batch, preds, probs)
    else:
        current_metrics = compute_regression_metrics(y_batch, preds)

    cur_score = current_metrics.get(primary_metric, float("nan"))
    base_score = baseline_metrics.get(primary_metric, float("nan"))

    # Determine degradation
    degraded = False
    relative_drop = 0.0

    if np.isfinite(cur_score) and np.isfinite(base_score):
        if primary_metric in ("mae", "rmse", "mape"):
            # For error metrics, degradation occurs when current error is HIGHER than baseline
            if base_score > 0:
                relative_increase = (cur_score - base_score) / base_score
                if relative_increase >= degradation_threshold:
                    degraded = True
                    relative_drop = relative_increase
        else:
            # For score metrics (F1, Accuracy, R2), degradation occurs when score is LOWER
            if base_score > 0:
                relative_decrease = (base_score - cur_score) / base_score
                if relative_decrease >= degradation_threshold:
                    degraded = True
                    relative_drop = relative_decrease
            elif cur_score < base_score - degradation_threshold:
                degraded = True

    status_str = "degraded" if degraded else "healthy"
    message_str = (
        f"Performance degradation alert: {primary_metric} dropped from {base_score:.4f} to {cur_score:.4f}."
        if degraded
        else f"Model performance is within healthy tolerance ({primary_metric}: baseline {base_score:.4f} vs current {cur_score:.4f})."
    )

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "batch_name": batch_name,
        "has_ground_truth": True,
        "status": status_str,
        "message": message_str,
        "sample_count": len(valid_batch),
        "primary_metric": primary_metric,
        "baseline_score": base_score,
        "current_score": cur_score,
        "relative_drop": round(relative_drop, 4),
        "degradation_detected": degraded,
        "metrics": current_metrics,
        "baseline_metrics": baseline_metrics,
    }

    # Append to persistent log
    _append_monitoring_history(record)
    return record


def _append_monitoring_history(entry: dict[str, Any]) -> None:
    """Save monitoring event to metadata JSON history."""
    history = get_monitoring_history()
    history.append(entry)
    try:
        MONITORING_LOG_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")
    except Exception:
        pass


def get_monitoring_history() -> list[dict[str, Any]]:
    """Retrieve history of performance evaluations."""
    if not MONITORING_LOG_PATH.exists():
        return []
    try:
        return json.loads(MONITORING_LOG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
