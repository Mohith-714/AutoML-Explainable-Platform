"""Leakage-safe multi-algorithm AutoML training engine with timing and holdout ranking."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.class_weight import compute_sample_weight

from config import RANDOM_SEED
from modules.data_processing import build_preprocessor, split_features_target
from modules.model_evaluation import (
    compute_classification_metrics,
    compute_regression_metrics,
)
from modules.task_detection import TaskType, detect_task_type


@dataclass
class TrainingResult:
    """Artifacts, test sets, evaluation scores, and fitted pipelines from AutoML training."""

    target_column: str
    task_type: TaskType
    best_model_name: str
    best_model: Pipeline
    models: dict[str, Pipeline]
    metrics: dict[str, dict[str, float]]
    comparison: pd.DataFrame
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    train_duration_seconds: dict[str, float]
    primary_metric: str


def get_candidate_models(task_type: TaskType, random_seed: int = RANDOM_SEED) -> list[tuple[str, Any, bool]]:
    """Return list of (model_name, estimator_instance, needs_dense_preprocessor)."""
    if task_type == "classification":
        return [
            (
                "Logistic Regression",
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_seed),
                False,
            ),
            (
                "Decision Tree",
                DecisionTreeClassifier(class_weight="balanced", min_samples_leaf=3, random_state=random_seed),
                False,
            ),
            (
                "Random Forest",
                RandomForestClassifier(
                    n_estimators=100, class_weight="balanced", min_samples_leaf=3, random_state=random_seed, n_jobs=-1
                ),
                False,
            ),
            (
                "HistGradientBoosting",
                HistGradientBoostingClassifier(class_weight="balanced", min_samples_leaf=10, random_state=random_seed),
                True,
            ),
        ]
    return [
        (
            "Linear Regression",
            LinearRegression(),
            False,
        ),
        (
            "Decision Tree Regressor",
            DecisionTreeRegressor(random_state=random_seed),
            False,
        ),
        (
            "Random Forest Regressor",
            RandomForestRegressor(n_estimators=100, random_state=random_seed, n_jobs=-1),
            False,
        ),
        (
            "HistGradientBoosting Regressor",
            HistGradientBoostingRegressor(random_state=random_seed),
            True,
        ),
    ]


def train_models(
    frame: pd.DataFrame,
    target_column: str,
    *,
    feature_columns: list[str] | None = None,
    task_type: TaskType | Literal["auto"] = "auto",
    primary_metric: str | None = None,
    test_size: float = 0.2,
    random_seed: int = RANDOM_SEED,
    algorithms_to_run: list[str] | None = None,
) -> TrainingResult:
    """Execute end-to-end AutoML pipeline: train/test split, pipeline construction, fit, and evaluation."""
    if not (0.05 <= test_size <= 0.5):
        raise ValueError("Holdout test split ratio must be between 5% and 50%.")

    X, y = split_features_target(frame, target_column, feature_columns=feature_columns)

    # Detect or confirm task type
    if task_type == "auto":
        detection_res = detect_task_type(y)
        resolved_task: TaskType = detection_res["task_type"]
    else:
        resolved_task = task_type

    if resolved_task == "classification":
        if y.nunique() < 2:
            raise ValueError("Classification requires at least two distinct target classes.")
        default_primary = "f1_macro"
    else:
        if not pd.api.types.is_numeric_dtype(y):
            raise ValueError(f"Regression target '{target_column}' must be numeric.")
        default_primary = "r2"

    eval_metric = primary_metric or default_primary

    # Stratification check for classification
    stratify = None
    if resolved_task == "classification":
        class_counts = y.value_counts()
        test_samples = int(np.ceil(len(y) * test_size))
        train_samples = len(y) - test_samples
        n_classes = len(class_counts)
        if class_counts.min() >= 2 and test_samples >= n_classes and train_samples >= n_classes:
            stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=stratify
    )

    all_candidates = get_candidate_models(resolved_task, random_seed=random_seed)
    if algorithms_to_run:
        candidates = [c for c in all_candidates if c[0] in algorithms_to_run]
        if not candidates:
            candidates = all_candidates
    else:
        candidates = all_candidates

    fitted_pipelines: dict[str, Pipeline] = {}
    metrics_by_model: dict[str, dict[str, float]] = {}
    durations: dict[str, float] = {}
    leaderboard_rows: list[dict[str, Any]] = []
    failure_errors: list[str] = []

    for name, estimator, needs_dense in candidates:
        t0 = time.perf_counter()
        preprocessor = build_preprocessor(X_train, dense=needs_dense)
        pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])

        try:
            fit_kwargs: dict[str, Any] = {}
            if resolved_task == "classification" and name == "HistGradientBoosting":
                fit_kwargs["model__sample_weight"] = compute_sample_weight("balanced", y_train)

            pipeline.fit(X_train, y_train, **fit_kwargs)
            duration = time.perf_counter() - t0
            durations[name] = round(duration, 3)

            # Predict and evaluate on holdout set
            preds = pipeline.predict(X_test)
            probs = pipeline.predict_proba(X_test) if (resolved_task == "classification" and hasattr(pipeline, "predict_proba")) else None

            if resolved_task == "classification":
                scores = compute_classification_metrics(y_test, preds, probs)
            else:
                scores = compute_regression_metrics(y_test, preds)

            fitted_pipelines[name] = pipeline
            metrics_by_model[name] = scores

            row = {
                "model": name,
                "status": "success",
                "training_time_sec": durations[name],
                **scores,
            }
            leaderboard_rows.append(row)

        except Exception as exc:
            duration = time.perf_counter() - t0
            durations[name] = round(duration, 3)
            failure_errors.append(f"{name}: {exc}")
            leaderboard_rows.append({
                "model": name,
                "status": "failed",
                "training_time_sec": durations[name],
                "error": str(exc),
            })

    if not fitted_pipelines:
        raise RuntimeError("All candidate models failed during training: " + " | ".join(failure_errors))

    # Leaderboard DataFrame
    leaderboard_df = pd.DataFrame(leaderboard_rows)
    if not leaderboard_df.empty and eval_metric in leaderboard_df.columns:
        ascending = eval_metric in ("mae", "rmse", "mape")
        leaderboard_df = leaderboard_df.sort_values(by=eval_metric, ascending=ascending).reset_index(drop=True)

    # Select best model based on primary metric
    def get_score(m_name: str) -> float:
        val = metrics_by_model[m_name].get(eval_metric, float("nan"))
        # For error metrics, lower is better (negate score for sorting)
        if eval_metric in ("mae", "rmse", "mape"):
            return -val if np.isfinite(val) else -float("inf")
        return val if np.isfinite(val) else -float("inf")

    best_model_name = max(fitted_pipelines.keys(), key=get_score)

    return TrainingResult(
        target_column=target_column,
        task_type=resolved_task,
        best_model_name=best_model_name,
        best_model=fitted_pipelines[best_model_name],
        models=fitted_pipelines,
        metrics=metrics_by_model,
        comparison=leaderboard_df,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        train_duration_seconds=durations,
        primary_metric=eval_metric,
    )
