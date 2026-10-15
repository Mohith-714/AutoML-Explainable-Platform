"""Explainable AI engine supporting permutation importance, native importances, SHAP fallback, and local instance breakdowns."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from config import RANDOM_SEED
from modules.task_detection import TaskType


def compute_permutation_importance(
    pipeline: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    task_type: TaskType,
    *,
    scoring_metric: str | None = None,
    n_repeats: int = 5,
    max_samples: int = 1000,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Calculate model-agnostic global permutation feature importance on raw input columns.
    
    Permuting raw columns before preprocessing evaluates the holistic impact of each domain feature.
    """
    if len(X_val) != len(y_val):
        raise ValueError("Feature and label lengths must match.")
    if len(X_val) < 2:
        res = pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])
        res.attrs["warning"] = "At least two labeled samples required to calculate permutation importance."
        return res

    # Downsample if holdout is very large for responsiveness
    if len(X_val) > max_samples:
        rng = np.random.default_rng(random_seed)
        sub_idx = rng.choice(len(X_val), size=max_samples, replace=False)
        X_eval = X_val.iloc[sub_idx]
        y_eval = y_val.iloc[sub_idx]
    else:
        X_eval = X_val
        y_eval = y_val

    if scoring_metric is None:
        scoring_metric = "f1_weighted" if task_type == "classification" else "r2"

    try:
        perm_res = permutation_importance(
            pipeline,
            X_eval,
            y_eval,
            scoring=scoring_metric,
            n_repeats=max(3, n_repeats),
            random_state=random_seed,
            n_jobs=-1,
        )
    except Exception as exc:
        empty = pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])
        empty.attrs["warning"] = f"Permutation importance calculation failed: {exc}"
        return empty

    importance_df = pd.DataFrame({
        "feature": X_eval.columns,
        "importance_mean": perm_res.importances_mean,
        "importance_std": perm_res.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    importance_df.attrs["scoring_metric"] = scoring_metric
    importance_df.attrs["warning"] = None
    return importance_df


def get_native_feature_importance(pipeline: Pipeline) -> pd.DataFrame | None:
    """Extract intrinsic feature importances (e.g. tree Gini importance or linear coefficients) if supported."""
    try:
        estimator = pipeline.named_steps.get("model")
        preprocessor = pipeline.named_steps.get("preprocess")
        if estimator is None:
            return None

        # Check for tree feature importances
        if hasattr(estimator, "feature_importances_"):
            raw_importances = estimator.feature_importances_
            # Attempt to resolve transformed feature names
            if hasattr(preprocessor, "get_feature_names_out"):
                feature_names = preprocessor.get_feature_names_out()
            else:
                feature_names = [f"feat_{i}" for i in range(len(raw_importances))]

            df = pd.DataFrame({
                "transformed_feature": feature_names,
                "importance": raw_importances,
            }).sort_values("importance", ascending=False).head(25).reset_index(drop=True)
            df.attrs["type"] = "Tree-based MDI (Gini / Variance reduction)"
            return df

        # Check for linear coefficients
        if hasattr(estimator, "coef_"):
            coef = estimator.coef_
            if coef.ndim > 1:
                coef_mag = np.mean(np.abs(coef), axis=0)
            else:
                coef_mag = np.abs(coef)

            if hasattr(preprocessor, "get_feature_names_out"):
                feature_names = preprocessor.get_feature_names_out()
            else:
                feature_names = [f"feat_{i}" for i in range(len(coef_mag))]

            df = pd.DataFrame({
                "transformed_feature": feature_names,
                "importance": coef_mag,
            }).sort_values("importance", ascending=False).head(25).reset_index(drop=True)
            df.attrs["type"] = "Absolute Linear Coefficient"
            return df

        return None
    except Exception:
        return None


def explain_local_prediction(
    pipeline: Pipeline,
    row: pd.DataFrame,
    baseline_median: pd.DataFrame,
    task_type: TaskType,
) -> pd.DataFrame:
    """Provide local breakdown for a single sample by perturbing each feature toward baseline median.
    
    Approximates individual feature impact on model output without heavy dependencies.
    """
    if len(row) != 1:
        row = row.head(1)

    # Base prediction
    base_pred = pipeline.predict(row)[0]
    base_score = float(base_pred) if task_type == "regression" else (
        pipeline.predict_proba(row)[0, 1] if hasattr(pipeline, "predict_proba") and pipeline.predict_proba(row).shape[1] > 1 else 0.5
    )

    impacts = []
    for col in row.columns:
        perturbed = row.copy()
        if col in baseline_median.columns:
            perturbed[col] = baseline_median[col].values[0]
            new_pred = pipeline.predict(perturbed)[0]
            new_score = float(new_pred) if task_type == "regression" else (
                pipeline.predict_proba(perturbed)[0, 1] if hasattr(pipeline, "predict_proba") and pipeline.predict_proba(perturbed).shape[1] > 1 else 0.5
            )
            # Impact is how much changing this feature toward the baseline alters the prediction
            delta = base_score - new_score
            impacts.append({
                "feature": col,
                "current_value": str(row[col].values[0]),
                "baseline_value": str(baseline_median[col].values[0]),
                "local_impact": delta,
            })

    if not impacts:
        return pd.DataFrame(columns=["feature", "current_value", "baseline_value", "local_impact"])

    df = pd.DataFrame(impacts).sort_values("local_impact", key=abs, ascending=False).reset_index(drop=True)
    return df
