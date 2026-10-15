"""Task detection heuristics to identify classification or regression from target characteristics."""

from __future__ import annotations

from typing import Any, Literal
import numpy as np
import pandas as pd

TaskType = Literal["classification", "regression"]


def detect_task_type(target: pd.Series) -> dict[str, Any]:
    """Analyze target series to suggest classification or regression with reasoning and warnings.
    
    Heuristics:
    - Boolean, String, Categorical, Object types -> Classification
    - Numeric types with <= 2 unique values -> Binary Classification
    - Numeric types with low cardinality relative to dataset size -> Multiclass Classification
    - Continuous floats or high-cardinality numbers -> Regression
    """
    clean_target = target.dropna()
    if clean_target.empty:
        raise ValueError("Target column contains no labeled values to analyze.")

    n_samples = len(clean_target)
    n_unique = clean_target.nunique()
    unique_vals = clean_target.unique().tolist()[:10]  # sample values
    is_numeric = pd.api.types.is_numeric_dtype(clean_target)
    is_bool = pd.api.types.is_bool_dtype(clean_target)

    warnings: list[str] = []

    if is_bool or not is_numeric:
        task_type: TaskType = "classification"
        sub_type = "binary" if n_unique == 2 else "multiclass"
        confidence = "high"
        rationale = (
            f"Target has categorical/non-numeric type '{clean_target.dtype}' with {n_unique} distinct classes."
        )
    else:
        # Numeric target
        # Check integer vs float
        is_integer_like = np.all(np.equal(np.mod(clean_target, 1), 0))
        cardinality_threshold = max(2, min(20, int(np.sqrt(n_samples))))

        if n_unique == 2:
            task_type = "classification"
            sub_type = "binary"
            confidence = "high"
            rationale = "Target contains exactly 2 unique numerical values (likely binary flags 0 and 1)."
        elif n_unique <= cardinality_threshold and is_integer_like:
            task_type = "classification"
            sub_type = "multiclass"
            confidence = "moderate"
            rationale = (
                f"Numeric target has only {n_unique} distinct integer values, "
                f"suggesting discrete class codes or rating buckets."
            )
            warnings.append(
                "If these numbers represent continuous measurements or ranked scales, "
                "you can manually override to Regression."
            )
        else:
            task_type = "regression"
            sub_type = "continuous"
            confidence = "high"
            rationale = (
                f"Target is numeric with {n_unique} unique values across {n_samples} rows, "
                f"indicating a continuous numeric quantity."
            )

    # Sanity warnings
    if task_type == "classification" and n_unique < 2:
        warnings.append("Target has only 1 distinct class; supervised learning is not feasible without multiple classes.")
    if task_type == "classification" and n_unique > 50:
        warnings.append(
            f"High number of distinct classes ({n_unique}). Consider grouping categories or verifying if this is an ID column."
        )

    # Compute class distribution & imbalance metrics for classification
    class_distribution: list[dict[str, Any]] | None = None
    is_imbalanced: bool = False
    imbalance_ratio: float = 1.0
    minority_class: str | None = None
    majority_class: str | None = None

    if task_type == "classification":
        val_counts = clean_target.value_counts(dropna=False)
        class_distribution = [
            {
                "class": str(k),
                "count": int(v),
                "percentage": round(float(v) / n_samples * 100, 2),
            }
            for k, v in val_counts.items()
        ]
        if len(val_counts) >= 2:
            min_count = int(val_counts.min())
            max_count = int(val_counts.max())
            imbalance_ratio = round(max_count / max(min_count, 1), 2)
            minority_class = str(val_counts.idxmin())
            majority_class = str(val_counts.idxmax())
            minority_pct = round(min_count / n_samples * 100, 1)

            # Flag imbalance if max/min ratio >= 1.5 or minority < 25%
            if imbalance_ratio >= 1.5 or minority_pct < 25.0:
                is_imbalanced = True
                warnings.append(
                    f"Class imbalance detected (ratio {imbalance_ratio}:1, minority class '{minority_class}' represents only "
                    f"{minority_pct}% of records). Automated class weighting ('class_weight=balanced') and macro-F1 / minority recall optimization recommended."
                )

    return {
        "task_type": task_type,
        "sub_type": sub_type,
        "confidence": confidence,
        "unique_count": n_unique,
        "sample_values": unique_vals,
        "class_distribution": class_distribution,
        "is_imbalanced": is_imbalanced,
        "imbalance_ratio": imbalance_ratio,
        "minority_class": minority_class,
        "majority_class": majority_class,
        "minority_percentage": minority_pct if (task_type == "classification" and len(val_counts) >= 2) else 0.0,
        "rationale": rationale,
        "warnings": warnings,
        "limitations_note": (
            "This platform optimizes tabular classification and regression. "
            "Complex multi-label tasks, high-frequency time-series forecasting, and raw unstructured text "
            "are not supported in this tabular AutoML engine."
        ),
    }
