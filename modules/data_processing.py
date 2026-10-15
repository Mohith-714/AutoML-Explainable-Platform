"""CSV ingestion, data profiling, cleaning, and leakage-safe preprocessing pipelines."""

from __future__ import annotations

from typing import Any, Literal
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from config import MAX_UPLOAD_BYTES
from modules.data_validation import detect_identifier_columns, safe_read_csv, validate_target_column


def load_csv(source: Any, max_bytes: int = MAX_UPLOAD_BYTES) -> pd.DataFrame:
    """Safe ingestion of CSV files via validation module."""
    return safe_read_csv(source, max_bytes=max_bytes)


def profile_dataframe(frame: pd.DataFrame) -> dict[str, Any]:
    """Calculate comprehensive summary statistics, missing values, duplicates, and column types."""
    if frame.empty:
        raise ValueError("Cannot profile an empty dataset.")

    n_rows, n_cols = frame.shape
    missing_series = frame.isna().sum()
    missing_pct = (missing_series / n_rows) * 100

    numerical_cols = frame.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in frame.columns if c not in numerical_cols]
    constant_cols = [c for c in frame.columns if frame[c].nunique(dropna=False) <= 1]
    identifier_warnings = detect_identifier_columns(frame)

    missing_df = pd.DataFrame({
        "missing_count": missing_series,
        "missing_pct": missing_pct.round(2),
    }).sort_values("missing_count", ascending=False)

    dtypes_df = pd.DataFrame({
        "column": frame.columns,
        "dtype": frame.dtypes.astype(str),
        "nunique": frame.nunique(),
    })

    # Memory calculation
    mem_bytes = frame.memory_usage(deep=True).sum()
    mem_mb = mem_bytes / (1024 * 1024)

    return {
        "shape": (n_rows, n_cols),
        "n_rows": n_rows,
        "n_cols": n_cols,
        "memory_mb": round(mem_mb, 2),
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing_table": missing_df,
        "dtypes_table": dtypes_df,
        "summary": frame.describe(include="all").transpose(),
        "numerical_columns": numerical_cols,
        "categorical_columns": categorical_cols,
        "constant_columns": constant_cols,
        "identifier_warnings": identifier_warnings,
    }


def clean_dataframe(
    frame: pd.DataFrame,
    *,
    drop_duplicates: bool = True,
    target_column: str | None = None,
    drop_columns: list[str] | None = None,
    drop_constant_columns: bool = False,
    max_missing_col_pct: float | None = None,
) -> pd.DataFrame:
    """Apply deterministic data cleaning rules while preserving raw feature nulls for pipeline imputation."""
    cleaned = frame.copy()

    # Drop explicit unwanted columns (e.g. user selected ID columns)
    if drop_columns:
        valid_drops = [c for c in drop_columns if c in cleaned.columns and c != target_column]
        if valid_drops:
            cleaned = cleaned.drop(columns=valid_drops)

    # Drop constant columns if requested
    if drop_constant_columns:
        consts = [c for c in cleaned.columns if cleaned[c].nunique(dropna=False) <= 1 and c != target_column]
        if consts:
            cleaned = cleaned.drop(columns=consts)

    # Drop columns with excessive missing percentages if requested
    if max_missing_col_pct is not None and 0 < max_missing_col_pct < 100:
        high_missing = [
            c for c in cleaned.columns
            if (cleaned[c].isna().mean() * 100 > max_missing_col_pct) and c != target_column
        ]
        if high_missing:
            cleaned = cleaned.drop(columns=high_missing)

    # Drop rows missing the target variable
    if target_column is not None:
        if target_column not in cleaned.columns:
            raise ValueError(f"Target column '{target_column}' was not found in dataframe.")
        cleaned = cleaned.dropna(subset=[target_column])

    # Drop duplicate rows
    if drop_duplicates:
        cleaned = cleaned.drop_duplicates()

    if cleaned.empty:
        raise ValueError("Cleaning removed every row; please review selected cleaning criteria.")

    return cleaned.reset_index(drop=True)


def split_features_target(
    frame: pd.DataFrame,
    target_column: str,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate feature matrix X and target vector y, asserting non-empty subsets."""
    validate_target_column(frame, target_column)
    labeled = frame.dropna(subset=[target_column]).copy()
    if feature_columns is not None:
        valid_features = [c for c in feature_columns if c in labeled.columns and c != target_column]
        if not valid_features:
            raise ValueError(
                f"None of the specified feature columns are valid or present in dataset (excluding target '{target_column}')."
            )
        features = labeled[valid_features].copy()
    else:
        features = labeled.drop(columns=[target_column]).copy()
    if features.shape[1] == 0:
        raise ValueError("The dataset has no feature columns remaining after extracting the target.")
    return features, labeled[target_column].copy()


def get_dataset_recommendations(frame: pd.DataFrame, dataset_name: str | None = None) -> dict[str, Any]:
    """Provide dataset-aware recommendations for target variable, predictive features, and exclusions.

    Acts as an intelligent assistant for known benchmarks (such as AI4I 2020 Predictive Maintenance)
    while dynamically detecting identifiers and potential leakage on arbitrary user datasets.
    """
    cols = frame.columns.tolist()
    name_lower = (dataset_name or "").lower()

    # 1. AI4I 2020 Predictive Maintenance Benchmark check
    is_ai4i = "ai4i" in name_lower or {"Machine failure", "UDI", "TWF", "HDF"}.issubset(set(cols))
    if is_ai4i:
        rec_target = "Machine failure" if "Machine failure" in cols else None
        rec_features = [
            c for c in [
                "Type",
                "Air temperature [K]",
                "Process temperature [K]",
                "Rotational speed [rpm]",
                "Torque [Nm]",
                "Tool wear [min]",
            ]
            if c in cols
        ]
        excluded = [
            c for c in ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]
            if c in cols
        ]
        exclusion_reasons = {
            "UDI": "Row sequence identifier (100% unique ID)",
            "Product ID": "High-cardinality serial identifier",
            "TWF": "Tool Wear Failure mode (target-leaking indicator)",
            "HDF": "Heat Dissipation Failure mode (target-leaking indicator)",
            "PWF": "Power Failure mode (target-leaking indicator)",
            "OSF": "Overstrain Failure mode (target-leaking indicator)",
            "RNF": "Random Failure mode (target-leaking indicator)",
        }
        return {
            "is_known_benchmark": True,
            "benchmark_name": "AI4I 2020 Predictive Maintenance",
            "recommended_target": rec_target,
            "recommended_features": rec_features,
            "excluded_features": excluded,
            "exclusion_reasons": exclusion_reasons,
            "notes": (
                "For AI4I 2020, failure-mode indicators (TWF, HDF, PWF, OSF, RNF) and identifiers (UDI, Product ID) "
                "are excluded from predictive features to prevent data leakage and memorization."
            ),
        }

    # 2. Manufacturing Reference Benchmark
    if "manufacturing_reference" in name_lower or {"defect_detected", "vibration_amplitude"}.issubset(set(cols)):
        rec_target = "defect_detected" if "defect_detected" in cols else None
        id_info = detect_identifier_columns(frame)
        id_cols = {d["column"] for d in id_info}
        rec_features = [c for c in cols if c != rec_target and c not in id_cols]
        excluded = list(id_cols)
        exclusion_reasons = {d["column"]: d["reason"] for d in id_info}
        return {
            "is_known_benchmark": True,
            "benchmark_name": "Manufacturing Sensor Benchmark",
            "recommended_target": rec_target,
            "recommended_features": rec_features,
            "excluded_features": excluded,
            "exclusion_reasons": exclusion_reasons,
            "notes": "Target set to defect_detected with ID columns excluded.",
        }

    # 3. Customer Churn Benchmark
    if "churn" in name_lower or {"churn", "monthly_charges"}.issubset(set(cols)):
        rec_target = "churn" if "churn" in cols else None
        id_info = detect_identifier_columns(frame)
        id_cols = {d["column"] for d in id_info}
        rec_features = [c for c in cols if c != rec_target and c not in id_cols]
        return {
            "is_known_benchmark": True,
            "benchmark_name": "Customer Churn Benchmark",
            "recommended_target": rec_target,
            "recommended_features": rec_features,
            "excluded_features": list(id_cols),
            "exclusion_reasons": {d["column"]: d["reason"] for d in id_info},
            "notes": "Target set to churn with ID columns excluded.",
        }

    # 4. Housing Price Benchmark
    if "housing" in name_lower or {"price_usd", "square_meters"}.issubset(set(cols)):
        rec_target = "price_usd" if "price_usd" in cols else None
        id_info = detect_identifier_columns(frame)
        id_cols = {d["column"] for d in id_info}
        rec_features = [c for c in cols if c != rec_target and c not in id_cols]
        return {
            "is_known_benchmark": True,
            "benchmark_name": "Housing Valuation Benchmark",
            "recommended_target": rec_target,
            "recommended_features": rec_features,
            "excluded_features": list(id_cols),
            "exclusion_reasons": {d["column"]: d["reason"] for d in id_info},
            "notes": "Target set to price_usd with ID columns excluded.",
        }

    # 5. Generic Dataset: Analyze dynamically
    id_info = detect_identifier_columns(frame)
    id_cols = {d["column"] for d in id_info}
    target_candidates = [
        c for c in cols
        if any(term in str(c).lower() for term in ["target", "label", "outcome", "defect", "churn", "failure", "class", "status", "response"])
        and c not in id_cols
    ]
    rec_target = target_candidates[0] if target_candidates else None
    rec_features = [c for c in cols if c != rec_target and c not in id_cols]

    return {
        "is_known_benchmark": False,
        "benchmark_name": None,
        "recommended_target": rec_target,
        "recommended_features": rec_features,
        "excluded_features": list(id_cols),
        "exclusion_reasons": {d["column"]: d["reason"] for d in id_info},
        "notes": "Dynamic heuristic recommendations based on column profiling.",
    }


def build_preprocessor(
    X: pd.DataFrame,
    *,
    numerical_strategy: Literal["median", "mean"] = "median",
    scale_numeric: bool = True,
    dense: bool = False,
    max_categories: int | None = None,
) -> ColumnTransformer:
    """Construct an automated, leakage-free ColumnTransformer for mixed tabular data.
    
    Fits strictly on training splits to prevent data leakage.
    Handles unseen categories with handle_unknown='ignore'.
    """
    numerical_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numerical_cols]

    transformers: list[tuple[str, Any, list[str]]] = []

    if numerical_cols:
        num_steps: list[tuple[str, Any]] = [
            ("imputer", SimpleImputer(strategy=numerical_strategy, keep_empty_features=True))
        ]
        if scale_numeric:
            num_steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(num_steps), numerical_cols))

    if categorical_cols:
        cat_steps: list[tuple[str, Any]] = [
            ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=not dense,
                    max_categories=max_categories if dense else None,
                ),
            ),
        ]
        transformers.append(("cat", Pipeline(cat_steps), categorical_cols))

    return ColumnTransformer(transformers, remainder="drop")
