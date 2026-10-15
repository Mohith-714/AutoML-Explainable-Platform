"""Modular packages for Drift-Aware AutoML Platform."""

from modules.data_processing import clean_dataframe, get_dataset_recommendations, load_csv, profile_dataframe
from modules.data_validation import detect_identifier_columns, safe_read_csv
from modules.drift_detection import compare_feature_drift
from modules.explainability import compute_permutation_importance
from modules.model_evaluation import (
    compute_classification_metrics,
    compute_regression_metrics,
)
from modules.model_registry import get_approved_model, list_all_versions, save_model_version
from modules.model_training import TrainingResult, train_models
from modules.performance_monitoring import evaluate_batch_performance
from modules.retraining import check_retraining_trigger, validate_and_promote_candidate
from modules.task_detection import detect_task_type

__all__ = [
    "load_csv",
    "safe_read_csv",
    "profile_dataframe",
    "clean_dataframe",
    "get_dataset_recommendations",
    "detect_identifier_columns",
    "detect_task_type",
    "train_models",
    "TrainingResult",
    "compute_classification_metrics",
    "compute_regression_metrics",
    "compute_permutation_importance",
    "compare_feature_drift",
    "evaluate_batch_performance",
    "check_retraining_trigger",
    "validate_and_promote_candidate",
    "save_model_version",
    "get_approved_model",
    "list_all_versions",
]
