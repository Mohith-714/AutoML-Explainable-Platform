"""Centralized configuration for the Drift-Aware AutoML Platform.

Stores filesystem paths, statistical thresholds, model defaults, and random seeds.
"""

from __future__ import annotations

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
REFERENCE_DIR = DATA_DIR / "reference"
MONITORING_DIR = DATA_DIR / "monitoring"
SAMPLE_DIR = DATA_DIR / "sample"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METADATA_DIR = ARTIFACTS_DIR / "metadata"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure runtime directories exist
for directory in [
    UPLOADS_DIR,
    REFERENCE_DIR,
    MONITORING_DIR,
    SAMPLE_DIR,
    MODELS_DIR,
    METADATA_DIR,
    REPORTS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# General Limits & Constants
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
RANDOM_SEED = 42

# Drift Detection Defaults
DEFAULT_NUMERICAL_PSI_THRESHOLD = 0.2
DEFAULT_CATEGORICAL_TVD_THRESHOLD = 0.1
DEFAULT_P_VALUE_ALPHA = 0.05
DEFAULT_MINIMUM_DRIFT_SAMPLES = 20

# Performance Monitoring & Retraining Defaults
DEFAULT_PERFORMANCE_DEGRADATION_THRESHOLD = 0.05  # 5% relative drop
DEFAULT_MINIMUM_IMPROVEMENT_FOR_PROMOTION = 0.0   # Non-inferiority or improvement
DEFAULT_MINIMUM_LABELED_RETRAIN_SAMPLES = 20

# Supported Task & Algorithms
CLASSIFICATION_ALGORITHMS = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "HistGradientBoosting",
]

REGRESSION_ALGORITHMS = [
    "Linear Regression",
    "Decision Tree Regressor",
    "Random Forest Regressor",
    "HistGradientBoosting Regressor",
]
