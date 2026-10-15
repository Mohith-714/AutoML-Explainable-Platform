# Drift-Aware AutoML Platform with Explainable AI and Automated Model Lifecycle Management

A production-grade, modular, domain-independent machine learning web platform built from scratch in Python and Streamlit. The system unifies automated machine learning (AutoML), explainable AI (XAI), statistical feature drift detection, ground-truth performance tracking, controlled validation-gated candidate retraining, and an audited local model registry.

---

## 📑 Table of Contents

1. [Problem Statement & Motivation](#problem-statement--motivation)
2. [Key Capabilities & Innovations](#key-capabilities--innovations)
3. [Architecture & Folder Structure](#architecture--folder-structure)
4. [Technology Stack](#technology-stack)
5. [Windows Setup & Installation](#windows-setup--installation)
6. [Step-by-Step Demonstration Walkthrough](#step-by-step-demonstration-walkthrough)
7. [Mathematical & Statistical Formulations](#mathematical--statistical-formulations)
8. [Automated Test Suite](#automated-test-suite)
9. [Academic Research & Final-Year Project Guide](#academic-research--final-year-project-guide)
10. [Limitations & Responsible AI Disclaimer](#limitations--responsible-ai-disclaimer)

---

## 1. Problem Statement & Motivation

Traditional machine learning workflows focus predominantly on offline model training. However, in production domains such as manufacturing, healthcare, banking, retail, and energy, real-world data distributions change dynamically:
- Sensor calibration degrades over time.
- Economic conditions shift consumer habits.
- Operational environments change input distributions.

Blindly retraining a model whenever input shifts occur can introduce model instability or catastrophic forgetting. Conversely, ignoring drift causes silent performance degradation.

This platform bridges this gap by enforcing **responsible MLOps principles**:
- **Distinguishes three distinct phenomena**:
  1. *Data Drift*: Shifts in input feature distributions.
  2. *Concept Drift*: Changes in the relationship between input features and target outcomes.
  3. *Performance Degradation*: Measurable decline in evaluation metrics against actual ground-truth labels.
- **Enforces Validation Gates**: A newly retrained candidate model is **never** promoted merely because drift was detected. It must compete against the currently approved model on identical held-out validation data and satisfy non-inferiority criteria.
- **Guarantees Model Preservation**: If a candidate model fails validation, the existing approved model remains strictly intact.

---

## 2. Key Capabilities & Innovations

- **Automated Ingestion & Leakage Prevention**:
  - Safe parsing of CSV files with encoding detection.
  - Automatic detection of database keys and high-cardinality ID columns (`user_id`, `uuid`, `index`).
  - Preprocessing transformations (`ColumnTransformer`) fitted strictly on training data splits.
- **Smart Task Detection**:
  - Distinguishes Binary Classification, Multiclass Classification, and Regression using distribution heuristics.
  - Transparent rationale and user-override capabilities.
- **Multi-Algorithm AutoML Engine**:
  - Benchmarks Logistic/Linear Regression, Decision Trees, Random Forests, and HistGradientBoosting.
  - Automatically handles missing values and unseen categorical levels (`handle_unknown='ignore'`).
  - Computes Accuracy, F1 (weighted & macro), Precision, Recall, ROC-AUC, MAE, RMSE, and R².
- **Explainable AI (XAI)**:
  - Global Permutation Feature Importance with standard error bounds.
  - Native estimator feature importance (tree split importances & linear coefficients).
  - Local instance contribution breakdown for individual predictions.
- **Statistical Drift Detection**:
  - Numerical features: Two-Sample Kolmogorov-Smirnov test (p-value) + Population Stability Index (PSI).
  - Categorical features: Chi-Square contingency test + Total Variation Distance (TVD).
  - Missingness drift detection (shift in missing proportion $\ge 15\%$).
  - Schema drift alerts (missing columns, extra columns, data type mismatches).
- **Performance Degradation Tracking**:
  - Matches predictions to ground-truth labels when available.
  - Alerts on relative drops in primary evaluation metrics.
- **Controlled Retraining & Model Registry**:
  - Audit trail of model versions (`artifacts/models/v_*`).
  - Validation gate comparing candidate vs approved model.
  - Rollback to previous approved models with one click.
  - Single observation inference form and batch prediction CSV exporter.
- **Executive Reporting**:
  - Standalone, downloadable HTML audit report and CSV data exports.

---

## 3. Architecture & Folder Structure

```
AutoML-Explainable-Platform/
├── app.py                      # Multi-page Streamlit dashboard application
├── config.py                   # Centralized configuration (thresholds, paths, seeds)
├── requirements.txt            # Python package dependencies
├── README.md                   # Documentation, setup guide, and user manual
├── .gitignore                  # Git exclusions for models, logs, and uploads
│
├── modules/                    # Modular Python backend engines
│   ├── __init__.py             # Module exports
│   ├── data_validation.py      # File & schema validation, leakage detection, ID warnings
│   ├── data_processing.py      # Ingestion, profiling, cleaning, ColumnTransformer
│   ├── task_detection.py       # Heuristic ML task detection (Classification vs Regression)
│   ├── model_training.py       # AutoML training engine & candidate comparison
│   ├── model_evaluation.py     # Metrics (F1, ROC-AUC, R², MAE) and Plotly charts
│   ├── explainability.py       # Permutation importance, native weights, local breakdown
│   ├── drift_detection.py      # Statistical drift engine (KS, PSI, Chi², TVD)
│   ├── performance_monitoring.py # Labeled batch evaluation and degradation alerts
│   ├── retraining.py           # Controlled retraining engine & validation gates
│   ├── model_registry.py       # Versioned filesystem model registry & rollback
│   └── reporting.py            # Standalone HTML/Markdown executive summary generator
│
├── tests/                      # Automated test suite
│   ├── __init__.py
│   ├── test_data_processing.py # Ingestion, validation, profiling, and cleaning tests
│   ├── test_task_detection.py  # Task detection heuristic tests
│   ├── test_model_training.py  # Multi-algorithm training & metrics tests
│   ├── test_drift_detection.py # Statistical tests (KS, PSI, Chi², TVD, schema drift)
│   └── test_retraining.py      # Retraining triggers, validation gates, and rollback
│
├── data/                       # Tabular data storage
│   ├── uploads/                # User uploaded datasets
│   ├── reference/              # Established baseline reference datasets
│   ├── monitoring/             # Incoming batches for drift monitoring
│   └── sample/                 # Synthetic domain datasets (manufacturing, churn, housing)
│       ├── generate_samples.py # Script generating reproducible domain benchmarks
│       ├── manufacturing_reference.csv
│       ├── manufacturing_monitoring_stable.csv
│       ├── manufacturing_monitoring_drifted.csv
│       ├── customer_churn_sample.csv
│       └── housing_price_sample.csv
│
├── artifacts/                  # Persisted machine learning assets
│   ├── models/                 # Model versions & joblib pipelines (v_YYYYMMDD_.../model.joblib)
│   └── metadata/               # Registry pointers, monitoring history, and audit logs
│
└── reports/                    # Exported run reports & summaries
```

---

## 4. Technology Stack

- **Application Frontend**: Streamlit (v1.40+) with custom CSS design tokens and responsive cards.
- **Interactive Visualizations**: Plotly (v5.20+) for dark-themed heatmaps, ROC curves, and distribution histograms.
- **Core ML & Data Processing**: Scikit-learn (v1.4+), Pandas (v2.1+), NumPy (v1.26+).
- **Statistical Computing**: SciPy (v1.11+) (`scipy.stats.ks_2samp`, `scipy.stats.chi2_contingency`).
- **Model Serialization**: Joblib (v1.3+).
- **Quality Assurance**: Pytest (v8.0+).

---

## 5. Windows Setup & Installation

Follow these exact PowerShell commands in VS Code on Windows:

### Step 1: Open Terminal in Project Directory
Launch VS Code and open PowerShell in the project directory:
```powershell
cd "c:\Users\ADMIN\Desktop\AutoML Explainable Platform"
```

### Step 2: Virtual Environment Setup (Optional or using existing Python)
If PowerShell blocks virtual environment script activation (`Activate.ps1`), apply a temporary, process-scoped execution policy:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```
Create and activate your virtual environment:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Required Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Run Automated Tests
Verify complete platform health:
```powershell
python -m pytest tests -v
```

### Step 5: Launch the Streamlit Platform
```powershell
python -m streamlit run app.py
```
*(Alternatively, if `streamlit` is on your PATH: `streamlit run app.py`)*

The application will launch and open in your default browser at:
`http://localhost:8501`

---

## 6. Step-by-Step Demonstration Walkthrough

You can test the entire lifecycle in under 2 minutes using the built-in synthetic datasets:

### Phase 1: Ingestion & Profiling
1. On **Home Dashboard**, click **Load Manufacturing Sensor Dataset**.
2. Go to **Data Profiling**:
   - Inspect the 1,000 rows, distribution histograms, and correlation heatmap.
   - Note the detection of `machine_id` as an identifier column.
3. Go to **Data Cleaning**:
   - Confirm `machine_id` is selected for removal.
   - Click **Apply Cleaning Rules**.

### Phase 2: AutoML Model Training
4. Go to **AutoML Training**:
   - Target is auto-set to `defect_detected` (Classification).
   - Click **Run AutoML Model Benchmark**.
   - Watch the leaderboard train Logistic Regression, Decision Tree, Random Forest, and HistGradientBoosting.
   - Note the holdout F1-scores, accuracy, and training times.

### Phase 3: Explainability
5. Go to **Explainable AI**:
   - Inspect **Global Permutation Importance** to observe that `vibration_hz` and `temperature_c` are the primary predictive drivers.
   - Switch to **Local Instance Breakdown** to observe how individual feature values influence single defect predictions.

### Phase 4: Model Registration
6. Go to **Retraining & Registry** -> **Model Registry & Versions**:
   - Save the best candidate as the approved production version (`v1`).

### Phase 5: Drift Monitoring & Detection
7. Go to **Drift Monitoring**:
   - Baseline reference is established from the training data.
   - Under *Demonstration Batches*, click **Test Drifted Batch (Manufacturing Sensors)**.
   - Observe that `vibration_hz` and `temperature_c` cross the PSI ($\ge 0.20$) and KS-test ($p < 0.001$) thresholds.
   - Notice the system status changes to **DRIFT DETECTED** and shows overlaid distribution histograms.

### Phase 6: Controlled Retraining & Validation Gate
8. Go to **Retraining & Registry**:
   - Notice the diagnostic card indicates: *Retraining Condition Met*.
   - Check *Use loaded monitoring batch as labeled retraining dataset*.
   - Click **Train Candidate & Run Validation Gate**.
   - Observe the validation gate: the candidate model is evaluated on held-out validation data alongside the currently approved model.
   - If the candidate improves on the baseline, it is promoted; if it is inferior, the existing approved model is **strictly preserved**.

### Phase 7: Export Executive Report
9. Go to **Reports & Export**:
   - Click **Generate Executive Report**.
   - Download the standalone HTML report and review the complete audit summary.

---

## 7. Mathematical & Statistical Formulations

### Population Stability Index (PSI)
Used to quantify numerical distribution shifts without requiring parametric assumptions:
$$\text{PSI} = \sum_{b=1}^{B} \left( P_b - Q_b \right) \times \ln\left(\frac{P_b}{Q_b}\right)$$
Where:
- $P_b$ is the proportion of observations in quantile bin $b$ in the current batch.
- $Q_b$ is the proportion in quantile bin $b$ in the reference baseline.
- Interpretation: $\text{PSI} < 0.10$ (stable), $0.10 \le \text{PSI} < 0.20$ (moderate shift), $\text{PSI} \ge 0.20$ (significant drift).

### Kolmogorov-Smirnov Test
Two-sample non-parametric test comparing continuous empirical cumulative distribution functions $F_1(x)$ and $F_2(x)$:
$$D = \sup_x |F_1(x) - F_2(x)|$$

### Total Variation Distance (TVD)
Used for categorical feature distributions:
$$\text{TVD} = \frac{1}{2} \sum_{c \in \mathcal{C}} |P(c) - Q(c)|$$

### Permutation Feature Importance
Measures feature importance as the decrease in model score $S$ when feature $j$ is randomly permuted:
$$I(f_j) = S(X, y) - S(X^{\text{perm } j}, y)$$

---

## 8. Automated Test Suite

The test suite covers:
- Ingestion of valid/malformed/empty CSVs and file size limits.
- High-cardinality identifier column detection and data leakage prevention.
- Schema comparison between reference and incoming datasets.
- Task detection heuristics (Binary Classification, Multiclass, Continuous Regression).
- End-to-end model training for both classification and regression pipelines.
- Statistical drift detection (PSI, KS test, Chi-Square, TVD, missingness shifts, and schema drift).
- Controlled retraining validation gates, non-inferiority checks, and approved model preservation.
- Model registry version tracking and rollback functionality.

Run all tests via:
```powershell
python -m pytest tests -v
```

---

## 9. Academic Research & Final-Year Project Guide

This platform is structured for final-year Computer Science / Data Science capstone projects and academic research papers:

### Potential Research Questions
1. *Effectiveness of Multi-Metric Drift Gating*: Does combining Statistical Significance ($p$-value) with Effect Size (PSI/TVD) reduce false retraining triggers compared to $p$-value tests alone?
2. *Non-Inferiority Retraining Thresholds*: What impact does the parameter $\epsilon$ (minimum improvement) have on model stability in high-frequency drift environments?
3. *Domain-Independent Heuristics*: How accurately do automated task detection heuristics perform across diverse real-world open-data repositories?

---

## 10. Limitations & Responsible AI Disclaimer

- **Batch vs Continuous Streaming**: This implementation is designed for batch-oriented monitoring via CSV uploads. It does not connect to continuous message queues (e.g., Apache Kafka) out of the box.
- **Local Filesystem Scope**: Designed for local execution on Windows without requiring cloud databases or paid APIs.
- **Privacy Notice**: The prototype is not pre-certified for regulated HIPAA or PCI-DSS production environments without supplementary encryption and authentication layers.
- **Correlation vs Causation**: Feature importances describe the internal behavior and sensitivities of the trained estimator; they do not establish real-world causal mechanisms.
