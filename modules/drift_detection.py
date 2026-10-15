"""Statistical feature drift detection (KS-test, PSI, Chi-Square, TVD, and schema drift)."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import chi2_contingency, ks_2samp

from config import (
    DEFAULT_CATEGORICAL_TVD_THRESHOLD,
    DEFAULT_MINIMUM_DRIFT_SAMPLES,
    DEFAULT_NUMERICAL_PSI_THRESHOLD,
    DEFAULT_P_VALUE_ALPHA,
)


def calculate_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Calculate Population Stability Index (PSI) for numerical distributions using reference quantile bins."""
    ref_clean = reference[np.isfinite(reference)]
    cur_clean = current[np.isfinite(current)]
    if len(ref_clean) < 2 or len(cur_clean) < 2:
        return float("nan")

    # Construct quantile bin edges from reference data
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    raw_edges = np.quantile(ref_clean, quantiles)
    unique_interior = np.unique(raw_edges[1:-1])
    edges = np.concatenate(([-np.inf], unique_interior, [np.inf]))

    ref_counts = np.histogram(ref_clean, bins=edges)[0].astype(float)
    cur_counts = np.histogram(cur_clean, bins=edges)[0].astype(float)

    ref_total = max(ref_counts.sum(), 1.0)
    cur_total = max(cur_counts.sum(), 1.0)

    # Use epsilon clipping to avoid division by zero or log(0)
    ref_share = np.clip(ref_counts / ref_total, 1e-4, None)
    cur_share = np.clip(cur_counts / cur_total, 1e-4, None)

    psi_val = np.sum((cur_share - ref_share) * np.log(cur_share / ref_share))
    return float(np.clip(psi_val, 0.0, 50.0))


def calculate_categorical_distance(ref_series: pd.Series, cur_series: pd.Series) -> tuple[float, float]:
    """Calculate Total Variation Distance (TVD) and Chi-Square p-value for categorical distributions."""
    ref_clean = ref_series.astype(str).fillna("<missing>")
    cur_clean = cur_series.astype(str).fillna("<missing>")

    all_cats = sorted(set(ref_clean.unique()) | set(cur_clean.unique()))
    ref_counts = ref_clean.value_counts().reindex(all_cats, fill_value=0).to_numpy(dtype=float)
    cur_counts = cur_clean.value_counts().reindex(all_cats, fill_value=0).to_numpy(dtype=float)

    ref_share = ref_counts / max(ref_counts.sum(), 1.0)
    cur_share = cur_counts / max(cur_counts.sum(), 1.0)

    # TVD = 0.5 * sum(|P - Q|)
    tvd = float(0.5 * np.sum(np.abs(ref_share - cur_share)))

    # Chi-Square test of independence
    contingency_table = np.vstack((ref_counts, cur_counts))
    # Chi2 requires at least 2 categories and non-zero row sums
    if len(all_cats) > 1 and np.all(contingency_table.sum(axis=1) > 0):
        try:
            res = chi2_contingency(contingency_table, correction=False)
            p_val = float(res.pvalue)
        except Exception:
            p_val = 1.0
    else:
        p_val = 1.0

    return tvd, p_val


def validate_incoming_batch(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> tuple[bool, list[str]]:
    """Validate incoming batch against reference schema before drift evaluation."""
    issues: list[str] = []
    if current is None or current.empty or len(current) == 0:
        issues.append("Incoming monitoring batch is empty (0 rows).")
        return False, issues

    ref_norm = {str(c).strip(): c for c in reference.columns}
    cur_norm = {str(c).strip(): c for c in current.columns}

    missing = [c for norm, c in ref_norm.items() if norm not in cur_norm]
    if missing:
        issues.append(f"Missing required feature(s): {', '.join(missing)}")

    for norm, orig_ref in ref_norm.items():
        if norm in cur_norm:
            orig_cur = cur_norm[norm]
            ref_num = pd.api.types.is_numeric_dtype(reference[orig_ref])
            cur_num = pd.api.types.is_numeric_dtype(current[orig_cur])
            if ref_num != cur_num:
                issues.append(
                    f"Data type mismatch on '{orig_ref}': reference is "
                    f"{'numeric' if ref_num else 'categorical'}, incoming is "
                    f"{'numeric' if cur_num else 'categorical'}"
                )

    return len(issues) == 0, issues


def compare_feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    numerical_psi_threshold: float = DEFAULT_NUMERICAL_PSI_THRESHOLD,
    categorical_tvd_threshold: float = DEFAULT_CATEGORICAL_TVD_THRESHOLD,
    alpha_p_value: float = DEFAULT_P_VALUE_ALPHA,
    minimum_samples: int = DEFAULT_MINIMUM_DRIFT_SAMPLES,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare input feature distributions between reference baseline and incoming monitoring batch.
    
    Safe column matching is performed via whitespace normalization (e.g. 'Air temperature [K] ' matches
    'Air temperature [K]'). Missing columns are reported explicitly and never treated as stable.
    """
    if reference is None or reference.empty:
        raise ValueError("Reference dataset must be non-empty.")

    schema_issues: list[str] = []

    # Handle empty current DataFrame gracefully without crashing
    if current is None or current.empty or len(current) == 0:
        empty_rows: list[dict[str, Any]] = []
        for col in reference.columns:
            ref_s = reference[col]
            empty_rows.append({
                "feature": col,
                "ref_samples": int(ref_s.notna().sum()),
                "cur_samples": 0,
                "ref_null_pct": round(float(ref_s.isna().mean() * 100), 1),
                "cur_null_pct": 100.0,
                "score": np.nan,
                "metric": "PSI & KS-test" if pd.api.types.is_numeric_dtype(ref_s) else "TVD & Chi-Square",
                "p_value": np.nan,
                "drift_detected": False,
                "status": "Batch is empty (0 samples)",
            })
        summary = {
            "status": "empty_batch",
            "compared_features": 0,
            "drifted_features": 0,
            "missing_features": list(reference.columns),
            "schema_issues": ["Incoming monitoring batch contains 0 rows."],
            "message": "Incoming monitoring batch is empty (0 rows). Cannot perform drift assessment.",
            "thresholds": {
                "numerical_psi": numerical_psi_threshold,
                "categorical_tvd": categorical_tvd_threshold,
                "alpha": alpha_p_value,
                "min_samples": minimum_samples,
            },
        }
        return pd.DataFrame(empty_rows), summary

    # Safe whitespace normalization mapping
    ref_col_map: dict[str, str] = {str(c).strip(): c for c in reference.columns}
    cur_col_map: dict[str, str] = {str(c).strip(): c for c in current.columns}

    missing_cols = [c for norm, c in ref_col_map.items() if norm not in cur_col_map]
    extra_cols = [c for norm, c in cur_col_map.items() if norm not in ref_col_map]

    if missing_cols:
        schema_issues.append(f"Missing in incoming: {', '.join(missing_cols)}")
    if extra_cols:
        schema_issues.append(f"Unexpected in incoming: {', '.join(extra_cols)}")

    feature_rows: list[dict[str, Any]] = []
    compared_count = 0
    drifted_count = 0

    for col in reference.columns:
        ref_s = reference[col]
        ref_n = int(ref_s.notna().sum())
        ref_null_pct = float(ref_s.isna().mean() * 100)
        norm_col = str(col).strip()

        base_record: dict[str, Any] = {
            "feature": col,
            "ref_samples": ref_n,
            "cur_samples": 0,
            "ref_null_pct": round(ref_null_pct, 1),
            "cur_null_pct": np.nan,
            "score": np.nan,
            "metric": "",
            "p_value": np.nan,
            "drift_detected": False,
            "status": "Pending",
        }

        if norm_col not in cur_col_map:
            base_record["cur_null_pct"] = 100.0
            base_record["status"] = "Missing from incoming batch"
            feature_rows.append(base_record)
            continue

        matched_cur_col = cur_col_map[norm_col]
        cur_s = current[matched_cur_col]
        cur_n = int(cur_s.notna().sum())
        cur_null_pct = float(cur_s.isna().mean() * 100)
        base_record["cur_samples"] = cur_n
        base_record["cur_null_pct"] = round(cur_null_pct, 1)

        ref_is_num = pd.api.types.is_numeric_dtype(ref_s)
        cur_is_num = pd.api.types.is_numeric_dtype(cur_s)

        if ref_is_num != cur_is_num:
            base_record["status"] = "Type mismatch (numeric vs categorical)"
            schema_issues.append(f"Type mismatch on feature '{col}'")
            feature_rows.append(base_record)
            continue

        if min(ref_n, cur_n) < minimum_samples:
            base_record["status"] = f"Insufficient sample size (< {minimum_samples})"
            feature_rows.append(base_record)
            continue

        # Missingness drift check: shift >= 15%
        null_diff = abs(ref_null_pct - cur_null_pct)
        null_drift = null_diff >= 15.0

        if ref_is_num:
            base_record["metric"] = "PSI & KS-test"
            ref_vals = pd.to_numeric(ref_s, errors="coerce").to_numpy(dtype=float)
            cur_vals = pd.to_numeric(cur_s, errors="coerce").to_numpy(dtype=float)
            ref_clean = ref_vals[np.isfinite(ref_vals)]
            cur_clean = cur_vals[np.isfinite(cur_vals)]

            if len(ref_clean) < 2 or len(cur_clean) < 2:
                base_record["status"] = "Insufficient valid numerical values"
                feature_rows.append(base_record)
                continue

            psi = calculate_psi(ref_clean, cur_clean)
            ks_res = ks_2samp(ref_clean, cur_clean)
            p_val = float(ks_res.pvalue)

            stat_drift = (psi >= numerical_psi_threshold) and (p_val < alpha_p_value)
            is_drift = bool(stat_drift or null_drift)

            base_record["score"] = round(psi, 4)
            base_record["p_value"] = round(p_val, 5)
            base_record["drift_detected"] = is_drift
            base_record["status"] = "Drift Detected" if is_drift else "Stable"

        else:
            base_record["metric"] = "TVD & Chi-Square"
            tvd, p_val = calculate_categorical_distance(ref_s, cur_s)
            stat_drift = (tvd >= categorical_tvd_threshold) and (p_val < alpha_p_value)
            is_drift = bool(stat_drift or null_drift)

            base_record["score"] = round(tvd, 4)
            base_record["p_value"] = round(p_val, 5)
            base_record["drift_detected"] = is_drift
            base_record["status"] = "Drift Detected" if is_drift else "Stable"

        compared_count += 1
        if base_record["drift_detected"]:
            drifted_count += 1
        feature_rows.append(base_record)

    # Determine overall status
    if compared_count == 0:
        overall_status = "needs_attention"
        message = f"No features could be monitored. {len(missing_cols)} feature(s) missing from incoming batch."
    elif missing_cols:
        # Schema issue: one or more reference columns are missing from incoming batch.
        # NEVER report 'no_drift_detected' if features are missing!
        if drifted_count > 0:
            overall_status = "drift_detected"
            message = (
                f"Distribution shift detected in {drifted_count} of {compared_count} monitored features, "
                f"and {len(missing_cols)} required feature(s) are missing from incoming batch."
            )
        else:
            overall_status = "schema_mismatch"
            message = (
                f"Schema mismatch: {len(missing_cols)} required feature(s) missing from incoming batch "
                f"({', '.join(missing_cols)}). Cannot verify overall stability."
            )
    elif drifted_count > 0:
        overall_status = "drift_detected"
        message = (
            f"Distribution shift detected in {drifted_count} of {compared_count} features. "
            "Data drift alone does not prove model performance has degraded."
        )
    else:
        overall_status = "no_drift_detected"
        message = f"All {compared_count} monitored features are within stable distribution thresholds."

    summary = {
        "status": overall_status,
        "compared_features": compared_count,
        "drifted_features": drifted_count,
        "missing_features": missing_cols,
        "schema_issues": schema_issues,
        "message": message,
        "thresholds": {
            "numerical_psi": numerical_psi_threshold,
            "categorical_tvd": categorical_tvd_threshold,
            "alpha": alpha_p_value,
            "min_samples": minimum_samples,
        },
    }

    return pd.DataFrame(feature_rows), summary


def generate_demo_monitoring_batch(
    reference: pd.DataFrame,
    *,
    drift: bool = False,
    n_samples: int = 350,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate a realistic monitoring batch from reference distributions.
    
    If drift=False, draws a stable bootstrap sample preserving baseline distributions (PSI < 0.05, p > 0.05).
    If drift=True, introduces statistically significant distribution shifts (PSI > 0.25, TVD > 0.20, p < 0.001)
    specifically calibrated for industrial sensor/manufacturing telemetry or arbitrary tabular features.
    """
    if reference is None or reference.empty:
        raise ValueError("Reference dataset must be non-empty to generate demo batches.")

    rng = np.random.default_rng(random_seed)
    n = max(n_samples, 20)

    # Sample base rows from reference
    base_indices = rng.choice(len(reference), size=n, replace=True)
    sampled = reference.iloc[base_indices].reset_index(drop=True).copy()

    if not drift:
        return sampled

    # If drift=True, apply statistically significant, realistic shifts
    drifted_df = sampled.copy()
    for col in drifted_df.columns:
        s = drifted_df[col]
        ref_s = reference[col].dropna()

        if pd.api.types.is_numeric_dtype(s):
            vals = s.to_numpy(dtype=float)
            col_std = float(ref_s.std()) if len(ref_s) > 1 else 1.0
            if col_std <= 0:
                col_std = 1.0

            # Domain-calibrated shifts for AI4I sensor features if present
            norm_name = str(col).strip().lower()
            if "air temperature" in norm_name:
                vals = vals + rng.normal(5.5, 0.4, n)
            elif "process temperature" in norm_name:
                vals = vals + rng.normal(5.0, 0.4, n)
            elif "rotational speed" in norm_name:
                vals = vals - rng.normal(250.0, 30.0, n)
                vals = np.clip(vals, 500.0, None)
            elif "torque" in norm_name:
                vals = vals + rng.normal(15.0, 1.5, n)
                vals = np.clip(vals, 0.0, None)
            elif "tool wear" in norm_name:
                vals = vals + rng.normal(75.0, 5.0, n)
                vals = np.clip(vals, 0.0, None)
            else:
                # Generic numerical shift: +1.8 standard deviations with mild noise
                shift_mag = 1.8 * col_std
                vals = vals + rng.normal(shift_mag, 0.1 * col_std, n)

            drifted_df[col] = np.round(vals, 2)

        else:
            # Categorical drift
            cats = list(ref_s.unique())
            if len(cats) >= 2:
                norm_name = str(col).strip().lower()
                if norm_name == "type" and set(cats) == {"L", "M", "H"}:
                    # Invert AI4I Type distribution (Baseline is L: ~60%, M: ~30%, H: ~10%)
                    drifted_df[col] = rng.choice(["L", "M", "H"], size=n, p=[0.05, 0.20, 0.75])
                else:
                    # Invert empirical frequency: rarest category gets highest weight
                    val_counts = ref_s.value_counts(ascending=True)
                    ordered_cats = list(val_counts.index)
                    weights = np.linspace(0.8, 0.2, len(ordered_cats))
                    probs = weights / weights.sum()
                    drifted_df[col] = rng.choice(ordered_cats, size=n, p=probs)

    return drifted_df


def build_distribution_comparison_plot(
    reference: pd.DataFrame, current: pd.DataFrame, feature_name: str
) -> go.Figure:
    """Generate high-clarity Plotly chart comparing reference vs current distribution for a feature."""
    fig = go.Figure()

    # Normalization lookup for feature name
    ref_norm = {str(c).strip(): c for c in reference.columns}
    cur_norm = {str(c).strip(): c for c in current.columns}
    norm_feat = str(feature_name).strip()

    if norm_feat not in ref_norm or norm_feat not in cur_norm:
        fig.add_annotation(text="Feature missing from reference or current dataset", showarrow=False)
        return fig

    ref_col = reference[ref_norm[norm_feat]].dropna()
    cur_col = current[cur_norm[norm_feat]].dropna()

    is_num = pd.api.types.is_numeric_dtype(ref_col) and pd.api.types.is_numeric_dtype(cur_col)

    if is_num:
        fig.add_trace(go.Histogram(x=ref_col, name="Reference Baseline", opacity=0.6, marker_color="#2563eb"))
        fig.add_trace(go.Histogram(x=cur_col, name="Current Batch", opacity=0.6, marker_color="#dc2626"))
        fig.update_layout(
            barmode="overlay",
            title=f"Numerical Distribution Comparison: {feature_name}",
            xaxis_title=feature_name,
            yaxis_title="Count",
            template="plotly_white",
            margin=dict(l=40, r=40, t=50, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
    else:
        ref_counts = ref_col.astype(str).value_counts(normalize=True).reset_index()
        ref_counts.columns = ["category", "ref_share"]

        cur_counts = cur_col.astype(str).value_counts(normalize=True).reset_index()
        cur_counts.columns = ["category", "cur_share"]

        merged = pd.merge(ref_counts, cur_counts, on="category", how="outer").fillna(0)
        fig.add_trace(go.Bar(x=merged["category"], y=merged["ref_share"], name="Reference Share", marker_color="#2563eb"))
        fig.add_trace(go.Bar(x=merged["category"], y=merged["cur_share"], name="Current Share", marker_color="#dc2626"))
        fig.update_layout(
            barmode="group",
            title=f"Categorical Frequency Distribution: {feature_name}",
            xaxis_title="Category",
            yaxis_title="Relative Proportion",
            template="plotly_white",
            margin=dict(l=40, r=40, t=50, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

    return fig
