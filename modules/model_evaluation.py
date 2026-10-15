"""Evaluation metrics, scoring calculators, and visual analysis generators."""

from __future__ import annotations

from typing import Any, Literal
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    root_mean_squared_error,
)

TaskType = Literal["classification", "regression"]


def compute_classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute comprehensive classification performance metrics including per-class recall and macro F1."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    classes = np.unique(y_t)

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_t, y_p)),
        "f1_macro": float(f1_score(y_t, y_p, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_t, y_p, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_t, y_p, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_t, y_p, average="weighted", zero_division=0)),
        "recall_macro": float(recall_score(y_t, y_p, average="macro", zero_division=0)),
        "recall_weighted": float(recall_score(y_t, y_p, average="weighted", zero_division=0)),
    }

    # Binary classification specific metrics (especially positive / defect class 1)
    if len(classes) == 2:
        labels = [0, 1] if set(classes) == {0, 1} else list(classes)
        cm = confusion_matrix(y_t, y_p, labels=labels)
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        metrics["true_negatives"] = float(tn)
        metrics["false_positives"] = float(fp)
        metrics["false_negatives"] = float(fn)
        metrics["true_positives"] = float(tp)

        # Class 1 (Minority / Positive / Defect class)
        rec1 = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        prec1 = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        f1_1 = float(2 * (prec1 * rec1) / (prec1 + rec1)) if (prec1 + rec1) > 0 else 0.0
        metrics["recall_class_1"] = rec1
        metrics["precision_class_1"] = prec1
        metrics["f1_class_1"] = f1_1

        # Class 0 (Negative / Normal class)
        rec0 = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        prec0 = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
        f1_0 = float(2 * (prec0 * rec0) / (prec0 + rec0)) if (prec0 + rec0) > 0 else 0.0
        metrics["recall_class_0"] = rec0
        metrics["precision_class_0"] = prec0
        metrics["f1_class_0"] = f1_0

    if y_prob is not None:
        try:
            if len(classes) == 2:
                # Binary classification
                prob_vec = y_prob[:, 1] if y_prob.ndim == 2 and y_prob.shape[1] >= 2 else y_prob
                metrics["roc_auc"] = float(roc_auc_score(y_t, prob_vec))
            elif len(classes) > 2 and y_prob.ndim == 2 and y_prob.shape[1] == len(classes):
                # Multiclass OvR
                metrics["roc_auc"] = float(
                    roc_auc_score(y_t, y_prob, multi_class="ovr", average="weighted")
                )
        except Exception:
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")

    return metrics


def compute_regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute comprehensive regression performance metrics."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    return {
        "rmse": float(root_mean_squared_error(y_t, y_p)),
        "mae": float(mean_absolute_error(y_t, y_p)),
        "r2": float(r2_score(y_t, y_p)),
        "mape": float(mean_absolute_percentage_error(y_t, y_p)),
    }


def build_per_class_metrics_table(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """Generate detailed per-class precision, recall, F1-score, and support table."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    classes = np.unique(y_t)
    total = len(y_t)
    rows: list[dict[str, Any]] = []

    for cls in classes:
        cls_name = str(cls)
        if set(classes) == {0, 1}:
            cls_name = "Class 1 (Defect / Positive)" if cls == 1 else "Class 0 (Normal / Negative)"

        sup = int(np.sum(y_t == cls))
        pct = round(sup / total * 100, 1)
        prec = float(precision_score(y_t == cls, y_p == cls, zero_division=0))
        rec = float(recall_score(y_t == cls, y_p == cls, zero_division=0))
        f1 = float(f1_score(y_t == cls, y_p == cls, zero_division=0))

        rows.append({
            "Class": cls_name,
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "Support": sup,
            "Share": f"{pct}%",
        })

    # Add Macro & Weighted Averages
    macro_p = float(precision_score(y_t, y_p, average="macro", zero_division=0))
    macro_r = float(recall_score(y_t, y_p, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_t, y_p, average="macro", zero_division=0))
    rows.append({
        "Class": "Macro Average",
        "Precision": round(macro_p, 4),
        "Recall": round(macro_r, 4),
        "F1-Score": round(macro_f1, 4),
        "Support": total,
        "Share": "100.0%",
    })

    w_p = float(precision_score(y_t, y_p, average="weighted", zero_division=0))
    w_r = float(recall_score(y_t, y_p, average="weighted", zero_division=0))
    w_f1 = float(f1_score(y_t, y_p, average="weighted", zero_division=0))
    rows.append({
        "Class": "Weighted Average",
        "Precision": round(w_p, 4),
        "Recall": round(w_r, 4),
        "F1-Score": round(w_f1, 4),
        "Support": total,
        "Share": "100.0%",
    })

    return pd.DataFrame(rows)


def build_confusion_matrix_plot(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """Generate high-contrast Plotly heatmap for confusion matrix with explicit TN, FP, FN, TP cell annotations."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    classes = np.unique(y_t)

    is_binary = set(classes) == {0, 1}
    labels = [0, 1] if is_binary else sorted(list(set(y_t) | set(y_p)))
    cm = confusion_matrix(y_t, y_p, labels=labels)

    if is_binary:
        labels_str = ["0 (Normal)", "1 (Defect)"]
        text_annotations = [
            [f"TN: {cm[0, 0]}", f"FP: {cm[0, 1]}"],
            [f"FN: {cm[1, 0]}", f"TP: {cm[1, 1]}"],
        ]
    else:
        labels_str = [f"Class {l}" for l in labels]
        text_annotations = [[str(val) for val in row] for row in cm]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=labels_str,
            y=labels_str,
            hoverongaps=False,
            colorscale="Blues",
            text=text_annotations,
            texttemplate="%{text}",
            textfont={"size": 15, "color": "#172033"},
        )
    )
    fig.update_layout(
        title="Confusion Matrix (Actual vs Predicted)",
        xaxis_title="Predicted Class",
        yaxis_title="Actual Class",
        yaxis_autorange="reversed",
        template="plotly_white",
        margin=dict(l=50, r=50, t=50, b=50),
    )
    return fig


def build_roc_curve_plot(y_true: pd.Series, y_prob: np.ndarray) -> go.Figure | None:
    """Generate Plotly ROC Curve for binary classification."""
    classes = np.unique(y_true)
    if len(classes) != 2 or y_prob is None:
        return None

    try:
        prob_vec = y_prob[:, 1] if y_prob.ndim == 2 and y_prob.shape[1] >= 2 else y_prob
        # Binarize if needed
        pos_label = classes[1]
        fpr, tpr, _ = roc_curve(y_true == pos_label, prob_vec)
        auc_val = roc_auc_score(y_true == pos_label, prob_vec)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC = {auc_val:.3f})", line=dict(color="#3b82f6", width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Chance", line=dict(color="#9ca3af", dash="dash")))
        fig.update_layout(
            title=f"Receiver Operating Characteristic (ROC)",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            template="plotly_white",
            margin=dict(l=40, r=40, t=50, b=40),
        )
        return fig
    except Exception:
        return None


def build_regression_residual_plot(y_true: pd.Series, y_pred: np.ndarray) -> go.Figure:
    """Generate interactive scatter plot of actual vs predicted and residual distribution."""
    residuals = np.asarray(y_true) - y_pred
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=residuals,
            mode="markers",
            marker=dict(color="#3b82f6", opacity=0.7, size=7),
            name="Residuals",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#ef4444", annotation_text="Zero Residual")
    fig.update_layout(
        title="Residual Plot (Predicted vs Residual)",
        xaxis_title="Predicted Value",
        yaxis_title="Residual (Actual - Predicted)",
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig
