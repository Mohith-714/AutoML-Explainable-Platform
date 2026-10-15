"""Executive report generation in HTML, Markdown, and tabular CSV formats."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import pandas as pd


def generate_executive_html_report(
    dataset_name: str,
    dataset_shape: tuple[int, int],
    target_column: str,
    task_type: str,
    leaderboard_df: pd.DataFrame | None,
    best_model_name: str | None,
    best_metrics: dict[str, float] | None,
    importance_df: pd.DataFrame | None,
    drift_summary: dict[str, Any] | None,
    drift_table: pd.DataFrame | None,
    monitoring_record: dict[str, Any] | None,
    registry_versions: list[dict[str, Any]] | None,
) -> str:
    """Generate a responsive, beautifully styled HTML executive summary report."""
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Table formatting helpers
    leaderboard_html = (
        leaderboard_df.to_html(classes="styled-table", index=False)
        if leaderboard_df is not None and not leaderboard_df.empty
        else "<p class='empty-text'>No AutoML training leaderboard available.</p>"
    )

    importance_html = (
        importance_df.head(10).to_html(classes="styled-table", index=False)
        if importance_df is not None and not importance_df.empty
        else "<p class='empty-text'>No explainability data available.</p>"
    )

    drift_html = (
        drift_table.to_html(classes="styled-table", index=False)
        if drift_table is not None and not drift_table.empty
        else "<p class='empty-text'>No drift monitoring records available.</p>"
    )

    drift_status_badge = (
        f"<span class='badge badge-{drift_summary.get('status', 'none')}'>{drift_summary.get('status', 'Not evaluated').upper()}</span>"
        if drift_summary
        else "<span class='badge badge-none'>NOT EVALUATED</span>"
    )

    perf_status_badge = (
        f"<span class='badge badge-{monitoring_record.get('status', 'none')}'>{monitoring_record.get('status', 'No Ground Truth').upper()}</span>"
        if monitoring_record
        else "<span class='badge badge-none'>NOT EVALUATED</span>"
    )

    versions_rows = ""
    if registry_versions:
        for v in registry_versions:
            status_cls = "badge-success" if v.get("status") == "approved" else "badge-neutral"
            versions_rows += f"""
            <tr>
                <td><code>{v.get('version_id')}</code></td>
                <td>{v.get('model_name')}</td>
                <td><span class='badge {status_cls}'>{v.get('status')}</span></td>
                <td>{v.get('primary_metric')}: {v.get('candidate_score', v.get('metrics', {}).get(v.get('primary_metric'), 'N/A'))}</td>
                <td>{v.get('created_at', '')[:19]}</td>
            </tr>
            """
    else:
        versions_rows = "<tr><td colspan='5' class='empty-text'>No registered versions.</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AutoML Executive Lifecycle Report - {dataset_name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background-color: #0f172a;
            color: #f1f5f9;
            margin: 0;
            padding: 30px;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
            background-color: #1e293b;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
        }}
        h1, h2, h3 {{ color: #38bdf8; margin-top: 0; }}
        h2 {{ border-bottom: 2px solid #334155; padding-bottom: 8px; margin-top: 35px; }}
        .header-meta {{ color: #94a3b8; font-size: 14px; margin-bottom: 25px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 15px; margin-bottom: 25px; }}
        .card {{ background-color: #0f172a; padding: 18px; border-radius: 8px; border: 1px solid #334155; }}
        .card-label {{ color: #94a3b8; font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; }}
        .card-value {{ font-size: 20px; font-weight: bold; color: #f8fafc; margin-top: 5px; }}
        .styled-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            background-color: #0f172a;
            border-radius: 6px;
            overflow: hidden;
            font-size: 13px;
        }}
        .styled-table th, .styled-table td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #334155; }}
        .styled-table th {{ background-color: #1e293b; color: #38bdf8; font-weight: 600; }}
        .badge {{
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .badge-approved, .badge-no_drift_detected, .badge-healthy, .badge-success {{ background-color: #065f46; color: #34d399; }}
        .badge-drift_detected, .badge-degraded, .badge-warning {{ background-color: #7f1d1d; color: #f87171; }}
        .badge-needs_attention, .badge-insufficient_samples {{ background-color: #78350f; color: #fbbf24; }}
        .badge-none, .badge-neutral {{ background-color: #334155; color: #94a3b8; }}
        .empty-text {{ color: #64748b; font-style: italic; }}
        footer {{ margin-top: 40px; text-align: center; color: #64748b; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Drift-Aware AutoML Executive Report</h1>
        <div class="header-meta">Generated: {now_str} | Platform: Domain-Independent Drift-Aware AutoML</div>

        <div class="grid">
            <div class="card">
                <div class="card-label">Dataset Name</div>
                <div class="card-value">{dataset_name}</div>
            </div>
            <div class="card">
                <div class="card-label">Dimensions</div>
                <div class="card-value">{dataset_shape[0]:,} rows × {dataset_shape[1]:,} cols</div>
            </div>
            <div class="card">
                <div class="card-label">Target Column</div>
                <div class="card-value">{target_column or 'None'}</div>
            </div>
            <div class="card">
                <div class="card-label">ML Task</div>
                <div class="card-value">{task_type.capitalize()}</div>
            </div>
        </div>

        <h2>AutoML Model Performance</h2>
        <div class="grid">
            <div class="card">
                <div class="card-label">Best Candidate Model</div>
                <div class="card-value">{best_model_name or 'N/A'}</div>
            </div>
            <div class="card">
                <div class="card-label">Primary Metric Score</div>
                <div class="card-value">
                    {round(list(best_metrics.values())[0], 4) if best_metrics else 'N/A'}
                </div>
            </div>
        </div>
        {leaderboard_html}

        <h2>Explainable AI Insights (Top Feature Importance)</h2>
        <p style="color: #94a3b8; font-size: 13px;">
            Permutation importance estimates the dependency of the model on raw domain features.
            It does not establish real-world causal mechanisms.
        </p>
        {importance_html}

        <h2>Drift Monitoring & Quality Assessment</h2>
        <div class="grid">
            <div class="card">
                <div class="card-label">Feature Drift Status</div>
                <div class="card-value">{drift_status_badge}</div>
            </div>
            <div class="card">
                <div class="card-label">Performance Status</div>
                <div class="card-value">{perf_status_badge}</div>
            </div>
        </div>
        <p style="color: #94a3b8; font-size: 13px;">
            {drift_summary.get('message', '') if drift_summary else ''}
        </p>
        {drift_html}

        <h2>Model Lifecycle & Registry History</h2>
        <table class="styled-table">
            <thead>
                <tr>
                    <th>Version ID</th>
                    <th>Model</th>
                    <th>Status</th>
                    <th>Score</th>
                    <th>Timestamp</th>
                </tr>
            </thead>
            <tbody>
                {versions_rows}
            </tbody>
        </table>

        <footer>
            Automated Report generated by Drift-Aware AutoML Platform. Designed for Windows & VS Code.
        </footer>
    </div>
</body>
</html>
"""
    return html
