"""Prediction monitoring: score-distribution drift + evaluation snapshot.

After every write, the pipeline records summary statistics of the predictions
for the activity and compares them with the previous monitor row, so drift in
score mean / band mix is visible over time without touching the app.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import psycopg

from . import db


def _band_distribution(series: pd.Series) -> dict[str, float]:
    counts = series.astype(str).value_counts(dropna=True)
    total = float(counts.sum()) or 1.0
    return {str(k): round(float(v) / total, 4) for k, v in counts.items()}


def record(
    conn: psycopg.Connection,
    activity: str,
    model_version: str,
    predictions: pd.DataFrame,
    score_col: str,
    band_col: str | None = None,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute prediction stats, diff against the previous run, and store."""
    if predictions.empty or score_col not in predictions.columns:
        return {}

    scores = pd.to_numeric(predictions[score_col], errors="coerce").dropna()
    stats: dict[str, Any] = {
        "prediction_count": int(len(predictions)),
        "score_mean": round(float(scores.mean()), 6) if len(scores) else None,
        "score_std": round(float(scores.std(ddof=0)), 6) if len(scores) else None,
        "score_p50": round(float(scores.quantile(0.50)), 6) if len(scores) else None,
        "score_p95": round(float(scores.quantile(0.95)), 6) if len(scores) else None,
    }
    bands = (
        _band_distribution(predictions[band_col])
        if band_col and band_col in predictions.columns
        else {}
    )

    prev_rows = db.read_rows(
        conn,
        """
        SELECT score_mean, band_distribution
        FROM ml_prediction_monitor
        WHERE activity = %s
        ORDER BY monitor_date DESC, monitor_id DESC
        LIMIT 1
        """,
        (activity,),
    )
    drift: dict[str, Any] = {"has_previous": bool(prev_rows)}
    if prev_rows:
        prev_mean = float(prev_rows[0][0]) if prev_rows[0][0] is not None else None
        prev_bands = prev_rows[0][1] if isinstance(prev_rows[0][1], dict) else {}
        if prev_mean and stats["score_mean"] is not None:
            shift = (stats["score_mean"] - prev_mean) / prev_mean
            drift["score_mean_shift_pct"] = round(shift, 4)
            drift["score_mean_shift_flag"] = abs(shift) > 0.25
        if bands and prev_bands:
            drift["band_share_delta"] = {
                k: round(bands.get(k, 0.0) - float(prev_bands.get(k, 0.0)), 4)
                for k in sorted(set(bands) | set(prev_bands))
            }

    db.execute(
        conn,
        """
        INSERT INTO ml_prediction_monitor
            (activity, monitor_date, model_version, prediction_count,
             score_mean, score_std, score_p50, score_p95,
             band_distribution, drift_vs_previous, evaluation_metrics)
        VALUES (%s, CURRENT_DATE, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            activity,
            model_version,
            stats["prediction_count"],
            stats["score_mean"],
            stats["score_std"],
            stats["score_p50"],
            stats["score_p95"],
            json.dumps(bands),
            json.dumps(drift),
            json.dumps(metrics or {}, default=str),
        ),
    )
    conn.commit()
    return {**stats, "bands": bands, "drift": drift}
