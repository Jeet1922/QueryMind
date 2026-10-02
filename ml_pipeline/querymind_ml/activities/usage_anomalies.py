"""Activity 4 — Usage anomaly detection (ml_usage_anomalies).

Problem: flag team x tool weeks where usage volume deviates from the local
baseline. Grain: weekly (metric_name = 'usage_volume_weekly').
Approach: IsolationForest over a focused residual space — actual value,
4-week trailing baseline, residual, baseline ratio and z-score (the view
keeps session/quality context columns for EDA, but sparse-cell noise in those
dims would drown the spike signal). expected_value is the leakage-free
4-week trailing mean; a z-score >= 3 baseline is reported for comparison.
Ground truth: surge weeks injected in 01_raw_data_generators.sql.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import psycopg
from sklearn.ensemble import IsolationForest

from .. import db
from .base import (
    Activity,
    ActivityResult,
    ActivitySpec,
    make_model_version,
)

SPEC = ActivitySpec(
    name="usage_anomalies",
    title="Usage anomaly detection",
    task_type="anomaly_detection",
    output_table="ml_usage_anomalies",
    algorithm="IsolationForest + leakage-free 4-week rolling baseline",
    label_description="Injected surge weeks from ml_raw_weekly_usage.is_injected_anomaly",
    output_score_col="anomaly_score",
)

NUMERIC = [
    "actual_value",
    "roll_mean_4",
    "roll_std_4",
    "residual_4w",
    "ratio_4w",
    "z_score",
]
METRIC_NAME = "usage_volume_weekly"
WRITE_WEEKS = 26
CONTAMINATION = 0.01


class UsageAnomalies(Activity):
    spec = SPEC
    output_columns = [
        "metric_date",
        "team_id",
        "tool_id",
        "metric_name",
        "actual_value",
        "expected_value",
        "anomaly_score",
        "is_anomaly",
        "model_version",
    ]

    def load_dataset(self, conn: psycopg.Connection) -> pd.DataFrame:
        return db.read_frame(
            conn,
            "SELECT * FROM ml_features_usage_anomalies ORDER BY team_id, tool_id, week_start",
        )

    # ------------------------------------------------------------------ train
    def fit(self, df: pd.DataFrame) -> ActivityResult:
        rows = df[df["roll_mean_4"].notna()].copy()
        if len(rows) < 30:
            raise RuntimeError(
                "Not enough weekly rows with a rolling baseline. "
                "Run --step sql (01_raw_data_generators.sql) first."
            )
        X_raw = rows[NUMERIC].astype(float)
        X_raw = X_raw.fillna(X_raw.median(numeric_only=True)).fillna(0.0)

        model = IsolationForest(
            n_estimators=200,
            contamination=CONTAMINATION,
            random_state=42,
        )
        model.fit(X_raw)
        raw_scores = -model.score_samples(X_raw)  # higher = more anomalous
        predictions = model.predict(X_raw)  # -1 anomaly, 1 normal
        is_anom = predictions == -1

        score_min = float(raw_scores.min())
        score_max = float(raw_scores.max())
        span = (score_max - score_min) or 1.0
        norm = (raw_scores - score_min) / span

        metrics: dict[str, Any] = {
            "rows": int(len(rows)),
            "anomaly_rate": round(float(is_anom.mean()), 6),
            "score_min": round(score_min, 6),
            "score_max": round(score_max, 6),
            "contamination": CONTAMINATION,
        }

        label = rows["label_is_anomaly"]
        if label.notna().any() and label.nunique() > 1:
            from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

            precision, recall, f1, _ = precision_recall_fscore_support(
                label.astype(bool), is_anom, average="binary", zero_division=0
            )
            metrics["vs_ground_truth"] = {
                "precision": round(float(precision), 6),
                "recall": round(float(recall), 6),
                "f1": round(float(f1), 6),
                "prevalence": round(float(label.astype(bool).mean()), 6),
            }
            try:
                metrics["vs_ground_truth"]["roc_auc"] = round(
                    float(roc_auc_score(label.astype(bool), norm)), 6
                )
            except ValueError:
                pass

            # Transparent z-score baseline for comparison.
            z = pd.to_numeric(rows["z_score"], errors="coerce")
            if z.notna().any():
                z_flag = z.fillna(0.0) >= 3.0
                tp = int((z_flag & label.astype(bool)).sum())
                metrics["z_baseline"] = {
                    "flagged": int(z_flag.sum()),
                    "precision": round(tp / max(1, int(z_flag.sum())), 6),
                    "recall": round(tp / max(1, int(label.astype(bool).sum())), 6),
                }

        return ActivityResult(
            model_version=make_model_version(self.spec.name),
            bundle={
                "model": model,
                "feature_columns": NUMERIC,
                "score_min": score_min,
                "score_max": score_max,
            },
            metrics=metrics,
            feature_columns=list(NUMERIC),
            training_rows=int(len(rows)),
            extras={"score_min": score_min, "score_max": score_max},
        )

    # ---------------------------------------------------------------- predict
    def build_output(self, df: pd.DataFrame, result: ActivityResult) -> pd.DataFrame:
        df = df.copy()
        df["week_start"] = pd.to_datetime(df["week_start"])
        rows = df[df["roll_mean_4"].notna()].copy()
        if rows.empty:
            return pd.DataFrame(columns=self.output_columns)

        model = result.bundle["model"]
        score_min = float(result.bundle["score_min"])
        score_max = float(result.bundle["score_max"])
        span = (score_max - score_min) or 1.0

        X = rows[NUMERIC].astype(float)
        X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
        raw = -model.score_samples(X)
        is_anom = model.predict(X) == -1
        norm = np.clip((raw - score_min) / span, 0.0, 1.0)

        cutoff = rows["week_start"].max() - pd.Timedelta(weeks=WRITE_WEEKS - 1)
        keep = rows["week_start"] >= cutoff

        out = pd.DataFrame(
            {
                "metric_date": rows.loc[keep, "week_start"],
                "team_id": rows.loc[keep, "team_id"].astype(int),
                "tool_id": rows.loc[keep, "tool_id"].astype(int),
                "metric_name": METRIC_NAME,
                "actual_value": pd.to_numeric(rows.loc[keep, "actual_value"]).round(4),
                "expected_value": pd.to_numeric(rows.loc[keep, "roll_mean_4"]).round(4),
                "anomaly_score": pd.Series(norm[keep.to_numpy()], index=rows.index[keep]).round(4),
                "is_anomaly": is_anom[keep.to_numpy()],
                "model_version": result.model_version,
            }
        )
        return out[self.output_columns]

    def delete_statement(
        self, result: ActivityResult, output: pd.DataFrame
    ) -> tuple[str, list[Any]]:
        start = (
            pd.Timestamp(output["metric_date"].min()).date() if len(output) else None
        )
        return (
            "DELETE FROM ml_usage_anomalies WHERE metric_date >= %s AND metric_name = %s",
            [start, METRIC_NAME],
        )
