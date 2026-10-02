"""Shared types and helpers for the six ML activities.

Each activity implements: load_dataset -> fit -> build_output ->
(delete + insert) write_predictions. fit/build_output are pure DataFrame
functions so they can be unit-tested without a database.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

import numpy as np
import pandas as pd
import psycopg

from .. import db

MODEL_VERSION_FMT = "%Y%m%d-%H%M%S"


@dataclass(frozen=True)
class ActivitySpec:
    name: str
    title: str
    task_type: str  # forecasting | classification | clustering | anomaly_detection | recommendation
    output_table: str
    algorithm: str
    label_description: str
    output_score_col: str
    output_band_col: str | None = None


@dataclass
class ActivityResult:
    model_version: str
    bundle: dict[str, Any]
    metrics: dict[str, Any]
    feature_columns: list[str]
    training_rows: int
    extras: dict[str, Any] = field(default_factory=dict)


def make_model_version(activity: str) -> str:
    return f"{activity}-{datetime.now():{MODEL_VERSION_FMT}}"


# --------------------------------------------------------------------------
# Month / seasonality helpers
# --------------------------------------------------------------------------

def as_timestamp(value: Any) -> pd.Timestamp:
    return pd.Timestamp(value)


def month_start(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.normalize().replace(day=1)


def current_month_start() -> pd.Timestamp:
    return month_start(pd.Timestamp.today())


def add_months(value: Any, months: int) -> pd.Timestamp:
    ts = month_start(value)
    year = ts.year + (ts.month - 1 + months) // 12
    month = 1 + (ts.month - 1 + months) % 12
    return pd.Timestamp(year=year, month=month, day=1)


def month_index(value: Any) -> int:
    ts = month_start(value)
    return ts.year * 12 + (ts.month - 1)


def season_sin(mi: int) -> float:
    return math.sin(2 * math.pi * (mi % 12) / 12)


def season_cos(mi: int) -> float:
    return math.cos(2 * math.pi * (mi % 12) / 12)


# --------------------------------------------------------------------------
# Static categorical encoding (one-hot with stable columns across refits)
# --------------------------------------------------------------------------

def fit_static_columns(df: pd.DataFrame, cols: Sequence[str]) -> list[str]:
    return list(pd.get_dummies(df[list(cols)], columns=list(cols), dtype=float).columns)


def transform_static(static_cols: list[str], df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    dummies = pd.get_dummies(df[list(cols)], columns=list(cols), dtype=float)
    return dummies.reindex(columns=static_cols, fill_value=0.0)


def split_time(
    df: pd.DataFrame,
    time_col: str,
    test_periods: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on everything before the last `test_periods` distinct periods."""
    periods = sorted(df[time_col].dropna().unique())
    if len(periods) <= 1:
        return df, df.iloc[0:0]
    cutoff = periods[-test_periods] if len(periods) > test_periods else periods[0]
    train = df[df[time_col] < cutoff]
    test = df[df[time_col] >= cutoff]
    if train.empty:  # degenerate case: keep at least one row in train
        return df, df.iloc[0:0]
    return train, test


# --------------------------------------------------------------------------
# Metrics (JSON-serialisable floats only)
# --------------------------------------------------------------------------

def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return {"n": 0}
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    ss_res = float(np.sum(err ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    mask = np.abs(y_true) > 1e-6
    mape = float(np.mean(np.abs(err[mask] / y_true[mask])) * 100.0) if mask.any() else float("nan")
    return {
        "n": int(y_true.size),
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "r2": round(r2, 6),
        "mape_pct": round(mape, 4),
    }


def classification_metrics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    y_proba: np.ndarray | None = None,
    classes: Sequence[Any] | None = None,
) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return {"n": 0}
    metrics: dict[str, Any] = {
        "n": int(y_true.size),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 6),
        "macro_precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "macro_recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 6),
    }
    y_true_s = np.asarray([str(v) for v in y_true])
    y_pred_s = np.asarray([str(v) for v in y_pred])
    cm_labels = sorted(set(y_true_s) | set(y_pred_s))
    metrics["per_class_f1"] = {
        lab: round(float(f1_score(y_true_s == lab, y_pred_s == lab, zero_division=0)), 6)
        for lab in cm_labels
    }
    metrics["confusion_matrix"] = confusion_matrix(y_true_s, y_pred_s, labels=cm_labels).tolist()
    metrics["confusion_labels"] = cm_labels
    if y_proba is not None and classes is not None and len(set(map(str, y_true))) > 1:
        try:
            metrics["roc_auc_ovr"] = round(
                float(roc_auc_score(y_true, y_proba, multi_class="ovr", labels=list(classes))),
                6,
            )
        except ValueError:
            pass
    return metrics


# --------------------------------------------------------------------------
# Base class
# --------------------------------------------------------------------------

class Activity:
    """Template: load -> fit -> build_output -> write."""

    spec: ActivitySpec
    output_columns: list[str]

    def load_dataset(self, conn: psycopg.Connection) -> pd.DataFrame:
        raise NotImplementedError

    def fit(self, df: pd.DataFrame) -> ActivityResult:
        raise NotImplementedError

    def build_output(self, df: pd.DataFrame, result: ActivityResult) -> pd.DataFrame:
        raise NotImplementedError

    def delete_statement(
        self, result: ActivityResult, output: pd.DataFrame
    ) -> tuple[str, list[Any]]:
        raise NotImplementedError

    def write_predictions(
        self,
        conn: psycopg.Connection,
        result: ActivityResult,
        output: pd.DataFrame,
    ) -> int:
        """Atomically replace the activity's output window with new predictions."""
        delete_sql, params = self.delete_statement(result, output)
        with conn.transaction():
            db.execute(conn, delete_sql, params)
            db.write_frame(conn, self.spec.output_table, self.output_columns, output)
        conn.commit()
        return len(output)

    @staticmethod
    def result_from_meta(meta: dict[str, Any], bundle: dict[str, Any]) -> ActivityResult:
        return ActivityResult(
            model_version=meta["model_version"],
            bundle=bundle,
            metrics=meta.get("metrics", {}),
            feature_columns=list(meta.get("feature_columns", [])),
            training_rows=int(meta.get("training_rows", 0) or 0),
        )
