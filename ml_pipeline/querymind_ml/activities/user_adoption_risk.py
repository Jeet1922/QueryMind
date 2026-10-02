"""Activity 2 — User adoption risk (ml_user_adoption_risk).

Problem: for the coming month, classify each employee's adoption risk as
LOW / MEDIUM / HIGH from trailing-month behaviour (labels come from
ml_raw_user_adoption_labels: zero usage or <=50% MoM = HIGH, <=85% = MEDIUM).
Approach: gradient-boosted trees on trailing usage aggregates + MoM deltas +
one-hot employee profile; probability-weighted risk score in [0, 1].
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import psycopg
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

from .. import db
from .base import (
    Activity,
    ActivityResult,
    ActivitySpec,
    classification_metrics,
    fit_static_columns,
    make_model_version,
    transform_static,
)

SPEC = ActivitySpec(
    name="user_adoption_risk",
    title="User adoption risk",
    task_type="classification",
    output_table="ml_user_adoption_risk",
    algorithm="HistGradientBoostingClassifier (multiclass LOW/MEDIUM/HIGH)",
    label_description="Risk band from month-over-month usage decline (ml_raw_user_adoption_labels)",
    output_score_col="risk_score",
    output_band_col="risk_band",
)

NUMERIC = [
    "prev_month_events",
    "prev_month_active_days",
    "prev_month_distinct_tools",
    "prev_month_distinct_tasks",
    "prev_month_success_rate",
    "prev_month_avg_session",
    "prev2_events",
    "prev3_events",
    "prev2_active_days",
    "prev2_distinct_tools",
    "prev2_success_rate",
    "prev2_avg_session",
    "events_mom_delta",
    "events_trend",
    "prior6_mean_events",
]
CATEGORICAL = ["role", "location", "employment_type", "department_id", "team_id", "practice_id"]
CLASS_ORDER = ["LOW", "MEDIUM", "HIGH"]
CLASS_WEIGHTS = {"LOW": 0.0, "MEDIUM": 0.5, "HIGH": 1.0}


class UserAdoptionRisk(Activity):
    spec = SPEC
    output_columns = ["user_id", "prediction_month", "risk_score", "risk_band", "model_version"]

    def load_dataset(self, conn: psycopg.Connection) -> pd.DataFrame:
        return db.read_frame(
            conn,
            "SELECT * FROM ml_features_user_adoption_risk ORDER BY user_id, prediction_month",
        )

    # ------------------------------------------------------------------ train
    def fit(self, df: pd.DataFrame) -> ActivityResult:
        df = df.copy()
        df["prediction_month"] = pd.to_datetime(df["prediction_month"])

        static_cols = fit_static_columns(df, CATEGORICAL)
        feature_columns = NUMERIC + static_cols
        X = pd.concat(
            [df[NUMERIC].astype(float), transform_static(static_cols, df, CATEGORICAL)],
            axis=1,
        )

        labeled = df[df["label_risk_band"].notna()].copy()
        if labeled.empty:
            raise RuntimeError(
                "No labelled rows in ml_features_user_adoption_risk. "
                "Run --step sql (01_raw_data_generators.sql) first."
            )
        y = labeled["label_risk_band"].astype(str)

        periods = sorted(labeled["prediction_month"].unique())
        if len(periods) > 1:
            train_mask = labeled["prediction_month"] < periods[-1]
            train_idx = labeled.index[train_mask]
            test_idx = labeled.index[~train_mask]
        else:
            idx = np.arange(len(labeled))
            train_sub, test_sub = train_test_split(
                idx, test_size=0.3, random_state=42, stratify=y.values
            )
            train_idx = labeled.index[train_sub]
            test_idx = labeled.index[test_sub]

        model = HistGradientBoostingClassifier(
            random_state=42, max_iter=300, learning_rate=0.08, min_samples_leaf=10
        )
        model.fit(X.loc[train_idx], y.loc[train_idx])

        classes = [str(c) for c in model.classes_]
        y_test = y.loc[test_idx]
        pred = model.predict(X.loc[test_idx])
        proba = model.predict_proba(X.loc[test_idx])
        metrics: dict[str, Any] = classification_metrics(
            y_test, pred, y_proba=proba, classes=classes
        )
        metrics["train_months"] = len({p for p in periods[:-1]}) if len(periods) > 1 else 1
        metrics["test_month"] = str(periods[-1].date()) if periods else ""

        return ActivityResult(
            model_version=make_model_version(self.spec.name),
            bundle={"model": model, "feature_columns": feature_columns,
                    "static_cols": static_cols, "classes": classes},
            metrics=metrics,
            feature_columns=feature_columns,
            training_rows=int(len(train_idx)),
            extras={"classes": classes},
        )

    # ---------------------------------------------------------------- predict
    def build_output(self, df: pd.DataFrame, result: ActivityResult) -> pd.DataFrame:
        df = df.copy()
        df["prediction_month"] = pd.to_datetime(df["prediction_month"])
        model = result.bundle["model"]
        feature_columns = result.bundle["feature_columns"]
        static_cols = result.bundle["static_cols"]
        classes = result.bundle["classes"]

        latest = df["prediction_month"].max()
        infer = df[df["prediction_month"] == latest]
        if infer.empty:
            return pd.DataFrame(columns=self.output_columns)

        X = pd.concat(
            [infer[NUMERIC].astype(float), transform_static(static_cols, infer, CATEGORICAL)],
            axis=1,
        )[feature_columns]
        proba = model.predict_proba(X)

        col_of = {c: i for i, c in enumerate(classes)}
        weights = np.array([CLASS_WEIGHTS.get(c, 0.0) for c in classes], dtype=float)
        risk_score = proba @ weights
        band_idx = proba.argmax(axis=1)
        bands = [classes[i] for i in band_idx]

        out = pd.DataFrame(
            {
                "user_id": infer["user_id"].astype(int).to_numpy(),
                "prediction_month": latest,
                "risk_score": np.clip(risk_score, 0.0, 1.0).round(4),
                "risk_band": bands,
                "model_version": result.model_version,
            }
        )
        _ = col_of  # kept for readability/debugging
        return out[self.output_columns]

    def delete_statement(
        self, result: ActivityResult, output: pd.DataFrame
    ) -> tuple[str, list[Any]]:
        month = pd.Timestamp(output["prediction_month"].iloc[0]).date() if len(output) else None
        return (
            "DELETE FROM ml_user_adoption_risk WHERE prediction_month = %s",
            [month],
        )
