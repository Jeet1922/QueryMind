"""Activity 5 — Team productivity forecast (ml_team_productivity_forecast).

Problem: forecast next months' completed story points per team (integer >= 0).
Approach: gradient-boosted trees on throughput lags (1..6 months), trailing
mean, delivery-health context (cycle/lead time, story counts) + static team /
department one-hots; recursive 6-month horizon with prediction intervals from
test residuals. Training uses complete months only (the current month row is
partial and excluded).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import psycopg
from sklearn.ensemble import HistGradientBoostingRegressor

from .. import db
from .base import (
    Activity,
    ActivityResult,
    ActivitySpec,
    add_months,
    current_month_start,
    fit_static_columns,
    make_model_version,
    month_index,
    regression_metrics,
    season_cos,
    season_sin,
    split_time,
    transform_static,
)

SPEC = ActivitySpec(
    name="team_productivity_forecast",
    title="Team productivity forecast",
    task_type="forecasting",
    output_table="ml_team_productivity_forecast",
    algorithm="HistGradientBoostingRegressor + recursive multi-step",
    label_description="Monthly completed_story_points from team_work_metrics",
    output_score_col="predicted_completed_story_points",
)

HORIZON_MONTHS = 6
TEST_MONTHS = 3
STATIC_COLS = ["team_name", "department_name"]
LAG_COLS = [
    "prev_1_points",
    "prev_2_points",
    "prev_3_points",
    "prev_4_points",
    "prev_5_points",
    "prev_6_points",
]
CONTEXT_COLS = [
    "roll_mean_3",
    "prev_1_cycle",
    "prev_1_lead",
    "prev_1_completed",
    "prev_1_story_count",
    "team_size",
]
SEASON_COLS = ["month_sin", "month_cos"]


class TeamProductivityForecast(Activity):
    spec = SPEC
    output_columns = [
        "team_id",
        "prediction_month",
        "predicted_completed_story_points",
        "lower_bound",
        "upper_bound",
        "model_version",
    ]

    def load_dataset(self, conn: psycopg.Connection) -> pd.DataFrame:
        return db.read_frame(
            conn,
            "SELECT * FROM ml_features_team_productivity ORDER BY team_id, metric_month",
        )

    # ------------------------------------------------------------------ train
    def fit(self, df: pd.DataFrame) -> ActivityResult:
        df = df.copy()
        df["metric_month"] = pd.to_datetime(df["metric_month"])
        m_idx = df["metric_month"].map(month_index)
        df["month_sin"] = m_idx.map(season_sin)
        df["month_cos"] = m_idx.map(season_cos)

        static_cols = fit_static_columns(df, STATIC_COLS)
        feature_columns = LAG_COLS + CONTEXT_COLS + SEASON_COLS + static_cols
        X = pd.concat(
            [
                df[LAG_COLS + CONTEXT_COLS + SEASON_COLS].astype(float),
                transform_static(static_cols, df, STATIC_COLS),
            ],
            axis=1,
        )
        y = pd.to_numeric(df["label_completed_story_points"], errors="coerce")

        complete = df[df["metric_month"] < current_month_start()]
        if len(complete) < 8:
            raise RuntimeError(
                "Not enough complete months in team_work_metrics. "
                "Run --step sql (01_raw_data_generators.sql) to extend history."
            )
        train, test = split_time(complete, "metric_month", TEST_MONTHS)

        model = HistGradientBoostingRegressor(
            random_state=42,
            max_iter=300,
            learning_rate=0.08,
            max_leaf_nodes=31,
            min_samples_leaf=10,
        )
        model.fit(X.loc[train.index], y.loc[train.index])

        metrics: dict[str, Any] = {}
        sigma = 25.0
        if not test.empty:
            pred_test = model.predict(X.loc[test.index])
            y_test = y.loc[test.index].to_numpy(dtype=float)
            metrics = regression_metrics(y_test, pred_test)
            sigma = float(np.std(y_test - pred_test, ddof=0))
        sigma = max(sigma, 5.0)
        metrics["interval_sigma"] = round(sigma, 4)
        metrics["train_months"] = int(train["metric_month"].nunique())
        metrics["test_months"] = int(test["metric_month"].nunique())

        return ActivityResult(
            model_version=make_model_version(self.spec.name),
            bundle={
                "model": model,
                "feature_columns": feature_columns,
                "static_cols": static_cols,
            },
            metrics=metrics,
            feature_columns=feature_columns,
            training_rows=int(len(train)),
            extras={"interval_sigma": sigma},
        )

    # ---------------------------------------------------------------- predict
    def build_output(self, df: pd.DataFrame, result: ActivityResult) -> pd.DataFrame:
        df = df.copy()
        df["metric_month"] = pd.to_datetime(df["metric_month"])
        model = result.bundle["model"]
        feature_columns = result.bundle["feature_columns"]
        static_cols = result.bundle["static_cols"]
        sigma = float(result.extras.get("interval_sigma", 25.0))
        version = result.model_version

        current = current_month_start()
        rows: list[dict[str, Any]] = []
        for team_id, group in df.groupby("team_id"):
            group = group.sort_values("metric_month")
            known = group[group["metric_month"] < current]
            if known.empty:
                continue
            points = known["label_completed_story_points"].astype(float).tolist()
            if len(points) < 6:
                points = [points[0]] * (6 - len(points)) + points
            series = points[-6:]
            tail = known.tail(1)
            static_vec = transform_static(static_cols, tail, STATIC_COLS)
            ctx = {
                "prev_1_cycle": float(tail["prev_1_cycle"].iloc[0])
                if pd.notna(tail["prev_1_cycle"].iloc[0])
                else 8.0,
                "prev_1_lead": float(tail["prev_1_lead"].iloc[0])
                if pd.notna(tail["prev_1_lead"].iloc[0])
                else 11.0,
                "prev_1_completed": float(tail["prev_1_completed"].iloc[0])
                if pd.notna(tail["prev_1_completed"].iloc[0])
                else 40.0,
                "prev_1_story_count": float(tail["prev_1_story_count"].iloc[0])
                if pd.notna(tail["prev_1_story_count"].iloc[0])
                else 70.0,
                "team_size": float(tail["team_size"].iloc[0]),
            }
            base_month_ts = known["metric_month"].max()

            for step in range(1, HORIZON_MONTHS + 1):
                target = add_months(current, step)
                mi = month_index(target)
                feat = {
                    "prev_1_points": series[-1],
                    "prev_2_points": series[-2],
                    "prev_3_points": series[-3],
                    "prev_4_points": series[-4],
                    "prev_5_points": series[-5],
                    "prev_6_points": series[-6],
                    "roll_mean_3": float(np.mean(series[-3:])),
                    "month_sin": season_sin(mi),
                    "month_cos": season_cos(mi),
                    **ctx,
                }
                x_row = pd.concat(
                    [
                        pd.DataFrame([feat])[LAG_COLS + CONTEXT_COLS + SEASON_COLS],
                        static_vec.reset_index(drop=True),
                    ],
                    axis=1,
                )[feature_columns]
                predicted = float(model.predict(x_row)[0])
                predicted = max(0.0, predicted)
                rows.append(
                    {
                        "team_id": int(team_id),
                        "prediction_month": target,
                        "predicted_completed_story_points": int(round(predicted)),
                        "lower_bound": int(max(0, round(predicted - 1.96 * sigma))),
                        "upper_bound": int(max(0, round(predicted + 1.96 * sigma))),
                        "model_version": version,
                    }
                )
                series = (series + [predicted])[-6:]
            _ = base_month_ts  # horizon anchors to the month after the current one

        return pd.DataFrame(rows, columns=self.output_columns)

    def delete_statement(
        self, result: ActivityResult, output: pd.DataFrame
    ) -> tuple[str, list[Any]]:
        start = pd.Timestamp(output["prediction_month"].min()).date() if len(output) else None
        return (
            "DELETE FROM ml_team_productivity_forecast WHERE prediction_month >= %s",
            [start],
        )
