"""Activity 1 — Tool adoption forecast (ml_tool_adoption_forecast).

Problem: monthly adoption rate per AI tool (share of all employees using the
tool that month) is a bounded time series in [0, 1].
Approach: gradient-boosted trees on lag features (prev 1/2/3 months, rolling
mean, first difference) + seasonality + static tool attributes, forecast
recursively 12 months ahead, prediction intervals from test residuals.
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
    name="tool_adoption_forecast",
    title="Tool adoption forecast",
    task_type="forecasting",
    output_table="ml_tool_adoption_forecast",
    algorithm="HistGradientBoostingRegressor + recursive multi-step",
    label_description="Monthly adoption rate = distinct tool users / all employees",
    output_score_col="predicted_adoption_rate",
)

HORIZON_MONTHS = 12
TEST_MONTHS = 3
STATIC_COLS = ["category", "vendor", "tool_name"]
BASE_NUMERIC = [
    "prev_1",
    "prev_2",
    "prev_3",
    "roll_mean_3",
    "diff_1",
    "months_since_release",
]
SEASON_COLS = ["month_sin", "month_cos"]


class ToolAdoptionForecast(Activity):
    spec = SPEC
    output_columns = [
        "tool_id",
        "forecast_month",
        "predicted_adoption_rate",
        "lower_bound",
        "upper_bound",
        "model_version",
    ]

    def load_dataset(self, conn: psycopg.Connection) -> pd.DataFrame:
        return db.read_frame(
            conn,
            "SELECT * FROM ml_features_tool_adoption ORDER BY tool_id, forecast_month",
        )

    # ------------------------------------------------------------------ train
    def fit(self, df: pd.DataFrame) -> ActivityResult:
        df = df.copy()
        df["forecast_month"] = pd.to_datetime(df["forecast_month"])
        m_idx = df["forecast_month"].map(month_index)
        df["month_sin"] = m_idx.map(season_sin)
        df["month_cos"] = m_idx.map(season_cos)

        static_cols = fit_static_columns(df, STATIC_COLS)
        feature_columns = BASE_NUMERIC + SEASON_COLS + static_cols

        X = pd.concat(
            [
                df[BASE_NUMERIC + SEASON_COLS].astype(float),
                transform_static(static_cols, df, STATIC_COLS),
            ],
            axis=1,
        )
        y = pd.to_numeric(df["label_adoption_rate"], errors="coerce")
        labeled_mask = y.notna()
        labeled = df.loc[labeled_mask]

        train, test = split_time(labeled, "forecast_month", TEST_MONTHS)
        model = HistGradientBoostingRegressor(
            random_state=42,
            max_iter=300,
            learning_rate=0.08,
            max_leaf_nodes=31,
            min_samples_leaf=10,
        )
        model.fit(X.loc[train.index], y.loc[train.index])

        metrics: dict[str, Any] = {}
        sigma = 0.05
        if not test.empty:
            pred_test = model.predict(X.loc[test.index])
            y_test = y.loc[test.index].to_numpy(dtype=float)
            metrics = regression_metrics(y_test, pred_test)
            sigma = float(np.std(y_test - pred_test, ddof=0))
        sigma = max(sigma, 0.01)

        metrics["interval_sigma"] = round(sigma, 6)
        metrics["train_months"] = int(train["forecast_month"].nunique())
        metrics["test_months"] = int(test["forecast_month"].nunique())

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
        df["forecast_month"] = pd.to_datetime(df["forecast_month"])
        model = result.bundle["model"]
        feature_columns = result.bundle["feature_columns"]
        static_cols = result.bundle["static_cols"]
        sigma = float(result.extras.get("interval_sigma", 0.05))
        version = result.model_version

        rows: list[dict[str, Any]] = []
        for tool_id, group in df.groupby("tool_id"):
            group = group.sort_values("forecast_month")
            last_month = group["forecast_month"].max()
            history = (
                group.dropna(subset=["label_adoption_rate"])["label_adoption_rate"]
                .astype(float)
                .tolist()
            )
            if not history:
                continue
            series = (history + [history[-1]] * 3)[-3:]
            while len(series) < 3:
                series.insert(0, series[0])

            tail = group.tail(1)
            static_vec = transform_static(static_cols, tail, STATIC_COLS)
            rel_mi = month_index(tail["release_date"].iloc[0])

            for step in range(1, HORIZON_MONTHS + 1):
                target = add_months(last_month, step)
                mi = month_index(target)
                feat = pd.DataFrame(
                    [
                        {
                            "prev_1": series[-1],
                            "prev_2": series[-2],
                            "prev_3": series[-3],
                            "roll_mean_3": float(np.mean(series[-3:])),
                            "diff_1": series[-1] - series[-2],
                            "months_since_release": mi - rel_mi,
                            "month_sin": season_sin(mi),
                            "month_cos": season_cos(mi),
                        }
                    ]
                )
                x_row = pd.concat(
                    [feat.reset_index(drop=True), static_vec.reset_index(drop=True)],
                    axis=1,
                )[feature_columns]
                rate = float(model.predict(x_row)[0])
                rate = float(min(1.0, max(0.0, rate)))
                rows.append(
                    {
                        "tool_id": int(tool_id),
                        "forecast_month": target,
                        "predicted_adoption_rate": round(rate, 4),
                        "lower_bound": round(float(min(1.0, max(0.0, rate - 1.96 * sigma))), 4),
                        "upper_bound": round(float(min(1.0, max(0.0, rate + 1.96 * sigma))), 4),
                        "model_version": version,
                    }
                )
                series = (series + [rate])[-3:]

        return pd.DataFrame(rows, columns=self.output_columns)

    def delete_statement(
        self, result: ActivityResult, output: pd.DataFrame
    ) -> tuple[str, list[Any]]:
        start = pd.Timestamp(output["forecast_month"].min()).date()
        return (
            "DELETE FROM ml_tool_adoption_forecast WHERE forecast_month >= %s",
            [start],
        )
