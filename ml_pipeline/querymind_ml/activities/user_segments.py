"""Activity 3 — User segmentation (ml_user_segments).

Problem: unsupervised behavioural segmentation of employees (no label).
Approach: StandardScaler + KMeans; k chosen by silhouette over 2..9 (preferring
the canonical k=5 when it is within 0.02 of the best score); clusters are then
named with the schema's canonical segment names using interpretable rules
(most active -> AI Power User, least active -> Occasional User, broadest tool
mix -> Multi-Tool User, highest concentration -> Specialized User, rest ->
AI Explorer). segment_score is the normalised membership strength.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import psycopg
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .. import db
from .base import (
    Activity,
    ActivityResult,
    ActivitySpec,
    current_month_start,
    make_model_version,
)

SPEC = ActivitySpec(
    name="user_segments",
    title="User segmentation",
    task_type="clustering",
    output_table="ml_user_segments",
    algorithm="StandardScaler + KMeans (silhouette-selected k), rule-based naming",
    label_description="Unsupervised: no label; behaviour features from full event history",
    output_score_col="segment_score",
)

NUMERIC = [
    "total_events",
    "distinct_tools",
    "distinct_tasks",
    "active_days",
    "active_months",
    "avg_session_seconds",
    "success_rate",
    "events_per_active_month",
    "top_tool_share",
    "distinct_categories",
    "recency_days",
]
CANONICAL_NAME_ORDER = [
    "AI Power User",
    "Multi-Tool User",
    "AI Explorer",
    "Specialized User",
    "Occasional User",
]


def _name_clusters(
    labels: np.ndarray,
    profile: pd.DataFrame,
) -> dict[int, str]:
    """Assign canonical segment names via interpretable ordering rules."""
    clusters = sorted(profile.index.tolist())
    by_activity = profile.sort_values("total_events", ascending=False).index.tolist()
    mapping: dict[int, str] = {}

    if len(by_activity) >= 1:
        mapping[int(by_activity[0])] = "AI Power User"
    if len(by_activity) >= 2:
        mapping[int(by_activity[-1])] = "Occasional User"

    middle = [c for c in by_activity[1:-1]]
    if middle:
        by_tools = profile.loc[middle].sort_values("distinct_tools", ascending=False)
        mapping[int(by_tools.index[0])] = "Multi-Tool User"
        remaining = [c for c in by_tools.index[1:]]
        if remaining:
            by_focus = profile.loc[remaining].sort_values("top_tool_share", ascending=False)
            mapping[int(by_focus.index[0])] = "Specialized User"
            for i, c in enumerate(by_focus.index[1:]):
                mapping[int(c)] = "Segment Focus" if i == 0 else f"Segment {int(c) + 1}"

    for c in clusters:
        mapping.setdefault(int(c), CANONICAL_NAME_ORDER[int(c) % len(CANONICAL_NAME_ORDER)])
    return mapping


class UserSegments(Activity):
    spec = SPEC
    output_columns = [
        "user_id",
        "prediction_month",
        "cluster_id",
        "segment_name",
        "segment_score",
        "model_version",
    ]

    def load_dataset(self, conn: psycopg.Connection) -> pd.DataFrame:
        return db.read_frame(
            conn,
            "SELECT * FROM ml_features_user_segments ORDER BY user_id",
        )

    # ------------------------------------------------------------------ train
    def fit(self, df: pd.DataFrame) -> ActivityResult:
        if df.empty:
            raise RuntimeError("ml_features_user_segments is empty. Run --step sql first.")

        X_raw = df[NUMERIC].astype(float)
        X_raw = X_raw.fillna(X_raw.median(numeric_only=True)).fillna(0.0)
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)

        n = X.shape[0]
        best: tuple[float, int, KMeans] | None = None
        score_5: float | None = None
        for k in range(2, min(10, n)):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            if len(set(labels)) < 2:
                continue
            score = float(silhouette_score(X, labels))
            if best is None or score > best[0]:
                best = (score, k, km)
            if k == 5:
                score_5 = score
        if best is None:  # extremely small input
            km = KMeans(n_clusters=1, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            best = (0.0, 1, km)
        score_best, k_best, kmeans = best
        if score_5 is not None and k_best != 5 and (score_best - score_5) <= 0.02:
            # Prefer the canonical 5-segment split when it is essentially tied.
            k_best = 5
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            kmeans.fit(X)
        labels = kmeans.labels_
        score_best = float(silhouette_score(X, labels)) if k_best > 1 else 0.0

        df = df.copy()
        df["_cluster"] = labels
        profile = df.groupby("_cluster")[["total_events", "distinct_tools", "top_tool_share"]].mean()
        cluster_names = _name_clusters(labels, profile)

        metrics: dict[str, Any] = {
            "k": int(k_best),
            "silhouette": round(float(score_best), 6),
            "inertia": round(float(kmeans.inertia_), 4),
            "rows": int(n),
            "cluster_sizes": {str(k): int(v) for k, v in df["_cluster"].value_counts().sort_index().items()},
            "cluster_profiles": {
                str(k): {
                    "segment_name": cluster_names.get(int(k), ""),
                    "avg_total_events": round(float(v["total_events"]), 2),
                    "avg_distinct_tools": round(float(v["distinct_tools"]), 3),
                    "avg_top_tool_share": round(float(v["top_tool_share"]), 4),
                }
                for k, v in profile.iterrows()
            },
        }

        return ActivityResult(
            model_version=make_model_version(self.spec.name),
            bundle={
                "scaler": scaler,
                "kmeans": kmeans,
                "feature_columns": NUMERIC,
                "cluster_names": {int(k): v for k, v in cluster_names.items()},
            },
            metrics=metrics,
            feature_columns=list(NUMERIC),
            training_rows=int(n),
            extras={"k": int(k_best)},
        )

    # ---------------------------------------------------------------- predict
    def build_output(self, df: pd.DataFrame, result: ActivityResult) -> pd.DataFrame:
        df = df.copy()
        scaler = result.bundle["scaler"]
        kmeans = result.bundle["kmeans"]
        cluster_names = result.bundle["cluster_names"]

        X_raw = df[NUMERIC].astype(float)
        X_raw = X_raw.fillna(X_raw.median(numeric_only=True)).fillna(0.0)
        X = scaler.transform(X_raw)

        labels = kmeans.predict(X)
        distances = kmeans.transform(X)
        assigned = distances[np.arange(len(df)), labels]

        # Membership strength: 1 at the centroid, 0 at the cluster's farthest point.
        scores = np.ones(len(df), dtype=float)
        for cluster in set(int(c) for c in labels):
            mask = labels == cluster
            d = assigned[mask]
            d_min, d_max = float(d.min()), float(d.max())
            if d_max > d_min:
                scores[mask] = 1.0 - (d - d_min) / (d_max - d_min)

        prediction_month = current_month_start()
        out = pd.DataFrame(
            {
                "user_id": df["user_id"].astype(int).to_numpy(),
                "prediction_month": prediction_month,
                "cluster_id": (labels + 1).astype(int),
                "segment_name": [
                    cluster_names.get(int(c), f"Segment {int(c) + 1}") for c in labels
                ],
                "segment_score": np.clip(scores, 0.0, 1.0).round(4),
                "model_version": result.model_version,
            }
        )
        return out[self.output_columns]

    def delete_statement(
        self, result: ActivityResult, output: pd.DataFrame
    ) -> tuple[str, list[Any]]:
        month = pd.Timestamp(output["prediction_month"].iloc[0]).date() if len(output) else None
        return (
            "DELETE FROM ml_user_segments WHERE prediction_month = %s",
            [month],
        )
