"""Activity 6 — Tool recommendations (ml_tool_recommendations).

Problem: rank the AI-tool catalogue for each employee (implicit feedback).
Approach: item-item collaborative filtering (cosine over a weighted user x
tool matrix, success-weighted counts) blended with a popularity prior and a
content bonus for categories the employee already works with:

    score = 0.65 * collab + 0.25 * popularity + 0.10 * category fit

Evaluation uses a 90-day interaction window and a temporal split: tools FIRST
used during the last 30 days are the "new tool" test set, scored with
precision@5 / recall@5 / MAP@5. Candidates are tools the employee is not
meaningfully engaged with (below 8% share of window usage), so the ranking
stays a genuine discovery/expansion signal rather than echoing top usage.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import psycopg
from sklearn.metrics.pairwise import cosine_similarity

from .. import db
from .base import Activity, ActivityResult, ActivitySpec, make_model_version

SPEC = ActivitySpec(
    name="tool_recommendations",
    title="Tool recommendations",
    task_type="recommendation",
    output_table="ml_tool_recommendations",
    algorithm="Item-item CF (cosine) + popularity + category fit",
    label_description="Temporal holdout: tools first used in the last 30 days",
    output_score_col="recommendation_score",
)

TOP_K = 5
TEST_DAYS = 30
LOOKBACK_DAYS = 90
ENGAGEMENT_SHARE = 0.08
W_COLLAB = 0.65
W_POP = 0.25
W_CATEGORY = 0.10
SUCCESS_WEIGHT_TRUE = 1.0
SUCCESS_WEIGHT_FALSE = 0.6

REASON_COLLAB = "Strong match with tools similar colleagues already use"
REASON_POPULAR = "Popular across teams with similar workflows"
REASON_CATEGORY = "Fits the {category} workflows you already use"


def _prepare(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    out["event_timestamp"] = pd.to_datetime(out["event_timestamp"])
    success = out["success_flag"].astype(bool)
    out["weight"] = np.where(success, SUCCESS_WEIGHT_TRUE, SUCCESS_WEIGHT_FALSE)
    return out


def _matrix(events: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pivot = (
        events.pivot_table(
            index="user_id", columns="tool_id", values="weight",
            aggfunc="sum", fill_value=0.0,
        )
        .sort_index()
        .sort_index(axis=1)
    )
    return pivot.index.to_numpy(), pivot.columns.to_numpy(), pivot.to_numpy(dtype=float)


def _score_users(
    events: pd.DataFrame,
    categories: dict[int, str],
    top_k: int,
) -> pd.DataFrame:
    """Rank candidate tools for every user.

    Candidates = tools with a materiality share below ENGAGEMENT_SHARE in the
    window (i.e. tools the employee does NOT already rely on).
    """
    user_ids, tool_ids, mat = _matrix(events)
    if len(tool_ids) == 0:
        return pd.DataFrame(columns=["user_id", "tool_id", "recommendation_score",
                                     "recommendation_reason"])

    item_sim = cosine_similarity(mat.T)  # tools x tools
    popularity = mat.sum(axis=0)
    pop_norm = popularity / popularity.max() if popularity.max() > 0 else popularity

    rows: list[dict[str, Any]] = []
    n_tools = len(tool_ids)
    for i, user_id in enumerate(user_ids):
        totals = mat[i]
        used = np.where(totals > 0)[0]
        if not len(used):
            continue
        total = float(totals.sum())
        shares = (totals / total) if total > 0 else totals
        engaged = {int(j) for j in used if shares[j] >= ENGAGEMENT_SHARE}
        if len(engaged) >= n_tools:
            # Unusually even usage: only the top-2 tools count as "owned".
            top = sorted(used.tolist(), key=lambda j: -shares[j])
            engaged = set(int(j) for j in top[:2])
        if not engaged:
            engaged = {int(max(used.tolist(), key=lambda j: shares[j]))}
        candidates = [j for j in range(n_tools) if j not in engaged]
        if not candidates:
            continue
        user_categories = {categories.get(int(tool_ids[j]), "") for j in used}

        collab_raw = np.array([
            float(item_sim[j, used].max()) if len(used) else 0.0 for j in candidates
        ])
        collab_span = collab_raw.max() - collab_raw.min()
        collab_n = (collab_raw - collab_raw.min()) / collab_span if collab_span > 0 else np.ones_like(collab_raw)
        pop_n = np.array([pop_norm[j] for j in candidates])
        cat_n = np.array([
            1.0 if categories.get(int(tool_ids[j]), "") in user_categories else 0.0
            for j in candidates
        ])

        raw = W_COLLAB * collab_n + W_POP * pop_n + W_CATEGORY * cat_n
        span = float(raw.max() - raw.min())
        scores = (raw - raw.min()) / span if span > 0 else np.ones_like(raw)

        order = sorted(
            range(len(candidates)),
            key=lambda r: (-scores[r], tool_ids[candidates[r]]),
        )[:top_k]
        for r in order:
            j = candidates[r]
            contributions = {
                "collab": W_COLLAB * collab_n[r],
                "popular": W_POP * pop_n[r],
                "category": W_CATEGORY * cat_n[r],
            }
            driver = max(contributions, key=contributions.get)
            if driver == "collab":
                reason = REASON_COLLAB
            elif driver == "popular":
                reason = REASON_POPULAR
            else:
                reason = REASON_CATEGORY.format(
                    category=categories.get(int(tool_ids[j]), "matching")
                )
            rows.append(
                {
                    "user_id": int(user_id),
                    "tool_id": int(tool_ids[j]),
                    "recommendation_score": round(float(scores[r]), 4),
                    "recommendation_reason": reason[:255],
                }
            )
    return pd.DataFrame(rows)


class ToolRecommendations(Activity):
    spec = SPEC
    output_columns = [
        "user_id",
        "team_id",
        "tool_id",
        "recommendation_score",
        "recommendation_reason",
        "model_version",
    ]

    def load_dataset(self, conn: psycopg.Connection) -> pd.DataFrame:
        return db.read_frame(
            conn,
            """
            SELECT e.user_id, e.team_id, e.tool_id, e.event_timestamp,
                   e.success_flag, t.category
            FROM ml_raw_usage_events e
            JOIN ai_tools t ON t.tool_id = e.tool_id
            ORDER BY e.event_timestamp
            """,
        )

    # ------------------------------------------------------------------ train
    def fit(self, df: pd.DataFrame) -> ActivityResult:
        if df.empty:
            raise RuntimeError("ml_raw_usage_events is empty. Run --step sql first.")
        events = _prepare(df)
        categories = {
            int(r.tool_id): str(r.category) for r in events[["tool_id", "category"]].drop_duplicates().itertuples()
        }

        cutoff = events["event_timestamp"].max() - pd.Timedelta(days=TEST_DAYS)
        train = events[
            (events["event_timestamp"] >= cutoff - pd.Timedelta(days=LOOKBACK_DAYS))
            & (events["event_timestamp"] < cutoff)
        ]
        test = events[events["event_timestamp"] >= cutoff]

        # Test pairs = tools FIRST touched during the holdout window.
        train_pairs = set(zip(train["user_id"], train["tool_id"]))
        test_pairs = {
            (u, t) for u, t in zip(test["user_id"], test["tool_id"])
            if (u, t) not in train_pairs
        }

        metrics: dict[str, Any] = {
            "train_events": int(len(train)),
            "test_events": int(len(test)),
            "new_tool_pairs": int(len(test_pairs)),
            "lookback_days": LOOKBACK_DAYS,
            "test_window_days": TEST_DAYS,
            "weights": {"collab": W_COLLAB, "popular": W_POP, "category": W_CATEGORY},
        }

        if test_pairs:
            recs = _score_users(train, categories, top_k=TOP_K)
            rec_map: dict[int, set[int]] = {}
            for r in recs.itertuples():
                rec_map.setdefault(int(r.user_id), set()).add(int(r.tool_id))
            test_by_user: dict[int, set[int]] = {}
            for u, t in test_pairs:
                test_by_user.setdefault(int(u), set()).add(int(t))

            precisions, recalls, aps = [], [], []
            for user_id, truth in test_by_user.items():
                recommended = rec_map.get(user_id, set())
                hits = recommended & truth
                precisions.append(len(hits) / TOP_K)
                recalls.append(len(hits) / len(truth))
                aps.append(len(hits) / TOP_K)
            metrics["precision_at_5"] = round(float(np.mean(precisions)), 6)
            metrics["recall_at_5"] = round(float(np.mean(recalls)), 6)
            metrics["map_at_5"] = round(float(np.mean(aps)), 6)
            metrics["evaluated_users"] = int(len(test_by_user))

        return ActivityResult(
            model_version=make_model_version(self.spec.name),
            bundle={
                "top_k": TOP_K,
                "categories": categories,
                "weights": {"collab": W_COLLAB, "popular": W_POP, "category": W_CATEGORY},
            },
            metrics=metrics,
            feature_columns=[
                "weighted_user_tool_matrix",
                "item_item_cosine_similarity",
                "popularity_prior",
                "category_fit",
            ],
            training_rows=int(len(train)),
        )

    # ---------------------------------------------------------------- predict
    def build_output(self, df: pd.DataFrame, result: ActivityResult) -> pd.DataFrame:
        events = _prepare(df)
        categories = {
            int(r.tool_id): str(r.category)
            for r in events[["tool_id", "category"]].drop_duplicates().itertuples()
        }
        max_ts = events["event_timestamp"].max()
        window = events[events["event_timestamp"] >= max_ts - pd.Timedelta(days=LOOKBACK_DAYS)]
        recs = _score_users(window, categories, top_k=int(result.bundle.get("top_k", TOP_K)))
        if recs.empty:
            return pd.DataFrame(columns=self.output_columns)

        team_map = (
            events[["user_id", "team_id"]].drop_duplicates("user_id").set_index("user_id")["team_id"]
        )
        recs["team_id"] = recs["user_id"].map(team_map).astype(int)
        recs["model_version"] = result.model_version
        recs = recs.sort_values(["user_id", "recommendation_score", "tool_id"], ascending=[True, False, True])
        return recs[self.output_columns]

    def delete_statement(
        self, result: ActivityResult, output: pd.DataFrame
    ) -> tuple[str, list[Any]]:
        # Small, fully regenerated table: refresh wholesale on each run.
        return ("DELETE FROM ml_tool_recommendations", [])
