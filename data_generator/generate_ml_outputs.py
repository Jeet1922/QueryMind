from __future__ import annotations

import os
from datetime import date, timedelta

import psycopg


DEFAULT_DSN = "postgresql://querymind:querymind@localhost:5432/querymind"


def _fetch_tool_ids(conn: psycopg.Connection) -> list[int]:
    with conn.cursor() as cur:
        cur.execute("SELECT tool_id FROM ai_tools ORDER BY tool_id")
        return [row[0] for row in cur.fetchall()]


def _fetch_user_ids(conn: psycopg.Connection) -> list[int]:
    with conn.cursor() as cur:
        cur.execute("SELECT user_id FROM users ORDER BY user_id")
        return [row[0] for row in cur.fetchall()]


def _fetch_team_ids(conn: psycopg.Connection) -> list[int]:
    with conn.cursor() as cur:
        cur.execute("SELECT team_id FROM teams ORDER BY team_id")
        return [row[0] for row in cur.fetchall()]


def generate_ml_outputs(conn: psycopg.Connection) -> None:
    tool_ids = _fetch_tool_ids(conn)
    user_ids = _fetch_user_ids(conn)
    team_ids = _fetch_team_ids(conn)

    with conn.cursor() as cur:
        cur.execute("DELETE FROM ml_tool_recommendations")
        cur.execute("DELETE FROM ml_team_productivity_forecast")
        cur.execute("DELETE FROM ml_usage_anomalies")
        cur.execute("DELETE FROM ml_user_segments")
        cur.execute("DELETE FROM ml_user_adoption_risk")
        cur.execute("DELETE FROM ml_tool_adoption_forecast")

        month_start = date(2026, 1, 1)
        for tool_id in tool_ids:
            for i in range(12):
                forecast_month = month_start.replace(year=month_start.year + (month_start.month + i - 1) // 12, month=1 + (month_start.month + i - 1) % 12)
                base = 0.18 + (tool_id % 6) * 0.08 + (i * 0.025)
                predicted = min(0.92, max(0.12, round(base, 4)))
                lower = round(max(0.01, predicted - 0.08), 4)
                upper = round(min(0.99, predicted + 0.08), 4)
                cur.execute(
                    """
                    INSERT INTO ml_tool_adoption_forecast (
                        tool_id, forecast_month, predicted_adoption_rate, lower_bound, upper_bound, model_version
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (tool_id, forecast_month, predicted, lower, upper, "adoption-forecast-v1"),
                )

        for user_id in user_ids:
            for i in range(6):
                month = month_start.replace(year=month_start.year + (month_start.month + i - 1) // 12, month=1 + (month_start.month + i - 1) % 12)
                risk_score = ((user_id * 17 + i * 13) % 100) / 100
                risk_band = "LOW" if risk_score < 0.35 else "MEDIUM" if risk_score < 0.7 else "HIGH"
                cur.execute(
                    """
                    INSERT INTO ml_user_adoption_risk (
                        user_id, prediction_month, risk_score, risk_band, model_version
                    ) VALUES (%s, %s, %s, %s, %s)
                    """,
                    (user_id, month, round(risk_score, 4), risk_band, "adoption-risk-v1"),
                )

                cluster_id = (user_id + i) % 5 + 1
                segment_names = ["AI Explorer", "AI Power User", "Occasional User", "Multi-Tool User", "Specialized User"]
                segment_name = segment_names[(user_id + i) % len(segment_names)]
                segment_score = ((user_id * 11 + i * 7) % 100) / 100
                cur.execute(
                    """
                    INSERT INTO ml_user_segments (
                        user_id, prediction_month, cluster_id, segment_name, segment_score, model_version
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (user_id, month, cluster_id, segment_name, round(segment_score, 4), "segmentation-v1"),
                )

        for team_id in team_ids:
            for i in range(6):
                month = month_start.replace(year=month_start.year + (month_start.month + i - 1) // 12, month=1 + (month_start.month + i - 1) % 12)
                predicted_story_points = 65 + (team_id * 17 + i * 11) % 180
                lower = max(0, predicted_story_points - 20)
                upper = predicted_story_points + 25
                cur.execute(
                    """
                    INSERT INTO ml_team_productivity_forecast (
                        team_id, prediction_month, predicted_completed_story_points, lower_bound, upper_bound, model_version
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (team_id, month, predicted_story_points, lower, upper, "throughput-forecast-v1"),
                )

        for team_id in team_ids:
            for i in range(6):
                metric_date = month_start + timedelta(days=i * 30)
                tool_id = (team_id + i) % len(tool_ids) + 1
                actual_value = 210 + (team_id * 13 + i * 11) % 160
                expected_value = actual_value - 25 + (i % 3)
                anomaly_score = abs(actual_value - expected_value) / max(1, expected_value)
                is_anomaly = anomaly_score > 0.22
                cur.execute(
                    """
                    INSERT INTO ml_usage_anomalies (
                        metric_date, team_id, tool_id, metric_name, actual_value, expected_value, anomaly_score, is_anomaly, model_version
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (metric_date, team_id, tool_id, "usage_volume", actual_value, expected_value, round(anomaly_score, 4), is_anomaly, "anomaly-v1"),
                )

        for user_id in user_ids[:30]:
            for tool_id in tool_ids[:3]:
                reason = ["Strong productivity alignment", "High research frequency", "Cross-functional reuse", "Content workflow match"][tool_id % 4]
                score = 0.45 + ((user_id * 11 + tool_id * 7) % 100) / 200
                cur.execute(
                    """
                    INSERT INTO ml_tool_recommendations (
                        user_id, team_id, tool_id, recommendation_score, recommendation_reason, model_version
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (user_id, team_ids[user_id % len(team_ids)], tool_id, round(score, 4), reason, "recommendation-v1"),
                )


def generate_ml_outputs_from_database(dsn: str | None = None) -> None:
    dsn = dsn or os.getenv("DATABASE_URL", DEFAULT_DSN)
    with psycopg.connect(dsn) as conn:
        generate_ml_outputs(conn)
        conn.commit()
        print("Synthetic ML outputs generated.")


if __name__ == "__main__":
    generate_ml_outputs_from_database()
