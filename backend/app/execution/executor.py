from __future__ import annotations

from typing import Any

import psycopg

from backend.app.db.dsn import get_postgres_url
from backend.app.db.sqlite import get_connection
from backend.app.procedures.registry import get_procedure


class ProcedureExecutor:
    """Execute approved procedures against Neon/PostgreSQL or local SQLite."""

    def __init__(self, connection: Any | None = None, database_url: str | None = None):
        self.connection = connection
        self.database_url = get_postgres_url(database_url)
        self.backend = "postgresql" if self.database_url else "sqlite"

    def execute_procedure(self, procedure_name: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        get_procedure(procedure_name)
        params = params or {}
        if self.backend == "postgresql":
            return self._execute_postgres_procedure(procedure_name, params)

        query, values = self._resolve_sql(procedure_name, params)

        with get_connection() as conn:
            cursor = conn.execute(query, values)
            rows = [dict(row) for row in cursor.fetchall()]

        return rows

    def _execute_postgres_procedure(self, procedure_name: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        query, values = self._resolve_postgres_call(procedure_name, params)
        connection = self.connection or psycopg.connect(self.database_url)
        owns_connection = self.connection is None
        try:
            cursor = connection.cursor()
            try:
                cursor.execute(query, values)
                columns = [getattr(column, "name", column[0]) for column in cursor.description or []]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
            finally:
                cursor.close()
        finally:
            if owns_connection:
                connection.close()

    @staticmethod
    def _resolve_postgres_call(procedure_name: str, params: dict[str, Any]) -> tuple[str, tuple[Any, ...]]:
        limit = min(max(int(params.get("limit", params.get("forecast_months", 10))), 1), 100)

        if procedure_name == "get_tool_ranking":
            time_period = str(params.get("time_period", "last_year")).lower()
            days = 30 if time_period in {"last_month", "30_days", "month"} else 90 if time_period in {"last_quarter", "90_days", "quarter"} else 365
            return "SELECT * FROM get_tool_ranking(%s, %s)", (days, limit)
        if procedure_name == "get_tool_usage_trend":
            tool_id = int(params.get("tool_id", 0))
            months = min(max(int(params.get("months", 12)), 1), 36)
            return "SELECT * FROM get_tool_usage_trend(%s, %s)", (tool_id, months)
        if procedure_name == "get_department_adoption":
            days = 30 if str(params.get("time_period", "last_year")).lower() == "last_month" else 365
            return "SELECT * FROM get_department_adoption(%s)", (days,)
        if procedure_name == "get_tool_adoption_forecast":
            tool_id = int(params.get("tool_id", 0))
            months = min(max(int(params.get("forecast_months", 6)), 1), 12)
            return "SELECT * FROM get_tool_adoption_forecast(%s, %s)", (tool_id, months)
        if procedure_name == "get_department_adoption_risk":
            prediction_month = params.get("prediction_month")
            if prediction_month == "current_month":
                prediction_month = None
            return "SELECT * FROM get_department_adoption_risk(%s)", (prediction_month,)
        if procedure_name == "get_user_segment_distribution":
            prediction_month = params.get("prediction_month")
            if prediction_month == "current_month":
                prediction_month = None
            return "SELECT * FROM get_user_segment_distribution(%s)", (prediction_month,)
        if procedure_name == "get_usage_anomalies":
            team_id = params.get("team_id")
            return "SELECT * FROM get_usage_anomalies(%s, %s)", (int(team_id) if team_id is not None else None, limit)
        if procedure_name == "get_team_productivity_forecast":
            team_id = params.get("team_id")
            return "SELECT * FROM get_team_productivity_forecast(%s, %s)", (int(team_id) if team_id is not None else None, limit)
        raise KeyError(f"Procedure '{procedure_name}' is not implemented for PostgreSQL execution.")

    def execute_step(self, step: dict[str, Any]) -> dict[str, Any]:
        procedure_name = step.get("procedure")
        parameters = step.get("parameters", {})
        rows = self.execute_procedure(procedure_name, parameters)
        return {
            "procedure": procedure_name,
            "parameters": parameters,
            "status": "ok",
            "rows": rows,
            "rows_returned": len(rows),
        }

    def execute_plan(self, plan: dict[str, Any]) -> list[dict[str, Any]]:
        results = []
        for step in plan.get("steps", []):
            resolved_step = {**step, "parameters": dict(step.get("parameters", {}))}
            for name, value in resolved_step["parameters"].items():
                if isinstance(value, str) and value.startswith("$"):
                    step_id = value[1:].split(".", 1)[0]
                    dependency = next((result for result in results if result.get("id") == step_id), None)
                    if dependency and dependency.get("rows"):
                        resolved_step["parameters"][name] = dependency["rows"][0].get("tool_id", 0)
            output = self.execute_step(resolved_step)
            output["id"] = step.get("id")
            results.append(output)
        return results

    @staticmethod
    def _resolve_sql(procedure_name: str, params: dict[str, Any]) -> tuple[str, tuple[Any, ...]]:
        limit = int(params.get("limit", params.get("forecast_months", params.get("forecast_window", 5))))
        if procedure_name == "get_tool_ranking":
            return (
                """
                  SELECT t.tool_id,
                      t.tool_name,
                       COUNT(au.usage_id) AS adoption_count,
                       ROUND(CAST(COUNT(au.usage_id) AS FLOAT) / NULLIF((SELECT COUNT(*) FROM ai_tool_usage), 0), 4) AS adoption_score
                FROM ai_tools t
                LEFT JOIN ai_tool_usage au ON au.tool_id = t.tool_id
                GROUP BY t.tool_id, t.tool_name
                ORDER BY adoption_count DESC, t.tool_name ASC
                LIMIT ?
                """,
                (limit,),
            )

        if procedure_name == "get_tool_usage_trend":
            tool_id = int(params.get("tool_id", 0))
            return (
                """
                SELECT strftime('%Y-%m', au.usage_timestamp) AS usage_month,
                       COUNT(au.usage_id) AS usage_count,
                       SUM(au.session_duration_seconds) AS total_session_seconds,
                       t.tool_name
                FROM ai_tool_usage au
                JOIN ai_tools t ON t.tool_id = au.tool_id
                WHERE au.tool_id = ?
                GROUP BY strftime('%Y-%m', au.usage_timestamp), t.tool_name
                ORDER BY usage_month ASC
                LIMIT 12
                """,
                (tool_id,),
            )

        if procedure_name == "get_department_adoption":
            return (
                """
                SELECT d.department_name,
                       d.department_id,
                       ROUND(CAST(COUNT(au.usage_id) AS FLOAT) / NULLIF(COUNT(DISTINCT u.user_id), 0), 4) AS adoption_rate
                FROM departments d
                LEFT JOIN users u ON u.department_id = d.department_id
                LEFT JOIN ai_tool_usage au ON au.user_id = u.user_id
                GROUP BY d.department_id, d.department_name
                ORDER BY adoption_rate DESC, d.department_name ASC
                LIMIT ?
                """,
                (limit,),
            )

        if procedure_name == "get_tool_adoption_forecast":
            tool_id = int(params.get("tool_id", 0))
            return (
                """
                SELECT t.tool_name,
                       f.forecast_month,
                       f.predicted_adoption_rate,
                       f.lower_bound,
                       f.upper_bound,
                       f.model_version
                FROM ml_tool_adoption_forecast f
                JOIN ai_tools t ON t.tool_id = f.tool_id
                WHERE f.tool_id = ?
                ORDER BY f.forecast_month ASC
                LIMIT ?
                """,
                (tool_id, limit),
            )

        if procedure_name == "get_department_adoption_risk":
            prediction_month = params.get("prediction_month", "2026-01-01")
            if prediction_month == "current_month":
                prediction_month = "2026-01-01"
            return (
                """
                SELECT d.department_name,
                       ROUND(AVG(r.risk_score), 4) AS risk_score,
                       CASE WHEN AVG(r.risk_score) >= 0.7 THEN 'HIGH' WHEN AVG(r.risk_score) >= 0.35 THEN 'MEDIUM' ELSE 'LOW' END AS risk_band
                FROM ml_user_adoption_risk r
                JOIN users u ON u.user_id = r.user_id
                JOIN departments d ON d.department_id = u.department_id
                WHERE r.prediction_month = ?
                GROUP BY d.department_id, d.department_name
                ORDER BY risk_score DESC
                LIMIT ?
                """,
                (prediction_month, limit),
            )

        if procedure_name == "get_user_segment_distribution":
            prediction_month = params.get("prediction_month", "2026-01-01")
            if prediction_month == "current_month":
                prediction_month = "2026-01-01"
            return (
                """
                SELECT s.segment_name,
                       COUNT(s.user_id) AS user_count,
                       ROUND(AVG(s.segment_score), 4) AS avg_score
                FROM ml_user_segments s
                WHERE s.prediction_month = ?
                GROUP BY s.segment_name
                ORDER BY user_count DESC
                LIMIT ?
                """,
                (prediction_month, limit),
            )

        if procedure_name == "get_usage_anomalies":
            team_id = params.get("team_id")
            if team_id is not None:
                query = """
                    SELECT t.team_name,
                           tool.tool_name,
                           a.anomaly_score,
                           a.is_anomaly,
                           a.metric_date
                    FROM ml_usage_anomalies a
                    JOIN teams t ON t.team_id = a.team_id
                    JOIN ai_tools tool ON tool.tool_id = a.tool_id
                    WHERE a.team_id = ?
                    ORDER BY a.anomaly_score DESC
                    LIMIT ?
                """
                return query, (int(team_id), limit)
            return (
                """
                SELECT t.team_name,
                       tool.tool_name,
                       a.anomaly_score,
                       a.is_anomaly,
                       a.metric_date
                FROM ml_usage_anomalies a
                JOIN teams t ON t.team_id = a.team_id
                JOIN ai_tools tool ON tool.tool_id = a.tool_id
                ORDER BY a.anomaly_score DESC
                LIMIT ?
                """,
                (limit,),
            )

        if procedure_name == "get_team_productivity_forecast":
            team_id = params.get("team_id")
            if team_id is not None:
                return (
                    """
                    SELECT t.team_name,
                           f.prediction_month,
                           f.predicted_completed_story_points,
                           f.lower_bound,
                           f.upper_bound,
                           f.model_version
                    FROM ml_team_productivity_forecast f
                    JOIN teams t ON t.team_id = f.team_id
                    WHERE f.team_id = ?
                    ORDER BY f.prediction_month ASC
                    LIMIT ?
                    """,
                    (int(team_id), limit),
                )
            return (
                """
                SELECT t.team_name,
                       f.prediction_month,
                       f.predicted_completed_story_points,
                       f.lower_bound,
                       f.upper_bound,
                       f.model_version
                FROM ml_team_productivity_forecast f
                JOIN teams t ON t.team_id = f.team_id
                ORDER BY f.prediction_month ASC
                LIMIT ?
                """,
                (limit,),
            )

        raise KeyError(f"Procedure '{procedure_name}' is not implemented for SQLite execution.")
