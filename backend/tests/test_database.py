import psycopg
import pytest

from backend.app.db.dsn import get_postgres_url


DATABASE_URL = get_postgres_url()
pytestmark = pytest.mark.skipif(DATABASE_URL is None, reason="A valid PostgreSQL DATABASE_URL is required for database integration tests.")


REQUIRED_TABLES = {
    "departments",
    "practices",
    "teams",
    "users",
    "ai_tools",
    "ai_tasks",
    "ai_tool_usage",
    "team_work_metrics",
    "ml_tool_adoption_forecast",
    "ml_user_adoption_risk",
    "ml_user_segments",
    "ml_usage_anomalies",
    "ml_team_productivity_forecast",
    "ml_tool_recommendations",
}


def get_connection():
    return psycopg.connect(DATABASE_URL)


def test_required_tables_exist():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name = ANY(%s)
                """,
                [sorted(REQUIRED_TABLES)],
            )
            table_names = {row[0] for row in cur.fetchall()}
    assert REQUIRED_TABLES.issubset(table_names)


def test_reference_data_loaded():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM departments")
            assert cur.fetchone()[0] >= 3

            cur.execute("SELECT COUNT(*) FROM ai_tools")
            assert cur.fetchone()[0] >= 5

            cur.execute("SELECT COUNT(*) FROM users")
            assert cur.fetchone()[0] >= 20


def test_ml_outputs_loaded():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM ml_tool_adoption_forecast")
            assert cur.fetchone()[0] > 0

            cur.execute("SELECT COUNT(*) FROM ml_user_adoption_risk")
            assert cur.fetchone()[0] > 0

            cur.execute("SELECT COUNT(*) FROM ml_team_productivity_forecast")
            assert cur.fetchone()[0] > 0
