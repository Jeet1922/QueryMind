from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT / "data" / "insightmesh.sqlite3"


def generate_sqlite_database(target_path: str | None = None) -> str:
    db_path = Path(target_path or str(DEFAULT_DB_PATH))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")

    conn.executescript(
        """
        DROP TABLE IF EXISTS query_history;
        DROP TABLE IF EXISTS ml_tool_recommendations;
        DROP TABLE IF EXISTS ml_team_productivity_forecast;
        DROP TABLE IF EXISTS ml_usage_anomalies;
        DROP TABLE IF EXISTS ml_user_segments;
        DROP TABLE IF EXISTS ml_user_adoption_risk;
        DROP TABLE IF EXISTS ml_tool_adoption_forecast;
        DROP TABLE IF EXISTS team_work_metrics;
        DROP TABLE IF EXISTS ai_tool_usage;
        DROP TABLE IF EXISTS ai_tasks;
        DROP TABLE IF EXISTS ai_tools;
        DROP TABLE IF EXISTS users;
        DROP TABLE IF EXISTS teams;
        DROP TABLE IF EXISTS practices;
        DROP TABLE IF EXISTS departments;

        CREATE TABLE departments (
            department_id INTEGER PRIMARY KEY AUTOINCREMENT,
            department_name TEXT NOT NULL UNIQUE,
            description TEXT NOT NULL
        );

        CREATE TABLE practices (
            practice_id INTEGER PRIMARY KEY AUTOINCREMENT,
            practice_name TEXT NOT NULL UNIQUE,
            description TEXT NOT NULL
        );

        CREATE TABLE teams (
            team_id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_name TEXT NOT NULL UNIQUE,
            department_id INTEGER NOT NULL,
            practice_id INTEGER NOT NULL,
            team_size INTEGER NOT NULL,
            FOREIGN KEY (department_id) REFERENCES departments(department_id),
            FOREIGN KEY (practice_id) REFERENCES practices(practice_id)
        );

        CREATE TABLE users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_code TEXT NOT NULL UNIQUE,
            department_id INTEGER NOT NULL,
            team_id INTEGER NOT NULL,
            practice_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            location TEXT NOT NULL,
            employment_type TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (department_id) REFERENCES departments(department_id),
            FOREIGN KEY (team_id) REFERENCES teams(team_id),
            FOREIGN KEY (practice_id) REFERENCES practices(practice_id)
        );

        CREATE TABLE ai_tools (
            tool_id INTEGER PRIMARY KEY AUTOINCREMENT,
            tool_name TEXT NOT NULL UNIQUE,
            vendor TEXT NOT NULL,
            category TEXT NOT NULL,
            description TEXT NOT NULL,
            release_date TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active'
        );

        CREATE TABLE ai_tasks (
            task_id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_category TEXT NOT NULL,
            task_name TEXT NOT NULL UNIQUE,
            description TEXT NOT NULL
        );

        CREATE TABLE ai_tool_usage (
            usage_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            tool_id INTEGER NOT NULL,
            team_id INTEGER NOT NULL,
            department_id INTEGER NOT NULL,
            task_id INTEGER NOT NULL,
            usage_timestamp TEXT NOT NULL,
            usage_count INTEGER NOT NULL DEFAULT 1,
            session_duration_seconds INTEGER NOT NULL DEFAULT 0,
            success_flag INTEGER NOT NULL DEFAULT 1,
            FOREIGN KEY (user_id) REFERENCES users(user_id),
            FOREIGN KEY (tool_id) REFERENCES ai_tools(tool_id),
            FOREIGN KEY (team_id) REFERENCES teams(team_id),
            FOREIGN KEY (department_id) REFERENCES departments(department_id),
            FOREIGN KEY (task_id) REFERENCES ai_tasks(task_id)
        );

        CREATE TABLE team_work_metrics (
            metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_id INTEGER NOT NULL,
            metric_month TEXT NOT NULL,
            team_size INTEGER NOT NULL,
            story_count INTEGER NOT NULL,
            completed_story_count INTEGER NOT NULL,
            estimated_story_count INTEGER NOT NULL,
            unestimated_story_count INTEGER NOT NULL,
            completed_story_points INTEGER NOT NULL,
            cycle_time_days REAL NOT NULL,
            lead_time_days REAL NOT NULL,
            FOREIGN KEY (team_id) REFERENCES teams(team_id)
        );

        CREATE TABLE ml_tool_adoption_forecast (
            forecast_id INTEGER PRIMARY KEY AUTOINCREMENT,
            tool_id INTEGER NOT NULL,
            forecast_month TEXT NOT NULL,
            predicted_adoption_rate REAL NOT NULL,
            lower_bound REAL NOT NULL,
            upper_bound REAL NOT NULL,
            model_version TEXT NOT NULL,
            generated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (tool_id) REFERENCES ai_tools(tool_id)
        );

        CREATE TABLE ml_user_adoption_risk (
            risk_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            prediction_month TEXT NOT NULL,
            risk_score REAL NOT NULL,
            risk_band TEXT NOT NULL,
            model_version TEXT NOT NULL,
            generated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        );

        CREATE TABLE ml_user_segments (
            segment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            prediction_month TEXT NOT NULL,
            cluster_id INTEGER NOT NULL,
            segment_name TEXT NOT NULL,
            segment_score REAL NOT NULL,
            model_version TEXT NOT NULL,
            generated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        );

        CREATE TABLE ml_usage_anomalies (
            anomaly_id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_date TEXT NOT NULL,
            team_id INTEGER NOT NULL,
            tool_id INTEGER NOT NULL,
            metric_name TEXT NOT NULL,
            actual_value REAL NOT NULL,
            expected_value REAL NOT NULL,
            anomaly_score REAL NOT NULL,
            is_anomaly INTEGER NOT NULL,
            model_version TEXT NOT NULL,
            generated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (team_id) REFERENCES teams(team_id),
            FOREIGN KEY (tool_id) REFERENCES ai_tools(tool_id)
        );

        CREATE TABLE ml_team_productivity_forecast (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_id INTEGER NOT NULL,
            prediction_month TEXT NOT NULL,
            predicted_completed_story_points INTEGER NOT NULL,
            lower_bound INTEGER NOT NULL,
            upper_bound INTEGER NOT NULL,
            model_version TEXT NOT NULL,
            generated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (team_id) REFERENCES teams(team_id)
        );

        CREATE TABLE ml_tool_recommendations (
            recommendation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            team_id INTEGER NOT NULL,
            tool_id INTEGER NOT NULL,
            recommendation_score REAL NOT NULL,
            recommendation_reason TEXT NOT NULL,
            model_version TEXT NOT NULL,
            generated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id),
            FOREIGN KEY (team_id) REFERENCES teams(team_id),
            FOREIGN KEY (tool_id) REFERENCES ai_tools(tool_id)
        );

        CREATE TABLE query_history (
            request_id TEXT PRIMARY KEY,
            question TEXT NOT NULL,
            intent TEXT NOT NULL,
            status TEXT NOT NULL,
            execution_time_ms INTEGER DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            payload TEXT
        );
        """
    )

    departments = [
        (1, 'Engineering', 'Product and platform delivery organization.'),
        (2, 'Operations', 'Service execution and operational delivery.'),
        (3, 'Sales', 'Revenue and customer growth organization.'),
        (4, 'Finance', 'Financial planning and governance.'),
    ]
    conn.executemany(
        'INSERT INTO departments (department_id, department_name, description) VALUES (?, ?, ?)',
        departments,
    )

    practices = [
        (1, 'Product Engineering', 'Engineering and platform delivery practice.'),
        (2, 'AI Enablement', 'AI adoption and transformation practice.'),
        (3, 'Customer Success', 'Customer-facing operational excellence.'),
        (4, 'Finance Ops', 'Financial automation and controls.'),
    ]
    conn.executemany(
        'INSERT INTO practices (practice_id, practice_name, description) VALUES (?, ?, ?)',
        practices,
    )

    teams = [
        (1, 'Platform Engineering', 1, 1, 18),
        (2, 'Data Platform', 1, 2, 18),
        (3, 'Support Operations', 2, 3, 16),
        (4, 'Field Enablement', 3, 2, 17),
        (5, 'Revenue Operations', 3, 3, 18),
        (6, 'Finance Analytics', 4, 4, 16),
        (7, 'Customer Intelligence', 2, 2, 17),
        (8, 'Product Ops', 1, 1, 20),
        (9, 'Enterprise GTM', 3, 1, 18),
        (10, 'Controls & Automation', 4, 4, 15),
    ]
    conn.executemany(
        'INSERT INTO teams (team_id, team_name, department_id, practice_id, team_size) VALUES (?, ?, ?, ?, ?)',
        teams,
    )

    users = []
    employee_code = 1000
    for department_id, _ in [(1, 'Engineering'), (2, 'Operations'), (3, 'Sales'), (4, 'Finance')]:
        for team in teams:
            if team[2] != department_id:
                continue
            for idx in range(1, 9):
                employee_code += 1
                users.append(
                    (
                        employee_code,
                        f'EMP-{employee_code:05d}',
                        department_id,
                        team[0],
                        team[3],
                        ['Engineer', 'Analyst', 'Manager', 'Developer', 'Product Manager'][idx % 5],
                        ['Seattle', 'Austin', 'New York', 'London', 'Berlin'][idx % 5],
                        ['Full-time', 'Contract'][idx % 2],
                        '2024-01-15 00:00:00',
                    )
                )
    conn.executemany(
        'INSERT INTO users (user_id, employee_code, department_id, team_id, practice_id, role, location, employment_type, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
        users,
    )

    tools = [
        (1, 'ChatGPT', 'OpenAI', 'General AI', 'General-purpose productivity assistant.', '2022-11-30', 'active'),
        (2, 'Microsoft Copilot', 'Microsoft', 'Developer Tool', 'Enterprise productivity and coding assistant.', '2023-09-15', 'active'),
        (3, 'GitHub Copilot', 'GitHub', 'Developer Tool', 'Coding assistance built into the developer workflow.', '2021-10-29', 'active'),
        (4, 'Claude', 'Anthropic', 'Research Assist', 'Long-context analysis and drafting assistant.', '2023-03-14', 'active'),
        (5, 'Gemini', 'Google', 'General AI', 'AI workspace and research companion.', '2023-12-06', 'active'),
        (6, 'Internal AI Assistant', 'QueryMind Labs', 'Internal AI', 'Synthetic internal AI service used by employees.', '2024-05-07', 'active'),
    ]
    conn.executemany(
        'INSERT INTO ai_tools (tool_id, tool_name, vendor, category, description, release_date, status) VALUES (?, ?, ?, ?, ?, ?, ?)',
        tools,
    )

    tasks = [
        (1, 'Coding', 'Code Review', 'Reviewing and improving implementation quality.'),
        (2, 'Documentation', 'Technical Writing', 'Drafting internal and customer documentation.'),
        (3, 'Research', 'Market Research', 'Gathering background information and analysis.'),
        (4, 'Data Analysis', 'Trend Analysis', 'Reviewing trend and performance metrics.'),
        (5, 'Customer Support', 'Issue Triage', 'Addressing support and issue intake.'),
        (6, 'Content Generation', 'Proposal Drafting', 'Writing proposals and summaries.'),
        (7, 'Meeting Summarization', 'Meeting Notes', 'Summarizing meetings and actions.'),
        (8, 'Translation', 'Localization', 'Translating text for global usage.'),
        (9, 'Automation', 'Workflow Automation', 'Creating automations and helpers.'),
    ]
    conn.executemany(
        'INSERT INTO ai_tasks (task_id, task_category, task_name, description) VALUES (?, ?, ?, ?)',
        tasks,
    )

    usage_rows = []
    for day in range(1, 181):
        for _ in range(40):
            user = users[(day * 11 + _) % len(users)]
            tool = tools[(day + _) % len(tools)]
            team_id = user[3]
            department_id = user[2]
            task = tasks[(day + _) % len(tasks)]
            usage_timestamp = f'2024-{(day % 12) + 1:02d}-{(day % 28) + 1:02d} 08:00:00'
            usage_rows.append(
                (
                    None,
                    user[0],
                    tool[0],
                    team_id,
                    department_id,
                    task[0],
                    usage_timestamp,
                    1 + ((day + _) % 12),
                    120 + ((day * 3 + _) % 2400),
                    1 if ((day + _) % 4) != 0 else 0,
                )
            )
    conn.executemany(
        'INSERT INTO ai_tool_usage (usage_id, user_id, tool_id, team_id, department_id, task_id, usage_timestamp, usage_count, session_duration_seconds, success_flag) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
        usage_rows,
    )

    metrics_rows = []
    for team in teams:
        for month_idx in range(1, 7):
            story_count = 42 + (team[0] * 5) + month_idx * 6
            completed = int(story_count * 0.72)
            estimated = int(story_count * 0.8)
            unestimated = max(0, story_count - estimated)
            metrics_rows.append(
                (
                    None,
                    team[0],
                    f'2024-{month_idx:02d}-01',
                    team[4],
                    story_count,
                    completed,
                    estimated,
                    unestimated,
                    completed * 12 + team[0] * 3,
                    7.8 + (team[0] % 5),
                    10.6 + (team[0] % 4),
                )
            )
    conn.executemany(
        'INSERT INTO team_work_metrics (metric_id, team_id, metric_month, team_size, story_count, completed_story_count, estimated_story_count, unestimated_story_count, completed_story_points, cycle_time_days, lead_time_days) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
        metrics_rows,
    )

    for tool in tools:
        for month_idx in range(1, 13):
            base = 0.15 + (tool[0] * 0.05) + (month_idx * 0.02)
            predicted = round(min(0.9, max(0.1, base)), 4)
            lower = round(max(0.02, predicted - 0.08), 4)
            upper = round(min(0.99, predicted + 0.08), 4)
            conn.execute(
                'INSERT INTO ml_tool_adoption_forecast (tool_id, forecast_month, predicted_adoption_rate, lower_bound, upper_bound, model_version) VALUES (?, ?, ?, ?, ?, ?)',
                (tool[0], f'2026-{month_idx:02d}-01', predicted, lower, upper, 'adoption-forecast-v1'),
            )

    for user in users[:120]:
        for month_idx in range(1, 7):
            risk_score = round(((user[0] * 13 + month_idx * 17) % 100) / 100, 4)
            risk_band = 'LOW' if risk_score < 0.35 else 'MEDIUM' if risk_score < 0.7 else 'HIGH'
            conn.execute(
                'INSERT INTO ml_user_adoption_risk (user_id, prediction_month, risk_score, risk_band, model_version) VALUES (?, ?, ?, ?, ?)',
                (user[0], f'2026-{month_idx:02d}-01', risk_score, risk_band, 'adoption-risk-v1'),
            )

    for user in users[:120]:
        for month_idx in range(1, 7):
            segment_name = ['AI Explorer', 'AI Power User', 'Occasional User', 'Multi-Tool User', 'Specialized User'][(user[0] + month_idx) % 5]
            score = round(((user[0] * 11 + month_idx * 7) % 100) / 100, 4)
            conn.execute(
                'INSERT INTO ml_user_segments (user_id, prediction_month, cluster_id, segment_name, segment_score, model_version) VALUES (?, ?, ?, ?, ?, ?)',
                (user[0], f'2026-{month_idx:02d}-01', (user[0] + month_idx) % 5 + 1, segment_name, score, 'segmentation-v1'),
            )

    for team in teams:
        for month_idx in range(1, 7):
            predicted_points = 60 + ((team[0] * 17 + month_idx * 11) % 190)
            conn.execute(
                'INSERT INTO ml_team_productivity_forecast (team_id, prediction_month, predicted_completed_story_points, lower_bound, upper_bound, model_version) VALUES (?, ?, ?, ?, ?, ?)',
                (team[0], f'2026-{month_idx:02d}-01', predicted_points, max(0, predicted_points - 20), predicted_points + 25, 'throughput-forecast-v1'),
            )

    for team in teams:
        for day in range(1, 7):
            actual = 180 + ((team[0] * 13 + day * 11) % 200)
            expected = actual - 25 + (day % 3)
            anomaly_score = round(abs(actual - expected) / max(1.0, expected), 4)
            conn.execute(
                'INSERT INTO ml_usage_anomalies (metric_date, team_id, tool_id, metric_name, actual_value, expected_value, anomaly_score, is_anomaly, model_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (f'2026-{day:02d}-15', team[0], (team[0] + day) % len(tools) + 1, 'usage_volume', actual, expected, anomaly_score, 1 if anomaly_score > 0.22 else 0, 'anomaly-v1'),
            )

    conn.commit()
    conn.close()
    return str(db_path)


if __name__ == '__main__':
    print(generate_sqlite_database())
