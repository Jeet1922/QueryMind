from __future__ import annotations

import os
from datetime import date

import psycopg


DEFAULT_DSN = "postgresql://querymind:querymind@localhost:5432/querymind"


def generate_reference_data(conn: psycopg.Connection) -> None:
    departments = [
        ("Engineering", "Product and platform delivery organization."),
        ("Operations", "Service execution and operational delivery."),
        ("Sales", "Revenue and customer growth organization."),
        ("Finance", "Financial planning and governance."),
    ]

    practices = [
        ("Product Engineering", "Engineering and platform delivery practice."),
        ("AI Enablement", "AI adoption and transformation practice."),
        ("Customer Success", "Customer-facing operational excellence."),
        ("Finance Ops", "Financial automation and controls."),
    ]

    tools = [
        ("ChatGPT", "OpenAI", "General AI", "General-purpose productivity assistant.", date(2022, 11, 30), "active"),
        ("Microsoft Copilot", "Microsoft", "Developer Tool", "Enterprise productivity and coding assistant.", date(2023, 9, 15), "active"),
        ("GitHub Copilot", "GitHub", "Developer Tool", "Coding assistance built into the developer workflow.", date(2021, 10, 29), "active"),
        ("Claude", "Anthropic", "Research Assist", "Long-context analysis and drafting assistant.", date(2023, 3, 14), "active"),
        ("Gemini", "Google", "General AI", "AI workspace and research companion.", date(2023, 12, 6), "active"),
        ("Internal AI Assistant", "QueryMind Labs", "Internal AI", "Synthetic internal AI service used by employees.", date(2024, 5, 7), "active"),
    ]

    tasks = [
        ("Coding", "Code Review", "Reviewing and improving implementation quality."),
        ("Documentation", "Technical Writing", "Drafting internal and customer documentation."),
        ("Research", "Market Research", "Gathering background information and analysis."),
        ("Data Analysis", "Trend Analysis", "Reviewing trend and performance metrics."),
        ("Customer Support", "Issue Triage", "Addressing support and issue intake."),
        ("Content Generation", "Proposal Drafting", "Writing proposals and summaries."),
        ("Meeting Summarization", "Meeting Notes", "Summarizing meetings and actions."),
        ("Translation", "Localization", "Translating text for global usage."),
        ("Automation", "Workflow Automation", "Creating automations and helpers."),
    ]

    team_specs = [
        ("Platform Engineering", "Engineering", "Product Engineering", 18),
        ("Data Platform", "Engineering", "AI Enablement", 18),
        ("Support Operations", "Operations", "Customer Success", 16),
        ("Field Enablement", "Sales", "AI Enablement", 17),
        ("Revenue Operations", "Sales", "Customer Success", 18),
        ("Finance Analytics", "Finance", "Finance Ops", 16),
        ("Customer Intelligence", "Operations", "AI Enablement", 17),
        ("Product Ops", "Engineering", "Product Engineering", 20),
        ("Enterprise GTM", "Sales", "Product Engineering", 18),
        ("Controls & Automation", "Finance", "Finance Ops", 15),
        ("AI Operations", "Operations", "AI Enablement", 19),
        ("Platform Experience", "Engineering", "Product Engineering", 18),
    ]

    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO departments (department_name, description) VALUES (%s, %s) ON CONFLICT (department_name) DO NOTHING",
            departments,
        )
        cur.executemany(
            "INSERT INTO practices (practice_name, description) VALUES (%s, %s) ON CONFLICT (practice_name) DO NOTHING",
            practices,
        )
        cur.executemany(
            "INSERT INTO ai_tools (tool_name, vendor, category, description, release_date, status) VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (tool_name) DO NOTHING",
            tools,
        )
        cur.executemany(
            "INSERT INTO ai_tasks (task_category, task_name, description) VALUES (%s, %s, %s) ON CONFLICT (task_name) DO NOTHING",
            tasks,
        )

        dept_lookup = {row[1]: row[0] for row in cur.execute("SELECT department_id, department_name FROM departments").fetchall()}
        practice_lookup = {row[1]: row[0] for row in cur.execute("SELECT practice_id, practice_name FROM practices").fetchall()}

        cur.executemany(
            "INSERT INTO teams (team_name, department_id, practice_id, team_size) VALUES (%s, %s, %s, %s) ON CONFLICT (team_name) DO NOTHING",
            [
                (team_name, dept_lookup[dept_name], practice_lookup[practice_name], team_size)
                for team_name, dept_name, practice_name, team_size in team_specs
            ],
        )

        team_lookup = {row[1]: row[0] for row in cur.execute("SELECT team_id, team_name FROM teams").fetchall()}

        user_rows = []
        employee_code = 1000
        for dept_name, _ in departments:
            for team_name, team_dept_name, practice_name, _ in team_specs:
                if team_dept_name != dept_name:
                    continue
                for idx in range(1, 9):
                    employee_code += 1
                    role_pool = ["Engineer", "Analyst", "Manager", "Developer", "Product Manager"]
                    location_pool = ["Seattle", "Austin", "New York", "London", "Berlin"]
                    employment_pool = ["Full-time", "Contract"]
                    user_rows.append(
                        (
                            f"EMP-{employee_code:05d}",
                            dept_lookup[dept_name],
                            team_lookup[team_name],
                            practice_lookup[practice_name],
                            role_pool[idx % len(role_pool)],
                            location_pool[(employee_code + idx) % len(location_pool)],
                            employment_pool[(employee_code + idx) % len(employment_pool)],
                        )
                    )

        cur.executemany(
            """
            INSERT INTO users (employee_code, department_id, team_id, practice_id, role, location, employment_type)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (employee_code) DO NOTHING
            """,
            user_rows,
        )

        team_rows = cur.execute("SELECT team_id FROM teams ORDER BY team_id").fetchall()
        months = [
            (date(2024, 1, 1), 42),
            (date(2024, 2, 1), 44),
            (date(2024, 3, 1), 52),
            (date(2024, 4, 1), 58),
            (date(2024, 5, 1), 61),
            (date(2024, 6, 1), 64),
        ]
        metrics_rows = []
        for team_id, in team_rows:
            for metric_month, base_story_count in months:
                completed = int(base_story_count * 0.71 + (team_id * 3))
                story_count = base_story_count + 8
                estimated = int(story_count * 0.78)
                unestimated = max(0, story_count - estimated)
                completed_story_points = int((completed * 12) + (team_id * 5) + 18)
                cycle = 8.4 + (team_id % 5) * 1.3 + (base_story_count % 7) * 0.4
                lead = cycle + 3.1
                metrics_rows.append(
                    (
                        team_id,
                        metric_month,
                        18 + (team_id % 5),
                        story_count,
                        completed,
                        estimated,
                        unestimated,
                        completed_story_points,
                        float(cycle),
                        float(lead),
                    )
                )

        cur.executemany(
            """
            INSERT INTO team_work_metrics (
                team_id,
                metric_month,
                team_size,
                story_count,
                completed_story_count,
                estimated_story_count,
                unestimated_story_count,
                completed_story_points,
                cycle_time_days,
                lead_time_days
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            metrics_rows,
        )


if __name__ == "__main__":
    dsn = os.getenv("DATABASE_URL", DEFAULT_DSN)
    with psycopg.connect(dsn) as conn:
        generate_reference_data(conn)
        conn.commit()
        print("Reference data generated.")
