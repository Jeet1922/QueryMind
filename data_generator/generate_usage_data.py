from __future__ import annotations

import argparse
import os
import random
from datetime import date, datetime, timedelta, timezone

import psycopg


DEFAULT_DSN = "postgresql://querymind:querymind@localhost:5432/querymind"
SCALE_CONFIG = {
    "small": 10_000,
    "medium": 1_000_000,
    "large": 10_000_000,
}


def _fetch_ids(conn: psycopg.Connection) -> tuple[list[int], list[int], list[int], list[int], list[int]]:
    with conn.cursor() as cur:
        cur.execute("SELECT user_id FROM users ORDER BY user_id")
        user_ids = [row[0] for row in cur.fetchall()]
        cur.execute("SELECT tool_id FROM ai_tools ORDER BY tool_id")
        tool_ids = [row[0] for row in cur.fetchall()]
        cur.execute("SELECT task_id FROM ai_tasks ORDER BY task_id")
        task_ids = [row[0] for row in cur.fetchall()]
        cur.execute("SELECT team_id FROM teams ORDER BY team_id")
        team_ids = [row[0] for row in cur.fetchall()]
        cur.execute("SELECT department_id FROM departments ORDER BY department_id")
        department_ids = [row[0] for row in cur.fetchall()]
    return user_ids, tool_ids, task_ids, team_ids, department_ids


def generate_usage_data(conn: psycopg.Connection, scale: str = "small") -> int:
    if scale not in SCALE_CONFIG:
        raise ValueError(f"Unsupported scale '{scale}'. Choose from: {sorted(SCALE_CONFIG)}")

    total_rows = SCALE_CONFIG[scale]
    user_ids, tool_ids, task_ids, team_ids, department_ids = _fetch_ids(conn)
    if not user_ids or not tool_ids or not task_ids:
        raise RuntimeError("Reference data is missing. Generate departments, users, tools, and tasks first.")

    rng = random.Random(42)
    batch_size = 1000
    start_day = datetime(2024, 1, 1, tzinfo=timezone.utc)

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM ai_tool_usage")
        existing_rows = cur.fetchone()[0]
        if existing_rows and total_rows <= existing_rows:
            return existing_rows

        if existing_rows:
            cur.execute("DELETE FROM ai_tool_usage")

        for offset in range(0, total_rows, batch_size):
            values: list[tuple] = []
            current_batch = min(batch_size, total_rows - offset)
            for _ in range(current_batch):
                user_id = rng.choice(user_ids)
                tool_id = rng.choice(tool_ids)
                task_id = rng.choice(task_ids)
                team_id = rng.choice(team_ids)
                department_id = rng.choice(department_ids)
                usage_timestamp = start_day + timedelta(
                    days=rng.randint(0, 365),
                    hours=rng.randint(0, 23),
                    minutes=rng.randint(0, 59),
                )
                usage_count = rng.randint(1, 12)
                session_duration_seconds = rng.randint(120, 5400)
                success_flag = rng.choice([True, False])
                values.append(
                    (
                        user_id,
                        tool_id,
                        team_id,
                        department_id,
                        task_id,
                        usage_timestamp,
                        usage_count,
                        session_duration_seconds,
                        success_flag,
                    )
                )
            cur.executemany(
                """
                INSERT INTO ai_tool_usage (
                    user_id,
                    tool_id,
                    team_id,
                    department_id,
                    task_id,
                    usage_timestamp,
                    usage_count,
                    session_duration_seconds,
                    success_flag
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                values,
            )

    return total_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic ai_tool_usage records.")
    parser.add_argument("--scale", choices=sorted(SCALE_CONFIG), default="small")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL", DEFAULT_DSN))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with psycopg.connect(args.database_url) as conn:
        rows = generate_usage_data(conn, scale=args.scale)
        conn.commit()
        print(f"Generated {rows} usage rows at scale={args.scale}.")
