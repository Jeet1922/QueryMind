#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import psycopg
from dotenv import load_dotenv
from backend.app.db.dsn import get_postgres_url

from data_generator.generate_ml_outputs import generate_ml_outputs
from data_generator.generate_reference_data import generate_reference_data
from data_generator.generate_usage_data import generate_usage_data


def get_dsn() -> str:
    load_dotenv(ROOT / ".env")
    dsn = get_postgres_url(os.getenv("DATABASE_URL"))
    if not dsn:
        raise RuntimeError("Set a valid Neon/PostgreSQL DATABASE_URL in the environment or repo-root .env file.")
    return dsn


def split_sql_statements(sql: str) -> list[str]:
    statements: list[str] = []
    buffer: list[str] = []
    in_single_quote = False
    in_double_quote = False

    for char in sql:
        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
        elif char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote

        if char == ";" and not in_single_quote and not in_double_quote:
            statement = "".join(buffer).strip()
            if statement:
                statements.append(statement)
            buffer = []
        else:
            buffer.append(char)

    remainder = "".join(buffer).strip()
    if remainder:
        statements.append(remainder)
    return statements


def apply_sql_file(conn: psycopg.Connection, sql_path: Path) -> None:
    sql = sql_path.read_text(encoding="utf-8")
    for statement in split_sql_statements(sql):
        with conn.cursor() as cur:
            cur.execute(statement)


def bootstrap_database(scale: str = "small") -> None:
    dsn = get_dsn()
    with psycopg.connect(dsn) as conn:
        migrations_dir = ROOT / "database" / "migrations"
        for migration_file in sorted(migrations_dir.glob("*.sql")):
            apply_sql_file(conn, migration_file)

        generate_reference_data(conn)
        generate_usage_data(conn, scale=scale)
        generate_ml_outputs(conn)
        conn.commit()

    print(f"Bootstrapped QueryMind database with scale={scale}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize the QueryMind PostgreSQL schema and seed data.")
    parser.add_argument(
        "--scale",
        choices=["small", "medium", "large"],
        default="small",
        help="Synthetic data size profile to generate for ai_tool_usage.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    bootstrap_database(scale=args.scale)
