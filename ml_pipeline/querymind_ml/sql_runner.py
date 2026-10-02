"""Execute the numbered SQL files under ml_pipeline/sql against the database.

Each file runs inside a single transaction (psycopg wraps it), so a syntax
error rolls the whole file back instead of leaving half-applied objects. The
files themselves are idempotent, so re-running after a fix is safe.
"""

from __future__ import annotations

from pathlib import Path

import psycopg

from . import config


def run_file(conn: psycopg.Connection, path: Path) -> None:
    sql = path.read_text(encoding="utf-8")
    with conn.transaction():
        conn.execute(sql)
    conn.commit()


def run_setup(conn: psycopg.Connection) -> list[str]:
    """Run 00 (MLOps tables), 01 (raw generators), 02 (feature views)."""
    executed = []
    for name in config.SETUP_SQL_FILES:
        run_file(conn, config.sql_path(name))
        executed.append(name)
    return executed


def run_eda(conn: psycopg.Connection) -> None:
    """03 is read-only SELECTs; executed as one batch for convenience."""
    run_file(conn, config.sql_path(config.EDA_SQL_FILE))
