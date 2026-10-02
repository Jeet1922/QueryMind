from __future__ import annotations

import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB_PATH = ROOT / "data" / "insightmesh.sqlite3"


def get_db_path() -> Path:
    return Path(__import__("os").getenv("SQLITE_DB_PATH", str(DEFAULT_DB_PATH)))


def ensure_database() -> Path:
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if not db_path.exists():
        from data_generator.generate_sqlite_db import generate_sqlite_database

        generate_sqlite_database(str(db_path))
    return db_path


def get_connection() -> sqlite3.Connection:
    db_path = ensure_database()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn
