from __future__ import annotations

import math
from contextlib import contextmanager
from datetime import date, datetime
from typing import Any, Iterator, Sequence

import pandas as pd
import psycopg

from . import config


@contextmanager
def connect(dsn: str | None = None) -> Iterator[psycopg.Connection]:
    """Open a psycopg connection (commit/rollback handled by callers)."""
    with psycopg.connect(config.get_dsn(dsn)) as conn:
        yield conn


def read_frame(
    conn: psycopg.Connection,
    sql: str,
    params: Sequence[Any] | None = None,
) -> pd.DataFrame:
    """Run a SELECT and return the result as a DataFrame."""
    with conn.cursor() as cur:
        cur.execute(sql, params)
        if cur.description is None:
            return pd.DataFrame()
        columns = [d.name for d in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=columns)


def read_rows(
    conn: psycopg.Connection,
    sql: str,
    params: Sequence[Any] | None = None,
) -> list[tuple]:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def execute(
    conn: psycopg.Connection,
    sql: str,
    params: Sequence[Any] | None = None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(sql, params)


def _clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, pd.Timestamp):
        # All pipeline output columns are DATE; a bare date parses cleanly
        # while a full timestamp string does not.
        return value.date()
    if isinstance(value, (datetime, date, str, int, float, bool)):
        return value
    if hasattr(value, "item"):  # numpy scalar
        return _clean_value(value.item())
    return value


def write_frame(
    conn: psycopg.Connection,
    table: str,
    columns: Sequence[str],
    df: pd.DataFrame,
    chunk_size: int = 2000,
) -> int:
    """Insert a DataFrame into `table` for the given column list. Returns row count."""
    if df.empty:
        return 0
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{table}: missing columns in frame: {missing}")
    sql = (
        f"INSERT INTO {table} ({', '.join(columns)}) "
        f"VALUES ({', '.join(['%s'] * len(columns))})"
    )
    values = [
        tuple(_clean_value(v) for v in row)
        for row in df.loc[:, list(columns)].itertuples(index=False, name=None)
    ]
    with conn.cursor() as cur:
        for start in range(0, len(values), chunk_size):
            cur.executemany(sql, values[start : start + chunk_size])
    return len(values)
