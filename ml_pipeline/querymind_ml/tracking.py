"""Run tracking: one row per pipeline run in ml_pipeline_run.

Writes are short, self-committing transactions so a failed run still leaves a
`failed` audit row behind (the pipeline's own work is rolled back first). If
the MLOps tables are missing (SQL setup not run yet), events fall back to a
local JSONL file under reports/ so tracking never blocks a run.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg

from . import config

_FALLBACK_LOG = config.REPORTS_DIR / "run_logs.jsonl"


def _fallback(event: dict[str, Any]) -> None:
    try:
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        with _FALLBACK_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, default=str) + "\n")
    except OSError:
        pass


def start_run(
    conn: psycopg.Connection,
    activity: str,
    step: str,
    model_version: str | None = None,
) -> int | None:
    """Insert a `running` row; commits immediately. Returns run_id or None."""
    try:
        conn.rollback()  # clear any aborted state before logging
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ml_pipeline_run (activity, step, status, model_version)
                VALUES (%s, %s, 'running', %s)
                RETURNING run_id
                """,
                (activity, step, model_version),
            )
            run_id = int(cur.fetchone()[0])
        conn.commit()
        return run_id
    except Exception as exc:  # noqa: BLE001 - tracking must never break a run
        _fallback(
            {
                "event": "start",
                "activity": activity,
                "step": step,
                "model_version": model_version,
                "error": str(exc),
                "at": datetime.now(timezone.utc).isoformat(),
            }
        )
        return None


def finish_run(
    conn: psycopg.Connection,
    run_id: int | None,
    status: str,
    details: dict[str, Any] | None = None,
) -> None:
    """Close a run row as succeeded/failed; commits immediately."""
    payload = details or {}
    if run_id is None:
        _fallback({"event": "finish", "status": status, "details": payload,
                   "at": datetime.now(timezone.utc).isoformat()})
        return
    try:
        conn.rollback()  # discard any failed transaction before logging
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE ml_pipeline_run
                SET status = %s, finished_at = NOW(), details = %s
                WHERE run_id = %s
                """,
                (status, json.dumps(payload, default=str), run_id),
            )
        conn.commit()
    except Exception as exc:  # noqa: BLE001
        _fallback(
            {
                "event": "finish",
                "run_id": run_id,
                "status": status,
                "details": payload,
                "error": str(exc),
                "at": datetime.now(timezone.utc).isoformat(),
            }
        )
