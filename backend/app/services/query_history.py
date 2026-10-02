from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

import psycopg
from psycopg.types.json import Jsonb

from backend.app.db.dsn import get_postgres_url
from backend.app.db.sqlite import get_connection


class QueryHistoryStore:
    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = get_postgres_url(database_url)
        self.backend = "postgresql" if self.database_url else "sqlite"
        self._ensure_schema()

    @contextmanager
    def _connection(self) -> Iterator:
        if self.database_url:
            with psycopg.connect(self.database_url) as conn:
                yield conn
            return

        conn = get_connection()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        with self._connection() as conn:
            if self.backend == "postgresql":
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS query_history (
                        request_id UUID PRIMARY KEY,
                        question TEXT NOT NULL,
                        intent VARCHAR(80) NOT NULL,
                        status VARCHAR(30) NOT NULL,
                        execution_time_ms INTEGER NOT NULL DEFAULT 0,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        payload JSONB NOT NULL DEFAULT '{}'::jsonb
                    )
                    """
                )
            else:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS query_history (
                        request_id TEXT PRIMARY KEY,
                        question TEXT NOT NULL,
                        intent TEXT NOT NULL,
                        status TEXT NOT NULL,
                        execution_time_ms INTEGER DEFAULT 0,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        payload TEXT
                    )
                    """
                )

    def save(self, request_id: str, question: str, intent: str, status: str, execution_time_ms: int = 0, payload: dict | None = None) -> None:
        with self._connection() as conn:
            if self.backend == "postgresql":
                conn.execute(
                    """
                    INSERT INTO query_history (request_id, question, intent, status, execution_time_ms, payload)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (request_id) DO UPDATE SET
                        question = EXCLUDED.question,
                        intent = EXCLUDED.intent,
                        status = EXCLUDED.status,
                        execution_time_ms = EXCLUDED.execution_time_ms,
                        payload = EXCLUDED.payload
                    """,
                    (uuid.UUID(request_id), question, intent, status, execution_time_ms, Jsonb(payload or {})),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO query_history (request_id, question, intent, status, execution_time_ms, created_at, payload)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(request_id) DO UPDATE SET
                        question = excluded.question,
                        intent = excluded.intent,
                        status = excluded.status,
                        execution_time_ms = excluded.execution_time_ms,
                        payload = excluded.payload
                    """,
                    (
                        request_id,
                        question,
                        intent,
                        status,
                        execution_time_ms,
                        datetime.now(timezone.utc).isoformat(timespec='seconds'),
                        json.dumps(payload or {}),
                    ),
                )

    def get_by_id(self, request_id: str) -> dict | None:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT request_id, question, intent, status, execution_time_ms, payload FROM query_history WHERE request_id = ?" if self.backend == "sqlite" else "SELECT request_id, question, intent, status, execution_time_ms, payload FROM query_history WHERE request_id = %s",
                (request_id if self.backend == "sqlite" else uuid.UUID(request_id),),
            ).fetchone()
        if not row:
            return None
        return {
            "request_id": str(row[0]),
            "question": row[1],
            "intent": row[2],
            "status": row[3],
            "execution_time_ms": row[4],
            "payload": json.loads(row[5] or '{}') if isinstance(row[5], str) else row[5],
        }

    def list_recent(self, limit: int = 20) -> list[dict]:
        with self._connection() as conn:
            rows = conn.execute(
            "SELECT request_id, question, intent, status, execution_time_ms FROM query_history ORDER BY created_at DESC LIMIT ?" if self.backend == "sqlite" else "SELECT request_id, question, intent, status, execution_time_ms FROM query_history ORDER BY created_at DESC LIMIT %s",
            (limit,),
            ).fetchall()
        return [
            {
                "request_id": str(row[0]),
                "question": row[1],
                "intent": row[2],
                "status": row[3],
                "execution_time_ms": row[4],
            }
            for row in rows
        ]
