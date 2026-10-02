from __future__ import annotations

import os
from urllib.parse import urlsplit, urlunsplit

from dotenv import load_dotenv


def get_postgres_url(database_url: str | None = None) -> str | None:
    if database_url is None:
        load_dotenv()
        database_url = os.getenv("DATABASE_URL")
    if not database_url:
        return None

    database_url = database_url.strip().strip('"').strip("'")
    if database_url.startswith("postgresql+psycopg://"):
        database_url = database_url.replace("postgresql+psycopg://", "postgresql://", 1)

    try:
        parsed = urlsplit(database_url)
        host = parsed.hostname
        user = parsed.username
        password = parsed.password
    except ValueError:
        return None

    if parsed.scheme not in {"postgres", "postgresql"} or not host or not user:
        return None
    if host.lower() in {"host", "your-host", "localhost-placeholder"}:
        return None
    if user.lower() in {"user", "username", "your-user"}:
        return None
    if password and password.lower() in {"password", "your-password", "replace-me"}:
        return None
    return urlunsplit(parsed)