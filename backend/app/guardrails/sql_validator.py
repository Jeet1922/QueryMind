from __future__ import annotations

DANGEROUS_KEYWORDS = {
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "truncate",
    "create",
    "grant",
    "revoke",
    "copy",
    "call",
    "execute",
    "do",
    "vacuum",
    "set",
    "reset",
    "begin",
    "commit",
    "rollback",
    "pg_catalog",
    "information_schema",
}


class SecurityViolation(ValueError):
    """Raised when a SQL statement violates the read-only guardrails."""


def validate_sql(sql: str) -> str:
    if sql is None or not str(sql).strip():
        raise SecurityViolation("SQL text is empty.")

    query = sql.strip()
    lowered = query.lower()

    if ";" in query:
        raise SecurityViolation("Multiple statements are not allowed.")

    if any(keyword in lowered for keyword in DANGEROUS_KEYWORDS):
        raise SecurityViolation(f"Dangerous SQL pattern detected: {query[:80]}")

    if not lowered.startswith(("select", "with")):
        raise SecurityViolation("Only read-only SELECT/WITH queries are permitted.")

    if "pg_" in lowered or "information_schema" in lowered or "pg_catalog" in lowered:
        raise SecurityViolation("PostgreSQL system metadata is not permitted.")

    return query
