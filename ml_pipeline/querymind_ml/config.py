from __future__ import annotations

import os
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PIPELINE_ROOT.parent
SQL_DIR = PIPELINE_ROOT / "sql"
ARTIFACTS_DIR = PIPELINE_ROOT / "artifacts"
REPORTS_DIR = PIPELINE_ROOT / "reports"

DEFAULT_DSN = "postgresql://querymind:querymind@localhost:5432/querymind"

RANDOM_SEED = 42

# Run order matters: MLOps tables -> raw generators -> feature/label views.
# 03_eda_queries.sql is read-only and optional (run via --step eda).
SETUP_SQL_FILES = (
    "00_ml_mlops_tables.sql",
    "01_raw_data_generators.sql",
    "02_features_labels.sql",
)
EDA_SQL_FILE = "03_eda_queries.sql"


def _load_repo_env() -> None:
    """Best-effort load of the repo-root .env files (never prints values)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for candidate in (REPO_ROOT / ".env.local", REPO_ROOT / ".env"):
        if candidate.exists():
            load_dotenv(candidate, override=False)


_load_repo_env()


def get_dsn(explicit: str | None = None) -> str:
    """Resolve the Neon/PostgreSQL connection string.

    Precedence: explicit argument > DATABASE_URL env (repo .env) > local default.
    """
    if explicit:
        return explicit
    dsn = os.getenv("DATABASE_URL", "").strip()
    if dsn and "placeholder" not in dsn.lower():
        return dsn
    return DEFAULT_DSN


def sql_path(filename: str) -> Path:
    return SQL_DIR / filename
