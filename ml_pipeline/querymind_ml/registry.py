"""Model registry: joblib artifacts on disk + ml_model_registry rows in Neon.

Versioning: `<activity>-YYYYmmdd-HHMMSS`. Every training run registers a
'challenger'; if the activity has no 'champion' yet the new version is
promoted automatically, so `--step predict` always has something to serve.
Use --promote to force a version to champion.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import psycopg

from . import config, db


def artifact_dir(activity: str, model_version: str) -> Path:
    return config.ARTIFACTS_DIR / activity / model_version


def save(
    conn: psycopg.Connection,
    activity: str,
    task_type: str,
    algorithm: str,
    model_version: str,
    bundle: dict[str, Any],
    metrics: dict[str, float],
    feature_columns: list[str],
    training_rows: int,
    notes: str | None = None,
) -> Path:
    """Persist the model bundle and register it. Commits the registry row."""
    target = artifact_dir(activity, model_version)
    target.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, target / "model.joblib")

    meta = {
        "activity": activity,
        "task_type": task_type,
        "algorithm": algorithm,
        "model_version": model_version,
        "metrics": metrics,
        "feature_columns": feature_columns,
        "training_rows": training_rows,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "notes": notes or "",
    }
    (target / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    has_champion = bool(
        db.read_rows(
            conn,
            "SELECT 1 FROM ml_model_registry WHERE activity = %s AND status = 'champion' LIMIT 1",
            (activity,),
        )
    )
    status = "challenger" if has_champion else "champion"
    rel_path = str(target.relative_to(config.PIPELINE_ROOT))
    db.execute(
        conn,
        """
        INSERT INTO ml_model_registry
            (activity, model_version, algorithm, task_type, artifact_path,
             metrics, feature_names, training_row_count, status, notes)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (activity, model_version) DO UPDATE
        SET metrics = EXCLUDED.metrics,
            feature_names = EXCLUDED.feature_names,
            training_row_count = EXCLUDED.training_row_count,
            algorithm = EXCLUDED.algorithm,
            trained_at = NOW()
        """,
        (
            activity,
            model_version,
            algorithm,
            task_type,
            rel_path,
            json.dumps(metrics, default=str),
            json.dumps(feature_columns),
            int(training_rows),
            status,
            notes,
        ),
    )
    conn.commit()
    return target


def load_latest(
    conn: psycopg.Connection,
    activity: str,
    prefer_champion: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Load (bundle, meta) for the champion, else the newest registered version."""
    order = (
        "CASE WHEN status = 'champion' THEN 0 ELSE 1 END, trained_at DESC"
        if prefer_champion
        else "trained_at DESC"
    )
    rows = db.read_rows(
        conn,
        f"""
        SELECT artifact_path, model_version, metrics, feature_names, training_row_count
        FROM ml_model_registry
        WHERE activity = %s
        ORDER BY {order}
        LIMIT 1
        """,
        (activity,),
    )
    if not rows:
        return None
    artifact_path, model_version, metrics, feature_names, training_rows = rows[0]
    target = config.PIPELINE_ROOT / artifact_path
    model_file = target / "model.joblib"
    if not model_file.exists():
        raise FileNotFoundError(
            f"Registry points at missing artifact: {model_file}. Re-run --step train."
        )
    bundle = joblib.load(model_file)
    meta = {
        "model_version": model_version,
        "metrics": metrics if isinstance(metrics, dict) else json.loads(metrics),
        "feature_columns": feature_names
        if isinstance(feature_names, list)
        else json.loads(feature_names),
        "training_rows": training_rows,
    }
    return bundle, meta


def promote(conn: psycopg.Connection, activity: str, model_version: str) -> None:
    """Mark one version champion and archive the previous champion."""
    db.execute(
        conn,
        "UPDATE ml_model_registry SET status = 'archived' "
        "WHERE activity = %s AND status = 'champion' AND model_version <> %s",
        (activity, model_version),
    )
    db.execute(
        conn,
        "UPDATE ml_model_registry SET status = 'champion' "
        "WHERE activity = %s AND model_version = %s",
        (activity, model_version),
    )
    conn.commit()
