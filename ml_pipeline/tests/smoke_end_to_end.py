"""End-to-end smoke test for the ML pipeline (no external Neon required).

Spins up an embedded PostgreSQL (pip install pgserver), loads the application
demo schema read-only from database/neon_demo_setup.sql, runs the pipeline SQL
(00/01/02), executes the EDA batch, then runs train+predict+monitor for all six
activities. Fails fast with the offending step in the traceback.

Run from ml_pipeline/:
    .venv/bin/python tests/smoke_end_to_end.py
"""

from __future__ import annotations

import shutil
import sys
import traceback
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ML_ROOT.parent
sys.path.insert(0, str(ML_ROOT))

import pgserver  # noqa: E402

from querymind_ml import db, sql_runner  # noqa: E402
from run_pipeline import main as pipeline_main  # noqa: E402

DATA_DIR = ML_ROOT / ".pgtest"


def load_sql_file(conn, path: Path) -> None:
    conn.execute(path.read_text(encoding="utf-8"))


def main() -> int:
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    server = pgserver.get_server(str(DATA_DIR))
    uri = server.get_uri()
    print(f"embedded postgres: {uri}")

    step = "load app demo schema"
    try:
        with db.connect(uri) as conn:
            conn.autocommit = True  # demo file has its own BEGIN/COMMIT
            load_sql_file(conn, REPO_ROOT / "database" / "neon_demo_setup.sql")
            print(f"ok: {step}")

            step = "pipeline SQL setup (00/01/02)"
            executed = sql_runner.run_setup(conn)
            print(f"ok: {step} -> {executed}")

            step = "row-count sanity"
            counts = db.read_rows(
                conn,
                """
                SELECT
                  (SELECT COUNT(*) FROM ml_raw_usage_events) AS raw_events,
                  (SELECT COUNT(*) FROM ml_raw_weekly_usage) AS weekly_cells,
                  (SELECT COUNT(*) FROM ml_raw_weekly_usage WHERE is_injected_anomaly) AS injected_cells,
                  (SELECT COUNT(*) FROM ml_raw_user_adoption_labels) AS risk_labels,
                  (SELECT COUNT(*) FROM ml_features_user_adoption_risk) AS risk_rows,
                  (SELECT COUNT(*) FROM ml_features_tool_adoption) AS adoption_rows,
                  (SELECT COUNT(*) FROM ml_features_team_productivity) AS productivity_rows,
                  (SELECT COUNT(*) FROM team_work_metrics) AS team_months
                """,
            )[0]
            print(
                "ok: row counts -> "
                f"events={counts[0]} weekly={counts[1]} injected={counts[2]} "
                f"labels={counts[3]} risk_rows={counts[4]} adoption={counts[5]} "
                f"productivity={counts[6]} team_months={counts[7]}"
            )
            assert counts[0] > 50_000, "too few raw events"
            assert counts[2] > 10, "no injected anomaly cells"
            assert counts[3] > 1_000, "too few risk labels"
            assert counts[7] >= 24 * 12 - 70, "team_work_metrics history missing"

            step = "EDA batch (03)"
            sql_runner.run_eda(conn)
            print(f"ok: {step}")

    except Exception:
        print(f"FAILED at: {step}")
        traceback.print_exc()
        return 1

    step = "run_pipeline --step all (six activities)"
    try:
        code = pipeline_main(["--dsn", uri, "--step", "all"])
        if code != 0:
            raise SystemExit(f"pipeline returned {code}")
        print(f"ok: {step}")
    except SystemExit as exc:
        if exc.code not in (0, None):
            print(f"FAILED at: {step}")
            return 1
    except Exception:
        print(f"FAILED at: {step}")
        traceback.print_exc()
        return 1

    # Verify outputs landed with sane values.
    step = "output verification"
    try:
        with db.connect(uri) as conn:
            checks = db.read_rows(
                conn,
                """
                SELECT
                  (SELECT COUNT(*) FROM ml_tool_adoption_forecast) AS adoption_rows,
                  (SELECT COUNT(*) FROM ml_user_adoption_risk WHERE prediction_month >= date_trunc('month', CURRENT_DATE)::date) AS risk_rows,
                  (SELECT COUNT(*) FROM ml_user_segments WHERE prediction_month >= date_trunc('month', CURRENT_DATE)::date) AS segment_rows,
                  (SELECT COUNT(*) FROM ml_usage_anomalies WHERE metric_name = 'usage_volume_weekly') AS anomaly_rows,
                  (SELECT COUNT(*) FROM ml_team_productivity_forecast) AS productivity_rows,
                  (SELECT COUNT(*) FROM ml_tool_recommendations) AS recommendation_rows,
                  (SELECT COUNT(*) FROM ml_model_registry) AS registry_rows,
                  (SELECT COUNT(*) FROM ml_pipeline_run WHERE status = 'failed') AS failed_runs,
                  (SELECT COUNT(*) FROM ml_prediction_monitor) AS monitor_rows,
                  (SELECT COUNT(*) FROM ml_user_adoption_risk WHERE risk_band NOT IN ('LOW','MEDIUM','HIGH')) AS bad_bands
                """,
            )[0]
            print(
                "outputs -> "
                f"adoption={checks[0]} risk={checks[1]} segments={checks[2]} "
                f"anomalies={checks[3]} productivity={checks[4]} recs={checks[5]} "
                f"registry={checks[6]} failed_runs={checks[7]} monitor={checks[8]} bad_bands={checks[9]}"
            )
            assert checks[0] == 6 * 12, "adoption forecast must be 6 tools x 12 months"
            assert checks[1] > 0, "no risk rows written"
            assert checks[2] > 0, "no segment rows written"
            assert checks[3] > 0, "no anomaly rows written"
            assert checks[4] == 12 * 6, "productivity forecast must be 12 teams x 6 months"
            assert checks[5] > 0, "no recommendation rows written"
            assert checks[6] >= 6, "registry should hold one row per activity"
            assert checks[7] == 0, "a pipeline run failed"
            assert checks[8] >= 6, "monitor rows missing"
            assert checks[9] == 0, "invalid risk bands written"
            print(f"ok: {step}")
    except Exception:
        print(f"FAILED at: {step}")
        traceback.print_exc()
        return 1

    server.cleanup()
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR, ignore_errors=True)
    print("\nSMOKE TEST PASSED: schema -> SQL -> train -> predict -> monitor -> verify")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
