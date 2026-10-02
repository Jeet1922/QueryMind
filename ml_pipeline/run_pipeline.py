#!/usr/bin/env python3
"""QueryMind ML pipeline — regular execution entry point.

Examples (run from ml_pipeline/):
    python run_pipeline.py --list
    python run_pipeline.py --step sql                       # load SQL into Neon
    python run_pipeline.py --activity all --step all        # full retrain + write
    python run_pipeline.py --activity user_adoption_risk --step train
    python run_pipeline.py --activity usage_anomalies --step predict
    python run_pipeline.py --activity all --step monitor

Steps:
    sql     run 00/01/02 SQL (MLOps tables, raw generators, feature views)
    train   fit + evaluate + register the model (no output rows written)
    predict load the champion model, generate predictions, replace output window
    monitor record prediction distribution/drift into ml_prediction_monitor
    eda     execute 03_eda_queries.sql (read-only validation)
    all     sql is skipped; for each activity: train + predict + monitor
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from querymind_ml import config, db, monitor, registry, sql_runner, tracking  # noqa: E402
from querymind_ml.activities import ACTIVITIES, activity_names  # noqa: E402


def log(message: str) -> None:
    print(message, flush=True)


def _headline(metrics: dict[str, Any]) -> str:
    for key in ("mae", "macro_f1", "silhouette", "precision_at_5", "accuracy", "r2"):
        if key in metrics:
            return f"{key}={metrics[key]}"
    if isinstance(metrics.get("vs_ground_truth"), dict):
        return f"f1={metrics['vs_ground_truth'].get('f1')}"
    return "-"


def _train_and_register(act, conn, do_register: bool, do_promote: bool):
    df = act.load_dataset(conn)
    result = act.fit(df)
    if do_register:
        artifact = registry.save(
            conn,
            act.spec.name,
            act.spec.task_type,
            act.spec.algorithm,
            result.model_version,
            result.bundle,
            result.metrics,
            result.feature_columns,
            result.training_rows,
        )
        if do_promote:
            registry.promote(conn, act.spec.name, result.model_version)
        log(f"    registered -> {artifact}")
    return df, result


def _load_champion(act, conn):
    loaded = registry.load_latest(conn, act.spec.name)
    if loaded is None:
        raise RuntimeError(
            f"No registered model for {act.spec.name}. Run --step train first."
        )
    bundle, meta = loaded
    return act.result_from_meta(meta, bundle)


def run_activity(
    name: str,
    step: str,
    conn,
    do_register: bool = True,
    do_promote: bool = False,
) -> dict[str, Any]:
    act = ACTIVITIES[name]
    run_id = tracking.start_run(conn, name, step)
    try:
        summary: dict[str, Any] = {"activity": name, "step": step}
        if step == "train":
            _, result = _train_and_register(act, conn, do_register, do_promote)
            summary.update(model_version=result.model_version, written=0)
        elif step in ("predict", "monitor"):
            result = _load_champion(act, conn)
            df = act.load_dataset(conn)
            output = act.build_output(df, result)
            written = 0
            if step == "predict":
                written = act.write_predictions(conn, result, output)
                log(f"    wrote {written} rows -> {act.spec.output_table}")
            monitor.record(
                conn, act.spec.name, result.model_version, output,
                act.spec.output_score_col, act.spec.output_band_col,
                result.metrics,
            )
            summary.update(model_version=result.model_version, written=written)
        else:  # full run
            df, result = _train_and_register(act, conn, do_register, do_promote)
            output = act.build_output(df, result)
            written = act.write_predictions(conn, result, output)
            log(f"    wrote {written} rows -> {act.spec.output_table}")
            monitor.record(
                conn, act.spec.name, result.model_version, output,
                act.spec.output_score_col, act.spec.output_band_col,
                result.metrics,
            )
            summary.update(model_version=result.model_version, written=written)

        summary["headline"] = _headline(result.metrics)
        summary["metrics"] = result.metrics
        tracking.finish_run(
            conn, run_id, "succeeded",
            {k: v for k, v in summary.items() if k != "metrics"}
            | {"metrics": result.metrics},
        )
        return summary
    except Exception as exc:  # noqa: BLE001
        tracking.finish_run(conn, run_id, "failed", {"error": str(exc)})
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QueryMind ML pipeline runner")
    parser.add_argument(
        "--activity",
        default="all",
        choices=["all", *activity_names()],
        help="Activity to run (default: all)",
    )
    parser.add_argument(
        "--step",
        default="all",
        choices=["all", "sql", "train", "predict", "monitor", "eda"],
        help="Pipeline step (default: all = train + predict + monitor)",
    )
    parser.add_argument("--dsn", default=None, help="Override DATABASE_URL")
    parser.add_argument("--no-register", action="store_true",
                        help="Skip writing model artifacts / registry rows")
    parser.add_argument("--promote", action="store_true",
                        help="Force the new version to champion status")
    parser.add_argument("--list", action="store_true", help="List activities and exit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.list:
        log(f"{'activity':30} {'task type':20} {'output table'}")
        log("-" * 78)
        for name in activity_names():
            spec = ACTIVITIES[name].spec
            log(f"{spec.name:30} {spec.task_type:20} {spec.output_table}")
        return 0

    dsn = config.get_dsn(args.dsn)
    log(f"database: {dsn.split('@')[-1]}")  # host/db only, never credentials
    failures: list[str] = []

    with db.connect(dsn) as conn:
        if args.step == "sql":
            executed = sql_runner.run_setup(conn)
            log(f"sql setup complete: {', '.join(executed)}")
            return 0
        if args.step == "eda":
            sql_runner.run_eda(conn)
            log("executed 03_eda_queries.sql (read-only)")
            return 0

        targets = activity_names() if args.activity == "all" else [args.activity]
        for name in targets:
            log(f"\n== {name} [{args.step}] ==")
            try:
                summary = run_activity(
                    name, args.step, conn,
                    do_register=not args.no_register,
                    do_promote=args.promote,
                )
                log(
                    f"    ok: version={summary['model_version']} "
                    f"headline={summary['headline']} written={summary['written']}"
                )
            except Exception as exc:  # noqa: BLE001
                failures.append(name)
                log(f"    FAILED: {exc}")
                traceback.print_exc()

    if failures:
        log(f"\n{len(failures)} activity/activities failed: {', '.join(failures)}")
        return 1
    log("\nAll requested pipeline steps completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
