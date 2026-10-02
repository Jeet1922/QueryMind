# QueryMind ML Pipeline (ml_pipeline/)

Machine-learning training, evaluation and MLOps for the six ML activities
defined by [`database/neon_demo_setup.sql`](../database/neon_demo_setup.sql).

This folder is the **separate ML project** the repository README refers to: the
QueryMind application itself only reads persisted `ml_*` outputs through its
approved database functions — nothing in `backend/`, `frontend/`, `database/`
or `scripts/` was modified to build this. Everything here is additive.

## The six ML activities and their algorithms

From the `ml_*` tables in `database/neon_demo_setup.sql`:

| # | Activity | Output table | Problem type | Algorithm | Label |
|---|----------|--------------|--------------|-----------|-------|
| 1 | Tool adoption forecast | `ml_tool_adoption_forecast` | Time-series regression | `HistGradientBoostingRegressor` on adoption lags + seasonality + static tool attributes; recursive 12-month horizon; intervals from test residuals | `label_adoption_rate` (distinct tool users / all employees per month) |
| 2 | User adoption risk | `ml_user_adoption_risk` | Multiclass classification | `HistGradientBoostingClassifier` (LOW/MEDIUM/HIGH); `risk_score` = probability-weighted expected risk | `label_risk_band` from MoM usage decline (`ml_raw_user_adoption_labels`) |
| 3 | User segmentation | `ml_user_segments` | Clustering (unsupervised) | `StandardScaler` + `KMeans`, k by silhouette (canonical k=5 preferred on ties), rule-based canonical segment names, membership-strength score | none |
| 4 | Usage anomaly detection | `ml_usage_anomalies` | Anomaly detection | `IsolationForest` over the weekly residual space (actual, 4-week mean, residual, ratio, z-score); `expected_value` = leakage-free trailing mean; z≥3 baseline reported for comparison | `label_is_anomaly` (injected surge weeks) |
| 5 | Team productivity forecast | `ml_team_productivity_forecast` | Time-series regression | `HistGradientBoostingRegressor` on throughput lags (1–6 months) + delivery health; recursive 6-month horizon; residual intervals; complete months only | `label_completed_story_points` |
| 6 | Tool recommendations | `ml_tool_recommendations` | Recommendation / ranking | Item-item collaborative filtering (cosine over success-weighted 90-day interactions) blended with popularity + category fit (`0.65/0.25/0.10`); engagement-based candidates | temporal holdout: tools first used in the last 30 days; precision@5 / recall@5 / MAP@5 |

## Layout

```text
ml_pipeline/
├── sql/
│   ├── 00_ml_mlops_tables.sql       # model registry, run log, prediction monitor (additive)
│   ├── 01_raw_data_generators.sql   # deterministic raw event stream + raw views + 24-month history
│   ├── 02_features_labels.sql       # one feature/label view per activity
│   └── 03_eda_queries.sql           # read-only EDA queries (numbered per activity)
├── querymind_ml/
│   ├── config.py                    # paths + DATABASE_URL resolution (reads repo .env)
│   ├── db.py                        # psycopg helpers (read_frame, write_frame)
│   ├── tracking.py                  # ml_pipeline_run audit rows (JSONL fallback)
│   ├── registry.py                  # joblib artifacts + ml_model_registry (champion/challenger)
│   ├── monitor.py                   # prediction distribution/drift -> ml_prediction_monitor
│   ├── sql_runner.py                # transactional execution of the numbered SQL files
│   └── activities/                  # one module per activity (load -> fit -> build -> write)
├── notebooks/
│   ├── 00_eda.ipynb                 # rich EDA across all six activities
│   └── 01_training_walkthrough.ipynb# trains every activity + MLOps operating guide
├── tests/
│   └── smoke_end_to_end.py          # embedded-Postgres end-to-end test (pip install pgserver)
├── run_pipeline.py                  # CLI for regular execution
└── requirements.txt                 # kept separate from the app requirements.txt
```

## Quick start

```bash
cd ml_pipeline
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# DATABASE_URL comes from the repo-root .env (Neon pooled URL, sslmode=require)
python run_pipeline.py --list
python run_pipeline.py --step sql     # create MLOps tables, raw data, feature views
python run_pipeline.py --step all     # train + register + predict + monitor, all six
```

### Loading the data on Neon (SQL Editor)

Run in order, after `database/neon_demo_setup.sql`:

1. `sql/00_ml_mlops_tables.sql` — additive MLOps tables.
2. `sql/01_raw_data_generators.sql` — **raw data generator**: ~112K deterministic
   events over 24 complete months (`ml_raw_usage_events`, tagged batches),
   anomaly surges, plus raw views (adoption actuals, weekly usage with injected
   ground truth, monthly activity, risk labels) and a 24-month
   `team_work_metrics` extension that only inserts missing months. Re-running is
   safe: batch rows are deleted/regenerated identically; demo rows are never
   modified.
3. `sql/02_features_labels.sql` — feature & label views (`CREATE OR REPLACE`).
4. `sql/03_eda_queries.sql` — optional read-only EDA (also in the notebook).

## Regular execution (MLOps)

```bash
python run_pipeline.py --activity all --step all      # full retrain + write
python run_pipeline.py --activity user_adoption_risk --step train
python run_pipeline.py --step predict                 # serve champion, no retrain
python run_pipeline.py --step monitor                 # record drift stats only
python run_pipeline.py --step eda                     # execute the EDA batch
python run_pipeline.py --promote                      # force new version to champion
```

- **Versioning**: `<activity>-YYYYmmdd-HHMMSS`; artifacts in
  `artifacts/<activity>/<version>/{model.joblib, meta.json}` and registered in
  `ml_model_registry` with metrics. The first model of an activity becomes
  `champion`; later ones register as `challenger` (`--promote` to switch).
- **Writing predictions** replaces only each activity's own window inside one
  transaction (e.g. risk/segments replace the current prediction month,
  forecasts replace their forward months, anomalies replace
  `metric_name='usage_volume_weekly'` for the last 26 weeks,
  recommendations refresh their small table). The demo `synthetic-demo-v1`
  rows outside those windows are left untouched.
- **Run audit**: every run writes `ml_pipeline_run` (succeeded/failed + details);
  if the MLOps tables are missing it falls back to `reports/run_logs.jsonl`
  instead of failing the run.
- **Drift monitoring**: after each write, `ml_prediction_monitor` stores score
  mean/p50/p95, band mix, and shift flags vs the previous run.

Suggested schedule (cron, on any host that reaches Neon):

```cron
0 6 * * *  cd ml_pipeline && python run_pipeline.py --step predict
0 7 * * 1  cd ml_pipeline && python run_pipeline.py --step all
```

## Verification

`tests/smoke_end_to_end.py` spins up an embedded PostgreSQL
(`pip install pgserver`), loads the app demo schema **read-only**, runs
`00/01/02/03`, executes train+predict+monitor for all six activities, and
asserts output volumes, registry/monitor rows, and zero failed runs:

```bash
pip install pgserver
python tests/smoke_end_to_end.py
```

Latest local run: all six activities green — adoption MAE ≈ 0.028, anomaly
f1 ≈ 0.40 vs injected ground truth (AUC ≈ 0.96, ~3.5× the naive z≥3 baseline
precision), recommendation precision@5 ≈ 0.26, segmentation silhouette ≈ 0.35,
risk macro-F1 ≈ 0.34 against a 0.24 majority baseline (labels are stochastic
by construction, so macro-F1 — not accuracy — is the metric to watch).

## Integration boundary

- The app (`backend/`) keeps consuming `ml_*` tables via its existing
  functions (`get_tool_adoption_forecast`, `get_department_adoption_risk`,
  `get_user_segment_distribution`, `get_usage_anomalies`,
  `get_team_productivity_forecast`, plus the recommendation table).
- Nothing in the app was edited: no schema, route, UI or config changes —
  this folder only reads raw tables and writes `ml_*` outputs + its own
  additive `ml_model_registry` / `ml_pipeline_run` / `ml_prediction_monitor`.
- Raw ML data lives in `ml_raw_*` objects so the application's demo dataset
  (`ai_tool_usage`) is never mutated by training.
