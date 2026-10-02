-- QueryMind ML pipeline — MLOps support tables
-- Additive only: creates new ml_* / ml_pipeline_* tables. It does not modify
-- any application table. Safe to re-run (all objects use IF NOT EXISTS).
-- Run order: 00 -> 01 -> 02 -> 03, after database/neon_demo_setup.sql.

CREATE TABLE IF NOT EXISTS ml_model_registry (
    registry_id BIGSERIAL PRIMARY KEY,
    activity VARCHAR(60) NOT NULL,
    model_version VARCHAR(120) NOT NULL,
    algorithm VARCHAR(120) NOT NULL,
    task_type VARCHAR(40) NOT NULL,
    artifact_path TEXT NOT NULL,
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    feature_names JSONB NOT NULL DEFAULT '[]'::jsonb,
    training_row_count INTEGER,
    status VARCHAR(20) NOT NULL DEFAULT 'challenger'
        CHECK (status IN ('champion', 'challenger', 'archived')),
    notes TEXT,
    trained_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (activity, model_version)
);

CREATE TABLE IF NOT EXISTS ml_pipeline_run (
    run_id BIGSERIAL PRIMARY KEY,
    activity VARCHAR(60) NOT NULL,
    step VARCHAR(40) NOT NULL,
    status VARCHAR(20) NOT NULL CHECK (status IN ('running', 'succeeded', 'failed')),
    model_version VARCHAR(120),
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    details JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_ml_pipeline_run_activity
    ON ml_pipeline_run(activity, started_at DESC);

CREATE TABLE IF NOT EXISTS ml_prediction_monitor (
    monitor_id BIGSERIAL PRIMARY KEY,
    activity VARCHAR(60) NOT NULL,
    monitor_date DATE NOT NULL DEFAULT CURRENT_DATE,
    model_version VARCHAR(120),
    prediction_count INTEGER NOT NULL,
    score_mean NUMERIC(10,6),
    score_std NUMERIC(10,6),
    score_p50 NUMERIC(10,6),
    score_p95 NUMERIC(10,6),
    band_distribution JSONB NOT NULL DEFAULT '{}'::jsonb,
    drift_vs_previous JSONB NOT NULL DEFAULT '{}'::jsonb,
    evaluation_metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ml_prediction_monitor_activity
    ON ml_prediction_monitor(activity, monitor_date DESC);
