CREATE TABLE IF NOT EXISTS ml_tool_adoption_forecast (
    forecast_id BIGSERIAL PRIMARY KEY,
    tool_id INTEGER NOT NULL REFERENCES ai_tools(tool_id),
    forecast_month DATE NOT NULL,
    predicted_adoption_rate NUMERIC(5,4) NOT NULL CHECK (predicted_adoption_rate >= 0 AND predicted_adoption_rate <= 1),
    lower_bound NUMERIC(5,4) NOT NULL CHECK (lower_bound >= 0 AND lower_bound <= 1),
    upper_bound NUMERIC(5,4) NOT NULL CHECK (upper_bound >= 0 AND upper_bound <= 1),
    model_version VARCHAR(100) NOT NULL,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ml_user_adoption_risk (
    risk_id BIGSERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(user_id),
    prediction_month DATE NOT NULL,
    risk_score NUMERIC(5,4) NOT NULL CHECK (risk_score >= 0 AND risk_score <= 1),
    risk_band VARCHAR(20) NOT NULL CHECK (risk_band IN ('LOW', 'MEDIUM', 'HIGH')),
    model_version VARCHAR(100) NOT NULL,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ml_user_segments (
    segment_id BIGSERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(user_id),
    prediction_month DATE NOT NULL,
    cluster_id INTEGER NOT NULL,
    segment_name VARCHAR(80) NOT NULL,
    segment_score NUMERIC(5,4) NOT NULL CHECK (segment_score >= 0 AND segment_score <= 1),
    model_version VARCHAR(100) NOT NULL,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ml_usage_anomalies (
    anomaly_id BIGSERIAL PRIMARY KEY,
    metric_date DATE NOT NULL,
    team_id INTEGER NOT NULL REFERENCES teams(team_id),
    tool_id INTEGER NOT NULL REFERENCES ai_tools(tool_id),
    metric_name VARCHAR(80) NOT NULL,
    actual_value NUMERIC(12,4) NOT NULL,
    expected_value NUMERIC(12,4) NOT NULL,
    anomaly_score NUMERIC(8,4) NOT NULL,
    is_anomaly BOOLEAN NOT NULL,
    model_version VARCHAR(100) NOT NULL,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ml_team_productivity_forecast (
    prediction_id BIGSERIAL PRIMARY KEY,
    team_id INTEGER NOT NULL REFERENCES teams(team_id),
    prediction_month DATE NOT NULL,
    predicted_completed_story_points INTEGER NOT NULL CHECK (predicted_completed_story_points >= 0),
    lower_bound INTEGER NOT NULL CHECK (lower_bound >= 0),
    upper_bound INTEGER NOT NULL CHECK (upper_bound >= 0),
    model_version VARCHAR(100) NOT NULL,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ml_tool_recommendations (
    recommendation_id BIGSERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(user_id),
    team_id INTEGER NOT NULL REFERENCES teams(team_id),
    tool_id INTEGER NOT NULL REFERENCES ai_tools(tool_id),
    recommendation_score NUMERIC(5,4) NOT NULL CHECK (recommendation_score >= 0 AND recommendation_score <= 1),
    recommendation_reason VARCHAR(255) NOT NULL,
    model_version VARCHAR(100) NOT NULL,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ml_tool_adoption_forecast_tool_id ON ml_tool_adoption_forecast(tool_id, forecast_month);
CREATE INDEX IF NOT EXISTS idx_ml_user_adoption_risk_user_id ON ml_user_adoption_risk(user_id, prediction_month);
CREATE INDEX IF NOT EXISTS idx_ml_user_segments_user_id ON ml_user_segments(user_id, prediction_month);
CREATE INDEX IF NOT EXISTS idx_ml_usage_anomalies_team_id ON ml_usage_anomalies(team_id, metric_date);
CREATE INDEX IF NOT EXISTS idx_ml_usage_anomalies_tool_id ON ml_usage_anomalies(tool_id, metric_date);
CREATE INDEX IF NOT EXISTS idx_ml_team_productivity_forecast_team_id ON ml_team_productivity_forecast(team_id, prediction_month);
