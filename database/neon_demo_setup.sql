-- QueryMind Neon demo setup
-- Safe for a fresh Neon database with a 300 MB storage allowance.
-- Seeds 250,000 synthetic usage events. No ML models are trained here; the
-- ml_* tables contain synthetic example outputs only.
-- Run once in Neon SQL Editor or with psql. Re-running does not add duplicate
-- reference data, usage events, or generated metric rows.

BEGIN;

CREATE TABLE IF NOT EXISTS departments (
    department_id SERIAL PRIMARY KEY,
    department_name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS practices (
    practice_id SERIAL PRIMARY KEY,
    practice_name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS teams (
    team_id SERIAL PRIMARY KEY,
    team_name VARCHAR(120) NOT NULL UNIQUE,
    department_id INTEGER NOT NULL REFERENCES departments(department_id),
    practice_id INTEGER NOT NULL REFERENCES practices(practice_id),
    team_size INTEGER NOT NULL CHECK (team_size > 0)
);

CREATE TABLE IF NOT EXISTS users (
    user_id SERIAL PRIMARY KEY,
    employee_code VARCHAR(50) NOT NULL UNIQUE,
    department_id INTEGER NOT NULL REFERENCES departments(department_id),
    team_id INTEGER NOT NULL REFERENCES teams(team_id),
    practice_id INTEGER NOT NULL REFERENCES practices(practice_id),
    role VARCHAR(80) NOT NULL,
    location VARCHAR(80) NOT NULL,
    employment_type VARCHAR(40) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ai_tools (
    tool_id SERIAL PRIMARY KEY,
    tool_name VARCHAR(120) NOT NULL UNIQUE,
    vendor VARCHAR(120) NOT NULL,
    category VARCHAR(80) NOT NULL,
    description TEXT NOT NULL,
    release_date DATE NOT NULL,
    status VARCHAR(30) NOT NULL DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS ai_tasks (
    task_id SERIAL PRIMARY KEY,
    task_category VARCHAR(80) NOT NULL,
    task_name VARCHAR(120) NOT NULL UNIQUE,
    description TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ai_tool_usage (
    usage_id BIGSERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(user_id),
    tool_id INTEGER NOT NULL REFERENCES ai_tools(tool_id),
    team_id INTEGER NOT NULL REFERENCES teams(team_id),
    department_id INTEGER NOT NULL REFERENCES departments(department_id),
    task_id INTEGER NOT NULL REFERENCES ai_tasks(task_id),
    usage_timestamp TIMESTAMPTZ NOT NULL,
    usage_count INTEGER NOT NULL DEFAULT 1 CHECK (usage_count >= 0),
    session_duration_seconds INTEGER NOT NULL DEFAULT 0 CHECK (session_duration_seconds >= 0),
    success_flag BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS team_work_metrics (
    metric_id BIGSERIAL PRIMARY KEY,
    team_id INTEGER NOT NULL REFERENCES teams(team_id),
    metric_month DATE NOT NULL,
    team_size INTEGER NOT NULL CHECK (team_size > 0),
    story_count INTEGER NOT NULL DEFAULT 0 CHECK (story_count >= 0),
    completed_story_count INTEGER NOT NULL DEFAULT 0 CHECK (completed_story_count >= 0),
    estimated_story_count INTEGER NOT NULL DEFAULT 0 CHECK (estimated_story_count >= 0),
    unestimated_story_count INTEGER NOT NULL DEFAULT 0 CHECK (unestimated_story_count >= 0),
    completed_story_points INTEGER NOT NULL DEFAULT 0 CHECK (completed_story_points >= 0),
    cycle_time_days NUMERIC(6,2) NOT NULL DEFAULT 0,
    lead_time_days NUMERIC(6,2) NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS ml_tool_adoption_forecast (
    forecast_id BIGSERIAL PRIMARY KEY,
    tool_id INTEGER NOT NULL REFERENCES ai_tools(tool_id),
    forecast_month DATE NOT NULL,
    predicted_adoption_rate NUMERIC(5,4) NOT NULL CHECK (predicted_adoption_rate BETWEEN 0 AND 1),
    lower_bound NUMERIC(5,4) NOT NULL CHECK (lower_bound BETWEEN 0 AND 1),
    upper_bound NUMERIC(5,4) NOT NULL CHECK (upper_bound BETWEEN 0 AND 1),
    model_version VARCHAR(100) NOT NULL,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ml_user_adoption_risk (
    risk_id BIGSERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(user_id),
    prediction_month DATE NOT NULL,
    risk_score NUMERIC(5,4) NOT NULL CHECK (risk_score BETWEEN 0 AND 1),
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
    segment_score NUMERIC(5,4) NOT NULL CHECK (segment_score BETWEEN 0 AND 1),
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
    recommendation_score NUMERIC(5,4) NOT NULL CHECK (recommendation_score BETWEEN 0 AND 1),
    recommendation_reason VARCHAR(255) NOT NULL,
    model_version VARCHAR(100) NOT NULL,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS query_history (
    request_id UUID PRIMARY KEY,
    question TEXT NOT NULL,
    intent VARCHAR(80) NOT NULL,
    status VARCHAR(30) NOT NULL,
    execution_time_ms INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    payload JSONB NOT NULL DEFAULT '{}'::jsonb
);

INSERT INTO departments (department_name, description) VALUES
    ('Engineering', 'Product and platform delivery organization.'),
    ('Operations', 'Service execution and operational delivery.'),
    ('Sales', 'Revenue and customer growth organization.'),
    ('Finance', 'Financial planning and governance.')
ON CONFLICT (department_name) DO NOTHING;

INSERT INTO practices (practice_name, description) VALUES
    ('Product Engineering', 'Engineering and platform delivery practice.'),
    ('AI Enablement', 'AI adoption and transformation practice.'),
    ('Customer Success', 'Customer-facing operational excellence.'),
    ('Finance Ops', 'Financial automation and controls.')
ON CONFLICT (practice_name) DO NOTHING;

INSERT INTO ai_tools (tool_name, vendor, category, description, release_date, status) VALUES
    ('ChatGPT', 'OpenAI', 'General AI', 'General-purpose productivity assistant.', DATE '2022-11-30', 'active'),
    ('Microsoft Copilot', 'Microsoft', 'Developer Tool', 'Enterprise productivity and coding assistant.', DATE '2023-09-15', 'active'),
    ('GitHub Copilot', 'GitHub', 'Developer Tool', 'Coding assistance built into the developer workflow.', DATE '2021-10-29', 'active'),
    ('Claude', 'Anthropic', 'Research Assist', 'Long-context analysis and drafting assistant.', DATE '2023-03-14', 'active'),
    ('Gemini', 'Google', 'General AI', 'AI workspace and research companion.', DATE '2023-12-06', 'active'),
    ('Internal AI Assistant', 'QueryMind Labs', 'Internal AI', 'Synthetic internal AI service used by employees.', DATE '2024-05-07', 'active')
ON CONFLICT (tool_name) DO NOTHING;

INSERT INTO ai_tasks (task_category, task_name, description) VALUES
    ('Coding', 'Code Review', 'Reviewing and improving implementation quality.'),
    ('Documentation', 'Technical Writing', 'Drafting internal and customer documentation.'),
    ('Research', 'Market Research', 'Gathering background information and analysis.'),
    ('Data Analysis', 'Trend Analysis', 'Reviewing trend and performance metrics.'),
    ('Customer Support', 'Issue Triage', 'Addressing support and issue intake.'),
    ('Content Generation', 'Proposal Drafting', 'Writing proposals and summaries.'),
    ('Meeting Summarization', 'Meeting Notes', 'Summarizing meetings and actions.'),
    ('Translation', 'Localization', 'Translating text for global usage.'),
    ('Automation', 'Workflow Automation', 'Creating automations and helpers.')
ON CONFLICT (task_name) DO NOTHING;

WITH team_seed(team_name, department_name, practice_name, team_size) AS (
    VALUES
        ('Platform Engineering', 'Engineering', 'Product Engineering', 18),
        ('Data Platform', 'Engineering', 'AI Enablement', 18),
        ('Support Operations', 'Operations', 'Customer Success', 16),
        ('Field Enablement', 'Sales', 'AI Enablement', 17),
        ('Revenue Operations', 'Sales', 'Customer Success', 18),
        ('Finance Analytics', 'Finance', 'Finance Ops', 16),
        ('Customer Intelligence', 'Operations', 'AI Enablement', 17),
        ('Product Ops', 'Engineering', 'Product Engineering', 20),
        ('Enterprise GTM', 'Sales', 'Product Engineering', 18),
        ('Controls & Automation', 'Finance', 'Finance Ops', 15),
        ('AI Operations', 'Operations', 'AI Enablement', 19),
        ('Platform Experience', 'Engineering', 'Product Engineering', 18)
)
INSERT INTO teams (team_name, department_id, practice_id, team_size)
SELECT s.team_name, d.department_id, p.practice_id, s.team_size
FROM team_seed s
JOIN departments d ON d.department_name = s.department_name
JOIN practices p ON p.practice_name = s.practice_name
ON CONFLICT (team_name) DO NOTHING;

INSERT INTO users (employee_code, department_id, team_id, practice_id, role, location, employment_type)
SELECT
    'EMP-' || LPAD((1000 + ROW_NUMBER() OVER (ORDER BY t.team_id, person.member_no))::text, 5, '0'),
    t.department_id,
    t.team_id,
    t.practice_id,
    (ARRAY['Engineer', 'Analyst', 'Manager', 'Developer', 'Product Manager'])[((person.member_no - 1) % 5) + 1],
    (ARRAY['Seattle', 'Austin', 'New York', 'London', 'Berlin'])[((t.team_id + person.member_no - 1) % 5) + 1],
    (ARRAY['Full-time', 'Contract'])[((person.member_no - 1) % 2) + 1]
FROM teams t
CROSS JOIN generate_series(1, 8) AS person(member_no)
WHERE NOT EXISTS (SELECT 1 FROM users)
ON CONFLICT (employee_code) DO NOTHING;

INSERT INTO team_work_metrics (
    team_id, metric_month, team_size, story_count, completed_story_count,
    estimated_story_count, unestimated_story_count, completed_story_points,
    cycle_time_days, lead_time_days
)
SELECT
    t.team_id,
    (date_trunc('month', CURRENT_DATE)::date - (month_no * INTERVAL '1 month'))::date,
    t.team_size,
    48 + ((t.team_id * 7 + month_no * 5) % 70),
    32 + ((t.team_id * 3 + month_no * 4) % 50),
    38 + ((t.team_id * 5 + month_no * 3) % 55),
    5 + ((t.team_id + month_no) % 12),
    120 + ((t.team_id * 19 + month_no * 13) % 260),
    5.5 + ((t.team_id * 13 + month_no * 7) % 90) / 10.0,
    8.0 + ((t.team_id * 11 + month_no * 9) % 120) / 10.0
FROM teams t
CROSS JOIN generate_series(0, 5) AS month_no
WHERE NOT EXISTS (SELECT 1 FROM team_work_metrics);

-- 250K event rows are substantial for a demo, but leave room for indexes and
-- Neon overhead. Lower 250000 to 100000 for a smaller first import.
WITH user_pool AS MATERIALIZED (
    SELECT user_id, team_id, department_id, ROW_NUMBER() OVER (ORDER BY user_id) AS rn FROM users
),
tool_pool AS MATERIALIZED (
    SELECT tool_id, ROW_NUMBER() OVER (ORDER BY tool_id) AS rn FROM ai_tools
),
task_pool AS MATERIALIZED (
    SELECT task_id, ROW_NUMBER() OVER (ORDER BY task_id) AS rn FROM ai_tasks
),
pool_sizes AS (
    SELECT (SELECT COUNT(*) FROM user_pool) AS user_count,
           (SELECT COUNT(*) FROM tool_pool) AS tool_count,
           (SELECT COUNT(*) FROM task_pool) AS task_count
)
INSERT INTO ai_tool_usage (
    user_id, tool_id, team_id, department_id, task_id, usage_timestamp,
    usage_count, session_duration_seconds, success_flag
)
SELECT
    u.user_id,
    t.tool_id,
    u.team_id,
    u.department_id,
    task.task_id,
    NOW() - (((g.i * 37) % 730) * INTERVAL '1 day') - (((g.i * 19) % 1440) * INTERVAL '1 minute'),
    1 + ((g.i * 7) % 12),
    120 + ((g.i * 53) % 5280),
    (g.i % 5) <> 0
FROM generate_series(1, 250000) AS g(i)
CROSS JOIN pool_sizes n
JOIN user_pool u ON u.rn = ((g.i - 1) % n.user_count) + 1
JOIN tool_pool t ON t.rn = ((g.i * 7 - 1) % n.tool_count) + 1
JOIN task_pool task ON task.rn = ((g.i * 11 - 1) % n.task_count) + 1
WHERE NOT EXISTS (SELECT 1 FROM ai_tool_usage);

INSERT INTO ml_tool_adoption_forecast (
    tool_id, forecast_month, predicted_adoption_rate, lower_bound, upper_bound, model_version
)
SELECT
    t.tool_id,
    (date_trunc('month', CURRENT_DATE) + (m.month_no * INTERVAL '1 month'))::date,
    rate.value,
    GREATEST(0.01, rate.value - 0.08),
    LEAST(0.99, rate.value + 0.08),
    'synthetic-demo-v1'
FROM ai_tools t
CROSS JOIN generate_series(1, 12) AS m(month_no)
CROSS JOIN LATERAL (SELECT (0.20 + ((t.tool_id * 7 + m.month_no * 3) % 55) / 100.0)::numeric(5,4) AS value) rate
WHERE NOT EXISTS (SELECT 1 FROM ml_tool_adoption_forecast);

INSERT INTO ml_user_adoption_risk (user_id, prediction_month, risk_score, risk_band, model_version)
SELECT
    u.user_id,
    (date_trunc('month', CURRENT_DATE) + (m.month_no * INTERVAL '1 month'))::date,
    score.value,
    CASE WHEN score.value < 0.35 THEN 'LOW' WHEN score.value < 0.70 THEN 'MEDIUM' ELSE 'HIGH' END,
    'synthetic-demo-v1'
FROM users u
CROSS JOIN generate_series(0, 5) AS m(month_no)
CROSS JOIN LATERAL (SELECT (((u.user_id * 17 + m.month_no * 13) % 100) / 100.0)::numeric(5,4) AS value) score
WHERE NOT EXISTS (SELECT 1 FROM ml_user_adoption_risk);

INSERT INTO ml_user_segments (user_id, prediction_month, cluster_id, segment_name, segment_score, model_version)
SELECT
    u.user_id,
    (date_trunc('month', CURRENT_DATE) + (m.month_no * INTERVAL '1 month'))::date,
    ((u.user_id + m.month_no) % 5) + 1,
    (ARRAY['AI Explorer', 'AI Power User', 'Occasional User', 'Multi-Tool User', 'Specialized User'])[((u.user_id + m.month_no) % 5) + 1],
    (((u.user_id * 11 + m.month_no * 7) % 100) / 100.0)::numeric(5,4),
    'synthetic-demo-v1'
FROM users u
CROSS JOIN generate_series(0, 5) AS m(month_no)
WHERE NOT EXISTS (SELECT 1 FROM ml_user_segments);

INSERT INTO ml_usage_anomalies (
    metric_date, team_id, tool_id, metric_name, actual_value, expected_value,
    anomaly_score, is_anomaly, model_version
)
SELECT
    (date_trunc('month', CURRENT_DATE) - (m.month_no * INTERVAL '1 month'))::date,
    t.team_id,
    ((t.team_id + m.month_no - 1) % (SELECT COUNT(*) FROM ai_tools)) + 1,
    'usage_volume',
    actual.value,
    actual.value - 20,
    (20.0 / GREATEST(1, actual.value - 20))::numeric(8,4),
    (t.team_id + m.month_no) % 4 = 0,
    'synthetic-demo-v1'
FROM teams t
CROSS JOIN generate_series(0, 5) AS m(month_no)
CROSS JOIN LATERAL (SELECT 180 + ((t.team_id * 17 + m.month_no * 23) % 240) AS value) actual
WHERE NOT EXISTS (SELECT 1 FROM ml_usage_anomalies);

INSERT INTO ml_team_productivity_forecast (
    team_id, prediction_month, predicted_completed_story_points, lower_bound, upper_bound, model_version
)
SELECT
    t.team_id,
    (date_trunc('month', CURRENT_DATE) + (m.month_no * INTERVAL '1 month'))::date,
    points.value,
    GREATEST(0, points.value - 20),
    points.value + 25,
    'synthetic-demo-v1'
FROM teams t
CROSS JOIN generate_series(1, 6) AS m(month_no)
CROSS JOIN LATERAL (SELECT 60 + ((t.team_id * 17 + m.month_no * 11) % 190) AS value) points
WHERE NOT EXISTS (SELECT 1 FROM ml_team_productivity_forecast);

INSERT INTO ml_tool_recommendations (
    user_id, team_id, tool_id, recommendation_score, recommendation_reason, model_version
)
SELECT
    u.user_id,
    u.team_id,
    t.tool_id,
    (0.45 + ((u.user_id * 11 + t.tool_id * 7) % 50) / 100.0)::numeric(5,4),
    (ARRAY['Productivity workflow match', 'Research task fit', 'Cross-team reuse'])[((u.user_id + t.tool_id) % 3) + 1],
    'synthetic-demo-v1'
FROM users u
CROSS JOIN ai_tools t
WHERE u.user_id <= 30 AND t.tool_id <= 3
  AND NOT EXISTS (SELECT 1 FROM ml_tool_recommendations);

CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON ai_tool_usage(usage_timestamp);
CREATE INDEX IF NOT EXISTS idx_usage_tool_timestamp ON ai_tool_usage(tool_id, usage_timestamp);
CREATE INDEX IF NOT EXISTS idx_usage_user ON ai_tool_usage(user_id);
CREATE INDEX IF NOT EXISTS idx_usage_team ON ai_tool_usage(team_id);
CREATE INDEX IF NOT EXISTS idx_usage_department ON ai_tool_usage(department_id);
CREATE INDEX IF NOT EXISTS idx_risk_department_month ON ml_user_adoption_risk(prediction_month, user_id);
CREATE INDEX IF NOT EXISTS idx_segment_month ON ml_user_segments(prediction_month, segment_name);
CREATE INDEX IF NOT EXISTS idx_forecast_tool_month ON ml_tool_adoption_forecast(tool_id, forecast_month);
CREATE INDEX IF NOT EXISTS idx_team_forecast_month ON ml_team_productivity_forecast(team_id, prediction_month);
CREATE INDEX IF NOT EXISTS idx_anomaly_date_team ON ml_usage_anomalies(metric_date, team_id);
CREATE INDEX IF NOT EXISTS idx_query_history_created_at ON query_history(created_at DESC);

CREATE OR REPLACE FUNCTION get_tool_ranking(p_days INTEGER DEFAULT 365, p_limit INTEGER DEFAULT 10)
RETURNS TABLE(tool_id INTEGER, tool_name TEXT, adoption_count BIGINT, adoption_score NUMERIC)
LANGUAGE sql STABLE AS $$
    SELECT t.tool_id, t.tool_name::text, COUNT(u.usage_id),
           ROUND(COUNT(u.usage_id)::numeric / NULLIF((SELECT COUNT(*) FROM ai_tool_usage x WHERE x.usage_timestamp >= NOW() - make_interval(days => GREATEST(1, p_days))), 0), 4)
    FROM ai_tools t
    JOIN ai_tool_usage u ON u.tool_id = t.tool_id
    WHERE u.usage_timestamp >= NOW() - make_interval(days => GREATEST(1, p_days))
    GROUP BY t.tool_id, t.tool_name
    ORDER BY COUNT(u.usage_id) DESC, t.tool_name
    LIMIT LEAST(GREATEST(p_limit, 1), 50)
$$;

CREATE OR REPLACE FUNCTION get_tool_usage_trend(p_tool_id INTEGER, p_months INTEGER DEFAULT 12)
RETURNS TABLE(usage_month DATE, usage_count BIGINT, total_session_seconds BIGINT, tool_name TEXT)
LANGUAGE sql STABLE AS $$
    SELECT date_trunc('month', u.usage_timestamp)::date, COUNT(*), SUM(u.session_duration_seconds)::bigint, t.tool_name::text
    FROM ai_tool_usage u
    JOIN ai_tools t ON t.tool_id = u.tool_id
    WHERE u.tool_id = p_tool_id
      AND u.usage_timestamp >= date_trunc('month', CURRENT_DATE) - make_interval(months => LEAST(GREATEST(p_months, 1), 36))
    GROUP BY 1, t.tool_name
    ORDER BY 1
$$;

CREATE OR REPLACE FUNCTION get_department_adoption(p_days INTEGER DEFAULT 365)
RETURNS TABLE(department_id INTEGER, department_name TEXT, adoption_rate NUMERIC)
LANGUAGE sql STABLE AS $$
    SELECT d.department_id, d.department_name::text,
           ROUND(COUNT(DISTINCT u.user_id)::numeric / NULLIF(COUNT(DISTINCT emp.user_id), 0), 4)
    FROM departments d
    JOIN users emp ON emp.department_id = d.department_id
    LEFT JOIN ai_tool_usage u ON u.user_id = emp.user_id
        AND u.usage_timestamp >= NOW() - make_interval(days => GREATEST(1, p_days))
    GROUP BY d.department_id, d.department_name
    ORDER BY 3 DESC, d.department_name
$$;

CREATE OR REPLACE FUNCTION get_tool_adoption_forecast(p_tool_id INTEGER, p_months INTEGER DEFAULT 6)
RETURNS TABLE(tool_name TEXT, forecast_month DATE, predicted_adoption_rate NUMERIC, lower_bound NUMERIC, upper_bound NUMERIC, model_version TEXT)
LANGUAGE sql STABLE AS $$
    SELECT t.tool_name::text, f.forecast_month, f.predicted_adoption_rate::numeric,
           f.lower_bound::numeric, f.upper_bound::numeric, f.model_version::text
    FROM ml_tool_adoption_forecast f
    JOIN ai_tools t ON t.tool_id = f.tool_id
    WHERE f.tool_id = p_tool_id AND f.forecast_month >= date_trunc('month', CURRENT_DATE)::date
    ORDER BY f.forecast_month
    LIMIT LEAST(GREATEST(p_months, 1), 12)
$$;

CREATE OR REPLACE FUNCTION get_department_adoption_risk(p_prediction_month DATE DEFAULT NULL)
RETURNS TABLE(department_name TEXT, risk_score NUMERIC, risk_band TEXT)
LANGUAGE sql STABLE AS $$
    SELECT d.department_name::text, ROUND(AVG(r.risk_score), 4)::numeric,
           CASE WHEN AVG(r.risk_score) >= 0.70 THEN 'HIGH' WHEN AVG(r.risk_score) >= 0.35 THEN 'MEDIUM' ELSE 'LOW' END::text
    FROM ml_user_adoption_risk r
    JOIN users u ON u.user_id = r.user_id
    JOIN departments d ON d.department_id = u.department_id
    WHERE r.prediction_month = COALESCE(p_prediction_month, date_trunc('month', CURRENT_DATE)::date)
    GROUP BY d.department_id, d.department_name
    ORDER BY AVG(r.risk_score) DESC
$$;

CREATE OR REPLACE FUNCTION get_user_segment_distribution(p_prediction_month DATE DEFAULT NULL)
RETURNS TABLE(segment_name TEXT, user_count BIGINT, avg_score NUMERIC)
LANGUAGE sql STABLE AS $$
    SELECT s.segment_name::text, COUNT(DISTINCT s.user_id), ROUND(AVG(s.segment_score), 4)::numeric
    FROM ml_user_segments s
    WHERE s.prediction_month = COALESCE(p_prediction_month, date_trunc('month', CURRENT_DATE)::date)
    GROUP BY s.segment_name
    ORDER BY COUNT(DISTINCT s.user_id) DESC
$$;

CREATE OR REPLACE FUNCTION get_usage_anomalies(p_team_id INTEGER DEFAULT NULL, p_limit INTEGER DEFAULT 20)
RETURNS TABLE(team_name TEXT, tool_name TEXT, anomaly_score NUMERIC, is_anomaly BOOLEAN, metric_date DATE)
LANGUAGE sql STABLE AS $$
    SELECT t.team_name::text, tool.tool_name::text, a.anomaly_score::numeric, a.is_anomaly, a.metric_date
    FROM ml_usage_anomalies a
    JOIN teams t ON t.team_id = a.team_id
    JOIN ai_tools tool ON tool.tool_id = a.tool_id
    WHERE p_team_id IS NULL OR a.team_id = p_team_id
    ORDER BY a.anomaly_score DESC
    LIMIT LEAST(GREATEST(p_limit, 1), 100)
$$;

CREATE OR REPLACE FUNCTION get_team_productivity_forecast(p_team_id INTEGER DEFAULT NULL, p_limit INTEGER DEFAULT 30)
RETURNS TABLE(team_name TEXT, prediction_month DATE, predicted_completed_story_points INTEGER, lower_bound INTEGER, upper_bound INTEGER, model_version TEXT)
LANGUAGE sql STABLE AS $$
    SELECT t.team_name::text, f.prediction_month, f.predicted_completed_story_points,
           f.lower_bound, f.upper_bound, f.model_version::text
    FROM ml_team_productivity_forecast f
    JOIN teams t ON t.team_id = f.team_id
    WHERE p_team_id IS NULL OR f.team_id = p_team_id
    ORDER BY f.prediction_month, t.team_name
    LIMIT LEAST(GREATEST(p_limit, 1), 100)
$$;

COMMIT;

-- Verify data size and row counts before increasing the generator volume.
SELECT pg_size_pretty(pg_database_size(current_database())) AS database_size;
SELECT 'ai_tool_usage' AS table_name, COUNT(*) AS row_count, pg_size_pretty(pg_total_relation_size('ai_tool_usage')) AS total_size FROM ai_tool_usage
UNION ALL SELECT 'users', COUNT(*), pg_size_pretty(pg_total_relation_size('users')) FROM users
UNION ALL SELECT 'ml_user_adoption_risk', COUNT(*), pg_size_pretty(pg_total_relation_size('ml_user_adoption_risk')) FROM ml_user_adoption_risk;

-- Example calls for the QueryMind analytics API.
-- SELECT * FROM get_tool_ranking(365, 5);
-- SELECT * FROM get_tool_usage_trend(1, 12);
-- SELECT * FROM get_department_adoption(365);
-- SELECT * FROM get_tool_adoption_forecast(1, 6);
-- SELECT * FROM get_department_adoption_risk();
-- SELECT * FROM get_user_segment_distribution();
-- SELECT * FROM get_usage_anomalies(NULL, 20);
-- SELECT * FROM get_team_productivity_forecast(NULL, 30);
