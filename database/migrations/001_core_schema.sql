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

CREATE INDEX IF NOT EXISTS idx_ai_tool_usage_user_id ON ai_tool_usage(user_id);
CREATE INDEX IF NOT EXISTS idx_ai_tool_usage_tool_id ON ai_tool_usage(tool_id);
CREATE INDEX IF NOT EXISTS idx_ai_tool_usage_team_id ON ai_tool_usage(team_id);
CREATE INDEX IF NOT EXISTS idx_ai_tool_usage_department_id ON ai_tool_usage(department_id);
CREATE INDEX IF NOT EXISTS idx_ai_tool_usage_timestamp ON ai_tool_usage(usage_timestamp);
CREATE INDEX IF NOT EXISTS idx_team_work_metrics_team_id ON team_work_metrics(team_id);
CREATE INDEX IF NOT EXISTS idx_team_work_metrics_metric_month ON team_work_metrics(metric_month);
CREATE INDEX IF NOT EXISTS idx_users_department_id ON users(department_id);
CREATE INDEX IF NOT EXISTS idx_users_team_id ON users(team_id);
