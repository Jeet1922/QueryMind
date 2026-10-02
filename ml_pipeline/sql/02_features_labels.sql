-- QueryMind ML pipeline — feature & label views (run AFTER 01_raw_data_generators.sql)
--
-- One view per ML activity defined by database/neon_demo_setup.sql. Each view is
-- the single source of training data for its activity: it exposes raw features,
-- the label where one exists, and the keys needed to write predictions back.
-- All views are CREATE OR REPLACE (idempotent) and touch no application table.
--
--   ml_features_tool_adoption        -> features + label for tool adoption forecast
--   ml_features_user_adoption_risk   -> features + label for adoption risk (risk_band)
--   ml_features_user_segments        -> behaviour features (unsupervised, no label)
--   ml_features_usage_anomalies      -> weekly features + injected-anomaly ground truth
--   ml_features_team_productivity    -> features + label for throughput forecast
--   ml_features_tool_recommendations -> user x tool interaction features

-- ---------------------------------------------------------------------------
-- 1) Tool adoption forecast: lags of monthly adoption rate + static tool attrs
--    Label: label_adoption_rate (known for past months only).
--    Future months are generated recursively in Python from the last lags.
-- ---------------------------------------------------------------------------

CREATE OR REPLACE VIEW ml_features_tool_adoption AS
WITH actual AS (
    SELECT
        a.tool_id,
        a.actual_month,
        a.adoption_rate,
        a.active_users,
        a.usage_events,
        a.adoption_rate_active,
        LAG(a.adoption_rate, 1) OVER w AS prev_1,
        LAG(a.adoption_rate, 2) OVER w AS prev_2,
        LAG(a.adoption_rate, 3) OVER w AS prev_3,
        AVG(a.adoption_rate) OVER (
            PARTITION BY a.tool_id ORDER BY a.actual_month
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ) AS roll_mean_3,
        a.adoption_rate - LAG(a.adoption_rate, 1) OVER w AS diff_1
    FROM ml_raw_tool_adoption_actuals a
    WINDOW w AS (PARTITION BY a.tool_id ORDER BY a.actual_month)
)
SELECT
    act.tool_id,
    t.tool_name,
    t.vendor,
    t.category,
    t.release_date,
    act.actual_month AS forecast_month,
    act.adoption_rate AS label_adoption_rate,
    act.prev_1,
    act.prev_2,
    act.prev_3,
    act.roll_mean_3,
    act.diff_1,
    act.active_users,
    act.usage_events,
    act.adoption_rate_active,
    (EXTRACT(YEAR FROM act.actual_month)::int * 12 + EXTRACT(MONTH FROM act.actual_month)::int) AS month_index,
    (EXTRACT(YEAR FROM act.actual_month)::int * 12 + EXTRACT(MONTH FROM act.actual_month)::int)
        - (EXTRACT(YEAR FROM t.release_date)::int * 12 + EXTRACT(MONTH FROM t.release_date)::int) AS months_since_release
FROM actual act
JOIN ai_tools t ON t.tool_id = act.tool_id;

-- ---------------------------------------------------------------------------
-- 2) User adoption risk: trailing-month behaviour of the months BEFORE the
--    prediction month (no leakage) + employee profile.
--    Label: risk_band from ml_raw_user_adoption_labels (NULL for the latest
--    prediction month, which is the real inference row).
-- ---------------------------------------------------------------------------

CREATE OR REPLACE VIEW ml_features_user_adoption_risk AS
WITH monthly AS (
    SELECT
        m.*,
        LAG(m.usage_events, 1) OVER w AS lag1_events,
        LAG(m.usage_events, 2) OVER w AS lag2_events,
        LAG(m.active_days, 1) OVER w AS lag1_active_days,
        LAG(m.distinct_tools, 1) OVER w AS lag1_distinct_tools,
        LAG(m.success_rate, 1) OVER w AS lag1_success_rate,
        LAG(m.avg_session_seconds, 1) OVER w AS lag1_avg_session,
        AVG(m.usage_events) OVER (
            PARTITION BY m.user_id ORDER BY m.activity_month
            ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING
        ) AS prior6_mean_events
    FROM ml_raw_user_monthly_activity m
    WINDOW w AS (PARTITION BY m.user_id ORDER BY m.activity_month)
)
SELECT
    c.user_id,
    (c.activity_month + interval '1 month')::date AS prediction_month,
    c.activity_month AS feature_month,
    c.usage_events AS prev_month_events,
    c.active_days AS prev_month_active_days,
    c.distinct_tools AS prev_month_distinct_tools,
    c.distinct_tasks AS prev_month_distinct_tasks,
    c.success_rate AS prev_month_success_rate,
    c.avg_session_seconds AS prev_month_avg_session,
    c.lag1_events AS prev2_events,
    c.lag2_events AS prev3_events,
    c.lag1_active_days AS prev2_active_days,
    c.lag1_distinct_tools AS prev2_distinct_tools,
    c.lag1_success_rate AS prev2_success_rate,
    c.lag1_avg_session AS prev2_avg_session,
    c.usage_events - c.lag1_events AS events_mom_delta,
    c.lag1_events - c.lag2_events AS events_trend,
    c.prior6_mean_events,
    lab.risk_band AS label_risk_band,
    lab.usage_ratio AS label_usage_ratio,
    lab.current_events AS label_current_events,
    lab.next_events AS label_next_events,
    u.role,
    u.location,
    u.employment_type,
    u.department_id,
    u.team_id,
    u.practice_id
FROM monthly c
JOIN users u ON u.user_id = c.user_id
LEFT JOIN ml_raw_user_adoption_labels lab
    ON lab.user_id = c.user_id
   AND lab.prediction_month = (c.activity_month + interval '1 month')::date
WHERE c.lag1_events IS NOT NULL
  AND c.lag2_events IS NOT NULL;

-- ---------------------------------------------------------------------------
-- 3) User segmentation: full-history behaviour features (unsupervised, no label)
-- ---------------------------------------------------------------------------

CREATE OR REPLACE VIEW ml_features_user_segments AS
WITH agg AS (
    SELECT
        e.user_id,
        COUNT(*) AS total_events,
        COUNT(DISTINCT e.tool_id) AS distinct_tools,
        COUNT(DISTINCT e.task_id) AS distinct_tasks,
        COUNT(DISTINCT date_trunc('day', e.event_timestamp)) AS active_days,
        COUNT(DISTINCT date_trunc('month', e.event_timestamp)) AS active_months,
        ROUND(AVG(e.session_duration_seconds), 1) AS avg_session_seconds,
        ROUND(AVG(e.success_flag::int)::numeric, 4) AS success_rate,
        MAX(e.event_timestamp) AS last_event_at
    FROM ml_raw_usage_events e
    GROUP BY 1
),
tool_counts AS (
    SELECT e.user_id, e.tool_id, COUNT(*) AS cnt
    FROM ml_raw_usage_events e
    GROUP BY 1, 2
),
top_tool AS (
    SELECT
        user_id,
        MAX(cnt) AS top_tool_events,
        SUM(cnt) AS user_total_events
    FROM tool_counts
    GROUP BY user_id
),
category_counts AS (
    SELECT e.user_id, t.category, COUNT(*) AS cnt
    FROM ml_raw_usage_events e
    JOIN ai_tools t ON t.tool_id = e.tool_id
    GROUP BY 1, 2
)
SELECT
    a.user_id,
    a.total_events,
    a.distinct_tools,
    a.distinct_tasks,
    a.active_days,
    a.active_months,
    a.avg_session_seconds,
    a.success_rate,
    ROUND(a.total_events::numeric / NULLIF(a.active_months, 0), 2) AS events_per_active_month,
    ROUND(tt.top_tool_events::numeric / NULLIF(tt.user_total_events, 0), 4) AS top_tool_share,
    (SELECT COUNT(DISTINCT cc.category) FROM category_counts cc WHERE cc.user_id = a.user_id) AS distinct_categories,
    GREATEST(0, EXTRACT(DAY FROM (CURRENT_TIMESTAMP - a.last_event_at))::int) AS recency_days,
    u.role,
    u.location,
    u.employment_type,
    u.department_id,
    u.team_id,
    u.practice_id
FROM agg a
JOIN top_tool tt ON tt.user_id = a.user_id
JOIN users u ON u.user_id = a.user_id;

-- ---------------------------------------------------------------------------
-- 4) Usage anomalies: weekly team x tool grain with rolling baselines.
--    Expected value = mean of the 4 weeks BEFORE the current week (no leakage).
--    Label: is_injected_anomaly (ground truth injected in 01_raw_data_generators).
--    z_score is a transparent baseline feature alongside the rolling stats.
-- ---------------------------------------------------------------------------

CREATE OR REPLACE VIEW ml_features_usage_anomalies AS
WITH base AS (
    SELECT
        w.week_start,
        w.team_id,
        w.tool_id,
        w.usage_events,
        w.distinct_users,
        w.avg_session_seconds,
        w.success_rate,
        w.is_injected_anomaly,
        LAG(w.usage_events, 1) OVER win AS prev_1w,
        LAG(w.usage_events, 2) OVER win AS prev_2w,
        LAG(w.usage_events, 3) OVER win AS prev_3w,
        LAG(w.usage_events, 4) OVER win AS prev_4w,
        AVG(w.usage_events) OVER win AS roll_mean_4,
        STDDEV_SAMP(w.usage_events) OVER win AS roll_std_4
    FROM ml_raw_weekly_usage w
    WINDOW win AS (
        PARTITION BY w.team_id, w.tool_id
        ORDER BY w.week_start
        ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING
    )
)
SELECT
    b.week_start,
    b.team_id,
    b.tool_id,
    t.team_name,
    tl.tool_name,
    b.usage_events AS actual_value,
    b.roll_mean_4,
    ROUND(b.roll_mean_4::numeric, 4) AS expected_value,
    b.roll_std_4,
    b.usage_events - b.roll_mean_4 AS residual_4w,
    b.usage_events / NULLIF(b.roll_mean_4, 0) AS ratio_4w,
    b.prev_1w,
    b.prev_2w,
    b.prev_3w,
    b.prev_4w,
    b.distinct_users,
    b.avg_session_seconds,
    b.success_rate,
    CASE
        WHEN b.roll_mean_4 IS NOT NULL AND b.roll_std_4 > 0
            THEN ROUND(((b.usage_events - b.roll_mean_4) / b.roll_std_4)::numeric, 4)
    END AS z_score,
    b.is_injected_anomaly AS label_is_anomaly
FROM base b
JOIN teams t ON t.team_id = b.team_id
JOIN ai_tools tl ON tl.tool_id = b.tool_id;

-- ---------------------------------------------------------------------------
-- 5) Team productivity forecast: lags of monthly completed story points +
--    delivery-health metrics. Label: label_completed_story_points.
-- ---------------------------------------------------------------------------

CREATE OR REPLACE VIEW ml_features_team_productivity AS
WITH base AS (
    SELECT
        m.team_id,
        m.metric_month,
        m.completed_story_points AS label_completed_story_points,
        m.story_count,
        m.completed_story_count,
        m.estimated_story_count,
        m.unestimated_story_count,
        m.cycle_time_days,
        m.lead_time_days,
        m.team_size,
        LAG(m.completed_story_points, 1) OVER win AS prev_1_points,
        LAG(m.completed_story_points, 2) OVER win AS prev_2_points,
        LAG(m.completed_story_points, 3) OVER win AS prev_3_points,
        LAG(m.completed_story_points, 4) OVER win AS prev_4_points,
        LAG(m.completed_story_points, 5) OVER win AS prev_5_points,
        LAG(m.completed_story_points, 6) OVER win AS prev_6_points,
        AVG(m.completed_story_points) OVER win AS roll_mean_3,
        LAG(m.cycle_time_days, 1) OVER win AS prev_1_cycle,
        LAG(m.lead_time_days, 1) OVER win AS prev_1_lead,
        LAG(m.completed_story_count, 1) OVER win AS prev_1_completed,
        LAG(m.story_count, 1) OVER win AS prev_1_story_count
    FROM team_work_metrics m
    WINDOW win AS (PARTITION BY m.team_id ORDER BY m.metric_month)
)
SELECT
    b.*,
    t.team_name,
    d.department_name
FROM base b
JOIN teams t ON t.team_id = b.team_id
JOIN departments d ON d.department_id = t.department_id;

-- ---------------------------------------------------------------------------
-- 6) Tool recommendations: user x tool interaction features (implicit feedback).
--    The Python recommender does its own temporal train/test split from
--    ml_raw_usage_events; this view powers EDA and feature inspection.
-- ---------------------------------------------------------------------------

CREATE OR REPLACE VIEW ml_features_tool_recommendations AS
SELECT
    e.user_id,
    e.tool_id,
    COUNT(*) AS usage_events,
    COUNT(DISTINCT date_trunc('day', e.event_timestamp)) AS active_days,
    ROUND(AVG(e.session_duration_seconds), 1) AS avg_session_seconds,
    ROUND(AVG(e.success_flag::int)::numeric, 4) AS success_rate,
    ROUND(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - MAX(e.event_timestamp))) / 86400, 1) AS days_since_last_use
FROM ml_raw_usage_events e
GROUP BY 1, 2;
