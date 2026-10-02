-- QueryMind ML pipeline — raw data generators (run on Neon AFTER neon_demo_setup.sql)
--
-- Creates the ML-ready raw layer for the six activities defined by
-- database/neon_demo_setup.sql:
--   1. ml_raw_usage_events      -> rich event stream (risk, segments, recommender,
--                                  adoption, anomalies inputs)
--   2. ml_raw_tool_adoption_actuals (view) -> ground-truth adoption rate per tool-month
--   3. ml_raw_weekly_usage      (view) -> weekly team x tool aggregates + injected
--                                  anomaly ground truth
--   4. ml_raw_user_monthly_activity (view) -> per-user monthly behaviour
--   5. ml_raw_user_adoption_labels  (view) -> risk/decline labels for training
--   6. team_work_metrics        -> extended to 24 months of history (missing months only)
--
-- Everything is deterministic: pseudo-random values come from hashtextextended
-- hashes, so re-running produces identical rows. Re-running first deletes the
-- batch rows (batch_tag) and only inserts missing team_work_metrics months.
-- This file never modifies existing application rows; it only adds ML rows.

-- ---------------------------------------------------------------------------
-- 1) ml_raw_usage_events: enriched, randomized usage events for ML training
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS ml_raw_usage_events (
    event_id BIGSERIAL PRIMARY KEY,
    batch_tag VARCHAR(40) NOT NULL,
    user_id INTEGER NOT NULL REFERENCES users(user_id),
    team_id INTEGER NOT NULL REFERENCES teams(team_id),
    department_id INTEGER NOT NULL REFERENCES departments(department_id),
    tool_id INTEGER NOT NULL REFERENCES ai_tools(tool_id),
    task_id INTEGER NOT NULL REFERENCES ai_tasks(task_id),
    event_timestamp TIMESTAMPTZ NOT NULL,
    session_duration_seconds INTEGER NOT NULL CHECK (session_duration_seconds > 0),
    success_flag BOOLEAN NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ml_raw_usage_events_ts
    ON ml_raw_usage_events(event_timestamp);
CREATE INDEX IF NOT EXISTS idx_ml_raw_usage_events_user
    ON ml_raw_usage_events(user_id, event_timestamp);
CREATE INDEX IF NOT EXISTS idx_ml_raw_usage_events_team_tool
    ON ml_raw_usage_events(team_id, tool_id, event_timestamp);
CREATE INDEX IF NOT EXISTS idx_ml_raw_usage_events_batch
    ON ml_raw_usage_events(batch_tag);

DELETE FROM ml_raw_usage_events
WHERE batch_tag IN ('ml-raw-v1', 'ml-raw-surge-v1');

-- Base event stream: ~24 complete months of events with
--   * per-user adoption propensity (power users vs light users),
--   * department weighting (Engineering heavier, Finance lighter),
--   * a per-user preferred tool plus a rising cross-tool exploration rate,
--   * a monthly tool-share drift (mature tools shrink, newer tools grow),
--   * weekly/monthly seasonality and an overall growth trend,
--   * improving success rates over time.
-- These structures give every activity a learnable signal.

WITH month_series AS (
    -- months_back runs 1..24 so only COMPLETE past months are generated
    -- (m = 0 would be the current, partial month and could emit future events).
    SELECT
        m AS months_back,
        24 - m AS trend_index,
        (date_trunc('month', CURRENT_DATE) - make_interval(months => m))::date AS month_start,
        EXTRACT(MONTH FROM date_trunc('month', CURRENT_DATE) - make_interval(months => m))::int AS calendar_month,
        EXTRACT(DAY FROM (date_trunc('month', CURRENT_DATE) - make_interval(months => m)
                          + interval '1 month - 1 day'))::int AS days_in_month
    FROM generate_series(1, 24) AS m
),
tool_shares AS (
    -- Monthly per-tool share ladder, normalized via window sums.
    SELECT
        ms.months_back,
        s.tool_id,
        GREATEST(0.03, s.base_share + s.monthly_growth * ms.trend_index) AS share,
        SUM(GREATEST(0.03, s.base_share + s.monthly_growth * ms.trend_index))
            OVER (PARTITION BY ms.months_back) AS total_share,
        SUM(GREATEST(0.03, s.base_share + s.monthly_growth * ms.trend_index))
            OVER (PARTITION BY ms.months_back ORDER BY s.tool_id) AS cum_share
    FROM month_series ms
    CROSS JOIN (VALUES
        (1, 0.24, -0.0050),
        (2, 0.19, -0.0015),
        (3, 0.17,  0.0000),
        (4, 0.16,  0.0010),
        (5, 0.14,  0.0020),
        (6, 0.10,  0.0035)
    ) AS s(tool_id, base_share, monthly_growth)
),
user_prop AS (
    SELECT
        u.user_id,
        u.team_id,
        u.department_id,
        d.department_name,
        (abs(hashtextextended('qm-prop:' || u.user_id, 101)) % 10000) / 10000.0 AS r_prop,
        CASE d.department_name
            WHEN 'Engineering' THEN 1.25
            WHEN 'Operations'  THEN 1.05
            WHEN 'Sales'       THEN 0.95
            ELSE 0.80
        END AS dept_factor
    FROM users u
    JOIN departments d ON d.department_id = u.department_id
),
user_pref AS (
    SELECT
        up.user_id,
        up.team_id,
        up.department_id,
        up.department_name,
        up.r_prop,
        (0.15 + 2.1 * power(up.r_prop, 2)) * up.dept_factor AS propensity,
        CASE up.department_name
            WHEN 'Engineering' THEN (ARRAY[2, 3, 4])[1 + (abs(hashtextextended('qm-pref:' || up.user_id, 102)) % 3)]
            WHEN 'Operations'  THEN (ARRAY[1, 6, 4])[1 + (abs(hashtextextended('qm-pref:' || up.user_id, 102)) % 3)]
            WHEN 'Sales'       THEN (ARRAY[1, 5, 4])[1 + (abs(hashtextextended('qm-pref:' || up.user_id, 102)) % 3)]
            ELSE                   (ARRAY[1, 5, 6])[1 + (abs(hashtextextended('qm-pref:' || up.user_id, 102)) % 3)]
        END AS preferred_tool_id
    FROM user_prop up
),
user_month AS (
    SELECT
        uf.user_id,
        uf.team_id,
        uf.department_id,
        uf.department_name,
        uf.preferred_tool_id,
        uf.r_prop,
        ms.months_back,
        ms.trend_index,
        ms.month_start,
        ms.days_in_month,
        LEAST(300, GREATEST(0, ROUND(
            uf.propensity
            * (18 + 64 * ((abs(hashtextextended('qm-vol:' || uf.user_id || ':' || ms.month_start, 103)) % 10000) / 10000.0))
            * (1 + 0.18 * sin(2 * pi() * (ms.calendar_month - 1) / 12))
            * (1 + 0.02 * ms.trend_index)
        )::int)) AS event_count
    FROM user_pref uf
    CROSS JOIN month_series ms
),
expanded AS (
    SELECT um.*, g.seq
    FROM user_month um
    CROSS JOIN LATERAL generate_series(1, um.event_count) AS g(seq)
),
event_parts AS (
    SELECT
        e.user_id,
        e.team_id,
        e.department_id,
        e.department_name,
        e.preferred_tool_id,
        e.r_prop,
        e.months_back,
        e.trend_index,
        e.month_start,
        e.days_in_month,
        e.seq,
        1 + (abs(hashtextextended('qm-day:' || e.user_id || ':' || e.month_start || ':' || e.seq, 111)) % e.days_in_month::bigint)::int AS event_day,
        (6 + (abs(hashtextextended('qm-hour:' || e.user_id || ':' || e.month_start || ':' || e.seq, 112)) % 15))::int AS event_hour,
        (abs(hashtextextended('qm-min:' || e.user_id || ':' || e.month_start || ':' || e.seq, 113)) % 60)::int AS event_minute,
        (abs(hashtextextended('qm-cross:' || e.user_id || ':' || e.month_start || ':' || e.seq, 114)) % 10000) / 10000.0 AS r_cross,
        (abs(hashtextextended('qm-share:' || e.user_id || ':' || e.month_start || ':' || e.seq, 115)) % 10000) / 10000.0 AS r_share,
        (abs(hashtextextended('qm-task:' || e.user_id || ':' || e.month_start || ':' || e.seq, 116)) % 10000) / 10000.0 AS r_task,
        (abs(hashtextextended('qm-task2:' || e.user_id || ':' || e.month_start || ':' || e.seq, 117)) % 9)::int AS task_uniform,
        (abs(hashtextextended('qm-taskp:' || e.user_id || ':' || e.month_start || ':' || e.seq, 118)) % 2)::int AS task_pref_idx,
        (abs(hashtextextended('qm-dur:' || e.user_id || ':' || e.month_start || ':' || e.seq, 119)) % 10000) / 10000.0 AS r_dur,
        (abs(hashtextextended('qm-succ:' || e.user_id || ':' || e.month_start || ':' || e.seq, 120)) % 10000) / 10000.0 AS r_succ
    FROM expanded e
),
tool_pick AS (
    SELECT
        p.*,
        CASE
            WHEN p.r_cross < (0.05 + 0.005 * p.trend_index)
                THEN COALESCE(ts.share_tool_id, p.preferred_tool_id)
            ELSE p.preferred_tool_id
        END AS tool_id
    FROM event_parts p
    LEFT JOIN LATERAL (
        SELECT sh.tool_id AS share_tool_id
        FROM tool_shares sh
        WHERE sh.months_back = p.months_back
          AND (sh.cum_share / sh.total_share) >= p.r_share
        ORDER BY sh.cum_share
        LIMIT 1
    ) ts ON true
)
INSERT INTO ml_raw_usage_events (
    batch_tag, user_id, team_id, department_id, tool_id, task_id,
    event_timestamp, session_duration_seconds, success_flag
)
SELECT
    'ml-raw-v1',
    tp.user_id,
    tp.team_id,
    tp.department_id,
    tp.tool_id,
    CASE
        WHEN tp.r_task < 0.55 THEN
            CASE tp.department_name
                WHEN 'Engineering' THEN (ARRAY[1, 2])[1 + tp.task_pref_idx]
                WHEN 'Operations'  THEN (ARRAY[5, 7])[1 + tp.task_pref_idx]
                WHEN 'Sales'       THEN (ARRAY[3, 6])[1 + tp.task_pref_idx]
                ELSE                   (ARRAY[4, 9])[1 + tp.task_pref_idx]
            END
        ELSE 1 + tp.task_uniform
    END AS task_id,
    tp.month_start::timestamp
        + make_interval(days => tp.event_day - 1, hours => tp.event_hour, mins => tp.event_minute) AS event_timestamp,
    GREATEST(60, (90 + power(tp.r_dur, 3) * 4500)::int) AS session_duration_seconds,
    tp.r_succ < LEAST(0.99, GREATEST(0.40,
        CASE tp.tool_id
            WHEN 1 THEN 0.87 WHEN 2 THEN 0.84 WHEN 3 THEN 0.89
            WHEN 4 THEN 0.86 WHEN 5 THEN 0.83 ELSE 0.91
        END
        + 0.003 * tp.trend_index
        + 0.06 * (tp.r_prop - 0.33)
    )) AS success_flag
FROM tool_pick tp;

-- Anomaly ground truth: ~0.9% of team x tool x week cells get a demand surge
-- (extra events, tagged 'ml-raw-surge-v1'). ml_usage_anomalies training treats
-- these injected weeks as the labelled anomaly class for evaluation.

INSERT INTO ml_raw_usage_events (
    batch_tag, user_id, team_id, department_id, tool_id, task_id,
    event_timestamp, session_duration_seconds, success_flag
)
WITH base_range AS (
    -- Bound surge weeks so they never spill outside the 24 complete months
    -- (a partial edge month would corrupt lag features and labels).
    SELECT
        (date_trunc('week', date_trunc('month', MIN(event_timestamp)))
         + CASE WHEN date_trunc('week', date_trunc('month', MIN(event_timestamp)))
                     < date_trunc('month', MIN(event_timestamp))
                THEN interval '7 days' ELSE interval '0 days' END)::date AS first_week,
        date_trunc('week', GREATEST(
            date_trunc('month', MIN(event_timestamp)),
            date_trunc('month', MAX(event_timestamp)) + interval '1 month' - interval '7 days'
        ))::date AS last_week
    FROM ml_raw_usage_events
    WHERE batch_tag = 'ml-raw-v1'
),
week_series AS (
    SELECT generate_series(br.first_week, br.last_week, interval '7 days')::date AS week_start
    FROM base_range br
),
surge_cells AS (
    SELECT
        t.team_id,
        tk.tool_id,
        w.week_start,
        12 + (abs(hashtextextended('qm-surge-n:' || t.team_id || ':' || tk.tool_id || ':' || w.week_start, 122)) % 24) AS n_events
    FROM week_series w
    CROSS JOIN teams t
    CROSS JOIN ai_tools tk
    WHERE (abs(hashtextextended('qm-surge:' || t.team_id || ':' || tk.tool_id || ':' || w.week_start, 121)) % 1000) < 9
),
surge_seq AS (
    SELECT c.team_id, c.tool_id, c.week_start, c.n_events, g.seq
    FROM surge_cells c
    CROSS JOIN LATERAL generate_series(1, c.n_events) AS g(seq)
),
surge_member AS (
    -- Pick a deterministic member of the team for each surge event.
    SELECT s.team_id, s.tool_id, s.week_start, s.seq, m.user_id, m.department_id
    FROM surge_seq s
    LEFT JOIN LATERAL (
        SELECT x.user_id, x.department_id
        FROM (
            SELECT
                u.user_id,
                u.department_id,
                ROW_NUMBER() OVER (
                    ORDER BY abs(hashtextextended(
                        'qm-surge-o:' || s.team_id || ':' || s.tool_id || ':' || s.week_start || ':' || s.seq || ':' || u.user_id,
                        123))
                ) AS rn,
                COUNT(*) OVER () AS member_count
            FROM users u
            WHERE u.team_id = s.team_id
        ) x
        WHERE x.rn = 1 + (abs(hashtextextended(
            'qm-surge-p:' || s.team_id || ':' || s.tool_id || ':' || s.week_start || ':' || s.seq, 124)) % x.member_count)
    ) m ON true
)
SELECT
    'ml-raw-surge-v1',
    sm.user_id,
    sm.team_id,
    sm.department_id,
    sm.tool_id,
    1 + (abs(hashtextextended('qm-surge-t:' || sm.team_id || ':' || sm.tool_id || ':' || sm.week_start || ':' || sm.seq, 125)) % 9)::int,
    sm.week_start::timestamp
        + make_interval(
            days => (abs(hashtextextended('qm-surge-d:' || sm.team_id || ':' || sm.tool_id || ':' || sm.week_start || ':' || sm.seq, 126)) % 7)::int,
            hours => (6 + (abs(hashtextextended('qm-surge-h:' || sm.team_id || ':' || sm.tool_id || ':' || sm.week_start || ':' || sm.seq, 127)) % 15))::int,
            mins => (abs(hashtextextended('qm-surge-m:' || sm.team_id || ':' || sm.tool_id || ':' || sm.week_start || ':' || sm.seq, 128)) % 60)::int
        ) AS event_timestamp,
    GREATEST(90, (140 + power((abs(hashtextextended('qm-surge-s:' || sm.team_id || ':' || sm.tool_id || ':' || sm.week_start || ':' || sm.seq, 129)) % 10000) / 10000.0, 3) * 5200)::int) AS session_duration_seconds,
    (abs(hashtextextended('qm-surge-u:' || sm.team_id || ':' || sm.tool_id || ':' || sm.week_start || ':' || sm.seq, 130)) % 10000) / 10000.0 < 0.85 AS success_flag
FROM surge_member sm
WHERE sm.user_id IS NOT NULL;

-- ---------------------------------------------------------------------------
-- 2) ml_raw_tool_adoption_actuals (view): ground truth for adoption forecasting
--    adoption_rate = share of ALL employees who used the tool that month.
-- ---------------------------------------------------------------------------

CREATE OR REPLACE VIEW ml_raw_tool_adoption_actuals AS
WITH monthly AS (
    SELECT
        e.tool_id,
        date_trunc('month', e.event_timestamp)::date AS actual_month,
        COUNT(DISTINCT e.user_id) AS active_users,
        COUNT(*) AS usage_events
    FROM ml_raw_usage_events e
    GROUP BY 1, 2
),
month_totals AS (
    SELECT
        date_trunc('month', e.event_timestamp)::date AS actual_month,
        COUNT(DISTINCT e.user_id) AS active_employees
    FROM ml_raw_usage_events e
    GROUP BY 1
),
total_users AS (
    SELECT COUNT(*) AS n FROM users
)
SELECT
    m.tool_id,
    m.actual_month,
    m.active_users,
    m.usage_events,
    mt.active_employees,
    tu.n AS total_employees,
    ROUND(m.active_users::numeric / NULLIF(tu.n, 0), 4) AS adoption_rate,
    ROUND(m.active_users::numeric / NULLIF(mt.active_employees, 0), 4) AS adoption_rate_active
FROM monthly m
JOIN month_totals mt USING (actual_month)
CROSS JOIN total_users tu;

-- ---------------------------------------------------------------------------
-- 3) ml_raw_weekly_usage (view): weekly team x tool grain for anomaly detection.
--    is_injected_anomaly marks weeks where surge rows were added (ground truth).
-- ---------------------------------------------------------------------------

CREATE OR REPLACE VIEW ml_raw_weekly_usage AS
WITH base AS (
    SELECT
        date_trunc('week', e.event_timestamp)::date AS week_start,
        e.team_id,
        e.tool_id,
        COUNT(*) AS usage_events,
        COUNT(DISTINCT e.user_id) AS distinct_users,
        ROUND(AVG(e.session_duration_seconds), 1) AS avg_session_seconds,
        ROUND(AVG(e.success_flag::int)::numeric, 4) AS success_rate
    FROM ml_raw_usage_events e
    GROUP BY 1, 2, 3
),
surge AS (
    SELECT
        date_trunc('week', e.event_timestamp)::date AS week_start,
        e.team_id,
        e.tool_id,
        COUNT(*) AS surge_events
    FROM ml_raw_usage_events e
    WHERE e.batch_tag = 'ml-raw-surge-v1'
    GROUP BY 1, 2, 3
)
SELECT
    COALESCE(b.week_start, s.week_start) AS week_start,
    COALESCE(b.team_id, s.team_id) AS team_id,
    COALESCE(b.tool_id, s.tool_id) AS tool_id,
    COALESCE(b.usage_events, 0) AS usage_events,
    COALESCE(b.distinct_users, 0) AS distinct_users,
    COALESCE(b.avg_session_seconds, 0) AS avg_session_seconds,
    COALESCE(b.success_rate, 0) AS success_rate,
    COALESCE(s.surge_events, 0) AS surge_events,
    (s.surge_events IS NOT NULL) AS is_injected_anomaly
FROM base b
FULL OUTER JOIN surge s
    ON s.week_start = b.week_start
   AND s.team_id = b.team_id
   AND s.tool_id = b.tool_id;

-- ---------------------------------------------------------------------------
-- 4) ml_raw_user_monthly_activity (view): per-user monthly behaviour.
-- ---------------------------------------------------------------------------

CREATE OR REPLACE VIEW ml_raw_user_monthly_activity AS
SELECT
    e.user_id,
    date_trunc('month', e.event_timestamp)::date AS activity_month,
    COUNT(*) AS usage_events,
    COUNT(DISTINCT date_trunc('day', e.event_timestamp)) AS active_days,
    COUNT(DISTINCT e.tool_id) AS distinct_tools,
    COUNT(DISTINCT e.task_id) AS distinct_tasks,
    ROUND(AVG(e.session_duration_seconds), 1) AS avg_session_seconds,
    ROUND(AVG(e.success_flag::int)::numeric, 4) AS success_rate
FROM ml_raw_usage_events e
GROUP BY 1, 2;

-- ---------------------------------------------------------------------------
-- 5) ml_raw_user_adoption_labels (view): risk/decline labels for training.
--    For each user-month A, the label for prediction month A+1 compares
--    activity in A+1 against activity in A:
--      HIGH   -> zero activity or <= 50% of the previous month
--      MEDIUM -> <= 85% of the previous month
--      LOW    -> otherwise
--    The final month of each user has no next month and is excluded.
-- ---------------------------------------------------------------------------

CREATE OR REPLACE VIEW ml_raw_user_adoption_labels AS
WITH with_next AS (
    SELECT
        a.user_id,
        a.activity_month,
        a.usage_events AS current_events,
        LEAD(a.usage_events) OVER (PARTITION BY a.user_id ORDER BY a.activity_month) AS next_events
    FROM ml_raw_user_monthly_activity a
)
SELECT
    w.user_id,
    (w.activity_month + interval '1 month')::date AS prediction_month,
    w.activity_month AS base_month,
    w.current_events,
    w.next_events,
    ROUND(w.next_events::numeric / NULLIF(w.current_events, 0), 4) AS usage_ratio,
    CASE
        WHEN w.next_events = 0 THEN 'HIGH'
        WHEN w.next_events::numeric <= w.current_events * 0.5 THEN 'HIGH'
        WHEN w.next_events::numeric <= w.current_events * 0.85 THEN 'MEDIUM'
        ELSE 'LOW'
    END AS risk_band
FROM with_next w
WHERE w.next_events IS NOT NULL;

-- ---------------------------------------------------------------------------
-- 6) team_work_metrics: extend history to 24 complete months per team so the
--    productivity forecast has enough lag features. Only inserts months that do
--    not exist yet, so demo rows are never modified and re-runs add nothing.
--    History carries a mild upward trend plus per-team noise for learnability.
-- ---------------------------------------------------------------------------

INSERT INTO team_work_metrics (
    team_id, metric_month, team_size, story_count, completed_story_count,
    estimated_story_count, unestimated_story_count, completed_story_points,
    cycle_time_days, lead_time_days
)
SELECT
    t.team_id,
    (date_trunc('month', CURRENT_DATE) - make_interval(months => h.months_back))::date,
    t.team_size,
    h.story_count,
    d.completed_count,
    d.estimated_count,
    GREATEST(0, h.story_count - d.estimated_count),
    d.story_points,
    d.cycle_time,
    d.lead_time
FROM teams t
CROSS JOIN generate_series(0, 23) AS m(months_back)
CROSS JOIN LATERAL (
    SELECT
        m.months_back AS months_back,
        45 + (abs(hashtextextended('qm-story:' || t.team_id || ':' || m.months_back, 141)) % 55)
            + (23 - m.months_back) AS story_count,
        (abs(hashtextextended('qm-rate:' || t.team_id || ':' || m.months_back, 142)) % 10000) / 10000.0 AS r_rate,
        (abs(hashtextextended('qm-est:' || t.team_id || ':' || m.months_back, 143)) % 10000) / 10000.0 AS r_est,
        (abs(hashtextextended('qm-pt:' || t.team_id || ':' || m.months_back, 144)) % 10000) / 10000.0 AS r_pt,
        (abs(hashtextextended('qm-cyc:' || t.team_id || ':' || m.months_back, 145)) % 10000) / 10000.0 AS r_cyc,
        (abs(hashtextextended('qm-lead:' || t.team_id || ':' || m.months_back, 146)) % 10000) / 10000.0 AS r_lead
) h
CROSS JOIN LATERAL (
    SELECT
        GREATEST(0, LEAST(h.story_count,
            (ROUND(h.story_count * (0.58 + 0.34 * h.r_rate))::int))) AS completed_count,
        GREATEST(0, LEAST(h.story_count,
            (ROUND(h.story_count * (0.70 + 0.25 * h.r_est))::int))) AS estimated_count,
        GREATEST(0,
            (ROUND(h.story_count * (0.58 + 0.34 * h.r_rate))::int) * (3 + ((h.r_pt * 5)::int))
            + ((h.r_pt * 20)::int)) AS story_points,
        ROUND((4.5 + h.r_cyc * 11.0)::numeric, 2) AS cycle_time,
        ROUND((4.5 + h.r_cyc * 11.0 + 2.0 + h.r_lead * 6.0)::numeric, 2) AS lead_time
) d
WHERE NOT EXISTS (
    SELECT 1
    FROM team_work_metrics tw
    WHERE tw.team_id = t.team_id
      AND tw.metric_month = (date_trunc('month', CURRENT_DATE) - make_interval(months => h.months_back))::date
);

-- ---------------------------------------------------------------------------
-- Verification (optional): run after loading to confirm volumes.
-- Expected: ~120K ml_raw_usage_events rows, 24 team_work_metrics months/team.
-- ---------------------------------------------------------------------------
-- SELECT batch_tag, COUNT(*) FROM ml_raw_usage_events GROUP BY batch_tag;
-- SELECT COUNT(*) AS months_per_team FROM (SELECT team_id, metric_month FROM team_work_metrics GROUP BY 1, 2) x;
-- SELECT COUNT(*) AS anomaly_cells FROM ml_raw_weekly_usage WHERE is_injected_anomaly;
-- SELECT risk_band, COUNT(*) FROM ml_raw_user_adoption_labels GROUP BY risk_band;
