-- QueryMind ML pipeline — EDA queries (run AFTER 01 + 02 in the Neon SQL Editor)
--
-- Numbered sections matching the six ML activities from neon_demo_setup.sql.
-- Run any section on its own; every query is read-only.

-- === 0. Dataset health =====================================================

-- 0.1 Row counts of every table the pipeline touches
SELECT 'ai_tool_usage' AS table_name, COUNT(*) AS row_count FROM ai_tool_usage
UNION ALL SELECT 'team_work_metrics', COUNT(*) FROM team_work_metrics
UNION ALL SELECT 'ml_raw_usage_events', COUNT(*) FROM ml_raw_usage_events
UNION ALL SELECT 'ml_raw_tool_adoption_actuals', COUNT(*) FROM ml_raw_tool_adoption_actuals
UNION ALL SELECT 'ml_raw_weekly_usage', COUNT(*) FROM ml_raw_weekly_usage
UNION ALL SELECT 'ml_raw_user_monthly_activity', COUNT(*) FROM ml_raw_user_monthly_activity
UNION ALL SELECT 'ml_raw_user_adoption_labels', COUNT(*) FROM ml_raw_user_adoption_labels
ORDER BY row_count DESC;

-- 0.2 Raw event coverage: months, batches, timestamp range
SELECT
    MIN(event_timestamp)::date AS first_event,
    MAX(event_timestamp)::date AS last_event,
    COUNT(DISTINCT date_trunc('month', event_timestamp)) AS months_covered,
    COUNT(*) AS total_events
FROM ml_raw_usage_events;

SELECT batch_tag, COUNT(*) AS events, COUNT(DISTINCT user_id) AS users
FROM ml_raw_usage_events
GROUP BY batch_tag
ORDER BY events DESC;

-- 0.3 Events per department and role (who generates the signal)
SELECT d.department_name, COUNT(*) AS events,
       COUNT(DISTINCT e.user_id) AS users,
       ROUND(AVG(e.success_flag::int)::numeric, 4) AS success_rate
FROM ml_raw_usage_events e
JOIN departments d ON d.department_id = e.department_id
GROUP BY d.department_name
ORDER BY events DESC;

-- === 1. Tool adoption forecast EDA =========================================

-- 1.1 Monthly adoption rate per tool (training series)
SELECT tool_id, forecast_month, label_adoption_rate, prev_1, roll_mean_3
FROM ml_features_tool_adoption
ORDER BY tool_id, forecast_month;

-- 1.2 Trend direction: first vs last adoption rate per tool
SELECT tool_id,
       MIN(label_adoption_rate) AS min_rate,
       MAX(label_adoption_rate) AS max_rate,
       ROUND((MAX(label_adoption_rate) - MIN(label_adoption_rate))::numeric, 4) AS lift,
       COUNT(*) AS months
FROM ml_features_tool_adoption
GROUP BY tool_id
ORDER BY lift DESC;

-- === 2. User adoption risk EDA =============================================

-- 2.1 Label class balance (target balance check before training)
SELECT label_risk_band, COUNT(*) AS rows,
       ROUND(COUNT(*)::numeric / SUM(COUNT(*)) OVER (), 4) AS share
FROM ml_features_user_adoption_risk
WHERE label_risk_band IS NOT NULL
GROUP BY label_risk_band
ORDER BY rows DESC;

-- 2.2 Label balance by prediction month (drift over time)
SELECT prediction_month, label_risk_band, COUNT(*) AS users
FROM ml_features_user_adoption_risk
WHERE label_risk_band IS NOT NULL
GROUP BY prediction_month, label_risk_band
ORDER BY prediction_month, label_risk_band;

-- 2.3 Feature distribution by risk band (do features separate the classes?)
SELECT label_risk_band,
       ROUND(AVG(prev_month_events)::numeric, 2) AS avg_prev_events,
       ROUND(AVG(prev_month_active_days)::numeric, 2) AS avg_active_days,
       ROUND(AVG(prev_month_distinct_tools)::numeric, 2) AS avg_distinct_tools,
       ROUND(AVG(prev_month_success_rate)::numeric, 4) AS avg_success_rate,
       ROUND(AVG(events_mom_delta)::numeric, 2) AS avg_mom_delta
FROM ml_features_user_adoption_risk
WHERE label_risk_band IS NOT NULL
GROUP BY label_risk_band
ORDER BY avg_prev_events DESC;

-- === 3. User segmentation EDA ==============================================

-- 3.1 Behaviour distribution (scale/shape for clustering)
SELECT
       ROUND(AVG(total_events)::numeric, 1) AS avg_events,
       ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_events)::numeric, 1) AS median_events,
       ROUND(AVG(distinct_tools)::numeric, 2) AS avg_tools,
       ROUND(AVG(top_tool_share)::numeric, 4) AS avg_top_tool_share,
       ROUND(AVG(success_rate)::numeric, 4) AS avg_success_rate,
       ROUND(AVG(recency_days)::numeric, 1) AS avg_recency_days
FROM ml_features_user_segments;

-- 3.2 Power users vs light users (raw k-means input check)
SELECT user_id, total_events, distinct_tools, top_tool_share, success_rate, recency_days
FROM ml_features_user_segments
ORDER BY total_events DESC
LIMIT 15;

-- === 4. Usage anomaly EDA ==================================================

-- 4.1 Injected anomaly ground-truth rate
SELECT label_is_anomaly, COUNT(*) AS cells,
       ROUND(COUNT(*)::numeric / SUM(COUNT(*)) OVER (), 4) AS share
FROM ml_features_usage_anomalies
GROUP BY label_is_anomaly;

-- 4.2 Highest z-score weeks vs baseline (are spikes visible to the eye?)
SELECT week_start, team_name, tool_name, actual_value, expected_value, z_score, label_is_anomaly
FROM ml_features_usage_anomalies
WHERE z_score IS NOT NULL
ORDER BY z_score DESC
LIMIT 20;

-- 4.3 Weekly seasonality check (mean events by ISO week part)
SELECT EXTRACT(ISODOW FROM week_start) AS iso_dow, ROUND(AVG(actual_value)::numeric, 2) AS avg_events
FROM ml_features_usage_anomalies
GROUP BY 1
ORDER BY 1;

-- === 5. Team productivity forecast EDA =====================================

-- 5.1 Throughput trend per team
SELECT team_name,
       MIN(metric_month) AS first_month,
       MAX(metric_month) AS last_month,
       ROUND(AVG(label_completed_story_points)::numeric, 1) AS avg_points,
       ROUND(AVG(prev_1_points)::numeric, 1) AS avg_prev_points,
       ROUND(AVG(prev_1_cycle)::numeric, 2) AS avg_cycle_time
FROM ml_features_team_productivity
GROUP BY team_name
ORDER BY avg_points DESC;

-- 5.2 Correlation preview: does cycle time track throughput?
SELECT
    ROUND(CORR(label_completed_story_points, prev_1_cycle)::numeric, 4) AS corr_points_prev_cycle,
    ROUND(CORR(label_completed_story_points, prev_1_points)::numeric, 4) AS corr_points_prev_points,
    ROUND(CORR(label_completed_story_points, prev_1_story_count)::numeric, 4) AS corr_points_prev_stories
FROM ml_features_team_productivity
WHERE prev_1_points IS NOT NULL;

-- === 6. Tool recommendation EDA ============================================

-- 6.1 User x tool interaction density (collaborative filtering feasibility)
SELECT COUNT(DISTINCT user_id) AS users,
       COUNT(DISTINCT tool_id) AS tools,
       COUNT(*) AS observed_pairs,
       ROUND(COUNT(*)::numeric / NULLIF(COUNT(DISTINCT user_id) * COUNT(DISTINCT tool_id), 0), 4) AS density
FROM ml_features_tool_recommendations;

-- 6.2 Tool popularity and quality (popularity prior + content signal)
SELECT t.tool_name, t.category,
       SUM(r.usage_events) AS total_events,
       COUNT(DISTINCT r.user_id) AS distinct_users,
       ROUND(AVG(r.success_rate)::numeric, 4) AS avg_success_rate,
       ROUND(AVG(r.days_since_last_use)::numeric, 1) AS avg_recency_days
FROM ml_features_tool_recommendations r
JOIN ai_tools t ON t.tool_id = r.tool_id
GROUP BY t.tool_id, t.tool_name, t.category
ORDER BY total_events DESC;

-- 6.3 Cross-tool users (users with >= 3 tools = recommendable population)
SELECT COUNT(*) AS users_with_3plus_tools
FROM (
    SELECT user_id FROM ml_features_tool_recommendations GROUP BY user_id HAVING COUNT(*) >= 3
) x;
