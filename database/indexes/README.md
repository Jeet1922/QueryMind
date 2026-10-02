# Index strategy

The database includes a targeted set of indexes for the high-volume analytics paths:

- ai_tool_usage by user_id, tool_id, team_id, department_id, usage_timestamp
- team_work_metrics by team_id and metric_month
- ML output tables by tool, user, team, and prediction month

The design supports analytical queries without pre-empting future tuning work.
