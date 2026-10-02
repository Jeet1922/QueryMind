from __future__ import annotations


def detect_intent(question: str) -> dict:
    normalized = (question or "").strip()
    lower = normalized.lower()

    entities: dict[str, str] = {}
    if "chatgpt" in lower:
        entities["tool"] = "ChatGPT"
    elif "copilot" in lower:
        entities["tool"] = "Microsoft Copilot"
    elif "claude" in lower:
        entities["tool"] = "Claude"
    elif "gemini" in lower:
        entities["tool"] = "Gemini"

    if "forecast" in lower or "predicted" in lower or "expected" in lower:
        if "adoption" in lower:
            return {
                "intent": "ADOPTION_FORECAST",
                "entities": entities,
                "time_range": "next six months" if "six months" in lower else "forecast window",
                "grouping": "tool",
                "filters": {"tool": entities.get("tool")},
                "metrics": ["predicted_adoption_rate"],
                "requested_visualization": "line",
            }

    if "risk" in lower:
        return {
            "intent": "ADOPTION_RISK",
            "entities": entities,
            "time_range": "current forecast window",
            "grouping": "department",
            "filters": {},
            "metrics": ["risk_score"],
            "requested_visualization": "donut",
        }

    if "segment" in lower or "user segment" in lower:
        return {
            "intent": "USER_SEGMENTATION",
            "entities": entities,
            "time_range": "current month",
            "grouping": "segment",
            "filters": {},
            "metrics": ["user_count"],
            "requested_visualization": "bar",
        }

    if "anomaly" in lower or "unusual" in lower:
        return {
            "intent": "USAGE_ANOMALY",
            "entities": entities,
            "time_range": "current month",
            "grouping": "team",
            "filters": {},
            "metrics": ["anomaly_score"],
            "requested_visualization": "scatter",
        }

    if "throughput" in lower or "productivity" in lower:
        return {
            "intent": "PRODUCTIVITY_FORECAST",
            "entities": entities,
            "time_range": "next month",
            "grouping": "team",
            "filters": {},
            "metrics": ["predicted_completed_story_points"],
            "requested_visualization": "bar",
        }

    if "compare" in lower and ("tool" in lower or "copilot" in lower or "chatgpt" in lower):
        return {
            "intent": "TOOL_COMPARISON",
            "entities": entities,
            "time_range": "current period",
            "grouping": "tool",
            "filters": {"tool": entities.get("tool")},
            "metrics": ["usage_count"],
            "requested_visualization": "bar",
        }

    if "department" in lower and ("adoption" in lower or "usage" in lower):
        return {
            "intent": "DEPARTMENT_ANALYSIS",
            "entities": entities,
            "time_range": "current period",
            "grouping": "department",
            "filters": {},
            "metrics": ["adoption_rate"],
            "requested_visualization": "bar",
        }

    if "highest adoption" in lower or "most adopted" in lower or "adoption" in lower:
        if not entities:
            entities["scope"] = "AI tools"
        return {
            "intent": "TOOL_RANKING",
            "entities": entities,
            "time_range": "last month",
            "grouping": "tool",
            "filters": {},
            "metrics": ["adoption_score"],
            "requested_visualization": "horizontal_bar",
        }

    return {
        "intent": "GENERAL_ANALYTICS",
        "entities": entities,
        "time_range": "current period",
        "grouping": "none",
        "filters": {},
        "metrics": [],
        "requested_visualization": "table",
    }


def build_execution_plan(question: str, intent_override: str | None = None) -> dict:
    detected = detect_intent(question)
    intent = intent_override or detected["intent"]

    if intent == "ADOPTION_FORECAST":
        return {
            "intent": intent,
            "steps": [
                {"id": "step_1", "procedure": "get_tool_ranking", "parameters": {"time_period": "last_month", "limit": 1}},
                {"id": "step_2", "procedure": "get_tool_adoption_forecast", "parameters": {"tool_id": "$step_1.top_tool_id", "forecast_months": 6}},
            ],
        }

    if intent in {"TOOL_RANKING", "TOOL_ADOPTION"}:
        return {
            "intent": intent,
            "steps": [
                {"id": "step_1", "procedure": "get_tool_ranking", "parameters": {"time_period": "last_month", "limit": 5}},
            ],
        }

    if intent == "ADOPTION_RISK":
        return {
            "intent": intent,
            "steps": [
                {"id": "step_1", "procedure": "get_department_adoption_risk", "parameters": {"prediction_month": "current_month"}},
            ],
        }

    if intent == "DEPARTMENT_ANALYSIS":
        return {
            "intent": intent,
            "steps": [
                {"id": "step_1", "procedure": "get_department_adoption", "parameters": {"limit": 10}},
            ],
        }

    if intent == "USER_SEGMENTATION":
        return {
            "intent": intent,
            "steps": [
                {"id": "step_1", "procedure": "get_user_segment_distribution", "parameters": {"prediction_month": "current_month", "limit": 10}},
            ],
        }

    if intent == "USAGE_ANOMALY":
        return {
            "intent": intent,
            "steps": [
                {"id": "step_1", "procedure": "get_usage_anomalies", "parameters": {"limit": 10}},
            ],
        }

    if intent in {"PRODUCTIVITY_FORECAST", "PRODUCTIVITY_ANALYSIS"}:
        return {
            "intent": intent,
            "steps": [
                {"id": "step_1", "procedure": "get_team_productivity_forecast", "parameters": {"limit": 10}},
            ],
        }

    return {
        "intent": intent,
        "steps": [
            {"id": "step_1", "procedure": "get_tool_ranking", "parameters": {"time_period": "last_month", "limit": 5}},
            {"id": "step_2", "procedure": "get_tool_usage_trend", "parameters": {"tool_id": "$step_1.top_tool_id", "date_range": "last_six_months"}},
        ],
    }
