from backend.app.planner.intent import detect_intent, build_execution_plan


def test_detect_intent_for_tool_ranking():
    result = detect_intent("Which AI tools have the highest adoption?")
    assert result["intent"] in {"TOOL_RANKING", "TOOL_ADOPTION"}
    assert result["entities"]


def test_detect_intent_for_forecast_question():
    result = detect_intent("What is the predicted adoption of ChatGPT over the next six months?")
    assert result["intent"] == "ADOPTION_FORECAST"
    assert result["entities"]["tool"] == "ChatGPT"


def test_execution_plan_has_steps_for_multi_step_query():
    plan = build_execution_plan("What was the most adopted tool last month and what is its predicted adoption next month?")
    assert len(plan["steps"]) >= 2
    assert plan["steps"][0]["procedure"] == "get_tool_ranking"
