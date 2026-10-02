from backend.app.procedures.registry import PROCEDURE_REGISTRY, get_procedure


def test_registry_contains_expected_procedures():
    names = {p["procedure_name"] for p in PROCEDURE_REGISTRY}
    assert "get_tool_ranking" in names
    assert "get_tool_adoption_forecast" in names
    assert "get_department_adoption_risk" in names


def test_registry_entries_are_valid():
    procedure = get_procedure("get_tool_ranking")
    assert procedure["purpose"]
    assert "ai_tool_usage" in procedure["allowed_tables"]
    assert "TOOL_RANKING" in procedure["supported_intents"]
