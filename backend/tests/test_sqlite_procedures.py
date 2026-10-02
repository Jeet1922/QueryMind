import pytest

from backend.app.execution.executor import ProcedureExecutor
from backend.app.services.query_service import QueryService


def test_real_sqlite_procedure_returns_rows():
    executor = ProcedureExecutor()
    rows = executor.execute_procedure('get_tool_ranking', {'limit': 3})
    assert rows
    assert len(rows) <= 3
    assert 'tool_name' in rows[0]


def test_query_service_uses_real_sql_result():
    service = QueryService()
    result = service.process('Which AI tools have the highest adoption?')
    assert result['intent'] == 'TOOL_RANKING'
    assert result['supporting_data']
    assert result['answer']


def test_greeting_does_not_run_analytics_procedures():
    result = QueryService().process('whatsup!!')

    assert result['intent'] == 'CONVERSATION'
    assert result['answer'].startswith('Hey!')
    assert result['supporting_data'] == []
    assert result['execution_details']['selected_procedures'] == []


def test_non_analytics_question_does_not_return_unrelated_sql_data():
    result = QueryService().process('What is the capital of the United States?')

    assert result['intent'] == 'OUT_OF_SCOPE'
    assert result['supporting_data'] == []
    assert result['execution_details']['selected_procedures'] == []
    assert 'AI adoption' in result['answer']


@pytest.mark.parametrize(
    ('question', 'expected_intent', 'expected_chart'),
    [
        ('Compare AI adoption across departments', 'DEPARTMENT_ANALYSIS', 'bar'),
        ('Which departments have the highest adoption risk?', 'ADOPTION_RISK', 'donut'),
        ('Show the user segment distribution', 'USER_SEGMENTATION', 'bar'),
        ('Show unusual AI usage anomalies', 'USAGE_ANOMALY', 'bar'),
        ('Show team productivity forecast', 'PRODUCTIVITY_FORECAST', 'bar'),
        ('Show the adoption forecast for ChatGPT', 'ADOPTION_FORECAST', 'line'),
    ],
)
def test_dashboard_questions_return_chartable_sql_rows(question, expected_intent, expected_chart):
    result = QueryService().process(question)

    assert result['intent'] == expected_intent
    assert result['supporting_data']
    assert result['visualization']['type'] == expected_chart
