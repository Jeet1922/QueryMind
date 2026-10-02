from __future__ import annotations

import re
import threading
import time
import uuid

from backend.app.execution.executor import ProcedureExecutor
from backend.app.guardrails.question_guardrails import validate_user_question
from backend.app.planner.intent import build_execution_plan, detect_intent
from backend.app.llm.router import LLMRouter
from backend.app.rag.schema_knowledge import retrieve_knowledge
from backend.app.services.query_history import QueryHistoryStore


class QueryService:
    def __init__(self) -> None:
        self.history_store = QueryHistoryStore()
        self.llm_router = LLMRouter()

    def process(self, question: str) -> dict:
        started_at = time.perf_counter()
        cleaned = validate_user_question(question)
        if self._is_small_talk(cleaned):
            return self._respond_to_small_talk(cleaned, started_at)
        intent = detect_intent(cleaned)
        if not self._is_analytics_question(cleaned):
            return self._respond_out_of_scope(cleaned, started_at)
        fallback_intent = intent['intent'] if intent['intent'] != 'GENERAL_ANALYTICS' else 'OUT_OF_SCOPE'
        intent["intent"] = self.llm_router.classify_intent(cleaned, fallback_intent)
        if intent['intent'] == 'OUT_OF_SCOPE':
            return self._respond_out_of_scope(cleaned, started_at)
        plan = build_execution_plan(cleaned, intent["intent"])
        executor = ProcedureExecutor()
        execution = executor.execute_plan(plan)
        rows = next((step['rows'] for step in reversed(execution) if step.get('rows')), [])
        rag_sources = [entry['table'] for entry in retrieve_knowledge(cleaned)]
        answer = self.llm_router.generate_answer(cleaned, intent['intent'], rows) or self._build_answer(cleaned, intent, rows)
        insight_summary = self.llm_router.summarize_insight(cleaned, intent['intent'], rows)
        default_visualizations = {
            'TOOL_RANKING': 'horizontal_bar',
            'ADOPTION_FORECAST': 'line',
            'ADOPTION_RISK': 'donut',
            'USER_SEGMENTATION': 'bar',
            'USAGE_ANOMALY': 'bar',
            'PRODUCTIVITY_FORECAST': 'bar',
            'PRODUCTIVITY_ANALYSIS': 'bar',
            'DEPARTMENT_ANALYSIS': 'bar',
        }
        visualization_type = self.llm_router.recommend_visualization(cleaned, rows) or default_visualizations.get(
            intent['intent'], intent.get('requested_visualization', 'table')
        )
        visualization_title = {
            'TOOL_RANKING': 'AI tool adoption ranking',
            'ADOPTION_FORECAST': 'AI tool adoption forecast',
            'ADOPTION_RISK': 'Adoption risk by department',
            'USER_SEGMENTATION': 'AI user segment distribution',
            'USAGE_ANOMALY': 'Unusual AI usage activity',
            'PRODUCTIVITY_FORECAST': 'Team productivity forecast',
            'PRODUCTIVITY_ANALYSIS': 'Team productivity forecast',
            'DEPARTMENT_ANALYSIS': 'AI adoption by department',
        }.get(intent['intent'], 'AI adoption summary')
        execution_time_ms = int((time.perf_counter() - started_at) * 1000)
        request_id = str(uuid.uuid4())
        self.history_store.save(request_id, cleaned, intent['intent'], 'completed', execution_time_ms, {
            'answer': answer,
            'insight_summary': insight_summary,
            'visualization': {'type': visualization_type, 'title': visualization_title},
            'supporting_data': rows[:10],
        })
        return {
            'request_id': request_id,
            'question': cleaned,
            'intent': intent['intent'],
            'answer': answer,
            'insight_summary': insight_summary,
            'visualization': {
                'type': visualization_type,
                'title': visualization_title,
            },
            'supporting_data': rows[:10],
            'execution_details': {
                'intent': intent,
                'selected_procedures': [step['procedure'] for step in plan['steps']],
                'execution_order': [step['id'] for step in plan['steps']],
                'execution_time_ms': execution_time_ms,
                'rows_returned': len(rows),
            },
            'ragsources': rag_sources or ['ai_tool_usage'],
            'steps': execution,
        }

    def start_async_query(self, question: str) -> str:
        request_id = str(uuid.uuid4())
        self.history_store.save(request_id, question, 'PENDING', 'queued', 0, {'status': 'queued'})

        def worker() -> None:
            time.sleep(0.1)
            result = self.process(question)
            self.history_store.save(
                request_id,
                result['question'],
                result['intent'],
                'completed',
                result['execution_details']['execution_time_ms'],
                {k: v for k, v in result.items() if k not in {'question', 'intent', 'request_id'}},
            )

        threading.Thread(target=worker, daemon=True).start()
        return request_id

    def get_async_result(self, request_id: str) -> dict | None:
        return self.history_store.get_by_id(request_id)

    def get_history(self) -> list[dict]:
        return self.history_store.list_recent(20)

    def get_model_status(self) -> list[dict]:
        return self.llm_router.status()

    @staticmethod
    def get_database_backend() -> str:
        return ProcedureExecutor().backend

    def _respond_to_small_talk(self, question: str, started_at: float) -> dict:
        answer = "Hey! I'm QueryMind. I can help you explore AI adoption, teams, productivity, and forecasts. What would you like to look into?"
        return self._respond_without_analytics(question, 'CONVERSATION', answer, started_at)

    def _respond_out_of_scope(self, question: str, started_at: float) -> dict:
        answer = "I can answer questions about AI adoption, tool usage, teams, productivity, forecasts, and related business metrics. Try asking about one of those topics."
        return self._respond_without_analytics(question, 'OUT_OF_SCOPE', answer, started_at)

    def _respond_without_analytics(self, question: str, intent_name: str, answer: str, started_at: float) -> dict:
        intent = {
            'intent': intent_name,
            'entities': {},
            'time_range': 'none',
            'grouping': 'none',
            'filters': {},
            'metrics': [],
            'requested_visualization': 'none',
        }
        execution_time_ms = int((time.perf_counter() - started_at) * 1000)
        request_id = str(uuid.uuid4())
        self.history_store.save(request_id, question, intent_name, 'completed', execution_time_ms, {'answer': answer})
        return {
            'request_id': request_id,
            'question': question,
            'intent': intent_name,
            'answer': answer,
            'insight_summary': None,
            'visualization': {'type': 'none', 'title': 'Conversation'},
            'supporting_data': [],
            'execution_details': {
                'intent': intent,
                'selected_procedures': [],
                'execution_order': [],
                'execution_time_ms': execution_time_ms,
                'rows_returned': 0,
            },
            'ragsources': [],
            'steps': [],
        }

    @staticmethod
    def _is_small_talk(question: str) -> bool:
        normalized = re.sub(r'[^a-z\s]', '', question.lower())
        normalized = ' '.join(normalized.split())
        return normalized in {
            'hi', 'hello', 'hey', 'heya', 'yo', 'sup', 'whats up', 'whatsup',
            'how are you', 'how are you doing', 'good morning', 'good afternoon',
            'good evening', 'thanks', 'thank you', 'thanks so much', 'bye', 'goodbye',
        }

    @staticmethod
    def _is_analytics_question(question: str) -> bool:
        lower = question.lower()
        terms = (
            'ai', 'artificial intelligence', 'tool', 'copilot', 'chatgpt', 'claude', 'gemini',
            'adoption', 'usage', 'using', 'used', 'department', 'team', 'productivity',
            'forecast', 'predicted', 'risk', 'segment', 'anomaly', 'throughput', 'employee',
            'user', 'story point', 'business metric', 'kpi',
        )
        return any(term in lower for term in terms)

    @staticmethod
    def _build_answer(question: str, intent: dict, rows: list[dict]) -> str:
        if intent['intent'] == 'ADOPTION_FORECAST' and rows:
            row = rows[0]
            tool_name = row.get('tool_name', 'the selected tool')
            rate = row.get('predicted_adoption_rate', row.get('adoption_rate', 0))
            return f"The forecast for {tool_name} indicates an estimated adoption rate of {float(rate):.0%}, which should be interpreted as a model-generated forecast rather than a historical measurement."
        if intent['intent'] in {'TOOL_RANKING', 'TOOL_ADOPTION'} and rows:
            top = rows[0]
            tool_name = top.get('tool_name', 'the selected tool')
            value = top.get('adoption_count', top.get('usage_count', top.get('value', 0)))
            return f"The highest-adoption tool is {tool_name} with {value} observed usage events in the selected period. This is an observed adoption summary, not a causal claim."
        if intent['intent'] == 'ADOPTION_RISK' and rows:
            dept = rows[0]
            return f"The most significant adoption risk appears in {dept.get('department_name', 'the selected group')} with a model-based risk score of {dept.get('risk_score', 0):.2f}."
        if intent['intent'] == 'DEPARTMENT_ANALYSIS' and rows:
            department = rows[0]
            return f"{department.get('department_name', 'The leading department')} has the highest observed AI usage, averaging {department.get('adoption_rate', 0):.2f} usage events per employee in this dataset."
        if intent['intent'] == 'USER_SEGMENTATION' and rows:
            seg = rows[0]
            return f"The dominant user segment is {seg.get('segment_name', 'AI segment')} with {seg.get('count', seg.get('user_count', 0))} users in the selected period."
        if intent['intent'] == 'USAGE_ANOMALY' and rows:
            anomaly = rows[0]
            return f"The strongest unusual usage signal is for {anomaly.get('tool_name', 'an AI tool')} in {anomaly.get('team_name', 'the selected team')}, with an anomaly score of {anomaly.get('anomaly_score', 0):.2f}."
        if intent['intent'] in {'PRODUCTIVITY_FORECAST', 'PRODUCTIVITY_ANALYSIS'} and rows:
            forecast = rows[0]
            return f"The forecast for {forecast.get('team_name', 'the selected team')} is {forecast.get('predicted_completed_story_points', 0)} completed story points in {forecast.get('prediction_month', 'the next period')}. This is a model-generated estimate."
        return "The available enterprise analytics data indicates a relevant pattern for the requested AI usage and productivity question, and the result should be interpreted as a dataset-backed summary rather than a claim of causation."
