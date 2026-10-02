from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backend.app.guardrails.question_guardrails import validate_user_question
from backend.app.guardrails.sql_validator import SecurityViolation, validate_sql
from backend.app.schemas.query import QueryRequest, QueryResponse
from backend.app.services.query_service import QueryService

app = FastAPI(title="QueryMind API", version="0.1.0")
query_service = QueryService()


@app.get("/api/v1/health")
def health() -> dict:
    return {"status": "ok", "database_backend": query_service.get_database_backend()}


@app.get("/api/v1/ready")
def ready() -> dict:
    return {"status": "ready"}


@app.get("/api/v1/tools")
def list_tools() -> list[str]:
    return ["tool_ranking", "tool_trend", "department_adoption", "forecast", "risk"]


@app.get("/api/v1/schema")
def schema() -> dict:
    return {"tables": ["ai_tool_usage", "ml_tool_adoption_forecast", "ml_user_adoption_risk", "team_work_metrics"]}


@app.post("/api/v1/query")
def submit_query(request: QueryRequest) -> QueryResponse:
    try:
        question = validate_user_question(request.question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result = query_service.process(question)
    return QueryResponse(
        request_id=result["request_id"],
        question=result["question"],
        intent=result["intent"],
        answer=result["answer"],
        insight_summary=result.get("insight_summary"),
        visualization=result["visualization"],
        supporting_data=result["supporting_data"],
        execution_details=result["execution_details"],
        ragsources=result["ragsources"],
    )


@app.post("/api/v1/query/async")
def submit_async_query(request: QueryRequest) -> dict:
    return {"request_id": query_service.start_async_query(request.question), "status": "queued"}


@app.get("/api/v1/query/history")
def query_history() -> list[dict]:
    return query_service.get_history()


@app.get("/api/v1/query/{request_id}")
def get_query_result(request_id: str) -> dict:
    result = query_service.get_async_result(request_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Query result not found")
    return result


@app.get("/api/v1/models/status")
def model_status() -> list[dict]:
    return query_service.get_model_status()


@app.post("/api/v1/admin/reindex")
def reindex() -> dict:
    return {"status": "reindex_requested"}


@app.post("/api/v1/query/validate-sql")
def validate_sql_endpoint(payload: dict) -> dict:
    sql = payload.get("sql", "")
    try:
        cleaned = validate_sql(sql)
    except SecurityViolation as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"sql": cleaned, "status": "ok"}
