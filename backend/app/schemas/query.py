from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)


class ExecutionStepModel(BaseModel):
    id: str
    procedure: str
    parameters: dict[str, object] = {}


class QueryResponse(BaseModel):
    request_id: str
    question: str
    intent: str
    answer: str
    insight_summary: str | None = None
    visualization: dict[str, object]
    supporting_data: list[dict[str, object]]
    execution_details: dict[str, object]
    ragsources: list[str]
