"""
Pydantic API request and response schemas for KASPARRO service.
Enforces strict input validation, forbids unauthorized gold fields, and defines response contracts.
"""
import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "0.1.0"
    dataset_verified: bool = True


FORBIDDEN_FIELDS = {
    "gold_claims",
    "claims_gold",
    "expected_label",
    "score",
    "evaluation",
    "difficulty_tags",
    "gold_quote",
    "gold_spans",
    "evaluator_scores",
    "gold_explanations",
}


class ClassifyRequest(BaseModel):
    """Request payload for /classify endpoint."""

    model_config = ConfigDict(extra="forbid")

    answer_id: str = Field(..., min_length=1, description="Answer identifier (e.g. A01)")
    brand_id: str = Field(..., min_length=1, description="Primary brand identifier (e.g. cleartrail)")
    question: str = Field(..., min_length=1, description="User question prompt")
    ai_answer: str = Field(..., min_length=1, description="AI-generated answer text to verify")
    agent_version: Literal["v1", "v2"] = Field("v1", description="Prompt and agent strategy version")

    @field_validator("brand_id")
    @classmethod
    def normalize_brand_id(cls, v: str) -> str:
        return v.strip().lower()


class ClaimPredictionSchema(BaseModel):
    """Validated claim prediction item."""

    model_config = ConfigDict(from_attributes=True)

    claim_id: str = Field(..., description="Deterministic claim identifier format {answer_id}:{start}:{end}")
    start_char: int = Field(..., ge=0, description="0-based start character offset")
    end_char: int = Field(..., gt=0, description="0-based end character offset")
    quote: str = Field(..., min_length=1, description="Exact source substring slice")
    label: Literal["supported", "contradicted", "unknown"] = Field(..., description="Catalogue verification label")
    fact_ids: list[str] = Field(default_factory=list, description="Authoritative catalogue fact IDs cited")
    reason: str = Field(..., max_length=1000, description="Explanation capped at two sentences")


class ClassifyResponse(BaseModel):
    """Response payload for /classify endpoint."""

    answer_id: str
    claims: list[ClaimPredictionSchema]


class CreateEvaluationRunRequest(BaseModel):
    """Request payload for /evaluation-runs endpoint."""

    model_config = ConfigDict(extra="forbid")

    agent_version: Literal["v1", "v2"] = Field("v1", description="Agent prompt version to evaluate")
    split: Literal["development", "check"] = Field(
        "development",
        description="Dataset split to evaluate ('development' for A01-A40, 'check' for A41-A50)",
    )


class EvaluationRunSummaryResponse(BaseModel):
    """Summary response for newly created evaluation run."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    split: str
    agent_version: str
    status: str
    scheduled_count: int
    evaluated_count: int
    successful_count: int
    failed_count: int
    duration_seconds: float
    metrics: dict[str, Any] | None = None
    started_at: datetime
    completed_at: datetime | None = None


class ClaimResultDetailSchema(BaseModel):
    """Detailed claim prediction joined with ground-truth info (when allowed)."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    claim_id: str
    start_char: int
    end_char: int
    quote: str
    label: str
    fact_ids: list[str]
    reason: str
    is_exact_match: bool | None = None
    is_label_correct: bool | None = None
    gold_claim_id: str | None = None
    gold_expected_label: str | None = None


class AnswerResultDetailSchema(BaseModel):
    """Detailed answer execution outcome."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    answer_id: str
    brand_id: str
    success: bool
    duration_seconds: float
    error_type: str | None = None
    error_message: str | None = None
    trace_reference: str | None = None
    claims_count: int
    created_at: datetime
    claims: list[ClaimResultDetailSchema] = Field(default_factory=list)


class EvaluationRunDetailResponse(BaseModel):
    """Full evaluation run details including answers and predictions in stable order."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    split: str
    agent_version: str
    status: str
    provider: str
    model: str
    prompt_version: str
    prompt_hash: str
    fact_version: str
    evaluator_version: str
    source_revision: str | None = None
    settings: dict[str, Any]
    dataset_hashes: dict[str, Any]
    scheduled_count: int
    evaluated_count: int
    successful_count: int
    failed_count: int
    duration_seconds: float
    started_at: datetime
    completed_at: datetime | None = None
    metrics: dict[str, Any] | None = None
    answer_results: list[AnswerResultDetailSchema]
