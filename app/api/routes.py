"""
FastAPI route definitions for health checks, classification, and evaluation runs.
Keeps handlers thin by delegating business logic to EvaluationService.
"""
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    CreateEvaluationRunRequest,
    EvaluationRunDetailResponse,
    EvaluationRunSummaryResponse,
    HealthResponse,
)
from app.services.data_loader import verify_dataset_integrity
from app.services.evaluation_service import EvaluationService
from app.services.fact_store import FactStore

router = APIRouter()
fact_store = FactStore()
evaluation_service = EvaluationService()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check() -> HealthResponse:
    """Verifies service health and dataset checksum integrity."""
    integrity = verify_dataset_integrity()
    dataset_ok = all(integrity.values())
    return HealthResponse(
        status="healthy" if dataset_ok else "degraded",
        version="0.1.0",
        dataset_verified=dataset_ok,
    )


@router.get("/brands", tags=["Knowledge Base"])
def get_brands():
    """Lists all brands available in the verified brand facts store."""
    return {"brands": fact_store.list_brands()}


@router.get("/brands/{brand_id}/models", tags=["Knowledge Base"])
def get_brand_models(brand_id: str):
    """Lists models under a given brand."""
    models = fact_store.list_models_for_brand(brand_id)
    if not models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Brand '{brand_id}' not found.",
        )
    return {"brand_id": brand_id, "models": models}


# ---------------------------------------------------------------------------
# Core Assessment Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/classify",
    response_model=ClassifyResponse,
    status_code=status.HTTP_200_OK,
    tags=["Claim Verification"],
    summary="Extract and verify factual claims from an answer",
)
def classify_endpoint(req: ClassifyRequest) -> ClassifyResponse:
    """
    Accepts required agent input fields and executes extraction and verification.
    Returns HTTP 200 with validated predictions.
    Returns HTTP 404 for an unknown primary brand.
    Returns HTTP 422 for forbidden fields or malformed request.
    Returns HTTP 502 for provider errors or invalid output.
    Returns HTTP 504 for provider timeout.
    """
    return evaluation_service.classify_answer(req)


@router.post(
    "/evaluation-runs",
    response_model=EvaluationRunSummaryResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Evaluation"],
    summary="Create and execute an evaluation run over a dataset split",
)
def create_evaluation_run_endpoint(
    req: CreateEvaluationRunRequest,
    db: Session = Depends(get_db),
) -> EvaluationRunSummaryResponse:
    """
    Processes a fixed CSV split ('development' or 'check') using agent version v1 or v2.
    Preserves terminal records for every answer, including errors.
    Calculates deterministic metrics for development split; keeps metrics null for check split.
    """
    run = evaluation_service.create_evaluation_run(db=db, req=req)

    metrics = None
    if run.split == "development" and run.joint_f1 is not None:
        metrics = {
            "G": run.gold_claims_g,
            "P": run.accepted_predictions_p,
            "M": run.exact_span_matches_m,
            "C": run.correct_label_matches_c,
            "extraction_precision": run.extraction_precision,
            "extraction_recall": run.extraction_recall,
            "extraction_f1": run.extraction_f1,
            "joint_precision": run.joint_precision,
            "joint_recall": run.joint_recall,
            "joint_f1": run.joint_f1,
            "gold_claim_coverage": run.gold_claim_coverage,
            "matched_label_accuracy": run.matched_label_accuracy,
        }

    return EvaluationRunSummaryResponse(
        id=run.id,
        split=run.split,
        agent_version=run.agent_version,
        status=run.status,
        scheduled_count=run.scheduled_count,
        evaluated_count=run.evaluated_count,
        successful_count=run.successful_count,
        failed_count=run.failed_count,
        duration_seconds=run.duration_seconds,
        metrics=metrics,
        started_at=run.started_at,
        completed_at=run.completed_at,
    )


@router.get(
    "/evaluation-runs/{id}",
    response_model=EvaluationRunDetailResponse,
    status_code=status.HTTP_200_OK,
    tags=["Evaluation"],
    summary="Retrieve evaluation run details and predictions in stable order",
)
def get_evaluation_run_endpoint(
    id: uuid.UUID,
    db: Session = Depends(get_db),
) -> EvaluationRunDetailResponse:
    """
    Retrieves full evaluation run details, joined answer outcomes, and claim predictions.
    Returns 404 for missing run.
    Returns 422 for invalid UUID.
    Stable ordering: answers ordered by answer_id ASC, claims ordered by start_char ASC.
    """
    return evaluation_service.get_evaluation_run(db=db, run_id=id)
