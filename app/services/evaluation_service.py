"""
Evaluation Service orchestrating classification, persistent evaluation runs,
and deterministic metrics calculation.
"""
import hashlib
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.agent import AgentError, ClaimVerificationAgent, InvalidAgentOutputError
from app.evaluator import (
    AnswerExecutionOutcome,
    PredictionRecord,
    evaluate_predictions,
)
from app.fact_tool import FACT_VERSION, BrandFactTool
from app.models.database import AnswerResult, ClaimResult, EvaluationRun
from app.models.schemas import (
    AnswerResultDetailSchema,
    ClaimPredictionSchema,
    ClaimResultDetailSchema,
    ClassifyRequest,
    ClassifyResponse,
    CreateEvaluationRunRequest,
    EvaluationRunDetailResponse,
)
from app.providers import (
    BaseLLMProvider,
    ProviderError,
    ProviderTimeoutError,
    get_live_provider,
)
from app.services.data_loader import (
    load_answers,
    load_development_gold_for_eval_only,
)
from scripts.run_live_evaluation import DevelopmentBenchmarkProvider

ANSWERS_PATH = Path("data/answers.csv")
BRAND_FACTS_PATH = Path("data/brand_facts.csv")
DEV_GOLD_PATH = Path("data/development_gold.csv")


def compute_file_sha256(file_path: Path) -> str:
    """Computes SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_current_git_revision() -> str:
    """Gets current git revision hash if available."""
    import subprocess
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
        return rev
    except Exception:
        return "unknown"


def resolve_service_provider(agent_version: str) -> BaseLLMProvider:
    """Resolves live provider if configured, or falls back to benchmark simulator."""
    try:
        return get_live_provider()
    except ProviderError:
        return DevelopmentBenchmarkProvider(prompt_version=agent_version)


class EvaluationService:
    """Encapsulates business logic for claim verification and evaluation persistence."""

    def __init__(self, fact_tool: BrandFactTool | None = None):
        self._fact_tool = fact_tool or BrandFactTool()

    @property
    def fact_tool(self) -> BrandFactTool:
        return self._fact_tool

    def classify_answer(self, req: ClassifyRequest) -> ClassifyResponse:
        """Executes agent claim extraction and verification on a single answer."""
        # 1. Validate primary brand
        if req.brand_id not in self._fact_tool.known_brands:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unknown brand '{req.brand_id}'. Permitted brands: {sorted(self._fact_tool.known_brands)}",
            )

        prompt_path = Path(f"prompts/{req.agent_version}.txt")
        if not prompt_path.exists():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Prompt version '{req.agent_version}' does not exist.",
            )

        provider = resolve_service_provider(req.agent_version)
        agent = ClaimVerificationAgent(
            provider=provider,
            fact_tool=self._fact_tool,
            prompt_path=prompt_path,
        )

        try:
            result = agent.verify_answer(
                answer_id=req.answer_id,
                brand_id=req.brand_id,
                question=req.question,
                ai_answer=req.ai_answer,
            )
            predictions = [
                ClaimPredictionSchema(
                    claim_id=c.claim_id,
                    start_char=c.start_char,
                    end_char=c.end_char,
                    quote=c.quote,
                    label=c.label,
                    fact_ids=c.fact_ids,
                    reason=c.reason,
                )
                for c in result.claims
            ]
            return ClassifyResponse(answer_id=req.answer_id, claims=predictions)

        except ProviderTimeoutError as exc:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Gateway timeout: Model provider request timed out.",
            ) from exc

        except (ProviderError, InvalidAgentOutputError, AgentError) as exc:
            if exc.__cause__ and isinstance(exc.__cause__, ProviderTimeoutError):
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="Gateway timeout: Model provider request timed out.",
                ) from exc
            sanitized_msg = f"Bad gateway: Model provider failure or unparseable output ({type(exc).__name__})."
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=sanitized_msg,
            ) from exc

    def create_evaluation_run(
        self, db: Session, req: CreateEvaluationRunRequest
    ) -> EvaluationRun:
        """Executes and records a complete evaluation run over a fixed split."""
        prompt_path = Path(f"prompts/{req.agent_version}.txt")
        if not prompt_path.exists():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Prompt version '{req.agent_version}' not found.",
            )

        # 1. Dataset splits
        all_answers = load_answers()
        scheduled_answers = [a for a in all_answers if a.split == req.split]
        if not scheduled_answers:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"No answers found for split '{req.split}'.",
            )

        scheduled_ids = [a.answer_id for a in scheduled_answers]

        prompt_hash = compute_file_sha256(prompt_path)
        answers_hash = compute_file_sha256(ANSWERS_PATH)
        brand_facts_hash = compute_file_sha256(BRAND_FACTS_PATH)
        gold_hash = compute_file_sha256(DEV_GOLD_PATH) if DEV_GOLD_PATH.exists() else "none"

        provider = resolve_service_provider(req.agent_version)
        agent = ClaimVerificationAgent(
            provider=provider,
            fact_tool=self._fact_tool,
            prompt_path=prompt_path,
        )

        start_time = datetime.now(UTC)
        run_id = uuid.uuid4()

        run = EvaluationRun(
            id=run_id,
            split=req.split,
            agent_version=req.agent_version,
            status="processing",
            provider=provider.provider_name,
            model=provider.model_name,
            prompt_version=req.agent_version,
            prompt_hash=prompt_hash,
            fact_version=FACT_VERSION,
            evaluator_version="1.0.0",
            source_revision=get_current_git_revision(),
            settings={"max_turns": 3, "timeout": 35.0},
            dataset_hashes={
                "answers.csv": answers_hash,
                "brand_facts.csv": brand_facts_hash,
                "development_gold.csv": gold_hash,
            },
            scheduled_count=len(scheduled_answers),
            started_at=start_time,
        )
        db.add(run)
        db.commit()

        # 2. Process all answers
        outcomes: list[AnswerExecutionOutcome] = []
        answer_records: list[AnswerResult] = []
        claim_records: list[ClaimResult] = []

        total_t0 = time.perf_counter()

        try:
            for ans in scheduled_answers:
                t0 = time.perf_counter()
                try:
                    ver_res = agent.verify_answer(
                        answer_id=ans.answer_id,
                        brand_id=ans.brand_id,
                        question=ans.question,
                        ai_answer=ans.ai_answer,
                    )
                    duration = time.perf_counter() - t0

                    ans_rec = AnswerResult(
                        id=uuid.uuid4(),
                        run_id=run.id,
                        answer_id=ans.answer_id,
                        brand_id=ans.brand_id,
                        success=True,
                        duration_seconds=round(duration, 3),
                        claims_count=len(ver_res.claims),
                        created_at=datetime.now(UTC),
                    )
                    answer_records.append(ans_rec)

                    pred_recs = []
                    for c in ver_res.claims:
                        c_rec = ClaimResult(
                            id=uuid.uuid4(),
                            run_id=run.id,
                            answer_id=ans.answer_id,
                            claim_id=c.claim_id,
                            start_char=c.start_char,
                            end_char=c.end_char,
                            quote=c.quote,
                            label=c.label,
                            fact_ids=c.fact_ids,
                            reason=c.reason,
                            created_at=datetime.now(UTC),
                        )
                        claim_records.append(c_rec)
                        pred_recs.append(
                            PredictionRecord(
                                answer_id=ans.answer_id,
                                claim_id=c.claim_id,
                                start_char=c.start_char,
                                end_char=c.end_char,
                                quote=c.quote,
                                label=c.label,
                                fact_ids=c.fact_ids,
                                reason=c.reason,
                                is_valid_slice=True,
                            )
                        )

                    outcomes.append(
                        AnswerExecutionOutcome(
                            answer_id=ans.answer_id,
                            success=True,
                            duration_seconds=round(duration, 3),
                            predictions=pred_recs,
                        )
                    )

                except Exception as exc:
                    duration = time.perf_counter() - t0
                    err_type = type(exc).__name__
                    err_msg = str(exc)[:250]

                    ans_rec = AnswerResult(
                        id=uuid.uuid4(),
                        run_id=run.id,
                        answer_id=ans.answer_id,
                        brand_id=ans.brand_id,
                        success=False,
                        duration_seconds=round(duration, 3),
                        error_type=err_type,
                        error_message=err_msg,
                        claims_count=0,
                        created_at=datetime.now(UTC),
                    )
                    answer_records.append(ans_rec)
                    outcomes.append(
                        AnswerExecutionOutcome(
                            answer_id=ans.answer_id,
                            success=False,
                            duration_seconds=round(duration, 3),
                            error_message=f"{err_type}: {err_msg}",
                            predictions=[],
                        )
                    )

            total_elapsed = time.perf_counter() - total_t0

            # Persist answer outcomes and claim predictions
            db.add_all(answer_records)
            db.add_all(claim_records)
            db.flush()

            # 3. Post-run scoring for development split ONLY
            if req.split == "development":
                gold_claims = load_development_gold_for_eval_only(DEV_GOLD_PATH)
                eval_report = evaluate_predictions(
                    outcomes=outcomes,
                    gold_claims=gold_claims,
                    split_name="development",
                    scheduled_answer_ids=scheduled_ids,
                )
                m = eval_report.metrics
                run.gold_claims_g = m.G
                run.accepted_predictions_p = m.P
                run.exact_span_matches_m = m.M
                run.correct_label_matches_c = m.C
                run.extraction_precision = m.extraction_precision
                run.extraction_recall = m.extraction_recall
                run.extraction_f1 = m.extraction_f1
                run.joint_precision = m.joint_precision
                run.joint_recall = m.joint_recall
                run.joint_f1 = m.joint_f1
                run.gold_claim_coverage = m.gold_claim_coverage
                run.matched_label_accuracy = m.matched_label_accuracy

                run.error_breakdown = {
                    "missed_gold_claim_ids": eval_report.missed_gold_claim_ids,
                    "unmatched_prediction_ids": eval_report.unmatched_prediction_ids,
                    "label_mismatches_count": len(eval_report.label_mismatches or []),
                }

                # Map match details into claim records
                gold_span_map = {(g.answer_id, g.start_char, g.end_char): g for g in gold_claims}
                for cr in claim_records:
                    key = (cr.answer_id, cr.start_char, cr.end_char)
                    if key in gold_span_map:
                        g = gold_span_map[key]
                        cr.is_exact_match = True
                        cr.gold_claim_id = g.claim_id
                        cr.gold_expected_label = getattr(g.expected_label, "value", str(g.expected_label))
                        cr.is_label_correct = (cr.label == cr.gold_expected_label)
                    else:
                        cr.is_exact_match = False
                        cr.is_label_correct = False
            else:
                # Check split: Quality metrics and ground truth match fields MUST remain NULL
                run.gold_claims_g = None
                run.accepted_predictions_p = None
                run.exact_span_matches_m = None
                run.correct_label_matches_c = None
                run.extraction_precision = None
                run.extraction_recall = None
                run.extraction_f1 = None
                run.joint_precision = None
                run.joint_recall = None
                run.joint_f1 = None
                run.gold_claim_coverage = None
                run.matched_label_accuracy = None
                run.error_breakdown = None

            run.evaluated_count = len(answer_records)
            run.successful_count = sum(1 for a in answer_records if a.success)
            run.failed_count = sum(1 for a in answer_records if not a.success)
            run.duration_seconds = round(total_elapsed, 2)
            run.status = "completed"
            run.completed_at = datetime.now(UTC)

            db.commit()
            db.refresh(run)
            return run

        except Exception as infra_err:
            db.rollback()
            run.status = "failed"
            run.error_message = f"Infrastructure failure: {type(infra_err).__name__}: {infra_err}"
            run.completed_at = datetime.now(UTC)
            db.add(run)
            db.commit()
            raise

    def get_evaluation_run(self, db: Session, run_id: uuid.UUID) -> EvaluationRunDetailResponse:
        """Retrieves an evaluation run with joined answer and claim records in stable order."""
        run = db.query(EvaluationRun).filter(EvaluationRun.id == run_id).first()
        if not run:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Evaluation run '{run_id}' not found.",
            )

        # Retrieve answers in stable order (answer_id ASC)
        answers = (
            db.query(AnswerResult)
            .filter(AnswerResult.run_id == run_id)
            .order_by(AnswerResult.answer_id.asc())
            .all()
        )

        # Retrieve claims in stable span order (answer_id ASC, start_char ASC)
        claims = (
            db.query(ClaimResult)
            .filter(ClaimResult.run_id == run_id)
            .order_by(ClaimResult.answer_id.asc(), ClaimResult.start_char.asc())
            .all()
        )

        claims_by_answer: dict[str, list[ClaimResultDetailSchema]] = {}
        for c in claims:
            claims_by_answer.setdefault(c.answer_id, []).append(
                ClaimResultDetailSchema(
                    id=c.id,
                    claim_id=c.claim_id,
                    start_char=c.start_char,
                    end_char=c.end_char,
                    quote=c.quote,
                    label=c.label,
                    fact_ids=c.fact_ids or [],
                    reason=c.reason,
                    is_exact_match=c.is_exact_match if run.split == "development" else None,
                    is_label_correct=c.is_label_correct if run.split == "development" else None,
                    gold_claim_id=c.gold_claim_id if run.split == "development" else None,
                    gold_expected_label=c.gold_expected_label if run.split == "development" else None,
                )
            )

        answer_schemas = [
            AnswerResultDetailSchema(
                id=a.id,
                answer_id=a.answer_id,
                brand_id=a.brand_id,
                success=a.success,
                duration_seconds=a.duration_seconds,
                error_type=a.error_type,
                error_message=a.error_message,
                trace_reference=a.trace_reference,
                claims_count=a.claims_count,
                created_at=a.created_at,
                claims=claims_by_answer.get(a.answer_id, []),
            )
            for a in answers
        ]

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
                "error_breakdown": run.error_breakdown,
            }

        return EvaluationRunDetailResponse(
            id=run.id,
            split=run.split,
            agent_version=run.agent_version,
            status=run.status,
            provider=run.provider,
            model=run.model,
            prompt_version=run.prompt_version,
            prompt_hash=run.prompt_hash,
            fact_version=run.fact_version,
            evaluator_version=run.evaluator_version,
            source_revision=run.source_revision,
            settings=run.settings or {},
            dataset_hashes=run.dataset_hashes or {},
            scheduled_count=run.scheduled_count,
            evaluated_count=run.evaluated_count,
            successful_count=run.successful_count,
            failed_count=run.failed_count,
            duration_seconds=run.duration_seconds,
            started_at=run.started_at,
            completed_at=run.completed_at,
            metrics=metrics,
            answer_results=answer_schemas,
        )
