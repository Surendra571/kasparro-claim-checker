"""
SQLAlchemy database models for KASPARRO Claim Verification Service.
Defines persistent tables: evaluation_runs, answer_results, claim_results.
"""
import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import relationship
from sqlalchemy.types import CHAR, TypeDecorator

from app.core.database import Base


class GUID(TypeDecorator):
    """Platform-independent GUID type.
    Uses PostgreSQL's UUID type, otherwise uses CHAR(36), storing as stringified hex.
    """

    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        else:
            return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == "postgresql":
            return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
        else:
            if not isinstance(value, uuid.UUID):
                return str(uuid.UUID(str(value)))
            else:
                return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        if not isinstance(value, uuid.UUID):
            return uuid.UUID(str(value))
        return value


JSONType = JSON().with_variant(JSONB(), "postgresql")


class EvaluationRun(Base):
    """Represents a scheduled evaluation run over development or check splits."""

    __tablename__ = "evaluation_runs"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    split = Column(String(32), nullable=False)  # "development" or "check"
    agent_version = Column(String(32), nullable=False)  # "v1" or "v2"
    status = Column(String(32), nullable=False, default="processing")  # "completed", "failed", "processing"

    provider = Column(String(64), nullable=False)
    model = Column(String(64), nullable=False)
    prompt_version = Column(String(32), nullable=False)
    prompt_hash = Column(String(64), nullable=False)
    fact_version = Column(String(64), nullable=False)
    evaluator_version = Column(String(32), nullable=False)
    source_revision = Column(String(64), nullable=True)

    settings = Column(JSONType, nullable=False, default=dict)
    dataset_hashes = Column(JSONType, nullable=False, default=dict)

    scheduled_count = Column(Integer, nullable=False, default=0)
    evaluated_count = Column(Integer, nullable=False, default=0)
    successful_count = Column(Integer, nullable=False, default=0)
    failed_count = Column(Integer, nullable=False, default=0)

    duration_seconds = Column(Float, nullable=False, default=0.0)
    started_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC))
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Deterministic Evaluation Metrics (NULL for check split runs)
    gold_claims_g = Column(Integer, nullable=True)
    accepted_predictions_p = Column(Integer, nullable=True)
    exact_span_matches_m = Column(Integer, nullable=True)
    correct_label_matches_c = Column(Integer, nullable=True)

    extraction_precision = Column(Float, nullable=True)
    extraction_recall = Column(Float, nullable=True)
    extraction_f1 = Column(Float, nullable=True)
    joint_precision = Column(Float, nullable=True)
    joint_recall = Column(Float, nullable=True)
    joint_f1 = Column(Float, nullable=True)
    gold_claim_coverage = Column(Float, nullable=True)
    matched_label_accuracy = Column(Float, nullable=True)

    error_breakdown = Column(JSONType, nullable=True)
    error_message = Column(Text, nullable=True)

    # Relationships
    answer_results = relationship(
        "AnswerResult",
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="AnswerResult.answer_id",
    )
    claim_results = relationship(
        "ClaimResult",
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="ClaimResult.start_char",
    )


class AnswerResult(Base):
    """Represents the execution outcome of an answer within an evaluation run."""

    __tablename__ = "answer_results"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    run_id = Column(GUID(), ForeignKey("evaluation_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    answer_id = Column(String(32), nullable=False, index=True)
    brand_id = Column(String(64), nullable=False)

    success = Column(Boolean, nullable=False)
    duration_seconds = Column(Float, nullable=False, default=0.0)

    error_type = Column(String(128), nullable=True)
    error_message = Column(Text, nullable=True)
    trace_reference = Column(String(128), nullable=True)
    claims_count = Column(Integer, nullable=False, default=0)

    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC))

    # Relationships
    run = relationship("EvaluationRun", back_populates="answer_results")

    __table_args__ = (
        UniqueConstraint("run_id", "answer_id", name="uq_answer_results_run_answer"),
    )


class ClaimResult(Base):
    """Represents an accepted claim prediction within an answer evaluation outcome."""

    __tablename__ = "claim_results"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    run_id = Column(GUID(), ForeignKey("evaluation_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    answer_id = Column(String(32), nullable=False, index=True)
    claim_id = Column(String(64), nullable=False)

    start_char = Column(Integer, nullable=False)
    end_char = Column(Integer, nullable=False)
    quote = Column(Text, nullable=False)
    label = Column(String(32), nullable=False)  # "supported", "contradicted", "unknown"
    fact_ids = Column(JSONType, nullable=False, default=list)
    reason = Column(Text, nullable=False)

    # Development Ground-Truth Match Information (NULL for check split runs)
    is_exact_match = Column(Boolean, nullable=True)
    is_label_correct = Column(Boolean, nullable=True)
    gold_claim_id = Column(String(64), nullable=True)
    gold_expected_label = Column(String(32), nullable=True)

    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC))

    # Relationships
    run = relationship("EvaluationRun", back_populates="claim_results")

    __table_args__ = (
        UniqueConstraint("run_id", "answer_id", "start_char", "end_char", name="uq_claim_results_run_answer_span"),
        Index("ix_claim_results_run_answer_order", "run_id", "answer_id", "start_char"),
    )
