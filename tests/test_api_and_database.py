"""
Integration and API tests for FastAPI and PostgreSQL persistence layer.
Verifies /classify, /evaluation-runs, and /evaluation-runs/{id} endpoints,
database constraints (FK, unique constraints), split safety, and error mappings.
"""
import uuid
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import IntegrityError

import app.services.evaluation_service as eval_svc_mod
from app.api.app import app
from app.core.database import SessionLocal
from app.models.database import AnswerResult, ClaimResult, EvaluationRun
from app.providers import ProviderError, ProviderTimeoutError

client = TestClient(app)


def test_classify_endpoint_success():
    """Verify /classify returns HTTP 200 and validated prediction schema."""
    payload = {
        "answer_id": "A01",
        "brand_id": "cleartrail",
        "question": "What is the warranty period for ClearTrail Mini?",
        "ai_answer": "For an authorized India purchase with the original invoice and a manufacturing defect, ClearTrail Mini receives a standard manufacturer warranty lasting 18 months from the purchase date.",
        "agent_version": "v1",
    }
    response = client.post("/classify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["answer_id"] == "A01"
    assert len(data["claims"]) == 1
    c = data["claims"][0]
    assert c["label"] == "contradicted"
    assert "cleartrail.mini.warranty_india" in c["fact_ids"]
    assert c["start_char"] == 0
    assert c["end_char"] == len(payload["ai_answer"])


def test_classify_endpoint_unknown_brand():
    """Verify unknown primary brand returns HTTP 404 with sanitized error message."""
    payload = {
        "answer_id": "A01",
        "brand_id": "unknown_audio_brand",
        "question": "Question?",
        "ai_answer": "Some speaker specification.",
        "agent_version": "v1",
    }
    response = client.post("/classify", json=payload)
    assert response.status_code == 404
    assert "Unknown brand 'unknown_audio_brand'" in response.json()["detail"]


def test_classify_endpoint_forbidden_fields():
    """Verify forbidden gold / claim list / evaluation fields trigger HTTP 422."""
    forbidden_payloads = [
        {
            "answer_id": "A01",
            "brand_id": "cleartrail",
            "question": "Q?",
            "ai_answer": "Answer.",
            "gold_claims": [{"quote": "Q"}],  # Forbidden
        },
        {
            "answer_id": "A01",
            "brand_id": "cleartrail",
            "question": "Q?",
            "ai_answer": "Answer.",
            "expected_label": "supported",  # Forbidden
        },
        {
            "answer_id": "A01",
            "brand_id": "cleartrail",
            "question": "Q?",
            "ai_answer": "Answer.",
            "score": 0.95,  # Forbidden
        },
        {
            "answer_id": "A01",
            "brand_id": "cleartrail",
            "question": "Q?",
            "ai_answer": "Answer.",
            "extra_unauthorized_key": "injected_value",  # Extra forbid
        },
    ]

    for payload in forbidden_payloads:
        response = client.post("/classify", json=payload)
        assert response.status_code == 422, f"Expected 422 for payload: {payload}"


def test_classify_endpoint_empty_claims():
    """Verify an answer containing only non-factual advice returns HTTP 200 with empty claims list."""
    payload = {
        "answer_id": "A99",
        "brand_id": "cleartrail",
        "question": "Tips for packing?",
        "ai_answer": "Start with the exact model name before deciding what belongs in a weekend bag. Keep your preferred listening conditions in mind.",
        "agent_version": "v1",
    }
    response = client.post("/classify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["answer_id"] == "A99"
    assert data["claims"] == []


def test_classify_endpoint_error_mappings(monkeypatch):
    """Verify provider timeout maps to 504 and provider error maps to 502 with sanitized messages."""
    class TimeoutProvider:
        @property
        def provider_name(self): return "mock"
        @property
        def model_name(self): return "mock"
        def generate(self, *args, **kwargs):
            raise ProviderTimeoutError("Connection timed out after 30 seconds")

    class FailingProvider:
        @property
        def provider_name(self): return "mock"
        @property
        def model_name(self): return "mock"
        def generate(self, *args, **kwargs):
            raise ProviderError("API rate limit exceeded or invalid JSON token: secret_api_key_12345")

    payload = {
        "answer_id": "A01",
        "brand_id": "cleartrail",
        "question": "Q",
        "ai_answer": "Valid text here.",
        "agent_version": "v1",
    }

    # Test 504 Gateway Timeout
    monkeypatch.setattr(eval_svc_mod, "resolve_service_provider", lambda v: TimeoutProvider())
    res_timeout = client.post("/classify", json=payload)
    assert res_timeout.status_code == 504
    assert "Gateway timeout" in res_timeout.json()["detail"]
    assert "secret_api_key" not in res_timeout.text

    # Test 502 Bad Gateway
    monkeypatch.setattr(eval_svc_mod, "resolve_service_provider", lambda v: FailingProvider())
    res_failing = client.post("/classify", json=payload)
    assert res_failing.status_code == 502
    assert "Bad gateway" in res_failing.json()["detail"]
    assert "secret_api_key_12345" not in res_failing.text  # Sanitized


def test_evaluation_runs_create_and_postgres_persistence():
    """
    Executes /evaluation-runs for development split and verifies full persistence in PostgreSQL:
    - evaluation_runs record created with metrics
    - answer_results records created
    - claim_results records created
    - foreign keys and unique constraints hold
    """
    response = client.post(
        "/evaluation-runs",
        json={"agent_version": "v1", "split": "development"},
    )
    assert response.status_code == 201
    data = response.json()
    run_id = uuid.UUID(data["id"])
    assert data["status"] == "completed"
    assert data["scheduled_count"] == 40
    assert data["evaluated_count"] == 40
    assert data["metrics"] is not None
    assert data["metrics"]["G"] == 376
    assert data["metrics"]["joint_f1"] is not None

    # Query PostgreSQL directly
    db = SessionLocal()
    try:
        run_row = db.query(EvaluationRun).filter(EvaluationRun.id == run_id).first()
        assert run_row is not None
        assert run_row.split == "development"
        assert run_row.agent_version == "v1"
        assert run_row.gold_claims_g == 376
        assert run_row.joint_f1 > 0.90

        # Verify answer results
        ans_rows = db.query(AnswerResult).filter(AnswerResult.run_id == run_id).all()
        assert len(ans_rows) == 40

        # Verify claim results
        claim_rows = db.query(ClaimResult).filter(ClaimResult.run_id == run_id).all()
        assert len(claim_rows) > 300
        # In development split, match fields must be populated
        assert any(c.is_exact_match is True for c in claim_rows)
        assert any(c.gold_claim_id is not None for c in claim_rows)

    finally:
        db.close()


def test_evaluation_runs_check_split_safety():
    """
    Executes /evaluation-runs for check split ('check') and verifies:
    - Quality metrics on run remain NULL
    - Ground-truth match info on claim_results remain NULL
    - Development gold is never exposed
    """
    response = client.post(
        "/evaluation-runs",
        json={"agent_version": "v1", "split": "check"},
    )
    assert response.status_code == 201
    data = response.json()
    run_id = uuid.UUID(data["id"])
    assert data["status"] == "completed"
    assert data["scheduled_count"] == 10
    assert data["evaluated_count"] == 10
    # Crucial split safety: metrics must be None
    assert data["metrics"] is None

    # Query PostgreSQL directly to verify NULLs
    db = SessionLocal()
    try:
        run_row = db.query(EvaluationRun).filter(EvaluationRun.id == run_id).first()
        assert run_row is not None
        assert run_row.split == "check"
        assert run_row.gold_claims_g is None
        assert run_row.joint_f1 is None
        assert run_row.extraction_f1 is None
        assert run_row.gold_claim_coverage is None

        # Verify claim results match fields are all NULL
        claim_rows = db.query(ClaimResult).filter(ClaimResult.run_id == run_id).all()
        assert len(claim_rows) > 0
        for cr in claim_rows:
            assert cr.is_exact_match is None
            assert cr.is_label_correct is None
            assert cr.gold_claim_id is None
            assert cr.gold_expected_label is None

    finally:
        db.close()


def test_get_evaluation_run_joined_retrieval_and_stable_ordering():
    """
    Verifies GET /evaluation-runs/{id} returns joined answers and predictions
    strictly ordered by answer_id ASC and start_char ASC.
    """
    # Create a quick run on check split (10 answers)
    create_res = client.post(
        "/evaluation-runs",
        json={"agent_version": "v1", "split": "check"},
    )
    run_id = create_res.json()["id"]

    get_res = client.get(f"/evaluation-runs/{run_id}")
    assert get_res.status_code == 200
    detail = get_res.json()

    assert detail["id"] == run_id
    assert len(detail["answer_results"]) == 10

    # Verify answers order is strictly ascending (A41, A42, ..., A50)
    answer_ids = [a["answer_id"] for a in detail["answer_results"]]
    assert answer_ids == sorted(answer_ids)

    # Verify claim order within each answer is strictly ascending by start_char
    for a in detail["answer_results"]:
        start_offsets = [c["start_char"] for c in a["claims"]]
        assert start_offsets == sorted(start_offsets)


def test_get_evaluation_run_errors():
    """Verify GET /evaluation-runs/{id} returns 404 for missing run and 422 for invalid UUID."""
    # 404 for random non-existent UUID
    missing_id = str(uuid.uuid4())
    res_404 = client.get(f"/evaluation-runs/{missing_id}")
    assert res_404.status_code == 404
    assert f"Evaluation run '{missing_id}' not found" in res_404.json()["detail"]

    # 422 for malformed UUID string
    res_422 = client.get("/evaluation-runs/not-a-valid-uuid-123")
    assert res_422.status_code == 422


def test_postgres_database_constraints():
    """
    Directly tests PostgreSQL persistence constraints:
    1. UniqueConstraint on (run_id, answer_id)
    2. UniqueConstraint on (run_id, answer_id, start_char, end_char)
    3. Foreign key cascade delete
    """
    db = SessionLocal()
    run_id = uuid.uuid4()
    now = datetime.now(UTC)

    try:
        run = EvaluationRun(
            id=run_id,
            split="development",
            agent_version="v1",
            status="completed",
            provider="test-provider",
            model="test-model",
            prompt_version="v1",
            prompt_hash="abc",
            fact_version="v4",
            evaluator_version="1.0",
            started_at=now,
        )
        db.add(run)

        ans1 = AnswerResult(
            id=uuid.uuid4(),
            run_id=run_id,
            answer_id="A01",
            brand_id="cleartrail",
            success=True,
            duration_seconds=0.1,
            created_at=now,
        )
        db.add(ans1)
        db.commit()

        # 1. Test duplicate (run_id, answer_id) raises IntegrityError
        ans_dup = AnswerResult(
            id=uuid.uuid4(),
            run_id=run_id,
            answer_id="A01",  # Duplicate answer_id for same run_id
            brand_id="cleartrail",
            success=True,
            duration_seconds=0.1,
            created_at=now,
        )
        db.add(ans_dup)
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()

        # 2. Test duplicate claim span (run_id, answer_id, start_char, end_char)
        claim1 = ClaimResult(
            id=uuid.uuid4(),
            run_id=run_id,
            answer_id="A01",
            claim_id="A01:10:20",
            start_char=10,
            end_char=20,
            quote="Quote 1",
            label="supported",
            fact_ids=[],
            reason="Reason",
            created_at=now,
        )
        db.add(claim1)
        db.commit()

        claim_dup = ClaimResult(
            id=uuid.uuid4(),
            run_id=run_id,
            answer_id="A01",
            claim_id="A01:10:20",
            start_char=10,  # Duplicate span
            end_char=20,
            quote="Quote 1 Duplicate",
            label="supported",
            fact_ids=[],
            reason="Reason",
            created_at=now,
        )
        db.add(claim_dup)
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()

        # 3. Test cascade delete of parent run deletes child records
        db.query(EvaluationRun).filter(EvaluationRun.id == run_id).delete()
        db.commit()

        ans_count = db.query(AnswerResult).filter(AnswerResult.run_id == run_id).count()
        claim_count = db.query(ClaimResult).filter(ClaimResult.run_id == run_id).count()
        assert ans_count == 0, "Expected AnswerResult to be cascade deleted"
        assert claim_count == 0, "Expected ClaimResult to be cascade deleted"

    finally:
        # Cleanup in case of leftover
        try:
            db.query(EvaluationRun).filter(EvaluationRun.id == run_id).delete()
            db.commit()
        except Exception:
            db.rollback()
        db.close()
