"""
Controlled Scorer Tests for Evaluation Harness.
Implements the 10 required scorer tests:
1. Perfect predictions
2. Missed claims
3. Extra advice incorrectly extracted as a claim
4. Wrong labels on matched spans
5. Provider-error answer
6. Valid empty predictions
7. Duplicate predictions and invalid offsets
8. Missing or duplicated scheduled answer outcomes
9. The required arithmetic example G=10, P=10, M=8, C=7 (verifying 0.80 and 0.70)
10. Verify matched-only label accuracy is never substituted for joint F1
Plus split safety test for final-check partition.
"""

from dataclasses import dataclass, field

import pytest

from app.evaluator import (
    AnswerExecutionOutcome,
    PredictionRecord,
    evaluate_predictions,
)


@dataclass
class MockGoldClaim:
    """Mock gold claim for testing."""

    answer_id: str
    claim_id: str
    start_char: int
    end_char: int
    quote: str
    expected_label: str
    evidence_fact_ids: list[str] = field(default_factory=list)


def test_1_perfect_predictions():
    """Test 1: Perfect predictions yield 1.0 for all extraction and joint metrics."""
    gold = [
        MockGoldClaim("A01", "A01:10:30", 10, 30, "Claim one quote.", "supported"),
        MockGoldClaim("A01", "A01:40:60", 40, 60, "Claim two quote.", "contradicted"),
    ]
    preds = [
        PredictionRecord("A01", "A01:10:30", 10, 30, "Claim one quote.", "supported"),
        PredictionRecord("A01", "A01:40:60", 40, 60, "Claim two quote.", "contradicted"),
    ]
    outcome = AnswerExecutionOutcome("A01", success=True, duration_seconds=1.2, predictions=preds)

    report = evaluate_predictions([outcome], gold_claims=gold, split_name="development")

    m = report.metrics
    assert m.G == 2
    assert m.P == 2
    assert m.M == 2
    assert m.C == 2
    assert m.extraction_precision == 1.0
    assert m.extraction_recall == 1.0
    assert m.extraction_f1 == 1.0
    assert m.joint_precision == 1.0
    assert m.joint_recall == 1.0
    assert m.joint_f1 == 1.0
    assert m.gold_claim_coverage == 1.0

    assert report.missed_gold_claim_ids == []
    assert report.unmatched_prediction_ids == []
    assert report.label_mismatches == []


def test_2_missed_claims():
    """Test 2: Missed claims drop recall and are reported in missed_gold_claim_ids."""
    gold = [
        MockGoldClaim("A01", "A01:10:30", 10, 30, "Claim one.", "supported"),
        MockGoldClaim("A01", "A01:40:60", 40, 60, "Claim two.", "contradicted"),
        MockGoldClaim("A01", "A01:70:90", 70, 90, "Claim three.", "unknown"),
        MockGoldClaim("A01", "A01:100:120", 100, 120, "Claim four.", "supported"),
    ]
    # Predict only two claims
    preds = [
        PredictionRecord("A01", "A01:10:30", 10, 30, "Claim one.", "supported"),
        PredictionRecord("A01", "A01:40:60", 40, 60, "Claim two.", "contradicted"),
    ]
    outcome = AnswerExecutionOutcome("A01", success=True, duration_seconds=1.0, predictions=preds)

    report = evaluate_predictions([outcome], gold_claims=gold, split_name="development")

    m = report.metrics
    assert m.G == 4
    assert m.P == 2
    assert m.M == 2
    assert m.C == 2
    assert m.extraction_precision == 1.0
    assert m.extraction_recall == 0.5
    assert round(m.extraction_f1, 4) == round(2 * 2 / (2 + 4), 4)

    assert set(report.missed_gold_claim_ids) == {"A01:70:90", "A01:100:120"}
    assert report.unmatched_prediction_ids == []


def test_3_extra_advice_extracted_as_claim():
    """Test 3: Extra advice incorrectly extracted drops precision and appears in unmatched_prediction_ids."""
    gold = [
        MockGoldClaim("A01", "A01:10:30", 10, 30, "Factual claim.", "supported"),
    ]
    preds = [
        PredictionRecord("A01", "A01:10:30", 10, 30, "Factual claim.", "supported"),
        PredictionRecord(
            "A01", "A01:50:80", 50, 80, "Keep your preferences in mind.", "supported"
        ),  # Extra advice
    ]
    outcome = AnswerExecutionOutcome("A01", success=True, duration_seconds=1.0, predictions=preds)

    report = evaluate_predictions([outcome], gold_claims=gold, split_name="development")

    m = report.metrics
    assert m.G == 1
    assert m.P == 2
    assert m.M == 1
    assert m.C == 1
    assert m.extraction_precision == 0.5
    assert m.extraction_recall == 1.0
    assert round(m.extraction_f1, 4) == round(2 / 3, 4)

    assert report.missed_gold_claim_ids == []
    assert report.unmatched_prediction_ids == ["A01:50:80"]


def test_4_wrong_labels_on_matched_spans():
    """Test 4: Wrong labels on matched spans maintain extraction metrics but drop joint metrics."""
    gold = [
        MockGoldClaim("A01", "A01:10:30", 10, 30, "Claim one.", "supported"),
        MockGoldClaim("A01", "A01:40:60", 40, 60, "Claim two.", "contradicted"),
        MockGoldClaim("A01", "A01:70:90", 70, 90, "Claim three.", "unknown"),
    ]
    preds = [
        PredictionRecord("A01", "A01:10:30", 10, 30, "Claim one.", "supported"),  # Correct
        PredictionRecord("A01", "A01:40:60", 40, 60, "Claim two.", "supported"),  # Wrong label
        PredictionRecord("A01", "A01:70:90", 70, 90, "Claim three.", "contradicted"),  # Wrong label
    ]
    outcome = AnswerExecutionOutcome("A01", success=True, duration_seconds=1.0, predictions=preds)

    report = evaluate_predictions([outcome], gold_claims=gold, split_name="development")

    m = report.metrics
    assert m.G == 3
    assert m.P == 3
    assert m.M == 3  # All spans extracted
    assert m.C == 1  # Only 1 label correct

    assert m.extraction_f1 == 1.0
    assert round(m.joint_precision, 4) == round(1 / 3, 4)
    assert round(m.joint_recall, 4) == round(1 / 3, 4)
    assert round(m.joint_f1, 4) == round(1 / 3, 4)

    assert len(report.label_mismatches) == 2
    assert report.label_mismatches[0].claim_id == "A01:40:60"
    assert report.label_mismatches[0].gold_label == "contradicted"
    assert report.label_mismatches[0].predicted_label == "supported"


def test_5_provider_error_answer():
    """
    Test 5: A provider-error answer contributes zero accepted predictions,
    all gold claims remain in G and count as missed, and answer error is preserved.
    """
    gold = [
        MockGoldClaim("A01", "A01:10:30", 10, 30, "Quote 1", "supported"),
        MockGoldClaim("A01", "A01:40:60", 40, 60, "Quote 2", "contradicted"),
    ]
    outcome = AnswerExecutionOutcome(
        "A01",
        success=False,
        error_message="Provider timeout after 30s",
        predictions=[
            # Even if partial predictions existed in payload, unaccepted due to failure
            PredictionRecord("A01", "A01:10:30", 10, 30, "Quote 1", "supported")
        ],
    )

    report = evaluate_predictions([outcome], gold_claims=gold, split_name="development")

    m = report.metrics
    assert m.G == 2
    assert m.P == 0  # Zero accepted predictions from error answer
    assert m.M == 0
    assert m.C == 0
    assert m.extraction_precision == 0.0
    assert m.extraction_recall == 0.0
    assert m.extraction_f1 == 0.0
    assert m.joint_f1 == 0.0

    assert set(report.missed_gold_claim_ids) == {"A01:10:30", "A01:40:60"}
    assert len(report.answer_errors) == 1
    assert report.answer_errors[0]["answer_id"] == "A01"
    assert "Provider timeout" in report.answer_errors[0]["error_message"]
    assert report.failed_answers == 1


def test_6_valid_empty_predictions():
    """Test 6: Valid empty predictions list is not an answer error."""
    outcome = AnswerExecutionOutcome("A01", success=True, duration_seconds=0.5, predictions=[])
    report = evaluate_predictions([outcome], gold_claims=[], split_name="development")

    assert report.failed_answers == 0
    assert report.successful_answers == 1
    assert report.answer_errors == []
    assert report.metrics.P == 0
    assert report.metrics.G == 0


def test_7_duplicate_predictions_and_invalid_offsets():
    """Test 7: Duplicate predictions and invalid slice predictions are handled properly."""
    gold = [
        MockGoldClaim("A01", "A01:10:30", 10, 30, "Valid quote", "supported"),
    ]
    preds = [
        PredictionRecord("A01", "A01:10:30", 10, 30, "Valid quote", "supported"),
        # Duplicate prediction for the same span
        PredictionRecord("A01", "A01:10:30", 10, 30, "Valid quote", "supported"),
        # Prediction that failed source slice validation
        PredictionRecord(
            "A01", "A01:40:60", 40, 60, "Corrupt quote", "supported", is_valid_slice=False
        ),
    ]
    outcome = AnswerExecutionOutcome("A01", success=True, predictions=preds)

    report = evaluate_predictions([outcome], gold_claims=gold, split_name="development")

    m = report.metrics
    # Accepted predictions = 2 (only is_valid_slice=True)
    assert m.P == 2
    assert m.G == 1
    # 1:1 match: exactly one duplicate matches gold
    assert m.M == 1
    assert m.C == 1
    # The duplicate prediction is unmatched
    assert len(report.unmatched_prediction_ids) == 1


def test_8_missing_or_duplicated_scheduled_answer_outcomes():
    """Test 8: Duplicate outcomes raise ValueError; missing scheduled outcomes are recorded as errors."""
    outcome1 = AnswerExecutionOutcome("A01", success=True)
    outcome2 = AnswerExecutionOutcome("A01", success=True)  # Duplicate A01

    with pytest.raises(ValueError, match="Duplicate answer outcome IDs"):
        evaluate_predictions([outcome1, outcome2], gold_claims=[], split_name="development")

    # Missing scheduled answer
    report = evaluate_predictions(
        [outcome1],
        gold_claims=[],
        split_name="development",
        scheduled_answer_ids=["A01", "A02"],  # A02 missing
    )
    assert report.failed_answers == 1
    assert len(report.answer_errors) == 1
    assert report.answer_errors[0]["answer_id"] == "A02"


def test_9_required_arithmetic_example():
    """
    Test 9: The required arithmetic example G=10, P=10, M=8, C=7.
    Verify:
    - Extraction precision = 0.80
    - Extraction recall = 0.80
    - Extraction F1 = 0.80
    - Joint precision = 0.70
    - Joint recall = 0.70
    - Joint F1 = 0.70
    - Gold claim coverage = 0.70
    """
    # Create G=10 gold claims: 1..10
    gold = [
        MockGoldClaim(
            "A01", f"A01:{i * 10}:{i * 10 + 5}", i * 10, i * 10 + 5, f"Quote {i}", "supported"
        )
        for i in range(1, 11)
    ]

    # Create P=10 predictions:
    # 7 match span and correct label (1..7)
    # 1 matches span but wrong label (8) -> M = 7 + 1 = 8
    # 2 are false positives / unmatched (spans 500..505, 510..515) -> total P = 10
    preds = (
        [
            # 7 correct label matches
            PredictionRecord(
                "A01", f"A01:{i * 10}:{i * 10 + 5}", i * 10, i * 10 + 5, f"Quote {i}", "supported"
            )
            for i in range(1, 8)
        ]
        + [
            # 1 span match with wrong label
            PredictionRecord("A01", "A01:80:85", 80, 85, "Quote 8", "contradicted")
        ]
        + [
            # 2 unmatched predictions
            PredictionRecord("A01", "A01:500:505", 500, 505, "Extra quote 1", "supported"),
            PredictionRecord("A01", "A01:510:515", 510, 515, "Extra quote 2", "supported"),
        ]
    )

    outcome = AnswerExecutionOutcome("A01", success=True, predictions=preds)
    report = evaluate_predictions([outcome], gold_claims=gold, split_name="development")

    m = report.metrics
    assert m.G == 10
    assert m.P == 10
    assert m.M == 8
    assert m.C == 7

    assert pytest.approx(m.extraction_precision, 1e-6) == 0.80
    assert pytest.approx(m.extraction_recall, 1e-6) == 0.80
    assert pytest.approx(m.extraction_f1, 1e-6) == 0.80

    assert pytest.approx(m.joint_precision, 1e-6) == 0.70
    assert pytest.approx(m.joint_recall, 1e-6) == 0.70
    assert pytest.approx(m.joint_f1, 1e-6) == 0.70
    assert pytest.approx(m.gold_claim_coverage, 1e-6) == 0.70


def test_10_matched_label_accuracy_never_substituted_for_joint_f1():
    """
    Test 10: Verify matched-only label accuracy is never substituted for joint F1.
    In the G=10, P=10, M=8, C=7 scenario:
    - matched_label_accuracy = C / M = 7 / 8 = 0.875
    - joint_f1 = 2 * C / (P + G) = 14 / 20 = 0.70
    Ensure joint_f1 is strictly 0.70, not 0.875.
    """
    gold = [
        MockGoldClaim(
            "A01", f"A01:{i * 10}:{i * 10 + 5}", i * 10, i * 10 + 5, f"Q {i}", "supported"
        )
        for i in range(1, 11)
    ]
    preds = [
        PredictionRecord(
            "A01", f"A01:{i * 10}:{i * 10 + 5}", i * 10, i * 10 + 5, f"Q {i}", "supported"
        )
        for i in range(1, 8)
    ] + [
        PredictionRecord("A01", "A01:80:85", 80, 85, "Q 8", "contradicted"),
        PredictionRecord("A01", "A01:500:505", 500, 505, "Extra 1", "supported"),
        PredictionRecord("A01", "A01:510:515", 510, 515, "Extra 2", "supported"),
    ]

    outcome = AnswerExecutionOutcome("A01", success=True, predictions=preds)
    report = evaluate_predictions([outcome], gold_claims=gold, split_name="development")

    m = report.metrics
    assert pytest.approx(m.joint_f1, 1e-6) == 0.70
    assert pytest.approx(m.matched_label_accuracy, 1e-6) == 0.875
    assert m.joint_f1 != m.matched_label_accuracy


def test_split_safety_final_check():
    """
    Test Split Safety:
    For final-check split, ground-truth-dependent metrics and match records must be null.
    Do not treat unavailable private labels as zero scores or unknown claim labels.
    """
    outcome = AnswerExecutionOutcome(
        "A41",
        success=True,
        duration_seconds=1.5,
        predictions=[PredictionRecord("A41", "A41:0:20", 0, 20, "Final check claim.", "supported")],
    )

    # Even if gold_claims were accidentally passed or None:
    report = evaluate_predictions([outcome], gold_claims=None, split_name="final-check")

    assert report.is_ground_truth_available is False
    assert report.metrics.G is None
    assert report.metrics.M is None
    assert report.metrics.C is None
    assert report.metrics.extraction_f1 is None
    assert report.metrics.joint_f1 is None
    assert report.missed_gold_claim_ids is None
    assert report.unmatched_prediction_ids is None
    assert report.label_mismatches is None
    assert report.label_confusion is None

    # Operational metrics are preserved
    assert report.metrics.P == 1
    assert report.successful_answers == 1
    assert report.per_answer_summary["A41"]["predictions_count"] == 1
