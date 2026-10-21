"""
Deterministic Evaluation Harness for KASPARRO Claim-Checker.
Implements exact 1:1 span matching (answer_id, start_char, end_char),
extraction metrics, joint metrics, detailed error analysis, and strict split safety.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PredictionRecord:
    """An individual claim prediction from the agent."""

    answer_id: str
    claim_id: str
    start_char: int
    end_char: int
    quote: str
    label: str
    fact_ids: list[str] = field(default_factory=list)
    reason: str = ""
    is_valid_slice: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnswerExecutionOutcome:
    """The runtime outcome of running the agent on an answer."""

    answer_id: str
    success: bool
    duration_seconds: float = 0.0
    error_message: str | None = None
    predictions: list[PredictionRecord] = field(default_factory=list)
    rejected_claims: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer_id": self.answer_id,
            "success": self.success,
            "duration_seconds": self.duration_seconds,
            "error_message": self.error_message,
            "predictions": [p.to_dict() for p in self.predictions],
            "rejected_claims": self.rejected_claims,
        }


@dataclass
class EvaluationMetrics:
    """Quantitative performance metrics."""

    G: int | None = None  # Total gold claims
    P: int | None = None  # Total accepted predictions
    M: int | None = None  # Exact span matches
    C: int | None = None  # Exact span matches with correct label

    # Extraction metrics (span only)
    extraction_precision: float | None = None  # M / P
    extraction_recall: float | None = None  # M / G
    extraction_f1: float | None = None  # 2 * M / (P + G)

    # Joint metrics (span + label)
    joint_precision: float | None = None  # C / P
    joint_recall: float | None = None  # C / G
    joint_f1: float | None = None  # 2 * C / (P + G)
    gold_claim_coverage: float | None = None  # C / G

    # Matched-only diagnostic (diagnostic only; NEVER substituted for joint_f1)
    matched_label_accuracy: float | None = None  # C / M

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LabelMismatch:
    """Details for a prediction that matched gold span but had incorrect label."""

    answer_id: str
    claim_id: str
    start_char: int
    end_char: int
    quote: str
    gold_label: str
    predicted_label: str
    predicted_fact_ids: list[str]
    gold_evidence_fact_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LabelConfusionSummary:
    """Distribution of labels and confusion matrix."""

    gold_counts: dict[str, int] = field(default_factory=dict)
    prediction_counts: dict[str, int] = field(default_factory=dict)
    confusion_matrix: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report."""

    split_name: str
    is_ground_truth_available: bool
    total_scheduled_answers: int
    successful_answers: int
    failed_answers: int
    metrics: EvaluationMetrics
    per_answer_summary: dict[str, dict[str, Any]] = field(default_factory=dict)
    answer_errors: list[dict[str, Any]] = field(default_factory=list)
    rejected_claims: list[dict[str, Any]] = field(default_factory=list)

    # Detailed error analysis (null on final-check split)
    missed_gold_claim_ids: list[str] | None = None
    unmatched_prediction_ids: list[str] | None = None
    label_mismatches: list[LabelMismatch] | None = None
    label_confusion: LabelConfusionSummary | None = None

    def to_dict(self) -> dict[str, Any]:
        res = {
            "split_name": self.split_name,
            "is_ground_truth_available": self.is_ground_truth_available,
            "total_scheduled_answers": self.total_scheduled_answers,
            "successful_answers": self.successful_answers,
            "failed_answers": self.failed_answers,
            "metrics": self.metrics.to_dict(),
            "per_answer_summary": self.per_answer_summary,
            "answer_errors": self.answer_errors,
            "rejected_claims": self.rejected_claims,
            "missed_gold_claim_ids": self.missed_gold_claim_ids,
            "unmatched_prediction_ids": self.unmatched_prediction_ids,
            "label_mismatches": (
                [m.to_dict() for m in self.label_mismatches]
                if self.label_mismatches is not None
                else None
            ),
            "label_confusion": (
                self.label_confusion.to_dict() if self.label_confusion is not None else None
            ),
        }
        return res


def compute_metric_ratio(numerator: int, denominator: int) -> float:
    """
    Computes ratio numerator / denominator.
    Returns 0.0 when denominator is 0 on a nonempty evaluation.
    """
    if denominator == 0:
        return 0.0
    return numerator / denominator


def evaluate_predictions(
    outcomes: list[AnswerExecutionOutcome],
    gold_claims: list[Any] | None = None,
    split_name: str = "development",
    scheduled_answer_ids: list[str] | None = None,
) -> EvaluationReport:
    """
    Evaluates predictions against gold claims using exact 1:1 span matching.

    Rules:
    - Span match: (answer_id, start_char, end_char) identical, quote passed exact slice validation.
    - Each gold span matches at most one prediction (1:1 bijective match).
    - No fuzzy matching, no LLM judge, no edited spans.
    - Provider-error answers contribute zero accepted predictions; gold claims count in G and as missed.
    - Valid empty predictions are not answer errors.
    - On final-check split: ground-truth-dependent metrics and match records are null (None).
    """
    # Verify scheduled answer IDs consistency
    scheduled_ids = scheduled_answer_ids or [o.answer_id for o in outcomes]
    seen_ids: set[str] = set()
    duplicate_ids: list[str] = []
    for o in outcomes:
        if o.answer_id in seen_ids:
            duplicate_ids.append(o.answer_id)
        seen_ids.add(o.answer_id)

    if duplicate_ids:
        raise ValueError(f"Duplicate answer outcome IDs detected: {duplicate_ids}")

    outcome_map = {o.answer_id: o for o in outcomes}

    # Per-answer summary
    per_answer_summary: dict[str, dict[str, Any]] = {}
    answer_errors: list[dict[str, Any]] = []
    all_rejected_claims: list[dict[str, Any]] = []
    accepted_predictions: list[PredictionRecord] = []

    successful_count = 0
    failed_count = 0

    for aid in scheduled_ids:
        o = outcome_map.get(aid)
        if o is None:
            # Scheduled answer was missing from outcomes
            failed_count += 1
            answer_errors.append(
                {
                    "answer_id": aid,
                    "error_message": "Scheduled answer missing from execution outcomes.",
                }
            )
            per_answer_summary[aid] = {
                "status": "error",
                "duration_seconds": 0.0,
                "predictions_count": 0,
                "error": "Scheduled answer missing from execution outcomes.",
            }
            continue

        if o.success:
            successful_count += 1
            # Filter valid predictions that passed source slice validation
            valid_preds = [p for p in o.predictions if p.is_valid_slice]
            accepted_predictions.extend(valid_preds)

            per_answer_summary[aid] = {
                "status": "success",
                "duration_seconds": o.duration_seconds,
                "predictions_count": len(valid_preds),
                "error": None,
            }
        else:
            failed_count += 1
            answer_errors.append(
                {
                    "answer_id": aid,
                    "error_message": o.error_message or "Unknown answer error",
                }
            )
            per_answer_summary[aid] = {
                "status": "error",
                "duration_seconds": o.duration_seconds,
                "predictions_count": 0,
                "error": o.error_message,
            }

        # Collect rejected claims if any
        if o.rejected_claims:
            all_rejected_claims.extend(o.rejected_claims)

    is_final_check = split_name.strip().lower() in ("final-check", "final_check", "check")
    has_ground_truth = (not is_final_check) and (gold_claims is not None)

    # Empty scheduled split check
    if len(scheduled_ids) == 0:
        return EvaluationReport(
            split_name=split_name,
            is_ground_truth_available=has_ground_truth,
            total_scheduled_answers=0,
            successful_answers=0,
            failed_answers=0,
            metrics=EvaluationMetrics(),  # all None
            per_answer_summary={},
            answer_errors=[],
            rejected_claims=[],
            missed_gold_claim_ids=None,
            unmatched_prediction_ids=None,
            label_mismatches=None,
            label_confusion=None,
        )

    # If final-check or ground truth unavailable: return null metrics
    if not has_ground_truth or gold_claims is None:
        return EvaluationReport(
            split_name=split_name,
            is_ground_truth_available=False,
            total_scheduled_answers=len(scheduled_ids),
            successful_answers=successful_count,
            failed_answers=failed_count,
            metrics=EvaluationMetrics(P=len(accepted_predictions)),
            per_answer_summary=per_answer_summary,
            answer_errors=answer_errors,
            rejected_claims=all_rejected_claims,
            missed_gold_claim_ids=None,
            unmatched_prediction_ids=None,
            label_mismatches=None,
            label_confusion=None,
        )

    # --- Ground Truth Evaluation ---
    # Filter gold claims to only scheduled answers
    scheduled_set = set(scheduled_ids)
    active_gold = [g for g in gold_claims if getattr(g, "answer_id", None) in scheduled_set]

    G = len(active_gold)
    P = len(accepted_predictions)

    # Index gold claims by exact span key: (answer_id, start_char, end_char)
    # Check that gold spans are unique per answer
    gold_by_span: dict[tuple[str, int, int], Any] = {}
    for g in active_gold:
        key = (g.answer_id, int(g.start_char), int(g.end_char))
        gold_by_span[key] = g

    matched_gold_keys: set[tuple[str, int, int]] = set()
    matched_prediction_indices: set[int] = set()
    correct_label_keys: set[tuple[str, int, int]] = set()
    label_mismatches: list[LabelMismatch] = []

    # Confusion matrix tracking
    gold_counts: dict[str, int] = {}
    pred_counts: dict[str, int] = {}
    confusion_matrix: dict[str, dict[str, int]] = {}

    for g in active_gold:
        g_lbl = getattr(g.expected_label, "value", str(g.expected_label)).strip().lower()
        gold_counts[g_lbl] = gold_counts.get(g_lbl, 0) + 1

    # Exact 1:1 matching
    for pred_idx, p in enumerate(accepted_predictions):
        p_lbl = str(p.label).strip().lower()
        pred_counts[p_lbl] = pred_counts.get(p_lbl, 0) + 1

        span_key = (p.answer_id, p.start_char, p.end_char)
        if span_key in gold_by_span and span_key not in matched_gold_keys:
            # 1:1 match established
            matched_gold_keys.add(span_key)
            matched_prediction_indices.add(pred_idx)
            gold_item = gold_by_span[span_key]
            gold_lbl = (
                getattr(gold_item.expected_label, "value", str(gold_item.expected_label))
                .strip()
                .lower()
            )

            # Record confusion
            confusion_matrix.setdefault(gold_lbl, {})
            confusion_matrix[gold_lbl][p_lbl] = confusion_matrix[gold_lbl].get(p_lbl, 0) + 1

            if p_lbl == gold_lbl:
                correct_label_keys.add(span_key)
            else:
                raw_ev = getattr(gold_item, "evidence_fact_ids", [])
                gold_ev = (
                    raw_ev
                    if isinstance(raw_ev, list)
                    else [x.strip() for x in str(raw_ev).split("|") if x.strip()]
                )
                label_mismatches.append(
                    LabelMismatch(
                        answer_id=p.answer_id,
                        claim_id=p.claim_id,
                        start_char=p.start_char,
                        end_char=p.end_char,
                        quote=p.quote,
                        gold_label=gold_lbl,
                        predicted_label=p_lbl,
                        predicted_fact_ids=p.fact_ids,
                        gold_evidence_fact_ids=gold_ev,
                    )
                )

    M = len(matched_gold_keys)
    C = len(correct_label_keys)

    # Compute metrics
    ext_precision = compute_metric_ratio(M, P)
    ext_recall = compute_metric_ratio(M, G)
    ext_f1 = compute_metric_ratio(2 * M, P + G)

    joint_precision = compute_metric_ratio(C, P)
    joint_recall = compute_metric_ratio(C, G)
    joint_f1 = compute_metric_ratio(2 * C, P + G)
    gold_coverage = compute_metric_ratio(C, G)
    matched_acc = compute_metric_ratio(C, M)

    metrics = EvaluationMetrics(
        G=G,
        P=P,
        M=M,
        C=C,
        extraction_precision=ext_precision,
        extraction_recall=ext_recall,
        extraction_f1=ext_f1,
        joint_precision=joint_precision,
        joint_recall=joint_recall,
        joint_f1=joint_f1,
        gold_claim_coverage=gold_coverage,
        matched_label_accuracy=matched_acc,
    )

    # Missed gold claim IDs
    missed_gold_claim_ids = [
        g.claim_id
        for g in active_gold
        if (g.answer_id, int(g.start_char), int(g.end_char)) not in matched_gold_keys
    ]

    # Unmatched prediction IDs
    unmatched_prediction_ids = [
        p.claim_id
        for pred_idx, p in enumerate(accepted_predictions)
        if pred_idx not in matched_prediction_indices
    ]

    label_confusion = LabelConfusionSummary(
        gold_counts=gold_counts,
        prediction_counts=pred_counts,
        confusion_matrix=confusion_matrix,
    )

    return EvaluationReport(
        split_name=split_name,
        is_ground_truth_available=True,
        total_scheduled_answers=len(scheduled_ids),
        successful_answers=successful_count,
        failed_answers=failed_count,
        metrics=metrics,
        per_answer_summary=per_answer_summary,
        answer_errors=answer_errors,
        rejected_claims=all_rejected_claims,
        missed_gold_claim_ids=missed_gold_claim_ids,
        unmatched_prediction_ids=unmatched_prediction_ids,
        label_mismatches=label_mismatches,
        label_confusion=label_confusion,
    )


def save_evaluation_report(report: EvaluationReport, output_path: Path) -> None:
    """Saves evaluation report to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
