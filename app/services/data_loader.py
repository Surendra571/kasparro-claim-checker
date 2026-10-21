"""
Data loader service for KASPARRO Claim Checker.
Ensures UTF-8 decoding, character preservation, and dataset integrity checks.
"""

import csv
import hashlib
from pathlib import Path

from app.core.config import settings
from app.models.domain import AnswerRecord, BrandFact, GoldClaim


def compute_file_sha256(path: Path) -> str:
    """Calculates SHA-256 checksum of a file without altering it."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_dataset_integrity(
    answers_path: Path | None = None,
    brand_facts_path: Path | None = None,
    gold_path: Path | None = None,
) -> dict[str, bool]:
    """
    Verifies that the dataset files match the recorded SHA-256 manifests.
    """
    answers_p = answers_path or settings.ANSWERS_FILE
    facts_p = brand_facts_path or settings.BRAND_FACTS_FILE
    gold_p = gold_path or settings.DEVELOPMENT_GOLD_FILE

    results = {
        "answers.csv": compute_file_sha256(answers_p) == settings.EXPECTED_ANSWERS_SHA256,
        "brand_facts.csv": compute_file_sha256(facts_p) == settings.EXPECTED_BRAND_FACTS_SHA256,
    }
    if gold_p.exists():
        results["development_gold.csv"] = (
            compute_file_sha256(gold_p) == settings.EXPECTED_DEVELOPMENT_GOLD_SHA256
        )
    return results


def load_answers(path: Path | None = None) -> list[AnswerRecord]:
    """
    Loads answers from CSV preserving raw ai_answer characters and whitespace exactly.
    Never splits lines manually; uses csv.DictReader with utf-8 encoding.
    """
    file_path = path or settings.ANSWERS_FILE
    records: list[AnswerRecord] = []
    with open(file_path, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(
                AnswerRecord(
                    answer_id=row["answer_id"],
                    split=row["split"],
                    brand_id=row["brand_id"],
                    question=row["question"],
                    ai_answer=row["ai_answer"],  # Preserved verbatim
                )
            )
    return records


def load_brand_facts(path: Path | None = None) -> list[BrandFact]:
    """
    Loads brand facts knowledge base from CSV.
    """
    file_path = path or settings.BRAND_FACTS_FILE
    facts: list[BrandFact] = []
    with open(file_path, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            facts.append(
                BrandFact(
                    fact_id=row["fact_id"],
                    brand_id=row["brand_id"],
                    model_id=row["model_id"],
                    field=row["field"],
                    value=row["value"],
                    value_type=row["value_type"],
                    unit=row.get("unit", "") or "",
                    condition=row.get("condition", "") or "",
                    valid_from=row["valid_from"],
                    valid_to=row["valid_to"],
                )
            )
    return facts


def load_development_gold_for_eval_only(path: Path | None = None) -> list[GoldClaim]:
    """
    STRICT EVALUATION ONLY:
    Loads development gold labels for testing and benchmarking.
    MUST NEVER be invoked by inference pipelines or runtime lookup tools.
    """
    file_path = path or settings.DEVELOPMENT_GOLD_FILE
    claims: list[GoldClaim] = []
    with open(file_path, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            claims.append(
                GoldClaim(
                    claim_id=row["claim_id"],
                    answer_id=row["answer_id"],
                    start_char=int(row["start_char"]),
                    end_char=int(row["end_char"]),
                    quote=row["quote"],
                    expected_label=row["expected_label"],
                    evidence_fact_ids=row.get("evidence_fact_ids", ""),
                    explanation=row.get("explanation", "") or "",
                    difficulty_tags=row.get("difficulty_tags", "") or "",
                )
            )
    return claims
