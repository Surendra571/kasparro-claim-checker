"""
Comprehensive test suite for Phase 1 Data Audit.
Validates SHA-256 hashes, schema integrity, quote equality, and relationship constraints.
"""

import json
from pathlib import Path

from app.services.data_loader import compute_file_sha256, verify_dataset_integrity


def test_file_existence_and_hashes():
    """Verify all 3 files exist and match their SHA-256 digests in manifest."""
    manifest_path = Path("reports/dataset_manifest.json")
    assert manifest_path.exists(), "Manifest file reports/dataset_manifest.json must exist"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for key, file_info in manifest["files"].items():
        p = Path(file_info["path"])
        assert p.exists(), f"File {p} does not exist"
        computed_hash = compute_file_sha256(p)
        assert computed_hash == file_info["sha256"], (
            f"Hash mismatch for {key}: expected {file_info['sha256']}, got {computed_hash}"
        )


def test_integrity_service():
    """Verify the data_loader verify_dataset_integrity helper returns all True."""
    integrity = verify_dataset_integrity()
    assert all(integrity.values()), f"Integrity check failed: {integrity}"


def test_answers_counts_and_splits(answers):
    """
    Verify:
    - Exactly 50 records in answers.csv
    - Development answers are A01-A40 (40 records)
    - Check / final evaluation answers are A41-A50 (10 records)
    """
    assert len(answers) == 50, f"Expected 50 answers, got {len(answers)}"

    dev_answers = [a for a in answers if a.split == "development"]
    check_answers = [a for a in answers if a.split in ("check", "final-check")]

    assert len(dev_answers) == 40, f"Expected 40 development answers, got {len(dev_answers)}"
    assert len(check_answers) == 10, f"Expected 10 check answers, got {len(check_answers)}"

    expected_dev_ids = [f"A{i:02d}" for i in range(1, 41)]
    expected_check_ids = [f"A{i:02d}" for i in range(41, 51)]

    actual_dev_ids = [a.answer_id for a in dev_answers]
    actual_check_ids = [a.answer_id for a in check_answers]

    assert actual_dev_ids == expected_dev_ids
    assert actual_check_ids == expected_check_ids


def test_brand_facts_schema_and_hierarchy(brand_facts, fact_store):
    """
    Verify:
    - 132 brand facts, all unique fact_ids
    - 3 brands: cleartrail, cloudnote, stonewave
    - 2 models per brand (total 6 models)
    """
    assert len(brand_facts) == 132
    assert len({f.fact_id for f in brand_facts}) == 132

    expected_hierarchy = {
        "cleartrail": ["mini", "plus"],
        "cloudnote": ["air", "studio"],
        "stonewave": ["go", "max"],
    }

    assert fact_store.list_brands() == sorted(expected_hierarchy.keys())

    for brand, expected_models in expected_hierarchy.items():
        actual_models = fact_store.list_models_for_brand(brand)
        assert sorted(actual_models) == sorted(expected_models)


def test_brand_facts_fields_and_types(brand_facts):
    """Verify value types and date formatting in brand facts."""
    valid_types = {"boolean", "number", "string"}
    for f in brand_facts:
        assert f.value_type in valid_types, f"Invalid value_type: {f.value_type} in {f.fact_id}"
        # Validate date formats YYYY-MM-DD
        assert len(f.valid_from) == 10 and f.valid_from[4] == "-" and f.valid_from[7] == "-"
        assert len(f.valid_to) == 10 and f.valid_to[4] == "-" and f.valid_to[7] == "-"
        assert f.valid_from <= f.valid_to


def test_development_gold_counts_and_labels(gold_claims):
    """
    Verify:
    - Total 376 labelled claims
    - Expected labels: supported (120), contradicted (120), unknown (136)
    - All 376 claim IDs follow pattern {answer_id}:{start_char}:{end_char}
    """
    assert len(gold_claims) == 376

    label_counts = {}
    for c in gold_claims:
        label_counts[c.expected_label.value] = label_counts.get(c.expected_label.value, 0) + 1

    assert label_counts == {
        "supported": 120,
        "contradicted": 120,
        "unknown": 136,
    }

    for c in gold_claims:
        expected_cid = f"{c.answer_id}:{c.start_char}:{c.end_char}"
        assert c.claim_id == expected_cid, f"Claim ID format error: {c.claim_id} != {expected_cid}"


def test_gold_quote_equality(answers, gold_claims):
    """
    CRITICAL AUDIT STEP:
    Validates ai_answer[start_char:end_char] == quote for every single gold claim.
    Zero quote mismatches allowed.
    """
    answers_map = {a.answer_id: a.ai_answer for a in answers}

    mismatches = []
    for c in gold_claims:
        assert c.answer_id in answers_map, f"Answer ID {c.answer_id} not in answers"
        full_text = answers_map[c.answer_id]
        extracted_slice = full_text[c.start_char : c.end_char]
        if extracted_slice != c.quote:
            mismatches.append(
                {
                    "claim_id": c.claim_id,
                    "expected_quote": c.quote,
                    "actual_slice": extracted_slice,
                    "start": c.start_char,
                    "end": c.end_char,
                }
            )

    assert len(mismatches) == 0, f"Found {len(mismatches)} quote mismatches: {mismatches[:3]}"


def test_gold_evidence_facts_validity(brand_facts, gold_claims):
    """Verify all evidence fact references in gold claims exist in brand_facts."""
    fact_ids = {f.fact_id for f in brand_facts}

    missing_references = []
    for c in gold_claims:
        for f_id in c.evidence_fact_ids:
            if f_id not in fact_ids:
                missing_references.append((c.claim_id, f_id))

    assert len(missing_references) == 0, f"Missing evidence fact references: {missing_references}"


def test_no_overlapping_claims_in_gold(gold_claims):
    """Verify that within any answer, gold claim spans do not overlap."""
    by_answer = {}
    for c in gold_claims:
        by_answer.setdefault(c.answer_id, []).append(c)

    overlaps = []
    for aid, claims in by_answer.items():
        claims_sorted = sorted(claims, key=lambda x: (x.start_char, x.end_char))
        for i in range(len(claims_sorted) - 1):
            c1 = claims_sorted[i]
            c2 = claims_sorted[i + 1]
            if c1.end_char > c2.start_char:
                overlaps.append((aid, c1.claim_id, c2.claim_id))

    assert len(overlaps) == 0, f"Found overlapping claims: {overlaps}"
