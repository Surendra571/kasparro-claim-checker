"""
Dataset Audit Script for KASPARRO Claim Checker Phase 1.
"""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_audit():
    data_dir = Path("data")
    answers_path = data_dir / "answers.csv"
    brand_facts_path = data_dir / "brand_facts.csv"
    gold_path = data_dir / "development_gold.csv"

    # 1. SHA-256 Hashes
    hashes = {
        "answers.csv": compute_sha256(answers_path),
        "brand_facts.csv": compute_sha256(brand_facts_path),
        "development_gold.csv": compute_sha256(gold_path),
    }

    # 2. Inspect answers.csv
    with open(answers_path, "r", encoding="utf-8") as f:
        answers_reader = csv.DictReader(f)
        answers_headers = answers_reader.fieldnames
        answers_rows = list(answers_reader)

    answers_by_id = {row["answer_id"]: row for row in answers_rows}
    splits = Counter(row["split"] for row in answers_rows)
    brands_in_answers = sorted({row["brand_id"] for row in answers_rows})

    dev_answer_ids = [row["answer_id"] for row in answers_rows if row["split"] == "development"]
    check_answer_ids = [row["answer_id"] for row in answers_rows if row["split"] == "check"]
    other_answer_ids = [
        row["answer_id"] for row in answers_rows if row["split"] not in ("development", "check")
    ]

    expected_dev_ids = [f"A{i:02d}" for i in range(1, 41)]
    expected_final_ids = [f"A{i:02d}" for i in range(41, 51)]

    # 3. Inspect brand_facts.csv
    with open(brand_facts_path, "r", encoding="utf-8") as f:
        facts_reader = csv.DictReader(f)
        facts_headers = facts_reader.fieldnames
        facts_rows = list(facts_reader)

    facts_by_id = {row["fact_id"]: row for row in facts_rows}
    brands_in_facts = sorted({row["brand_id"] for row in facts_rows})
    sorted({row["model_id"] for row in facts_rows})
    fields_in_facts = sorted({row["field"] for row in facts_rows})
    value_types = Counter(row["value_type"] for row in facts_rows)
    valid_from_set = sorted({row["valid_from"] for row in facts_rows})
    valid_to_set = sorted({row["valid_to"] for row in facts_rows})

    # Brand / Model mapping in facts
    brand_models = {}
    for r in facts_rows:
        b = r["brand_id"]
        m = r["model_id"]
        brand_models.setdefault(b, set()).add(m)
    brand_models = {b: sorted(models) for b, models in brand_models.items()}

    # 4. Inspect development_gold.csv
    with open(gold_path, "r", encoding="utf-8") as f:
        gold_reader = csv.DictReader(f)
        gold_headers = gold_reader.fieldnames
        gold_rows = list(gold_reader)

    gold_labels = Counter(row["expected_label"] for row in gold_rows)
    gold_answer_ids = sorted({row["answer_id"] for row in gold_rows})
    gold_claim_ids = [row["claim_id"] for row in gold_rows]
    unique_claim_ids = set(gold_claim_ids)

    # Claim ID pattern check
    claim_id_pattern_mismatches = []
    for r in gold_rows:
        expected_cid = f"{r['answer_id']}:{r['start_char']}:{r['end_char']}"
        if r["claim_id"] != expected_cid:
            claim_id_pattern_mismatches.append((r["claim_id"], expected_cid))

    # Quote equality: ai_answer[start_char:end_char] == quote
    quote_matches = 0
    quote_failures = []
    for idx, r in enumerate(gold_rows):
        aid = r["answer_id"]
        s = int(r["start_char"])
        e = int(r["end_char"])
        q = r["quote"]
        ans = answers_by_id[aid]["ai_answer"]
        actual_slice = ans[s:e]
        if actual_slice == q:
            quote_matches += 1
        else:
            quote_failures.append(
                {
                    "row_index": idx,
                    "claim_id": r["claim_id"],
                    "answer_id": aid,
                    "start_char": s,
                    "end_char": e,
                    "expected_quote": q,
                    "actual_slice": actual_slice,
                }
            )

    # Evidence fact IDs check
    evidence_fact_references = []
    missing_evidence_facts = []
    claims_with_evidence = 0
    for r in gold_rows:
        raw_ev = r["evidence_fact_ids"].strip()
        if not raw_ev:
            continue
        claims_with_evidence += 1
        facts = [p.strip() for p in raw_ev.split("|") if p.strip()]
        for f_id in facts:
            evidence_fact_references.append(f_id)
            if f_id not in facts_by_id:
                missing_evidence_facts.append((r["claim_id"], f_id))

    unique_evidence_facts = sorted(set(evidence_fact_references))
    unreferenced_facts = sorted(set(facts_by_id.keys()) - set(unique_evidence_facts))

    # Claims per answer distribution
    claims_per_answer = Counter(row["answer_id"] for row in gold_rows)

    # Overlapping claim spans
    overlapping_spans_by_answer = {}
    for aid in dev_answer_ids:
        c_for_aid = [r for r in gold_rows if r["answer_id"] == aid]
        c_for_aid.sort(key=lambda x: (int(x["start_char"]), int(x["end_char"])))
        overlaps = []
        for i in range(len(c_for_aid) - 1):
            curr = c_for_aid[i]
            nxt = c_for_aid[i + 1]
            if int(curr["end_char"]) > int(nxt["start_char"]):
                overlaps.append(
                    {
                        "claim_1": curr["claim_id"],
                        "span_1": (int(curr["start_char"]), int(curr["end_char"])),
                        "claim_2": nxt["claim_id"],
                        "span_2": (int(nxt["start_char"]), int(nxt["end_char"])),
                    }
                )
        if overlaps:
            overlapping_spans_by_answer[aid] = overlaps

    report = {
        "hashes": hashes,
        "answers": {
            "path": str(answers_path),
            "columns": answers_headers,
            "row_count": len(answers_rows),
            "splits": dict(splits),
            "brands": brands_in_answers,
            "dev_answer_ids_match_A01_A40": dev_answer_ids == expected_dev_ids,
            "check_answer_ids_match_A41_A50": check_answer_ids == expected_final_ids,
            "other_answer_ids": other_answer_ids,
            "split_inconsistency": "Split column in answers.csv uses 'check' instead of 'final-check' for A41-A50"
            if "check" in splits
            else None,
        },
        "brand_facts": {
            "path": str(brand_facts_path),
            "columns": facts_headers,
            "row_count": len(facts_rows),
            "unique_fact_ids": len(facts_by_id),
            "brands": brands_in_facts,
            "models_by_brand": brand_models,
            "distinct_fields": fields_in_facts,
            "value_types": dict(value_types),
            "valid_from_values": valid_from_set,
            "valid_to_values": valid_to_set,
        },
        "development_gold": {
            "path": str(gold_path),
            "columns": gold_headers,
            "row_count": len(gold_rows),
            "unique_claim_ids": len(unique_claim_ids),
            "claim_id_pattern_mismatches": claim_id_pattern_mismatches,
            "labels_distribution": dict(gold_labels),
            "answers_covered": len(gold_answer_ids),
            "gold_answer_ids": gold_answer_ids,
            "quote_matches": quote_matches,
            "quote_failures_count": len(quote_failures),
            "quote_failures": quote_failures,
            "claims_with_evidence": claims_with_evidence,
            "unique_facts_referenced": len(unique_evidence_facts),
            "missing_evidence_facts": missing_evidence_facts,
            "unreferenced_facts_count": len(unreferenced_facts),
            "claims_per_answer_summary": {
                "min": min(claims_per_answer.values()) if claims_per_answer else 0,
                "max": max(claims_per_answer.values()) if claims_per_answer else 0,
                "avg": (sum(claims_per_answer.values()) / len(claims_per_answer))
                if claims_per_answer
                else 0,
            },
            "overlapping_spans_count": sum(len(v) for v in overlapping_spans_by_answer.values()),
            "overlapping_spans_by_answer": overlapping_spans_by_answer,
        },
    }

    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run_audit()
