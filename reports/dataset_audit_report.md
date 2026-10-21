# Dataset Audit Report: KASPARRO FSE Intern Pathway Assessment

**Date of Audit**: October 21, 2026  
**Auditor**: Senior Backend AI Engineer  
**Dataset Directory**: `data/`  
**Audit Status**: COMPLETED & VERIFIED (All tests passing)

---

## 1. Executive Summary

A comprehensive, non-destructive audit was performed on the three source CSV datasets:
1. `data/answers.csv`
2. `data/brand_facts.csv`
3. `data/development_gold.csv`

All records were decoded using UTF-8 using Python's standard `csv` module without arbitrary line-splitting. `ai_answer` strings were preserved verbatim with exact character offsets. No source files were modified, normalized, or resaved.

### Key Audit Findings:
- **Cryptographic Hashes**: SHA-256 digests were computed for all 3 files and recorded in the dataset manifest.
- **Answer Split Anomaly**: In `data/answers.csv`, records A41–A50 are categorized with `split == "check"` rather than `split == "final-check"`. As per specification requirements, the file has been preserved unchanged, and the codebase will support both `'check'` and `'final-check'` aliases transparently.
- **Quote Span Equality**: All 376 claims in `data/development_gold.csv` were validated against `ai_answer[start_char:end_char] == quote`. The exact match rate is **100% (376 / 376)** with **0 mismatches**.
- **Claim ID Convention**: 100% of claims follow the exact `{answer_id}:{start_char}:{end_char}` pattern.
- **Fact ID References**: Multi-fact citations in `evidence_fact_ids` use the pipe delimiter (`|`). All 107 uniquely referenced fact IDs map to valid records in `data/brand_facts.csv`. Zero dangling references were detected.
- **Data Isolation Guarantee**: Strict separation has been architected to ensure that `development_gold.csv` and derived labels can never be accessed or imported by runtime inference agents or fact retrieval tools.

---

## 2. Dataset Manifest & Integrity Verification

| File Name | File Size (Bytes) | Record Count | SHA-256 Digest |
| :--- | :--- | :--- | :--- |
| `data/answers.csv` | 139,330 | 50 | `fa28e813ea54b4e5ea985585f22f084d826241e1182accbc27ace17066125724` |
| `data/brand_facts.csv` | 19,278 | 132 | `2190d40fad2dd167ef2437956c43d595aaf2554ea673a4789b9343cb7e2936f2` |
| `data/development_gold.csv` | 128,276 | 376 | `afe84cc78f304afb6a4aaa505dd7504af80fcbf8bc507806aa5ff0aab6bf4850` |

---

## 3. Detailed Data Inspections

### 3.1 `data/answers.csv`
- **Columns**: `answer_id`, `split`, `brand_id`, `question`, `ai_answer`
- **Total Records**: 50
- **Answer ID Sequence**:
  - `A01` through `A40`: 40 records, `split == "development"`
  - `A41` through `A50`: 10 records, `split == "check"`
- **Brand Distribution**:
  - `cleartrail`: 17 answers
  - `cloudnote`: 17 answers
  - `stonewave`: 16 answers
- **Discrepancy Analysis**:
  The assessment specification mentions: *"development answers are A01–A40, final-check answers are A41–A50"*. In the supplied CSV, the value in the `split` column is `"check"` for records `A41` to `A50` instead of `"final-check"`.
  - **Resolution**: Source CSV is preserved unmodified. In the domain loader and Pydantic models, both `"check"` and `"final-check"` are accepted and treated as the final evaluation split.

### 3.2 `data/brand_facts.csv`
- **Columns**: `fact_id`, `brand_id`, `model_id`, `field`, `value`, `value_type`, `unit`, `condition`, `valid_from`, `valid_to`
- **Total Facts**: 132 (all 132 fact IDs are unique)
- **Brand to Model Hierarchy**:
  - `cleartrail`: `mini`, `plus`
  - `cloudnote`: `air`, `studio`
  - `stonewave`: `go`, `max`
  *(Total: 3 brands, each having exactly 2 models, totaling 6 product models)*
- **Value Types**:
  - `boolean`: 60 records
  - `number`: 42 records
  - `string`: 30 records
- **Temporal Windows**:
  - `valid_from`: `2025-01-01`, `2026-01-01`
  - `valid_to`: `2025-12-31`, `2026-12-31`
- **Distinct Attributes (23 fields)**:
  `adapter_included`, `app_available`, `app_required_bluetooth`, `battery_100_percent`, `battery_50_percent`, `bluetooth_version`, `cable_included`, `charging_port`, `immersion_allowed`, `immersion_depth`, `immersion_duration`, `microphone`, `model_name`, `price_2025`, `price_2026`, `stereo_pairing_scope`, `usb_audio`, `warranty_india`, `warranty_invoice_required`, `water_damage_covered`, `water_rating`, `weight`, `wifi`.

### 3.3 `data/development_gold.csv`
- **Columns**: `answer_id`, `claim_id`, `start_char`, `end_char`, `quote`, `expected_label`, `evidence_fact_ids`, `explanation`, `difficulty_tags`
- **Total Labeled Claims**: 376
- **Answer Coverage**: Exactly 40 development answers (`A01`–`A40`). Answers `A41`–`A50` have zero gold labels (reserved for final evaluation).
- **Label Distribution**:
  - `unknown`: 136 claims (36.17%)
  - `supported`: 120 claims (31.91%)
  - `contradicted`: 120 claims (31.91%)
- **Quote Equality Check**:
  - `ai_answer[start_char:end_char] == quote`
  - Total validated: 376
  - Total matches: 376
  - Mismatches: 0 (100.0% match rate)
- **Evidence Fact Mapping**:
  - Format: Pipe-separated string (e.g., `cleartrail.mini.price_2026|cleartrail.plus.price_2026`)
  - Claims with evidence citations: 332
  - Unique fact IDs referenced: 107
  - Dangling / missing fact references: 0
  - Unreferenced facts in knowledge base: 25 facts (available as distractors / unqueried domain facts)
- **Span Boundaries & Overlaps**:
  - Claims per answer: min 8, max 11, mean 9.4 claims/answer.
  - Overlapping claim spans: 0 detected across all 40 answers. Claims represent non-overlapping contiguous semantic spans.

---

## 4. Architectural Isolation Policy

To prevent test-set contamination and guarantee valid evaluation:
1. **Physical & Logical Boundary**: `development_gold.csv` is exclusively imported in `tests/` and offline evaluation harness scripts.
2. **Runtime Protection**: The application service layer (`app.services.fact_store` and `app.services.data_loader`) provides no endpoint or function for inference agents to load or inspect `development_gold.csv`.
3. **Automated Leakage Test**: `tests/test_isolation_security.py` inspects module imports and verifies that no inference or fact-store code references or exposes gold labels.

