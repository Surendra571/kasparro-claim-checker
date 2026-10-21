# KASPARRO AI Claim-Extraction and Fact-Verification Service

**Official Repository**: [https://github.com/Surendra571/kasparro-claim-checker](https://github.com/Surendra571/kasparro-claim-checker)  
**Role**: KASPARRO FSE Intern Pathway Assessment  
**Python Compatibility**: Python >= 3.11 (tested on Python 3.12.10)  
**Database**: PostgreSQL 17 / 16 (local or Docker Compose)  
**Evaluator Version**: `1.0.0` | **Fact Version**: `speaker-catalogue-v4`  

---

## 1. Executive Summary & Assessment Deliverables

This repository contains the complete, production-grade AI claim-extraction and fact-verification backend built for the **KASPARRO FSE Intern Pathway Assessment**. It parses AI-generated answers, extracts complete factual assertions with exact character quote boundaries, verifies claims against an authoritative brand catalogue via live tool calling, and persists evaluation outcomes in PostgreSQL.

### Key Assessment Deliverables:
- **Git Repository**: [https://github.com/Surendra571/kasparro-claim-checker](https://github.com/Surendra571/kasparro-claim-checker) (Collaborator access granted for hiring team).
- **Dataset Provenance**: All 3 source CSVs preserved with 100% exact hashes (`fa28e813...`, `2190d40f...`, `afe84cc7...`).
- **Four Completed Evaluation Runs**:
  1. `v1` Baseline Development (40 answers `A01`–`A40`): [`reports/v1_development.json`](reports/v1_development.json) & [`reports/v1_development.md`](reports/v1_development.md)
  2. `v2` Targeted Development (40 answers `A01`–`A40`): [`reports/v2_development.json`](reports/v2_development.json) & [`reports/v2_development.md`](reports/v2_development.md)
  3. `v1` Baseline Final-Check (10 answers `A41`–`A50`, blind): [`reports/v1_check.json`](reports/v1_check.json) & [`reports/v1_check.md`](reports/v1_check.md)
  4. `v2` Targeted Final-Check (10 answers `A41`–`A50`, blind): [`reports/v2_check.json`](reports/v2_check.json) & [`reports/v2_check.md`](reports/v2_check.md)
- **Side-by-Side Comparison**: [`reports/development_comparison.md`](reports/development_comparison.md) documenting a **+1.60% gain in Joint F1 (0.9493 to 0.9653)** with 0 regressions.
- **Actual Terminal Test Results**: [`reports/test_results.txt`](reports/test_results.txt) verifying all **90 passing tests** (100% pass rate).
- **Final Audit Report**: [`reports/final_review.md`](reports/final_review.md) containing the pass/fail compliance matrix.

---

## 2. Dataset Integrity & SHA-256 Hashes

The three required CSV files in `data/` were audited and preserved with zero line modifications:

| File Name | SHA-256 Checksum | Record Count | Description |
| :--- | :--- | :---: | :--- |
| `data/answers.csv` | `fa28e813ea54b4e5ea985585f22f084d826241e1182accbc27ace17066125724` | 50 rows | 40 development (`A01`–`A40`), 10 check (`A41`–`A50`) |
| `data/brand_facts.csv` | `2190d40fad2dd167ef2437956c43d595aaf2554ea673a4789b9343cb7e2936f2` | 132 rows | Authoritative catalogue for 3 brands and 6 models |
| `data/development_gold.csv` | `afe84cc78f304afb6a4aaa505dd7504af80fcbf8bc507806aa5ff0aab6bf4850` | 376 rows | Ground truth claims (100% exact quote slice verified) |

All 376 development gold quotes strictly satisfy `ai_answer[start_char:end_char] == quote` with 0 mismatches.

---

## 3. Four-Run Protocol Results & Comparison

### 3.1 Side-by-Side Development Quality Metrics (v1 vs. v2)

| Metric | Baseline (`prompts/v1.txt`) | Targeted (`prompts/v2.txt`) | Delta (v2 - v1) | Evaluation Definition |
| :--- | :---: | :---: | :---: | :--- |
| **Gold Claims ($G$)** | 376 | 376 | 0 | Ground-truth benchmark claims across A01–A40 |
| **Accepted Predictions ($P$)** | 374 | 374 | 0 | Extracted predictions passing slice validation |
| **Exact Span Matches ($M$)** | 370 | 370 | 0 | Exact `(answer_id, start_char, end_char)` matches |
| **Correct Label Matches ($C$)** | **356** | **362** | **+6** | Span matches with matching ground-truth label |
| **Extraction Precision** | 0.9893 | 0.9893 | +0.0000 | $M / P$ |
| **Extraction Recall** | 0.9840 | 0.9840 | +0.0000 | $M / G$ |
| **Extraction F1** | 0.9867 | 0.9867 | +0.0000 | $2 \cdot M / (P + G)$ |
| **Joint Precision** | **0.9519** | **0.9679** | **+1.60%** | $C / P$ |
| **Joint Recall** | **0.9468** | **0.9628** | **+1.60%** | $C / G$ |
| **Joint F1** | **0.9493** | **0.9653** | **+1.60%** | $2 \cdot C / (P + G)$ |
| **Gold Claim Coverage** | **0.9468** | **0.9628** | **+1.60%** | $C / G$ |
| **Matched Label Accuracy** | 0.9622 | 0.9784 | +1.62% | $C / M$ (Diagnostic only, not substituted for Joint F1) |

### 3.2 Final-Check Blind Evaluation Invariants (`A41`–`A50`)
In strict adherence to the assessment requirements:
- **Zero Gold Leakage**: `development_gold.csv` was never loaded during check-split inference or evaluation.
- **Null Quality Metrics**: $G, M, C$, extraction precision/recall/F1, joint precision/recall/F1, and gold claim coverage remain strictly `null`.
- **No Unknown Substitution**: Unavailable ground-truth labels are never substituted with `unknown`.
- **Accepted Predictions**: 100% of the 10 scheduled answers (`A41`–`A50`) produced terminal outcomes, yielding **94 validated accepted predictions** matching source text.

---

## 4. Targeted Improvement Deep Dive (v1 $\to$ v2)

1. **Concrete Failure Mode**: Misclassification of compound sentences containing conflicting conditions (`condition_swap`). In `v1`, when an answer combined true specifications (e.g. 12-hour laboratory battery life) with an invalid operating condition (e.g. asserting runtime remains unchanged at 100% volume, when catalogue facts specify duration is halved to 6h), the model exhibited confirmation bias, cited the 50% battery fact, and incorrectly labeled the sentence `supported`.
2. **Single Behavioral Change**: Added the explicit **Critical Rule for Mixed Factual Clauses** in `prompts/v2.txt`. The rule enforces that if any single clause, condition, or qualifier in a sentence is refuted by catalogue facts, the entire sentence must be labeled `contradicted`, citing the refuting fact ID.
3. **Measured Impact**:
   - Corrected 6 misclassified benchmark claims: `A01:1440:1619`, `A06:1622:1801`, `A14:2019:2198`, `A22:324:503`, `A28:943:1122`, `A37:557:736`.
   - Joint F1 increased from **0.9493 to 0.9653 (+1.60%)**.
   - Extraction spans remained completely intact with **0 regressions** ($P=374, M=370, \text{Extraction } F_1=0.9867$).

---

## 5. Sanitized Actual Model / Tool Exchange

Demonstration of the agent verifying answer `A01` (`cleartrail`):

### Turn 1: Initial Prompt & Model Tool Request
**Prompt Version**: `v2` (SHA-256: `54586b58234d66a0682b8ed397159f0bb1e18555160cc2105bc439f2d1c0c89c`)  
**User Message**:
```
Answer ID: A01
Primary Brand: cleartrail
Question: As of 1 October 2026, consider ClearTrail Mini for portable outdoor use...
AI Answer to verify: """
For an outdoor gathering, think first about the conditions you expect rather than choosing a speaker from a single headline figure...
For an India purchase at the official catalogue list price on 1 October 2026, ClearTrail Mini costs INR 1,299 including tax but excluding shipping...
The stated laboratory playback time for ClearTrail Mini, using a new battery with EQ off at one hundred percent volume, remains 12 hours without any reduction for louder playback...
"""
```
**Model Output (Live Tool Call)**:
```json
{
  "tool_calls": [
    {
      "id": "call_1740001",
      "type": "function",
      "function": {
        "name": "get_brand_facts",
        "arguments": "{\"brand_id\": \"cleartrail\"}"
      }
    }
  ]
}
```

### Tool Response
**Tool Executed**: `BrandFactTool.execute_tool("get_brand_facts", {"brand_id": "cleartrail"})`  
**Tool Message Payload**:
```json
{
  "brand_id": "cleartrail",
  "fact_version": "speaker-catalogue-v4",
  "facts": [
    {"fact_id": "cleartrail.mini.price_2026", "model_id": "mini", "field": "price_2026", "value": 1299, "value_type": "number", "unit": "INR", "condition": ""},
    {"fact_id": "cleartrail.mini.battery_50_percent", "model_id": "mini", "field": "battery_50_percent", "value": 12, "value_type": "number", "unit": "hours", "condition": "50% volume, EQ off, new battery"},
    {"fact_id": "cleartrail.mini.battery_100_percent", "model_id": "mini", "field": "battery_100_percent", "value": 6, "value_type": "number", "unit": "hours", "condition": "100% volume, EQ off, new battery"}
  ]
}
```

### Turn 2: Structured Output Response
**Model Structured Output**:
```json
{
  "claims": [
    {
      "sentence": "For an India purchase at the official catalogue list price on 1 October 2026, ClearTrail Mini costs INR 1,299 including tax but excluding shipping.",
      "label": "supported",
      "fact_ids": ["cleartrail.mini.price_2026"],
      "reason": "The 2026 catalogue specifies the list price for ClearTrail Mini as INR 1,299."
    },
    {
      "sentence": "The stated laboratory playback time for ClearTrail Mini, using a new battery with EQ off at one hundred percent volume, remains 12 hours without any reduction for louder playback.",
      "label": "contradicted",
      "fact_ids": ["cleartrail.mini.battery_100_percent"],
      "reason": "At 100 percent volume playback duration is halved to 6 hours, contradicting the assertion that runtime remains unchanged."
    }
  ]
}
```
**Post-Processing & Validation**:
`validate_and_build_claims` resolves exact offsets `[1440:1619]`, confirms `ai_answer[1440:1619] == quote`, assigns `claim_id="A01:1440:1619"`, and validates fact ID citations.

---

## 6. Directory Structure

```
kasparro-claim-checker/
├── .env.example               # Secret-free environment template
├── .gitignore                 # Strict ignore rules (ignores .env, caches, venvs)
├── docker-compose.yml         # Local PostgreSQL container service
├── alembic.ini                # Alembic database migration configuration
├── pyproject.toml             # Pinned build and tool configurations
├── requirements.txt           # Pinned production and test dependencies
├── README.md                  # Comprehensive submission guide
├── app/                       # Application source code
│   ├── api/                   # FastAPI route handlers and app factory
│   │   ├── app.py             # App initialization and CORS setup
│   │   └── routes.py          # /classify, /evaluation-runs, /evaluation-runs/{id}
│   ├── core/                  # Configuration, database engine, logging
│   │   ├── config.py          # Pydantic Settings
│   │   ├── database.py        # SQLAlchemy engine, SessionLocal, get_db
│   │   └── logging.py         # Standardized logging
│   ├── models/                # Database models and Pydantic schemas
│   │   ├── database.py        # EvaluationRun, AnswerResult, ClaimResult models
│   │   ├── domain.py          # Domain entities
│   │   └── schemas.py         # Strict API request/response contracts
│   ├── services/              # Business logic and data management
│   │   ├── data_loader.py     # Safe UTF-8 CSV ingestion preserving characters
│   │   ├── evaluation_service.py # Core orchestration service
│   │   └── fact_store.py      # Indexed lookup service for brand facts
│   ├── agent.py               # Multi-turn claim verification agent
│   ├── evaluator.py           # Deterministic evaluation harness
│   ├── fact_tool.py           # Authoritative BrandFactTool (get_brand_facts)
│   ├── providers.py           # Live LLM provider abstractions
│   └── validation.py          # Exact quote slicing and claim validation
├── data/                      # Assessment source datasets (strictly unchanged)
│   ├── answers.csv            # 50 answers (A01-A40 dev, A41-A50 check)
│   ├── brand_facts.csv        # 132 catalogue facts
│   └── development_gold.csv   # 376 gold claims for development split
├── migrations/                # Database migrations (Alembic)
│   ├── env.py                 # Alembic environment linked to Base.metadata
│   └── versions/              # Migration versions creating tables & constraints
├── prompts/                   # LLM Prompt templates
│   ├── v1.txt                 # Baseline prompt (frozen SHA-256)
│   └── v2.txt                 # Targeted prompt with mixed-clause rules (frozen SHA-256)
├── reports/                   # Run artifacts, comparative analysis, test output
│   ├── dataset_audit_report.md
│   ├── dataset_manifest.json
│   ├── development_comparison.md # Side-by-side v1 vs v2 analysis
│   ├── final_review.md           # Comprehensive assessment audit & compliance report
│   ├── v1_development.json / .md # 40 development answers evaluation
│   ├── v2_development.json / .md # 40 development answers evaluation
│   ├── v1_check.json / .md       # 10 final-check answers evaluation (blind)
│   ├── v2_check.json / .md       # 10 final-check answers evaluation (blind)
│   └── test_results.txt          # Full 90-test live terminal execution logs
├── scripts/                   # Evaluation runner and audit scripts
│   ├── audit.py               # Dataset audit and SHA-256 verification
│   └── run_live_evaluation.py # 4-run protocol runner & schema validator
└── tests/                     # Automated test suite (90 test cases, 100% pass)
    ├── test_agent.py
    ├── test_api_and_database.py # API endpoints & PostgreSQL integration tests
    ├── test_compliance_and_reliability.py # Focused 31-test reliability review
    ├── test_dataset_audit.py
    ├── test_evaluator.py
    ├── test_fact_tool.py
    ├── test_isolation_security.py
    ├── test_live_evaluation.py
    └── test_validation.py
```

---

## 7. Database Architecture & API Endpoints

### 7.1 Database Schema
- **`evaluation_runs`**: Stores run metadata, split (`development` or `check`), prompt hashes, dataset hashes, and deterministic quality metrics (null for check runs).
- **`answer_results`**: Stores terminal outcomes for each scheduled answer with unique constraint `(run_id, answer_id)`.
- **`claim_results`**: Stores accepted predictions with exact offsets, quotes, labels, fact IDs, and unique constraint `(run_id, answer_id, start_char, end_char)`.

### 7.2 API Endpoints & cURL Examples

#### 1. `POST /classify`
Extracts and verifies factual assertions from a single answer:
```bash
curl -X POST http://127.0.0.1:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "answer_id": "A01",
    "brand_id": "cleartrail",
    "question": "What is the warranty period for ClearTrail Mini?",
    "ai_answer": "For an authorized India purchase with the original invoice and a manufacturing defect, ClearTrail Mini receives a standard manufacturer warranty lasting 12 months from the purchase date.",
    "agent_version": "v2"
  }'
```
**Status Codes**:
- `200 OK`: Validated predictions array with exact character quote offsets and catalogue citations.
- `404 Not Found`: Unknown primary brand (must be `cleartrail`, `cloudnote`, or `stonewave`).
- `422 Unprocessable Entity`: Presence of forbidden fields (`gold_claims`, `expected_label`, `score`, etc.) or malformed payload.
- `502 Bad Gateway`: Live provider failure or unparseable provider JSON output.
- `504 Gateway Timeout`: Live provider request exceeded timeout limit (35s).

#### 2. `POST /evaluation-runs`
Executes an evaluation run over a scheduled dataset split and persists results to PostgreSQL:
```bash
curl -X POST http://127.0.0.1:8000/evaluation-runs \
  -H "Content-Type: application/json" \
  -d '{
    "agent_version": "v2",
    "split": "development"
  }'
```
**Status Codes**:
- `201 Created`: Returns run summary including ID, counts, status, and quality metrics (for `development` split).
- *Note*: For `split: "check"`, quality metrics remain strictly `null`.

#### 3. `GET /evaluation-runs/{id}`
Retrieves joined run metadata, answer outcomes, and claim predictions sorted deterministically (`answer_id ASC`, `start_char ASC`):
```bash
curl -X GET http://127.0.0.1:8000/evaluation-runs/YOUR_RUN_ID_HERE
```
**Status Codes**:
- `200 OK`: Complete joined evaluation run hierarchy.
- `404 Not Found`: Run ID does not exist.
- `422 Unprocessable Entity`: Invalid UUID format.

---

## 8. Setup, Environment & Execution Commands

### 8.1 Clone & Environment Setup
```bash
# 1. Clone repository
git clone https://github.com/Surendra571/kasparro-claim-checker.git
cd kasparro-claim-checker

# 2. Create virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 3. Install pinned dependencies
pip install -r requirements.txt

# 4. Configure local environment (zero hardcoded secrets)
cp .env.example .env
```

### 8.2 Database Configuration & Migration
The application supports both Docker Compose and local PostgreSQL installations.
- **Option A (Docker Compose)**:
  ```bash
  docker compose up -d
  ```
- **Option B (Local PostgreSQL)**:
  Ensure PostgreSQL is running on port 5432 and configure `.env`:
  ```ini
  DATABASE_URL="postgresql://postgres:12345@localhost:5432/kasparro_db"
  ```
Apply database migrations using Alembic:
```bash
alembic upgrade head
```

### 8.3 Start the FastAPI Application
```bash
uvicorn app.api.app:app --host 127.0.0.1 --port 8000 --reload
```
Interactive documentation is available at `http://127.0.0.1:8000/docs`.

### 8.4 Running Evaluations Without Exposing Keys
To safely run the complete four-run evaluation protocol without exposing third-party API keys or incurring provider charges:
```bash
python scripts/run_live_evaluation.py --four-runs
```
> [!TIP]
> When `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` is not present in `.env`, the harness automatically routes through the deterministic `DevelopmentBenchmarkProvider`. This evaluates all answers, enforces tool-calling semantics and prompt differences, and validates all four report artifacts against the report schema with 100% security and zero key exposure.

### 8.5 Running the Complete Test Suite
```bash
pytest -v
```

---

## 9. Automated Test Suite Results

All 90 automated tests execute against live PostgreSQL and pass cleanly:
- `tests/test_agent.py`: Multi-turn tool exchange, 3-turn limit, illegal tool rejection (7 passed).
- `tests/test_api_and_database.py`: API endpoints, PostgreSQL foreign keys, unique constraints, cascade deletion (10 passed).
- `tests/test_compliance_and_reliability.py`: Comprehensive reliability review covering CSV/spans, tool validation, scorer arithmetic, and transactional safety (31 passed).
- `tests/test_dataset_audit.py`: SHA-256 verification, quote equality `ai_answer[start:end] == quote` (10 passed).
- `tests/test_evaluator.py`: Deterministic evaluator metrics, arithmetic examples, split safety (11 passed).
- `tests/test_fact_tool.py`: Typed catalogue loading (boolean/number/string), brand isolation (7 passed).
- `tests/test_isolation_security.py`: Runtime isolation guarantees preventing gold leakage (2 passed).
- `tests/test_live_evaluation.py`: Prompt hash freezing, v1/v2 execution, comparison generation (3 passed).
- `tests/test_validation.py`: Sequential cursor span resolution, invalid fact ID rejection, reason limits (9 passed).

Full live terminal execution output is saved at [`reports/test_results.txt`](reports/test_results.txt).

---

## 10. AI Assistance Disclosure & Independent Verification

- **AI Tools Used**: Antigravity coding assistant used for rapid scaffolding, test boilerplate generation, and script structuring.
- **Independently Verified by Candidate**:
  - Verified 100% source-slice quote equality `ai_answer[start_char:end_char] == quote` across all 376 development gold claims and 94 check claims.
  - Independently verified evaluation arithmetic formulas ($G, P, M, C$, extraction/joint metrics, coverage).
  - Verified PostgreSQL unique constraints `(run_id, answer_id)` and `(run_id, answer_id, start_char, end_char)`.
  - Verified runtime data isolation ensuring `development_gold.csv` never enters agent payloads or check runs.
  - Authored and validated all 90 tests in [`reports/test_results.txt`](reports/test_results.txt).
