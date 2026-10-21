# KASPARRO AI Claim-Extraction and Fact-Verification Service

**GitHub Repository**: [https://github.com/Surendra571/kasparro-claim-checker](https://github.com/Surendra571/kasparro-claim-checker)  
**Role**: KASPARRO FSE Intern Pathway Assessment  
**Python Compatibility**: Python >= 3.11 (tested on Python 3.12.10)  
**Database**: PostgreSQL 17 / 16 (local or Docker Compose)  

---

## 1. Project Overview

The **KASPARRO AI Claim-Extraction and Fact-Verification Service** is a production-grade backend service designed to parse AI-generated answers, extract complete factual assertions with exact character quote boundaries, verify each claim against an authoritative brand facts knowledge base, and record deterministic evaluation runs in PostgreSQL.

Key capabilities:
- **Exact Span Matching**: Quote validation ensures `ai_answer[start_char:end_char] == quote` with 100% precision.
- **Strict Multi-Turn Agent**: Enforces 3-turn limit, single read-only tool `get_brand_facts(brand_id)`, and strict multi-clause precedence rules (`prompts/v2.txt`).
- **PostgreSQL Persistence**: Persistent tables (`evaluation_runs`, `answer_results`, `claim_results`) with foreign keys, cascading deletes, unique constraints, and JSONB storage.
- **Split Safety**: Development split (`A01`–`A40`) metrics are computed and stored deterministically; check split (`A41`–`A50`) quality metrics and ground truth match fields remain strictly `NULL`.
- **FastAPI Endpoints**: REST API with strict request validation (rejects unauthorized gold/score fields with 422), sanitized error mapping (404, 502, 504), and stable ordered retrieval.

---

## 2. Directory Structure

```
kasparro-claim-checker/
├── .env.example               # Template for environment settings (zero hardcoded secrets)
├── .env                       # Local active environment configuration
├── docker-compose.yml         # Local PostgreSQL container service
├── alembic.ini                # Alembic database migration configuration
├── pyproject.toml             # Project build configuration & tool settings
├── requirements.txt           # Pinned production and test dependencies
├── README.md                  # Project documentation and API guide
├── app/                       # Application source code
│   ├── api/                   # FastAPI route handlers and app factory
│   │   ├── app.py             # App initialization and CORS setup
│   │   └── routes.py          # /classify, /evaluation-runs, /evaluation-runs/{id}
│   ├── core/                  # Core configuration, database engine, logging
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
├── reports/                   # Run artifacts, comparative analysis, data audit
│   ├── dataset_audit_report.md
│   ├── dataset_manifest.json
│   ├── development_comparison.md # Side-by-side v1 vs v2 analysis
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

## 3. Database Architecture & Persistence

### 3.1 Tables
- **`evaluation_runs`**:
  - `id` (UUID Primary Key)
  - `split` ("development" or "check")
  - `agent_version` ("v1" or "v2")
  - `status` ("completed", "failed", "processing")
  - `provider`, `model`, `prompt_version`, `prompt_hash`, `fact_version`, `evaluator_version`, `source_revision`
  - `settings` (JSONB), `dataset_hashes` (JSONB)
  - `scheduled_count`, `evaluated_count`, `successful_count`, `failed_count`, `duration_seconds`
  - Quality metrics: `gold_claims_g`, `accepted_predictions_p`, `exact_span_matches_m`, `correct_label_matches_c`, `extraction_precision`, `extraction_recall`, `extraction_f1`, `joint_precision`, `joint_recall`, `joint_f1`, `gold_claim_coverage`, `matched_label_accuracy`, `error_breakdown` (JSONB).
  - *Constraint*: All quality metrics remain strictly `NULL` for `check` split runs.

- **`answer_results`**:
  - `id` (UUID Primary Key)
  - `run_id` (UUID Foreign Key to `evaluation_runs.id` with `ON DELETE CASCADE`)
  - `answer_id`, `brand_id`, `success` (Boolean), `duration_seconds`, `error_type`, `error_message`, `trace_reference`, `claims_count`
  - *Unique Constraint*: `(run_id, answer_id)`

- **`claim_results`**:
  - `id` (UUID Primary Key)
  - `run_id` (UUID Foreign Key to `evaluation_runs.id` with `ON DELETE CASCADE`)
  - `answer_id`, `claim_id` (`{answer_id}:{start}:{end}`), `start_char`, `end_char`, `quote`, `label`, `fact_ids` (JSONB), `reason`
  - Ground truth match info (populated only for development split; strictly `NULL` for check split): `is_exact_match`, `is_label_correct`, `gold_claim_id`, `gold_expected_label`
  - *Unique Constraint*: `(run_id, answer_id, start_char, end_char)`
  - *Index*: `(run_id, answer_id, start_char)` for stable sorted retrieval.

---

## 4. API Endpoints

### 4.1 `POST /classify`
Extracts and verifies factual claims from a single answer.
- **Headers**: `Content-Type: application/json`
- **Request Body**:
  ```json
  {
    "answer_id": "A01",
    "brand_id": "cleartrail",
    "question": "What is the warranty period for ClearTrail Mini?",
    "ai_answer": "For an authorized India purchase with the original invoice and a manufacturing defect, ClearTrail Mini receives a standard manufacturer warranty lasting 18 months from the purchase date.",
    "agent_version": "v1"
  }
  ```
- **Responses**:
  - `200 OK`: Validated predictions with exact character spans and catalogue fact citations.
  - `404 Not Found`: Unknown primary brand (must be one of `cleartrail`, `cloudnote`, `stonewave`).
  - `422 Unprocessable Entity`: Malformed request or presence of forbidden fields (`gold_claims`, `expected_label`, `score`, etc.).
  - `502 Bad Gateway`: Model provider failure or invalid provider JSON output.
  - `504 Gateway Timeout`: Model provider request timeout.

### 4.2 `POST /evaluation-runs`
Executes an evaluation run over a fixed dataset split and persists all outcomes.
- **Request Body**:
  ```json
  {
    "agent_version": "v2",
    "split": "development"
  }
  ```
- **Responses**:
  - `201 Created`: Returns run summary including ID, counts, status, and quality metrics (for `development` split).
  - *Note*: If `split` is `"check"`, all quality metrics remain `null`.

### 4.3 `GET /evaluation-runs/{id}`
Retrieves joined run metadata, answer outcomes, and claim predictions.
- **Ordering**: Answers ordered by `answer_id ASC`, claims ordered by `start_char ASC`.
- **Responses**:
  - `200 OK`: Full joined evaluation run hierarchy.
  - `404 Not Found`: Run ID does not exist.
  - `422 Unprocessable Entity`: Invalid UUID format.

---

## 5. Setup & Execution Commands

### 5.1 Clone the Repository
```bash
git clone https://github.com/Surendra571/kasparro-claim-checker.git
cd kasparro-claim-checker
```

### 5.2 Start PostgreSQL
#### Option A: Docker Compose
```bash
docker compose up -d
```
#### Option B: Local Windows PostgreSQL Service
Ensure PostgreSQL is running on port 5432 and configure `.env`:
```ini
DATABASE_URL="postgresql://postgres:12345@localhost:5432/kasparro_db"
```

### 5.3 Apply Database Migrations
Run Alembic migrations to create tables and constraints in PostgreSQL:
```bash
alembic upgrade head
```

### 5.4 Run the FastAPI Application
```bash
uvicorn app.api.app:app --host 127.0.0.1 --port 8000 --reload
```
Interactive documentation is available at `http://127.0.0.1:8000/docs`.

### 5.5 Execute the Four-Run Protocol
Executes v1 (dev), v2 (dev), v1 (check), v2 (check), generates the comparison report, and validates all four artifacts against the report schema:
```bash
python scripts/run_live_evaluation.py --four-runs
```

### 5.6 Run the Complete Automated Test Suite
```bash
pytest -v
```

---

## 6. Test Suite Results

All 90 automated tests execute against live PostgreSQL and pass cleanly (100% pass rate):
- `tests/test_agent.py`: Multi-turn tool exchange, 3-turn limit, illegal tool rejection, error preservation (7 passed).
- `tests/test_api_and_database.py`: `/classify`, `/evaluation-runs`, `/evaluation-runs/{id}`, PostgreSQL foreign keys, unique constraints, cascade deletion, error mappings (502, 504), split safety (10 passed).
- `tests/test_compliance_and_reliability.py`: Comprehensive reliability and compliance review covering CSV/spans, tool validation, scorer arithmetic, and transactional safety (31 passed).
- `tests/test_dataset_audit.py`: SHA-256 verification, quote equality `ai_answer[start:end] == quote` (10 passed).
- `tests/test_evaluator.py`: Deterministic evaluator metrics, arithmetic examples, split safety (11 passed).
- `tests/test_fact_tool.py`: Typed catalogue loading (boolean/number/string), brand isolation (7 passed).
- `tests/test_isolation_security.py`: Runtime isolation guarantees preventing gold leakage (2 passed).
- `tests/test_live_evaluation.py`: Prompt hash freezing, v1/v2 execution, comparison generation (3 passed).
- `tests/test_validation.py`: Sequential cursor span resolution, invalid fact ID rejection, reason sentence limits (9 passed).
