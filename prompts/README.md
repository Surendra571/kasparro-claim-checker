 # Prompts Directory

This directory stores prompts used by the AI Claim Checker service.
In accordance with assessment rules:
- Prompts use clear instruction boundaries and strict output JSON schemas.
- Offsets (`start_char` and `end_char`) must refer directly to character indices in the verbatim `ai_answer`.
- Ground truth (`development_gold.csv`) is never fed into prompts.

### Contents:
- `claim_extraction_prompt.txt`: Prompt template for segmenting `ai_answer` into atomic factual claims with exact character slices.
- `fact_verification_prompt.txt`: Prompt template for assessing whether a claim is `supported`, `contradicted`, or `unknown` given retrieved brand facts.
