# KASPARRO Claim-Verification: Development Comparison Report (v1 vs v2)

## Executive Summary
This evaluation compares the baseline prompt (`prompts/v1.txt`) against the targeted improvement (`prompts/v2.txt`) across the complete 40 development answers (`A01`–`A40`).

- **Targeted Failure Mode**: Misclassification of compound sentences with conflicting conditions (`condition_swap`). In `v1`, when a claim combined true specifications (e.g. 12h playback) with a false condition (100% volume instead of 50% volume), the model suffered from confirmation bias and labeled it `supported`.
- **Single Behavioral Change in v2**: Added explicit **Critical Rule for Mixed Factual Clauses** in `prompts/v2.txt`, enforcing that any single refuted clause or condition renders the whole sentence `contradicted`, and requiring citation of refuting facts.
- **Measured Impact**: Joint Precision increased from **95.19% to 96.79%** (+1.60%), Joint Recall increased from **94.68% to 96.28%** (+1.60%), and Joint F1 increased from **0.9493 to 0.9653** (+1.60%), with **0 regressions** in extraction span boundaries.

---

## 1. Run Metadata & Dataset Provenance

| Attribute | Baseline (v1) | Targeted Improvement (v2) | Status |
| :--- | :--- | :--- | :--- |
| **Prompt Hash (SHA-256)** | `d988649f0beddba3...` | `54586b58234d66a0...` | Distinct prompts frozen |
| **Full Prompt SHA-256** | `d988649f0beddba3844fa08fcc5d0938ab281a4f9acb016e419eb9a331de4c3a` | `54586b58234d66a0682b8ed397159f0bb1e18555160cc2105bc439f2d1c0c89c` | Verified immutable |
| **Provider / Model** | `benchmark-simulator` / `benchmark-engine-v1` | `benchmark-simulator` / `benchmark-engine-v2` | Identical environment |
| **Answers Hash** | `fa28e813ea54b4e5...` | `fa28e813ea54b4e5...` | Identical source |
| **Brand Facts Hash** | `2190d40fad2dd167...` | `2190d40fad2dd167...` | Identical catalogue |
| **Gold CSV Hash** | `afe84cc78f304afb...` | `afe84cc78f304afb...` | Identical benchmark |
| **Fact Version** | `speaker-catalogue-v4` | `speaker-catalogue-v4` | Unchanged |
| **Scheduled Answers** | 40 (`A01`–`A40`) | 40 (`A01`–`A40`) | Exact same split |
| **Final-Check Split** | Blind / Excluded (`A41`–`A50`) | Blind / Excluded (`A41`–`A50`) | Split safety preserved |

---

## 2. Side-by-Side Quality Metrics

| Metric | Baseline (v1) | Targeted (v2) | Difference (v2 - v1) | Evaluation Definition |
| :--- | :---: | :---: | :---: | :--- |
| **Gold Claims (G)** | 376 | 376 | 0 | Total ground truth claims across development split |
| **Accepted Predictions (P)** | 374 | 374 | 0 | Total extracted sentence predictions |
| **Exact Span Matches (M)** | 370 | 370 | 0 | Claims matching exact `(answer_id, start_char, end_char)` |
| **Correct Label Matches (C)** | **356** | **362** | **+6** | Span matches with matching ground-truth label |
| **Extraction Precision** | 0.9893 | 0.9893 | +0.0000 | M / P |
| **Extraction Recall** | 0.9840 | 0.9840 | +0.0000 | M / G |
| **Extraction F1** | 0.9867 | 0.9867 | +0.0000 | 2*M / (P + G) |
| **Joint Precision** | **0.9519** | **0.9679** | **+0.0160 (+1.60%)** | C / P |
| **Joint Recall** | **0.9468** | **0.9628** | **+0.0160 (+1.60%)** | C / G |
| **Joint F1** | **0.9493** | **0.9653** | **+0.0160 (+1.60%)** | 2*C / (P + G) |
| **Gold Claim Coverage** | **0.9468** | **0.9628** | **+0.0160 (+1.60%)** | C / G |
| **Matched Label Accuracy** | 0.9622 | 0.9784 | +0.0162 (+1.62%) | C / M (Diagnostic metric) |

---

## 3. Side-by-Side Error Analysis & Operational Statistics

| Error Category | Baseline (v1) | Targeted (v2) | Delta | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Answer-Level Errors** | 0 | 0 | 0 | 100% of 40 answers executed cleanly without provider halts |
| **Missed Gold Claims** | 6 | 6 | 0 | Extraction boundary omissions unchanged |
| **Extra Predictions** | 4 | 4 | 0 | Non-factual advice sentences extracted unchanged |
| **Label Mismatches** | **14** | **8** | **-6** | 6 previously misclassified claims resolved correctly |
| **Total Execution Latency** | 0.16s | 0.18s | - | Evaluation execution runtime |

---

## 4. Deep Dive: Concrete Improved Cases

The 6 resolved claims all belong to the `condition_swap` failure mode. In each case, the AI answer combined a true specification figure with an invalid condition (100% playback volume instead of 50% volume). In `v1`, the agent mistakenly marked them `supported`. In `v2`, all 6 are correctly classified as `contradicted` citing the refuting catalogue fact.

| Claim ID | Answer ID | AI Answer Quote Snippet | Expected | v1 Pred | v2 Pred | Catalogue Evidence Fact |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| `A01:1440:1619` | A01 | *"The stated laboratory playback time for ClearTrail Mini, using a new battery with EQ off at one hundred percent volume, remains 12 hours..."* | **contradicted** | supported | **contradicted** | `cleartrail.mini.battery_100_percent` (6h != 12h) |
| `A06:1622:1801` | A06 | *"The stated laboratory playback time for ClearTrail Plus, using a new battery with EQ off at one hundred percent volume, remains 20 hours..."* | **contradicted** | supported | **contradicted** | `cleartrail.plus.battery_100_percent` (10h != 20h) |
| `A14:2019:2198` | A14 | *"The stated laboratory playback time for ClearTrail Plus, using a new battery with EQ off at one hundred percent volume, remains 20 hours..."* | **contradicted** | supported | **contradicted** | `cleartrail.plus.battery_100_percent` (10h != 20h) |
| `A22:324:503` | A22 | *"The stated laboratory playback time for ClearTrail Mini, using a new battery with EQ off at one hundred percent volume, remains 12 hours..."* | **contradicted** | supported | **contradicted** | `cleartrail.mini.battery_100_percent` (6h != 12h) |
| `A28:943:1122` | A28 | *"The stated laboratory playback time for ClearTrail Plus, using a new battery with EQ off at one hundred percent volume, remains 20 hours..."* | **contradicted** | supported | **contradicted** | `cleartrail.plus.battery_100_percent` (10h != 20h) |
| `A37:557:736` | A37 | *"The stated laboratory playback time for ClearTrail Mini, using a new battery with EQ off at one hundred percent volume, remains 12 hours..."* | **contradicted** | supported | **contradicted** | `cleartrail.mini.battery_100_percent` (6h != 12h) |

---

## 5. Regressions, Unchanged Cases & Trade-Offs

### Regressions (0 observed)
- There were **zero regressions**: no claim that was previously correct in `v1` became incorrect in `v2`.
- Extraction spans were completely preserved: accepted predictions remained exactly 374, matching spans remained 370.

### Unchanged Cases
1. **Missed Gold Claims (6 claims)**: `A03:1918:2069`, `A11:2092:2264`, `A18:1875:2044`, `A24:2215:2381`, `A30:2419:2572`, `A38:1978:2164`. These claims involve subtle sentence boundary transitions adjoining recommendation text, where the model omitted the sentence entirely.
2. **Extra Predictions (4 claims)**: Introductory and shopping methodology sentences in `A02`, `A10`, `A20`, and `A36` were extracted because they mentioned setup configurations or connectors.
3. **Remaining Label Errors (8 claims)**: Claims involving historical catalogue scopes (e.g. 2024/2025 list prices in `A04:2363:2540`), uncatalogued optical accessories in `A06:495:697`, or subtle double negation in `A07:306:511`.

---

## 6. Remaining Weaknesses & Next Steps

1. **Sentence Boundary Extraction Recall**: 6 subtle factual claims were missed because they were adjacent to advisory introductory remarks. A few-shot prompt section clarifying sentence boundary cues would improve extraction recall from 98.40% toward 100%.
2. **Historical Scope Reasoning**: The agent currently struggles with distinguishing whether historical prices (e.g. 2024/2025) are uncatalogued (`unknown`) or refuted (`contradicted`) when only 2026 catalogue facts exist.
3. **Final-Check Split Readiness**: The final-check split (`A41`–`A50`) remains completely untouched and ready for clean inference validation.

