# KASPARRO Claim-Verification: Final-Check Evaluation (V2)

**Run Status**: COMPLETED (10/10 Answers)  
**Date**: 2026-10-21T11:15:47.970637+00:00  
**Split**: `final-check` (Answers `A41`–`A50`)  
**Provider**: `benchmark-simulator` | **Model**: `benchmark-engine-v2`  
**Prompt Version**: `v2` (SHA-256: `54586b58234d66a0682b8ed397159f0bb1e18555160cc2105bc439f2d1c0c89c`)  
**Fact Version**: `speaker-catalogue-v4`  
**Total Run Duration**: 0.03s  

---

## 1. Ground-Truth Independence & Metric Status

> [!NOTE]
> In strict accordance with the assessment specification, all ground-truth-dependent metrics
> (Extraction/Joint Precision, Recall, F1, and Gold Coverage) are strictly `null`.
> The final-check split is unlabelled and evaluated blindly using only `answers.csv` and `brand_facts.csv`.

| Evaluation Metric | Value | Assessment Policy |
| :--- | :---: | :--- |
| **Gold Claims (G)** | `null` | Ground-truth labels unavailable on final-check split |
| **Accepted Predictions (P)** | **94** | Fully validated claim predictions matching source text |
| **Exact Span Matches (M)** | `null` | Gold spans unavailable on final-check split |
| **Correct Label Matches (C)** | `null` | Expected labels unavailable on final-check split |
| **Extraction Precision / Recall / F1** | `null` | Metric undefined without ground-truth |
| **Joint Precision / Recall / F1** | `null` | Metric undefined without ground-truth |
| **Gold Claim Coverage** | `null` | Metric undefined without ground-truth |

---

## 2. Operational Statistics

- **Total Scheduled Answers**: 10 (`A41`–`A50`)
- **Successfully Evaluated Answers**: 10
- **Failed Answers**: 0
- **Total Accepted Predictions**: 94
- **Average Predictions per Answer**: 9.4

### Predictions Count by Answer:
| Answer ID | Status | Predictions Count | Duration |
| :--- | :---: | :---: | :---: |
| `A41` | success | 9 | 0.002s |
| `A42` | success | 10 | 0.002s |
| `A43` | success | 9 | 0.002s |
| `A44` | success | 10 | 0.003s |
| `A45` | success | 9 | 0.002s |
| `A46` | success | 11 | 0.002s |
| `A47` | success | 8 | 0.003s |
| `A48` | success | 10 | 0.004s |
| `A49` | success | 8 | 0.004s |
| `A50` | success | 10 | 0.002s |

